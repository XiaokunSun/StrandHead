import torch
from pytorch3d.io import load_obj
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.structures import Meshes
from torch import nn
from torch.nn import functional as F

import itertools
import pickle

import numpy as np
import torchvision

from .texture import UNet
from .strand_prior import Decoder

from torchvision.transforms import functional as TF
import sys

import accelerate
from copy import deepcopy
import os
import trimesh
import cv2

import cubvh
from src.utils.util import param_to_buffer, positional_encoding
from src.utils.geometry import barycentric_coordinates_of_projection, face_vertices
from src.utils.sample_points_from_meshes import sample_points_from_meshes


def downsample_texture(rect_size, downsample_size):
        b = torch.linspace(0, rect_size**2 - 1, rect_size**2).reshape(rect_size, rect_size)
        
        patch_size = rect_size // downsample_size
        unf = torch.nn.Unfold(
            kernel_size=patch_size,
            stride=patch_size
                             )
        unfo = unf(b[None, None]).reshape(-1, downsample_size**2)
        idx = torch.randint(low=0, high=patch_size**2, size=(1,))
        idx_ = idx.repeat(downsample_size**2,)
        choosen_val = unfo[idx_, :].diag()
        x = choosen_val // rect_size
        y = choosen_val % rect_size 
        return x.long(), y.long()

class OptimizableTexturedStrands(nn.Module):
    def __init__(self, 
                 path_to_mesh, 
                 num_strands,
                 max_num_strands,
                 texture_size,
                 geometry_descriptor_size,
                 appearance_descriptor_size,
                 decoder_checkpoint_path,
                 path_to_scale=None,
                 cut_scalp=None, 
                 diffusion_cfg=None,
                 scalp_vertex_idx_path=None,
                 scalp_faces_path=None,
                 scalp_uvcoords_path=None
                 ):
        super().__init__()
    
        scalp_vert_idx = torch.load(scalp_vertex_idx_path).long().cuda() # indices of scalp vertices
        scalp_faces = torch.load(scalp_faces_path)[None].cuda() # faces that form a scalp
        scalp_uvs = torch.load(scalp_uvcoords_path).cuda()[None] # generated in Blender uv map for the scalp

        # Load FLAME head mesh
        verts, faces, _ = load_obj(path_to_mesh, device='cuda')
        
        # Transform head mesh if it's not in unit sphere (same scale used for world-->unit_sphere transform)
        self.transform = None
        if path_to_scale:
            with open(path_to_scale, 'rb') as f:
                self.transform = pickle.load(f)
            verts = (verts - torch.tensor(self.transform['translation'], device=verts.device)) / self.transform['scale']
       
            
        head_mesh =  Meshes(verts=[(verts)], faces=[faces.verts_idx]).cuda()
        
        # Scaling factor, as decoder pretrained on synthetic data with fixed head scale
        usc_scale = torch.tensor([[0.2579, 0.4082, 0.2580]]).cuda()
        head_scale = head_mesh.verts_packed().max(0)[0] - head_mesh.verts_packed().min(0)[0]
        head_scale = head_scale[[1,2,0]]
        self.scale_decoder = (usc_scale / head_scale).mean()
        
        scalp_verts = head_mesh.verts_packed()[None, scalp_vert_idx]
        scalp_face_verts = face_vertices(scalp_verts, scalp_faces)[0]
        
        # Extract scalp mesh from head
        self.scalp_mesh = Meshes(verts=scalp_verts, faces=scalp_faces, textures=TexturesVertex(scalp_uvs)).cuda()
        
        # If we want to use different scalp vertices for scene
        if cut_scalp:
            with open(cut_scalp, 'rb') as f:
                full_scalp_list = sorted(pickle.load(f))
                
            a = np.array(full_scalp_list)
            b = np.arange(a.shape[0])
            d = dict(zip(a, b))
            
            faces_masked = []
            for face in self.scalp_mesh.faces_packed():
                if face[0] in full_scalp_list and face[1] in full_scalp_list and  face[2] in full_scalp_list:
                    faces_masked.append(torch.tensor([d[int(face[0])], d[int(face[1])], d[int(face[2])]]))

            scalp_uvs = scalp_uvs[:, full_scalp_list]
            self.scalp_mesh = Meshes(verts=self.scalp_mesh.verts_packed()[None, full_scalp_list].float(), faces=torch.stack(faces_masked)[None].cuda(), textures=TexturesVertex(scalp_uvs)).cuda()
        
        self.scalp_mesh.textures = TexturesVertex(scalp_uvs)

        self.num_strands = num_strands
        self.max_num_strands = max_num_strands
        self.geometry_descriptor_size = geometry_descriptor_size
        self.appearance_descriptor_size = appearance_descriptor_size

        mgrid = torch.stack(torch.meshgrid([torch.linspace(-1, 1, texture_size)]*2))[None].cuda()
        self.register_buffer('encoder_input', positional_encoding(mgrid, 6))
        
        # Initialize the texture decoder network
        self.geo_texture_decoder = UNet(self.encoder_input.shape[1], geometry_descriptor_size, bilinear=True)

        self.app_texture_decoder = UNet(self.encoder_input.shape[1], appearance_descriptor_size, bilinear=True)
        

        self.register_buffer('local2world', self.init_scalp_basis(scalp_uvs))

        self.register_buffer('local2world_customization', self.init_scalp_basis(scalp_uvs))

        # Sample fixed origin points
        origins, uvs, face_idx = sample_points_from_meshes(self.scalp_mesh, num_samples=max_num_strands, return_textures=True)
        self.register_buffer('origins', origins[0])
        self.register_buffer('uvs', uvs[0])

        # Get transforms for the samples
        self.local2world.data = self.local2world[face_idx[0]]
        
        # For uniform faces selection
        self.N_faces =  self.scalp_mesh.faces_packed()[None].shape[1]
        self.m, self.q = num_strands // self.N_faces, num_strands % self.N_faces
        
        self.faces_dict = {}
        for idx, f in enumerate(face_idx[0].cpu().numpy()):
            try:
                self.faces_dict[f].append(idx)
            except KeyError:
                self.faces_dict[f] = [idx]
        
        idxes, counts = face_idx[0].unique(return_counts=True)
        self.faces_count_dict = dict(zip(idxes.cpu().numpy(), counts.cpu().numpy()))
        
        # Decoder predicts the strands from the embeddings
        self.strand_decoder = Decoder(None, latent_dim=geometry_descriptor_size, length=99).eval()
        self.strand_decoder.load_state_dict(torch.load(decoder_checkpoint_path)['decoder'])
        param_to_buffer(self.strand_decoder)
    
    def init_scalp_basis(self, scalp_uvs):         

        scalp_verts, scalp_faces = self.scalp_mesh.verts_packed()[None], self.scalp_mesh.faces_packed()[None]
        scalp_face_verts = face_vertices(scalp_verts, scalp_faces)[0] 
        
        # Define normal axis
        origin_v = scalp_face_verts.mean(1)
        origin_n = self.scalp_mesh.faces_normals_packed()
        origin_n /= origin_n.norm(dim=-1, keepdim=True)
        
        # Define tangent axis
        full_uvs = scalp_uvs[0][scalp_faces[0]]
        bs = full_uvs.shape[0]
        concat_full_uvs = torch.cat((full_uvs, torch.zeros(bs, full_uvs.shape[1], 1, device=full_uvs.device)), -1)
        new_point = concat_full_uvs.mean(1).clone()
        new_point[:, 0] += 0.001
        bary_coords = barycentric_coordinates_of_projection(new_point, concat_full_uvs).unsqueeze(1)
        full_verts = scalp_verts[0][scalp_faces[0]]
        origin_t = (bary_coords @ full_verts).squeeze(1) - full_verts.mean(1)
        origin_t /= origin_t.norm(dim=-1, keepdim=True)
        
        assert torch.where((bary_coords.reshape(-1, 3) > 0).sum(-1) != 3)[0].shape[0] == 0
        
        # Define bitangent axis
        origin_b = torch.cross(origin_n, origin_t, dim=-1)
        origin_b /= origin_b.norm(dim=-1, keepdim=True)

        # Construct transform from global to local (for each point)
        R = torch.stack([origin_t, origin_b, origin_n], dim=1) 
        
        # local to global 
        R_inv = torch.linalg.inv(R) 
        
        return R_inv
        
    def init_strand(self, target_texture, target_uv, iter_num=1000, sample_num=10000, lr=0.001):
        optim = torch.optim.Adam(self.parameters(), lr=lr)
        optim.zero_grad()

        for _ in range(iter_num):
            origins, q_source_uv, _ = sample_points_from_meshes(self.scalp_mesh, num_samples=sample_num, return_textures=True)

            mesh_BVH = cubvh.cuBVH(self.scalp_mesh.verts_packed().detach().cpu().numpy(), self.scalp_mesh.faces_packed().detach().cpu().numpy())
            map_dist, map_face, map_uvw = mesh_BVH.signed_distance(origins[0], return_uvw=True, mode="raystab")

            q_target_uv = (target_uv[self.scalp_mesh.faces_packed().detach()[map_face], :] * map_uvw[:, :, None]).sum(1)

            source_texture = self.geo_texture_decoder(self.encoder_input)

            pre = F.grid_sample(source_texture, q_source_uv[0][None, None])[0, :, 0].transpose(0, 1)
            gt = F.grid_sample(target_texture, q_target_uv[None, None])[0, :, 0].transpose(0, 1)
            loss = F.mse_loss(pre, gt)
            print(loss.item())
            optim.zero_grad()
            loss.backward()
            optim.step()
        

    def forward(self, it=None, num_strands=None, sample_mode="uniform", run_batch=3000):

        if num_strands is None:
            num_strands = self.num_strands
        else:
            self.num_strands = num_strands
            self.m, self.q = num_strands // self.N_faces, num_strands % self.N_faces
        
        # Generate texture
        geo_texture = self.geo_texture_decoder(self.encoder_input)
        app_texture = self.app_texture_decoder(self.encoder_input)
        texture_res = geo_texture.shape[-1]
        
        # Use diffusion prior
        diffusion_dict = {}

        # Sample idxes from texture
        if sample_mode == "uniform":
            # If the #sampled strands > #scalp faces, then we try to sample more uniformly for better convergence
            f_idx, count = torch.cat((torch.arange(self.N_faces).repeat(self.m), torch.randperm(self.N_faces)[:self.q])).unique(return_counts=True)
            
            current_iter =  dict(zip(f_idx.cpu().numpy(), count.cpu().numpy()))
            iter_idx = []

            for i in range(self.N_faces):
                cur_idx_list = torch.tensor(self.faces_dict[i])[torch.randperm(self.faces_count_dict[i])[:current_iter[i]]].tolist()
                iter_idx.append(cur_idx_list)
            idx = torch.tensor(list(itertools.chain(*iter_idx)))
        elif sample_mode == "random":
            idx = torch.randperm(self.max_num_strands, device=geo_texture.device)[:self.num_strands]

        origins = self.origins[idx]
        uvs = self.uvs[idx]
        local2world = self.local2world[idx]

        # Get latents for the samples
        z_geom = F.grid_sample(geo_texture, uvs[None, None])[0, :, 0].transpose(0, 1)
        z_app = F.grid_sample(app_texture, uvs[None, None])[0, :, 0].transpose(0, 1)

        strands_list = []
        r = 0
        for i in range(self.num_strands // run_batch):
            l, r = i * run_batch, (i+1) * run_batch
            z_geom_batch = z_geom[l:r]
            v = self.strand_decoder(z_geom_batch) / self.scale_decoder # [num_strands, strand_length - 1, 3]
        
            p_local = torch.cat([
                    torch.zeros_like(v[:, -1:, :]), 
                    torch.cumsum(v, dim=1)
                ], 
                dim=1
            )
            p = (local2world[l:r][:, None] @ p_local[..., None])[:, :, :3, 0] + origins[l:r][:, None] # [num_strands, strang_length, 3]
            strands_list.append(p)

        if r != self.num_strands:
            l = r
            r = self.num_strands
            z_geom_batch = z_geom[l:r]
            v = self.strand_decoder(z_geom_batch) / self.scale_decoder # [num_strands, strand_length - 1, 3]
        
            p_local = torch.cat([
                    torch.zeros_like(v[:, -1:, :]), 
                    torch.cumsum(v, dim=1)
                ], 
                dim=1
            )
            p = (local2world[l:r][:, None] @ p_local[..., None])[:, :, :3, 0] + origins[l:r][:, None] # [num_strands, strang_length, 3]
            strands_list.append(p)
        
        return torch.cat(strands_list, dim=0), z_geom, z_app, diffusion_dict

    def forward_customization(self, origins, uvs, face_idx, run_batch=3000, scale_factor=1):
        # To sample more strands at inference stage
        geo_texture = self.geo_texture_decoder(self.encoder_input)
        app_texture = self.app_texture_decoder(self.encoder_input)

        local2world = self.local2world_customization[face_idx]
        num_strands = origins.shape[0]
        # Get latents for the samples
        z_geom = F.grid_sample(geo_texture, uvs[None, None])[0, :, 0].transpose(0, 1) 
        z_app = F.grid_sample(app_texture, uvs[None, None])[0, :, 0].transpose(0, 1) 

        strands_list = []
        r = 0
        for i in range(num_strands // run_batch):
            l, r = i * run_batch, (i+1) * run_batch
            z_geom_batch = z_geom[l:r]
            v = self.strand_decoder(z_geom_batch) / scale_factor # [num_strands, strand_length - 1, 3]
        
            p_local = torch.cat([
                    torch.zeros_like(v[:, -1:, :]), 
                    torch.cumsum(v, dim=1)
                ], 
                dim=1
            )
            p = (local2world[l:r][:, None] @ p_local[..., None])[:, :, :3, 0] + origins[l:r][:, None] # [num_strands, strang_length, 3]
            strands_list.append(p)

        if r != num_strands:
            l = r
            r = num_strands
            z_geom_batch = z_geom[l:r]
            v = self.strand_decoder(z_geom_batch) / scale_factor # [num_strands, strand_length - 1, 3]
        
            p_local = torch.cat([
                    torch.zeros_like(v[:, -1:, :]), 
                    torch.cumsum(v, dim=1)
                ], 
                dim=1
            )
            p = (local2world[l:r][:, None] @ p_local[..., None])[:, :, :3, 0] + origins[l:r][:, None] # [num_strands, strang_length, 3]
            strands_list.append(p)
        return torch.cat(strands_list, dim=0), z_geom, z_app, geo_texture, app_texture
    
    def forward_inference(self, num_strands=None, sample_mode="uniform", run_batch=3000): 
        
        # To sample more strands at inference stage
        geo_texture = self.geo_texture_decoder(self.encoder_input)
        app_texture = self.app_texture_decoder(self.encoder_input)
    
        if num_strands is None:
            num_strands = self.num_strands
            tmp_m, tmp_q = self.m, self.q
        else:
            tmp_m, tmp_q = num_strands // self.N_faces, num_strands % self.N_faces

        # Sample from the fixed origins
        torch.manual_seed(0)
        # Sample idxes from texture
        if sample_mode == "uniform":
            # If the #sampled strands > #scalp faces, then we try to sample more uniformly for better convergence
            f_idx, count = torch.cat((torch.arange(self.N_faces).repeat(tmp_m), torch.randperm(self.N_faces)[:tmp_q])).unique(return_counts=True)

            current_iter =  dict(zip(f_idx.cpu().numpy(), count.cpu().numpy()))
            iter_idx = []
            for i in range(self.N_faces):
                cur_idx_list = torch.tensor(self.faces_dict[i])[torch.randperm(self.faces_count_dict[i])[:current_iter[i]]].tolist()
                iter_idx.append(cur_idx_list)
            idx = torch.tensor(list(itertools.chain(*iter_idx)))
        elif sample_mode == "random":
            idx = torch.randperm(self.max_num_strands, device=geo_texture.device)[:num_strands]

        origins = self.origins[idx]
        uvs = self.uvs[idx]
        local2world = self.local2world[idx]

        # Get latents for the samples
        z_geom = F.grid_sample(geo_texture, uvs[None, None])[0, :, 0].transpose(0, 1) 
        z_app = F.grid_sample(app_texture, uvs[None, None])[0, :, 0].transpose(0, 1) 

        strands_list = []
        r = 0
        for i in range(num_strands // run_batch):
            l, r = i * run_batch, (i+1) * run_batch
            z_geom_batch = z_geom[l:r]
            v = self.strand_decoder(z_geom_batch) / self.scale_decoder # [num_strands, strand_length - 1, 3]
        
            p_local = torch.cat([
                    torch.zeros_like(v[:, -1:, :]), 
                    torch.cumsum(v, dim=1)
                ], 
                dim=1
            )
            p = (local2world[l:r][:, None] @ p_local[..., None])[:, :, :3, 0] + origins[l:r][:, None] # [num_strands, strang_length, 3]
            strands_list.append(p)

        if r != num_strands:
            l = r
            r = num_strands
            z_geom_batch = z_geom[l:r]
            v = self.strand_decoder(z_geom_batch) / self.scale_decoder # [num_strands, strand_length - 1, 3]
        
            p_local = torch.cat([
                    torch.zeros_like(v[:, -1:, :]), 
                    torch.cumsum(v, dim=1)
                ], 
                dim=1
            )
            p = (local2world[l:r][:, None] @ p_local[..., None])[:, :, :3, 0] + origins[l:r][:, None] # [num_strands, strang_length, 3]
            strands_list.append(p)
        return torch.cat(strands_list, dim=0), z_geom, z_app
    