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
import open3d as o3d

import cubvh
from src.utils.util import param_to_buffer, positional_encoding
from src.utils.geometry import barycentric_coordinates_of_projection, face_vertices
from src.utils.sample_points_from_meshes import sample_points_from_meshes
from collections import OrderedDict

def save_pc(pc, save_path, color=None):
    obj_str = ""
    for i in range(pc.shape[0]):
        if color is None:
            obj_str += f"v {pc[i][0]} {pc[i][1]} {pc[i][2]}"
        else:
            obj_str += f"v {pc[i][0]} {pc[i][1]} {pc[i][2]} {color[i][0]} {color[i][1]} {color[i][2]}"
        obj_str += "\n"
    obj_str += "\n"
    with open(save_path, "w") as f:
        f.write(obj_str)
    return 

def save_hair(hair, save_path, color=None):
    if color == None:
        color = np.concatenate((np.repeat(np.random.rand(hair.shape[0], 3)[:, None], 100, axis=1), np.ones((hair.shape[0], 100, 1))), axis=-1).reshape(-1, 4)        
    trimesh.PointCloud(hair.reshape(-1, 3), colors=color).export(save_path)

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
        self.register_buffer('scale_decoder', (usc_scale / head_scale).mean())
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
        self.N_faces = self.scalp_mesh.faces_packed()[None].shape[1]

        self.geometry_descriptor_size = geometry_descriptor_size
        self.appearance_descriptor_size = appearance_descriptor_size

        mgrid = torch.stack(torch.meshgrid([torch.linspace(-1, 1, texture_size)]*2))[None].cuda()
        self.register_buffer('encoder_input', positional_encoding(mgrid, 6))
        
        # Initialize the texture decoder network
        self.geo_texture_decoder = UNet(self.encoder_input.shape[1], geometry_descriptor_size, bilinear=True)

        self.app_texture_decoder = UNet(self.encoder_input.shape[1], appearance_descriptor_size, bilinear=True)
        

        self.register_buffer('local2world', self.init_scalp_basis(scalp_uvs))
        self.local2world = self.local2world.cuda()
        
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
        
    def init_scalp_basis_customization(self, scalp_mesh, scalp_uvs):         

        scalp_verts, scalp_faces = scalp_mesh.verts_packed()[None], scalp_mesh.faces_packed()[None]
        scalp_face_verts = face_vertices(scalp_verts, scalp_faces)[0] 
        
        # Define normal axis
        origin_v = scalp_face_verts.mean(1)
        origin_n = scalp_mesh.faces_normals_packed()
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

    def init_NHC(self, init_NHC_path, head_mesh, device):
        # Sample fixed origin points

        head_verts = torch.tensor(head_mesh.vertices).float().to(device)
        head_faces = torch.tensor(head_mesh.faces).long().to(device)

        strand_root_scalp_mesh = trimesh.load(os.path.join(init_NHC_path, "obj", "scalp.obj"))
        strand_root_faces = torch.tensor(strand_root_scalp_mesh.faces).long().to(device)
        self.strand_root_scalp_mesh = strand_root_scalp_mesh

        strand_root_uvs = torch.load(os.path.join(init_NHC_path, "model", "map_uv_tensor.pth")).to(device)
        strand_root_face_idx = torch.load(os.path.join(init_NHC_path, "model", "map_face_tensor.pth")).to(device)
        strand_root_face_uvw = torch.load(os.path.join(init_NHC_path, "model", "map_face_uvw_tensor.pth")).to(device)
        strand_random_idx = torch.load(os.path.join(init_NHC_path, "model", "random_idx.pth")).to(device)
        state_dict = torch.load(os.path.join(init_NHC_path, "model", "model_10000.pth"), map_location=device)
        ordered_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("geo_texture_decoder"):
                ordered_dict[k.replace("geo_texture_decoder.", "")] = v
        self.geo_texture_decoder.load_state_dict(ordered_dict, strict=True)

        strand_root_map_scalp_verts = (self.scalp_mesh.verts_packed().detach()[self.scalp_mesh.faces_packed().detach()[strand_root_face_idx], :] * strand_root_face_uvw[:, :, None]).sum(1)

        head_BVH = cubvh.cuBVH(head_mesh.vertices, head_mesh.faces)
        _, map_face, map_uvw = head_BVH.signed_distance(strand_root_map_scalp_verts, return_uvw=True, mode="raystab")
        origins = (head_verts[head_faces[map_face], :] * map_uvw[:, :, None]).sum(1)

        

        self.register_buffer('origins', origins)
        self.register_buffer('uvs', strand_root_uvs)
        self.register_buffer('face_idx', strand_root_face_idx)
        self.register_buffer('face_uvw', strand_root_face_uvw)
        self.register_buffer('random_idx', strand_random_idx)
        self.register_buffer("strand_root_faces", strand_root_faces)
        self.register_buffer("strand_radius", torch.tensor(np.sqrt((trimesh.Trimesh(vertices=origins.cpu().numpy(), faces=strand_root_faces.cpu().numpy()).area / strand_root_scalp_mesh.vertices.shape[0]) / np.pi)).float().to(device))
        
        self.num_strands = origins.shape[0]

        # Get transforms for the samples
        
        # self.register_buffer('local2world_customization', self.init_scalp_basis_customization(Meshes(verts=origins[None], faces=strand_root_faces[None], textures=TexturesVertex(strand_root_uvs[None])).to(device), strand_root_uvs[None]).to(device))
        self.local2world.data = self.local2world[self.face_idx]
        
        self.faces_dict = {}
        for idx, f in enumerate(self.face_idx.cpu().numpy()):
            try:
                self.faces_dict[f].append(idx)
            except KeyError:
                self.faces_dict[f] = [idx]
        
        idxes, counts = self.face_idx.unique(return_counts=True)
        self.faces_count_dict = dict(zip(idxes.cpu().numpy(), counts.cpu().numpy()))

        strand_root_verts_neigh = [torch.tensor(tmp).long().to(device) for tmp in strand_root_scalp_mesh.vertex_neighbors]
        self.strand_root_verts_neigh = strand_root_verts_neigh
        self_idx = []
        neighbors_idx = []
        for i, neighbors in enumerate(strand_root_scalp_mesh.vertex_neighbors):
            self_idx = self_idx + [i] * len(neighbors)
            neighbors_idx = neighbors_idx + neighbors
        self.strand_root_verts_self_idx = torch.tensor(self_idx).long().to(device)
        self.strand_root_verts_neighbors_idx = torch.tensor(neighbors_idx).long().to(device)

        return

    def init_sampled_strand_root_scalp_mesh(self, idx):

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.strand_root_scalp_mesh.vertices[idx])
        pcd.normals = o3d.utility.Vector3dVector(self.strand_root_scalp_mesh.vertex_normals[idx])

        size = 0.01 * np.sqrt(self.num_strands / idx.shape[0]) 
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([size, size, size])
        )
        sampled_strand_root_scalp_mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))

        self.sampled_strand_root_scalp_mesh = sampled_strand_root_scalp_mesh

        # print(f"{os.path.basename(hair_path).split('.')[0]} holes number : {count_holes(mesh)}")
        # o3d.io.write_triangle_mesh(f"{save_path}/{os.path.basename(hair_path).split('.')[0]}.obj", sampled_strand_root_scalp_mesh)

        sampled_strand_root_verts_neigh = [torch.tensor(tmp).long().to(self.origins.device) for tmp in sampled_strand_root_scalp_mesh.vertex_neighbors]
        self.sampled_strand_root_verts_neigh = sampled_strand_root_verts_neigh
        self_idx = []
        neighbors_idx = []
        for i, neighbors in enumerate(sampled_strand_root_scalp_mesh.vertex_neighbors):
            self_idx = self_idx + [i] * len(neighbors)
            neighbors_idx = neighbors_idx + neighbors
        self.sampled_strand_root_verts_self_idx = torch.tensor(self_idx).long().to(self.origins.device)
        self.sampled_strand_root_verts_neighbors_idx = torch.tensor(neighbors_idx).long().to(self.origins.device)

    def forward(self, num_strands=-1, sample_mode="fixed", run_batch=2000, scale_factor=None, reset_sampled_mesh=False):
        # Generate texture
        geo_texture = self.geo_texture_decoder(self.encoder_input)
        app_texture = self.app_texture_decoder(self.encoder_input)
        texture_res = geo_texture.shape[-1]

        if scale_factor is None:
            scale_factor = self.scale_decoder

        if num_strands == -1 or num_strands >= self.num_strands:
            num_strands = self.num_strands
            idx = torch.tensor(torch.arange(num_strands)).to(geo_texture.device)
        else:
            self.m, self.q = num_strands // self.N_faces, num_strands % self.N_faces

            # Sample idxes from texture
            if sample_mode == "random":
                # If the #sampled strands > #scalp faces, then we try to sample more uniformly for better convergence
                f_idx, count = torch.cat((torch.arange(self.N_faces).repeat(self.m), torch.randperm(self.N_faces)[:self.q])).unique(return_counts=True)
                current_iter =  dict(zip(f_idx.cpu().numpy(), count.cpu().numpy()))
                iter_idx = []
                for i in range(self.N_faces):
                    if i not in self.faces_dict.keys():
                        continue
                    cur_idx_list = torch.tensor(self.faces_dict[i])[torch.randperm(self.faces_count_dict[i])[:current_iter[i]]].tolist()
                    iter_idx.append(cur_idx_list)
                idx = torch.tensor(list(itertools.chain(*iter_idx)))
            elif sample_mode == "fixed":
                # idx = self.random_idx[:num_strands]
                if not hasattr(self, 'fixed_idx'):
                    idx = []
                    start_idx = 0
                    while True:
                        for k, v in self.faces_dict.items():
                            if start_idx >= self.faces_count_dict[k]:
                                continue
                            idx.append(v[start_idx])
                            if len(idx) >= num_strands:
                                break
                        if len(idx) >= num_strands:
                            break
                        start_idx += 1
                    idx = torch.tensor(idx).long().to(geo_texture.device)
                    self.register_buffer('fixed_idx', idx)
                elif num_strands != self.fixed_idx.shape[0]:
                    idx = []
                    start_idx = 0
                    while True:
                        for k, v in self.faces_dict.items():
                            if start_idx >= self.faces_count_dict[k]:
                                continue
                            idx.append(v[start_idx])
                            if len(idx) >= num_strands:
                                break
                        if len(idx) >= num_strands:
                            break
                        start_idx += 1
                    idx = torch.tensor(idx).long().to(geo_texture.device)
                    self.register_buffer('fixed_idx', idx)
                else:
                    idx = self.fixed_idx

        if not hasattr(self, 'sampled_strand_root_verts_neigh') or reset_sampled_mesh:
            self.init_sampled_strand_root_scalp_mesh(idx.cpu().numpy())

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
        
        return torch.cat(strands_list, dim=0), z_geom, z_app