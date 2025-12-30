import sys
import imageio
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import trimesh
from glob import glob
import numpy as np
from threestudio.utils.rasterize import NVDiffRasterizerContext
import torch
import cv2
from PIL import Image
from threestudio.models.mesh import Mesh
from threestudio.utils.typing import *
from threestudio.utils.rendering import *
from threestudio.utils.build_prisms import build_prisms
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

hair_head_dir_path_list = ["./outputs/strand_head/stage3-fine-texture/Brad_Pitt_with_medium-length_tousled_blonde_hair"] # Replace this with the path to your generated strand head
device = "cuda"
save_path = "./outputs/edit_strand_length"
radius_scale = 0.5
num_edges = 4
ctx = NVDiffRasterizerContext("cuda", device)
render_data = prepare_render_data(120, device)
for hair_head_dir_path in hair_head_dir_path_list:
    os.makedirs(os.path.join(save_path, os.path.basename(hair_head_dir_path), "render_rgb"), exist_ok=True)
    head_mesh = trimesh.load(os.path.join(hair_head_dir_path, "save/obj/result/comp_0.obj"))
    strand_mesh = trimesh.load(os.path.join(hair_head_dir_path, "save/obj/result/comp_1.obj"))
    strand_root_scalp_mesh = trimesh.load(os.path.join(hair_head_dir_path.replace("stage3-fine-texture", "stage1-geometry"), "save/obj/init/init_strand_root_scalp.obj"))
    strand_radius = torch.tensor(np.sqrt((strand_root_scalp_mesh.area / strand_root_scalp_mesh.vertices.shape[0]) / np.pi)).float().to(device)
    head_mesh_center = torch.from_numpy(((head_mesh.vertices).max(0) + (head_mesh.vertices).min(0)) / 2).float().to(device)
    strand_pc = trimesh.load(os.path.join(hair_head_dir_path, "save/obj/result/result_strand.obj"))
    strand = strand_pc.vertices.reshape(-1, 100, 3)
    if hasattr(strand_pc.visual, "vertex_colors"):
        strand_color = strand_pc.visual.vertex_colors[:, :3].reshape(-1, 100, 3) / 255
    else:
        strand_color = np.repeat(np.random.rand(strand.shape[0], 3)[:, None], 100, axis=1).reshape(-1, 100, 3)
    strand_root = strand[:, 0, :]
    new_strand = strand
    new_strand_color = strand_color
    all_strand_color_tesnor = torch.from_numpy(new_strand_color.reshape(-1, 100, 3)).float().to(device)

    with torch.no_grad():
        tmp_all_strand_color_tesnor = all_strand_color_tesnor
        ori = new_strand[:, 1:, :] - new_strand[:, :-1, :]
        norm = np.sqrt((ori ** 2).sum(-1))
        ori = ori / (norm[:, :, None] + 1e-20)

        for i in tqdm(range(0, 120, 1)):
            new_norm = norm * (0.5 + (i / 120))
            tmp_strand = np.concatenate((new_strand[:, 0:1, :], new_norm[:, :, None] * ori), axis=1)
            tmp_all_strand = np.cumsum(tmp_strand, axis=1)

            strand_mesh_verts, strand_mesh_faces, indices = build_prisms(strands=torch.from_numpy(tmp_all_strand.reshape(-1, 100, 3)).float().to(device), center=head_mesh_center, reverse_flag=True, w=np.sqrt(strand.shape[0]/strand.shape[0])*radius_scale*strand_radius.cpu().item(), num_edges=num_edges)
            strand_mesh_color = tmp_all_strand_color_tesnor[:, :, None, :].tile(1, 1, int(strand_mesh_verts.shape[0] / tmp_all_strand.shape[0] / tmp_all_strand.shape[1]) ,1).reshape(tmp_all_strand.shape[0], -1, 3)
            strand_mesh_color = torch.cat((strand_mesh_color, tmp_all_strand_color_tesnor[:, 0:1, :], tmp_all_strand_color_tesnor[:, -1:, :]), dim=1).reshape(-1, 3)

            all_strand_mesh = trimesh.Trimesh(vertices=strand_mesh_verts.reshape(-1, 3).detach().cpu().numpy(), faces=strand_mesh_faces.reshape(-1, 3).detach().cpu().numpy())
            all_strand_mesh.visual.vertex_colors = np.concatenate(((strand_mesh_color * 255).detach().cpu().numpy(), np.ones_like(strand_mesh_color.detach().cpu().numpy())[:, 0:1]), axis=-1).astype(np.uint8)

            all_mesh = trimesh.util.concatenate(head_mesh + [all_strand_mesh])
            
            all_mesh_verts_tensor = torch.from_numpy(all_mesh.vertices).to(device).float()
            all_mesh_faces_tensor = torch.from_numpy(all_mesh.faces).to(device).long()
            all_mesh_rgb_tensor = (torch.from_numpy(all_mesh.visual.vertex_colors[:, :3]) / 255).float().to(device)
            mesh = Mesh(v_pos=all_mesh_verts_tensor.contiguous(), t_pos_idx=all_mesh_faces_tensor.contiguous())
            data = sample_render_data(render_data, 0)
            render_out = render_mesh(ctx=ctx, mesh=mesh,
                mvp_mtx=data["mvp_mtx"],
                c2w=data["c2w"],
                camera_positions=data["camera_positions"],
                light_positions=data["light_positions"],
                height=data["height"],
                width=data["width"],
                render_rgb=True,
                mesh_rgb=all_mesh_rgb_tensor
                )
            normal_img = ((render_out["normal"][0] + (1 - render_out["mask"][0])) * 255).detach().cpu().numpy().astype(np.uint8)
            rgb_img = ((render_out["rgb"][0]) * 255).detach().cpu().numpy().astype(np.uint8)
            name = str(i).rjust(6,"0")
            Image.fromarray(rgb_img).save(os.path.join(save_path, os.path.basename(hair_head_dir_path), "render_rgb", f"{name}.png"))

        rgb_img_files = sorted(glob(os.path.join(f"{save_path}/{os.path.basename(hair_head_dir_path)}/render_rgb", '*.png')))

        rgb_imgs = [cv2.imread(f) for f in rgb_img_files]
        rgb_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in rgb_imgs]
        imageio.mimsave(f"{save_path}/{os.path.basename(hair_head_dir_path)}/rgb_render.mp4", rgb_imgs, fps=30)







