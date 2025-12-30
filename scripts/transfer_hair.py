import sys
import imageio
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import trimesh
from glob import glob
import cubvh
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
from threestudio.utils.config import load_config, parse_structured

source_hair_head_dir_path_list = ["./outputs/strand_head/stage3-fine-texture/Beyoncé_with_voluminous_curly_honey_blonde_hair"] # Replace this with the path to the source strand head

target_hair_head_dir_path_list = ["./outputs/strand_head/stage3-fine-texture/Brad_Pitt_with_medium-length_tousled_blonde_hair"] # Replace this with the path to the target strand head

device = "cuda"
save_path = "./outputs/transfer_hair"
radius_scale = 0.5
num_edges = 4
ctx = NVDiffRasterizerContext("cuda", device)
render_data = prepare_render_data(120, device)
for idx in range(len(source_hair_head_dir_path_list)):
    source_hair_head_dir_path = source_hair_head_dir_path_list[idx]
    target_hair_head_dir_path = target_hair_head_dir_path_list[idx]
    os.makedirs(os.path.join(save_path, os.path.basename(source_hair_head_dir_path_list[idx]), "render_rgb"), exist_ok=True)

    source_head_mesh = trimesh.load(os.path.join(source_hair_head_dir_path, "save/obj/result/comp_0.obj"))
    source_strand_root_scalp_mesh = trimesh.load(os.path.join(source_hair_head_dir_path.replace("stage3-fine-texture", "stage1-geometry"), "save/obj/init/init_strand_root_scalp.obj"))
    source_head_mesh_center = torch.from_numpy(((source_head_mesh.vertices).max(0) + (source_head_mesh.vertices).min(0)) / 2).float().to(device)
    source_cfg = load_config(
        os.path.join(source_hair_head_dir_path.replace("stage3-fine-texture", "stage1-geometry"), "configs/parsed.yaml")
        )
    source_flame_mesh = trimesh.load(source_cfg["system"]["geometry"]["NHC_config"]["init_mesh_path"])
    source_flame_verts_tensor = torch.from_numpy(source_flame_mesh.vertices).float().to(device)
    source_flame_faces_tensor = torch.from_numpy(source_flame_mesh.faces).long().to(device)

    target_strand_pc = trimesh.load(os.path.join(target_hair_head_dir_path, "save/obj/result/result_strand.obj"))
    target_strand = target_strand_pc.vertices.reshape(-1, 100, 3)
    target_strand_radius = torch.tensor(np.sqrt((source_strand_root_scalp_mesh.area / target_strand.shape[0]) / np.pi)).float().to(device)
    if hasattr(target_strand_pc.visual, "vertex_colors"):
        target_strand_color = target_strand_pc.visual.vertex_colors[:, :3].reshape(-1, 100, 3) / 255
    else:
        target_strand_color = np.repeat(np.random.rand(target_strand.shape[0], 3)[:, None], 100, axis=1).reshape(-1, 100, 3)
    target_strand_root = target_strand[:, 0, :]
    target_strand_root_tensor = torch.from_numpy(target_strand_root).float().to(device)
    target_strand_color_tensor = torch.from_numpy(target_strand_color.reshape(-1, 100, 3)).float().to(device)
    target_cfg = load_config(
        os.path.join(target_hair_head_dir_path.replace("stage3-fine-texture", "stage1-geometry"), "configs/parsed.yaml")
        )
    target_flame_mesh = trimesh.load(target_cfg["system"]["geometry"]["NHC_config"]["init_mesh_path"])
    target_flame_verts_tensor = torch.from_numpy(target_flame_mesh.vertices).float().to(device)
    target_flame_faces_tensor = torch.from_numpy(target_flame_mesh.faces).long().to(device)

    target_flame_BVH = cubvh.cuBVH(target_flame_mesh.vertices, target_flame_mesh.faces)
    _, map_face, map_uvw = target_flame_BVH.signed_distance(target_strand_root_tensor, return_uvw=True, mode="raystab")
    map_vert = (source_flame_verts_tensor[source_flame_faces_tensor[map_face], :] * map_uvw[:, :, None]).sum(1)
    
    result_strand = target_strand
    result_strand = result_strand - result_strand[:, 0:1, :] + map_vert[:, None, :].detach().cpu().numpy()

    strand_mesh_verts, strand_mesh_faces, indices = build_prisms(strands=torch.from_numpy(result_strand.reshape(-1, 100, 3)).float().to(device), center=source_head_mesh_center, reverse_flag=True, w=radius_scale*target_strand_radius.cpu().item(), num_edges=num_edges)
    strand_mesh_color = target_strand_color_tensor[:, :, None, :].tile(1, 1, int(strand_mesh_verts.shape[0] / result_strand.shape[0] / result_strand.shape[1]) ,1).reshape(result_strand.shape[0], -1, 3)
    strand_mesh_color = torch.cat((strand_mesh_color, target_strand_color_tensor[:, 0:1, :], target_strand_color_tensor[:, -1:, :]), dim=1).reshape(-1, 3)

    all_strand_mesh = trimesh.Trimesh(vertices=strand_mesh_verts.reshape(-1, 3).detach().cpu().numpy(), faces=strand_mesh_faces.reshape(-1, 3).detach().cpu().numpy())
    all_strand_mesh.visual.vertex_colors = np.concatenate(((strand_mesh_color * 255).detach().cpu().numpy(), np.ones_like(strand_mesh_color.detach().cpu().numpy())[:, 0:1]), axis=-1).astype(np.uint8)

    with torch.no_grad():
        all_mesh = trimesh.util.concatenate(source_head_mesh + [all_strand_mesh])
        all_mesh_verts_tensor = torch.from_numpy(all_mesh.vertices).to(device).float()
        all_mesh_faces_tensor = torch.from_numpy(all_mesh.faces).to(device).long()
        all_mesh_rgb_tensor = (torch.from_numpy(all_mesh.visual.vertex_colors[:, :3]) / 255).float().to(device)
        mesh = Mesh(v_pos=all_mesh_verts_tensor.contiguous(), t_pos_idx=all_mesh_faces_tensor.contiguous())
        for i in tqdm(range(0, 120, 1)):
            data = sample_render_data(render_data, i)
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
            Image.fromarray(rgb_img).save(os.path.join(save_path, os.path.basename(source_hair_head_dir_path), "render_rgb", f"{name}.png"))

        rgb_img_files = sorted(glob(os.path.join(f"{save_path}/{os.path.basename(source_hair_head_dir_path)}/render_rgb", '*.png')))

        rgb_imgs = [cv2.imread(f) for f in rgb_img_files]
        rgb_imgs = [cv2.cvtColor(i, cv2.COLOR_BGR2RGB) for i in rgb_imgs]
        imageio.mimsave(f"{save_path}/{os.path.basename(source_hair_head_dir_path)}/rgb_render.mp4", rgb_imgs, fps=30)






