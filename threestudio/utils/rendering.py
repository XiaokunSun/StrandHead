import os
import torch
import math
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
from threestudio.utils.typing import *
import torch.nn.functional as F
import numpy as np
import trimesh

def convert_proj(K, H, W, near, far):
    return [
        [2 * K[0, 0] / W, -2 * K[0, 1] / W, (W - 2 * K[0, 2]) / W, 0],
        [0, -2 * K[1, 1] / H, (H - 2 * K[1, 2]) / H, 0],
        [0, 0, (-far - near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],
    ]

def load_mesh_rgb(mesh_path):
    with open(mesh_path, "r") as f:
        lines = f.readlines()
    rgb = []
    for line in lines:
        info_list = line.replace("\n", "").split(" ")
        if info_list[0] != 'v':
            break
        rgb.append(np.array([float(info_list[-3]), float(info_list[-2]), float(info_list[-1])]))
    return np.array(rgb)

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
    if color is None:
        color = np.concatenate((np.repeat(np.random.rand(hair.shape[0], 3)[:, None], 100, axis=1), np.ones((hair.shape[0], 100, 1))), axis=-1).reshape(-1, 4)  
    else:
        color = np.concatenate((color, np.ones((hair.shape[0], 100, 1))), axis=-1).reshape(-1, 4)
    trimesh.PointCloud(hair.reshape(-1, 3), colors=color).export(save_path)

def save_hair(pc, save_path, color=None):
    obj_str = ""
    for i in range(pc.shape[0]):
        if color is None:
            obj_str += f"v {pc[i][0]} {pc[i][1]} {pc[i][2]}"
        else:
            obj_str += f"v {pc[i][0]} {pc[i][1]} {pc[i][2]} {color[i][0]} {color[i][1]} {color[i][2]}"
        obj_str += "\n"
    for i in range(1, pc.shape[0] + 1):
        if i % 100 != 0:
            obj_str += f"l {i} {i+1}"
            obj_str += "\n"
    obj_str += "\n"
    
    with open(save_path, "w") as f:
        f.write(obj_str)
    return 


def prepare_render_data(num_views, device):
    n_views = num_views
    azimuth_deg = torch.linspace(0, 360.0, n_views)
    camera_distance_range = [3, 3]
    elevation_range = [5, 5]
    eval_batch_size = 1
    eval_fovy_deg = 45.
    eval_height = 1024
    eval_width = 1024

    elevation_deg = (
        torch.rand(n_views)
        * (elevation_range[1] - elevation_range[0])
        + elevation_range[0]
    )
    camera_distances = (
        torch.rand(n_views)
        * (camera_distance_range[1] - camera_distance_range[0])
        + camera_distance_range[0]
    )
    elevation = elevation_deg * math.pi / 180
    azimuth = azimuth_deg * math.pi / 180
    # convert spherical coordinates to cartesian coordinates
    # right hand coordinate system, x back, y right, z up
    # elevation in (-90, 90), azimuth from +x to +y in (-180, 180)
    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )

    # default scene center at origin
    center = torch.zeros_like(camera_positions)
    # default camera up direction as +z
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
        None, :
    ].repeat(eval_batch_size, 1)

    fovy_deg: Float[Tensor, "B"] = torch.full_like(
        elevation_deg, eval_fovy_deg
    )
    fovy = fovy_deg * math.pi / 180
    light_positions: Float[Tensor, "B 3"] = camera_positions

    # get directions by dividing directions_unit_focal by focal length
    focal_length: Float[Tensor, "B"] = (
        0.5 * eval_height / torch.tan(0.5 * fovy)
    )


    cx = torch.full_like(focal_length, eval_width / 2)
    cy = torch.full_like(focal_length, eval_height / 2)
    
    intrinsic: Float[Tensor, "B 4 4"] = torch.eye(4)[None, :,:].repeat(n_views, 1, 1)
    intrinsic[:, 0, 0] = focal_length
    intrinsic[:, 1, 1] = focal_length
    intrinsic[:, 0, 2] = cx
    intrinsic[:, 1, 2] = cy

    proj_mtx = []
    directions = []
    for i in range(n_views):
        proj = convert_proj(intrinsic[i], eval_height, eval_width, 0.1, 1000.0)
        proj: Float[Tensor, "4 4"] = torch.FloatTensor(proj)
        proj_mtx.append(proj)

        direction: Float[Tensor, "H W 3"] = get_ray_directions(
            eval_height,
            eval_width,
            (intrinsic[i, 0, 0], intrinsic[i, 1, 1]),
            (intrinsic[i, 0, 2], intrinsic[i, 1, 2]),
            use_pixel_centers=False,
        )
        directions.append(direction)

    proj_mtx: Float[Tensor, "B 4 4"] = torch.stack(proj_mtx, dim=0)
    directions: Float[Tensor, "B H W 3"] = torch.stack(directions, dim=0)

    lookat: Float[Tensor, "B 3"] = F.normalize(center - camera_positions, dim=-1)
    right: Float[Tensor, "B 3"] = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w3x4: Float[Tensor, "B 3 4"] = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w: Float[Tensor, "B 4 4"] = torch.cat(
        [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
    )
    c2w[:, 3, 3] = 1.0

    rays_o, rays_d = get_rays(directions, c2w, keepdim=True)
    
    mvp_mtx: Float[Tensor, "B 4 4"] = get_mvp_matrix(c2w, proj_mtx)

    data = {}

    data["rays_o"], data["rays_d"] = rays_o.to(device), rays_d.to(device)
    data["mvp_mtx"] = mvp_mtx.to(device)
    data["c2w"] = c2w.to(device)
    data["camera_positions"] = camera_positions.to(device)
    data["light_positions"] = light_positions.to(device)
    data["elevation"], data["azimuth"] = elevation, azimuth.to(device)
    data["elevation_deg"], data["azimuth_deg"] = elevation_deg.to(device), azimuth_deg.to(device)
    data["camera_distances"] = camera_distances.to(device)
    data["focal_length"] = focal_length.to(device)
    data["cx"] = cx.to(device)
    data["cy"] = cy.to(device)
    data["up"] = up.to(device)
    data["fovy_deg"] = fovy_deg.to(device)
    data["center"] = center.to(device)
    data["eval_height"] = eval_height
    data["eval_width"] = eval_width
    data["n_views"] = n_views

    return data

def sample_render_data(render_data, batch_idx: int):
    return {
    "index": batch_idx,
    "rays_o": render_data["rays_o"][batch_idx][None],
    "rays_d": render_data["rays_d"][batch_idx][None],
    "mvp_mtx": render_data["mvp_mtx"][batch_idx][None],
    "c2w": render_data["c2w"][batch_idx][None],
    "camera_positions": render_data["camera_positions"][batch_idx][None],
    "light_positions": render_data["light_positions"][batch_idx][None],
    "elevation": render_data["elevation_deg"][batch_idx][None],
    "azimuth": render_data["azimuth_deg"][batch_idx][None],
    "camera_distances": render_data["camera_distances"][batch_idx][None],
    "height": render_data["eval_height"],
    "width": render_data["eval_width"],
    "focal": render_data["focal_length"][batch_idx][None],
    "cx": render_data["cx"][batch_idx][None],
    "cy": render_data["cy"][batch_idx][None],
    "up": render_data["up"][batch_idx][None],
    "fovy": render_data["fovy_deg"][batch_idx][None],
    "center": render_data["center"][batch_idx][None],
    "n_views": render_data["n_views"]
    }

def render_mesh(
    ctx,
    mesh,
    mvp_mtx: Float[Tensor, "B 4 4"],
    c2w: Float[Tensor, "B 4 4"],
    camera_positions: Float[Tensor, "B 3"],
    light_positions: Float[Tensor, "B 3"],
    height: int,
    width: int,
    render_rgb: bool = True,
    mesh_rgb = None
) -> Dict[str, Any]:
    normal_type = "camera"
    v_pos_clip: Float[Tensor, "B Nv 4"] = ctx.vertex_transform(
        mesh.v_pos, mvp_mtx
    )
    rast, _ = ctx.rasterize(v_pos_clip, mesh.t_pos_idx, (height, width))
    mask = rast[..., 3:] > 0
    mask_aa = ctx.antialias(mask.float(), rast, v_pos_clip, mesh.t_pos_idx)

    out = {"mask": mask_aa, "mesh": mesh}

    gb_normal, _ = ctx.interpolate_one(mesh.v_nrm, rast, mesh.t_pos_idx)

    if normal_type == 'world':
        gb_normal = F.normalize(gb_normal, dim=-1)
        gb_normal = torch.cat([gb_normal[:,:,:,1:2], gb_normal[:,:,:,2:3], gb_normal[:,:,:,0:1]], -1)
        gb_normal_aa = torch.lerp(
            torch.zeros_like(gb_normal), (gb_normal + 1.0) / 2.0, mask.float()
        )
    elif normal_type == 'camera':
        # world coord to cam coord
        gb_normal = gb_normal.view(-1, height*width, 3)
        gb_normal = torch.matmul(torch.linalg.inv(c2w[:,:3,:3]), gb_normal[0][:,:,None])
        gb_normal = gb_normal.view(-1, height, width, 3)
        gb_normal = F.normalize(gb_normal, dim=-1)
        bg_normal = torch.zeros_like(gb_normal)
        gb_normal_aa = torch.lerp(
            bg_normal, (gb_normal + 1.0) / 2.0, mask.float()
        )
    elif normal_type == 'controlnet':
        # world coord to cam coord
        gb_normal = gb_normal.view(-1, height*width, 3)
        gb_normal = torch.matmul(torch.linalg.inv(c2w[:,:3,:3]), gb_normal[0][:,:,None])
        gb_normal = gb_normal.view(-1, height, width, 3)
        gb_normal = F.normalize(gb_normal, dim=-1)
        # nerf coord to a special coord for controlnet
        gb_normal = torch.cat([-gb_normal[:,:,:,0:1], gb_normal[:,:,:,1:2], gb_normal[:,:,:,2:3]], -1)
        bg_normal = torch.zeros_like(gb_normal)
        bg_normal[~mask[..., 0]] = torch.Tensor([[126/255, 107/255, 1.0]]).to(bg_normal)

        gb_normal_aa = torch.lerp(
            bg_normal, (gb_normal + 1.0) / 2.0, mask.float()
        )
    else:
        raise ValueError(f"Unknown normal type: {normal_type}")

    gb_normal_aa = ctx.antialias(
        gb_normal_aa, rast, v_pos_clip, mesh.t_pos_idx
    )
    out.update({"normal": gb_normal_aa})  # in [0, 1]

    # gb_depth = rast[..., 2:3]
    # gb_depth = 1./(gb_depth + 1e-7)
    gb_depth, _ = ctx.interpolate_one(v_pos_clip[0,:, :3].contiguous(), rast, mesh.t_pos_idx)
    gb_depth = 1./(gb_depth[..., 2:3] + 1e-7)
    max_depth = torch.max(gb_depth[mask[..., 0]])
    min_depth = torch.min(gb_depth[mask[..., 0]])
    gb_depth_aa = torch.lerp(
            torch.zeros_like(gb_depth), (gb_depth - min_depth) / (max_depth - min_depth + 1e-7), mask.float()
        )
    gb_depth_aa = ctx.antialias(
        gb_depth_aa, rast, v_pos_clip, mesh.t_pos_idx
    )
    out.update({"depth":gb_depth_aa})  # in [0, 1]

    # TODO: make it clear whether to compute the normal, now we compute it in all cases
    # consider using: require_normal_computation = render_normal or (render_rgb and material.requires_normal)
    # or
    # render_normal = render_normal or (render_rgb and material.requires_normal)

    if render_rgb:
        gb_rgb_fg, _ = ctx.interpolate_one(mesh_rgb.contiguous(), rast, mesh.t_pos_idx)
        gb_rgb_bg = torch.ones(1, height, width, 3).to(rast.device)
        gb_rgb = torch.lerp(gb_rgb_bg, gb_rgb_fg, mask.float())
        gb_rgb_aa = ctx.antialias(gb_rgb, rast, v_pos_clip, mesh.t_pos_idx)
        out.update({"rgb": gb_rgb_aa, "rgb_bg": gb_rgb_bg})
    return out

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
