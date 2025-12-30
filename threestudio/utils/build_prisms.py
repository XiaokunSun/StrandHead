import torch
import numpy as np
from typing import Optional

def smooth(x: torch.Tensor, n: int = 3) -> torch.Tensor:
    ret = torch.cumsum(torch.concat((torch.repeat_interleave(x[:1], n - 1, dim=0), x)), 0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def xy_normal(strands: torch.Tensor, normalize: bool = True, smooth_degree: Optional[int] = None, xy_axis = [0, 1]):
    d = torch.empty(strands.shape[:2] + (2, ), device=strands.device)

    if smooth_degree is not None:
        strands = smooth(strands, smooth_degree)

    d[:, :-1, :] = strands[:, 1:, xy_axis] - strands[:, : -1, xy_axis]
    d[:, -1, :] = d[:, -2, :]

    n = torch.cat((d[:, :, [1]], -d[:, :, [0]]), dim=2)
    
    if normalize:
        n = n / (torch.linalg.norm(n, dim=2, keepdims=True) + 1e-20)
    
    return n

def rotate_vector(vector, axis, angle):
    axis = axis / torch.norm(axis, dim=1, keepdim=True)
    cos_theta = torch.cos(angle).unsqueeze(1)  # (N, 1)
    sin_theta = torch.sin(angle).unsqueeze(1)  # (N, 1)
    term1 = vector * cos_theta
    term2 = torch.cross(axis, vector, dim=1) * sin_theta
    term3 = axis * torch.sum(axis * vector, dim=1, keepdim=True) * (1 - cos_theta)
    rotated_vector = term1 + term2 + term3
    return rotated_vector
    
def build_prisms(
    strands: torch.Tensor,
    center: torch.Tensor,
    reverse_flag: bool = False ,
    w: float = 0.0001,
    num_edges: int = 2,
    close: bool = True,
    calculate_faces: bool = True
    ):

    n_strands, n_points = strands.shape[:2]

    d = torch.empty(strands.shape[:2] + (3, ), device=strands.device)
    d[:, :-1, :] = strands[:, 1:, :] - strands[:, : -1, :]
    d[:, -1, :] = d[:, -2, :]

    n_xyz = torch.cross(d, (strands - center[None, None]), dim=2)    
    n_xyz = n_xyz / (torch.linalg.norm(n_xyz, dim=2, keepdims=True) + 1e-20)
    if reverse_flag:
        n_xyz = -1 * n_xyz

    angle = torch.tensor([(float(i) * 2 * np.pi / num_edges) + (np.pi / 2) + (np.pi / num_edges) for i in range(num_edges)], device=strands.device)[None, None, :].tile(n_strands, n_points, 1)
    n_xyz_rotate = rotate_vector(n_xyz[:, :, None, :].tile(1, 1, num_edges, 1).reshape(-1, 3), d[:, :, None, :].tile(1, 1, num_edges, 1).reshape(-1, 3), angle.reshape(-1)).reshape(n_strands, n_points, num_edges, 3)
    verts = torch.empty((n_strands, n_points, num_edges, 3), device=strands.device)
    verts = strands[:, :, None, :].tile(1, 1, num_edges, 1) + w * n_xyz_rotate
    
    indices = torch.empty((n_strands, n_points, num_edges, 1), device=strands.device, dtype=verts.dtype)
    values = torch.tensor(list(range(n_strands * n_points)), dtype=indices.dtype, device=indices.device)
    values = values.view((n_strands, n_points, 1, 1)).tile(1, 1, num_edges, 1)
    indices = values
    indices = indices.reshape(-1, 1)
    
    if calculate_faces:
        faces = torch.empty(((n_points - 1), num_edges, 2, 3), dtype=torch.long, device=strands.device)
        for edge_idx in range(num_edges - 1):
            faces[:, edge_idx, 0, 0] = torch.arange(edge_idx, (n_points - 1) * num_edges + edge_idx, num_edges)
            faces[:, edge_idx, 0, 2] = torch.arange(edge_idx + num_edges, n_points * num_edges + edge_idx, num_edges)
            faces[:, edge_idx, 0, 1] = torch.arange(edge_idx + 1, (n_points - 1) * num_edges + edge_idx + 1, num_edges)
            faces[:, edge_idx, 1, 0] = torch.arange(edge_idx + 1, (n_points - 1) * num_edges + edge_idx + 1, num_edges)
            faces[:, edge_idx, 1, 2] = torch.arange(edge_idx + num_edges, n_points * num_edges + edge_idx, num_edges)
            faces[:, edge_idx, 1, 1] = torch.arange(edge_idx + num_edges + 1, n_points * num_edges + edge_idx + 1, num_edges)
        edge_idx = num_edges - 1
        faces[:, edge_idx, 0, 0] = torch.arange(edge_idx, (n_points - 1) * num_edges + edge_idx, num_edges)
        faces[:, edge_idx, 0, 2] = torch.arange(edge_idx + num_edges, n_points * num_edges + edge_idx, num_edges)
        faces[:, edge_idx, 0, 1] = torch.arange(edge_idx + 1 - num_edges, (n_points - 1) * num_edges + edge_idx + 1 - num_edges, num_edges)
        faces[:, edge_idx, 1, 0] = torch.arange(edge_idx + 1 - num_edges, (n_points - 1) * num_edges + edge_idx + 1 - num_edges, num_edges)
        faces[:, edge_idx, 1, 2] = torch.arange(edge_idx + num_edges, n_points * num_edges + edge_idx, num_edges)
        faces[:, edge_idx, 1, 1] = torch.arange(edge_idx + num_edges + 1 - num_edges, n_points * num_edges + edge_idx + 1 - num_edges, num_edges)
        
        if num_edges == 2:
            faces = faces[:, 1:2, :, :]
        faces = faces[None].tile(n_strands, 1, 1, 1, 1)
        full_faces_array = torch.arange(0, verts.shape[1] * verts.shape[2] * n_strands, verts.shape[1] * verts.shape[2], device=strands.device).reshape(n_strands, 1, 1, 1, 1) + faces
        
        if close and num_edges > 2:
            verts_reshape = torch.cat((verts.reshape(n_strands, -1, 3), strands[:, 0:1, :], strands[:, -1:, :]), dim=1).reshape(-1, 3)
            tmp_faces = torch.empty((2, num_edges, 3), dtype=torch.long, device=strands.device)
            for edge_idx in range(num_edges - 1):
                tmp_faces[0, edge_idx, 0] = verts.shape[1] * verts.shape[2]
                tmp_faces[0, edge_idx, 2] = edge_idx
                tmp_faces[0, edge_idx, 1] = edge_idx + 1
                tmp_faces[1, edge_idx, 0] = verts.shape[1] * verts.shape[2] + 1
                tmp_faces[1, edge_idx, 1] = verts.shape[1] * verts.shape[2] - num_edges + edge_idx
                tmp_faces[1, edge_idx, 2] = verts.shape[1] * verts.shape[2] - num_edges + edge_idx + 1
            edge_idx = num_edges - 1
            tmp_faces[0, edge_idx, 0] = verts.shape[1] * verts.shape[2]
            tmp_faces[0, edge_idx, 2] = edge_idx
            tmp_faces[0, edge_idx, 1] = edge_idx + 1 - num_edges
            tmp_faces[1, edge_idx, 0] = verts.shape[1] * verts.shape[2] + 1
            tmp_faces[1, edge_idx, 1] = verts.shape[1] * verts.shape[2] - num_edges + edge_idx
            tmp_faces[1, edge_idx, 2] = verts.shape[1] * verts.shape[2] - num_edges + edge_idx + 1 - num_edges
            tmp_faces = tmp_faces[None].tile(n_strands, 1, 1, 1)
            tmp_full_faces_array = torch.arange(0, (verts.shape[1] * verts.shape[2] + 2) * n_strands, (verts.shape[1] * verts.shape[2] + 2), device=strands.device).reshape(n_strands, 1, 1, 1) + tmp_faces
            full_faces_array = torch.arange(0, 2 * n_strands, 2, device=strands.device).reshape(n_strands, 1, 1, 1, 1) + full_faces_array
            faces_reshape = torch.cat((full_faces_array.reshape(-1, 3), tmp_full_faces_array.reshape(-1, 3)), dim=0)
            indices_reshape = indices.reshape(n_strands, -1, 1)
        else:
            verts_reshape = verts.reshape(-1, 3)
            faces_reshape = full_faces_array.reshape(-1, 3)
            indices_reshape = indices.reshape(-1, 1)

    if calculate_faces:
        return verts_reshape.reshape(-1, 3), faces_reshape.reshape(-1, 3), indices_reshape.reshape(-1, 1)
    else:
        return verts_reshape.reshape(-1, 3), indices_reshape.reshape(-1, 1)