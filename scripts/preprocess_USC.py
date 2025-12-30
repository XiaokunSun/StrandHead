import struct
import numpy as np
import os
import trimesh
from glob import glob
import torch
import cubvh
import open3d as o3d
import pymeshlab as ml
import copy

def count_holes(mesh):
    edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
    edge_map = {}
    for edge in edges:
        edge = tuple(sorted(edge))
        if edge in edge_map:
            edge_map[edge] += 1
        else:
            edge_map[edge] = 1
    boundary_edges = [edge for edge, count in edge_map.items() if count == 1]
    boundary_graph = {}
    for edge in boundary_edges:
        if edge[0] not in boundary_graph:
            boundary_graph[edge[0]] = []
        if edge[1] not in boundary_graph:
            boundary_graph[edge[1]] = []
        boundary_graph[edge[0]].append(edge[1])
        boundary_graph[edge[1]].append(edge[0])

    visited = set()
    num_holes = 0
    for start in boundary_graph:
        if start not in visited:
            num_holes += 1
            stack = [start]
            while stack:
                node = stack.pop()
                if node not in visited:
                    visited.add(node)
                    stack.extend(boundary_graph[node])

    return num_holes

def laplacian_smoothing(mesh, iterations=10, lambda_factor=0.5):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    vertex_neighbors = {i: set() for i in range(len(vertices))}
    for tri in triangles:
        for i in range(3):
            vertex_neighbors[tri[i]].update([tri[(i + 1) % 3], tri[(i + 2) % 3]])

    for _ in range(iterations):
        new_vertices = np.copy(vertices)
        
        for i in range(len(vertices)):
            neighbors = vertex_neighbors[i]
            if len(neighbors) > 0:

                neighbor_positions = np.array([vertices[j] for j in neighbors])
                centroid = np.mean(neighbor_positions, axis=0)
                
                laplacian_vector = centroid - vertices[i]
                new_vertices[i] += lambda_factor * laplacian_vector

        vertices = new_vertices

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh

def laplacian_smoothing(mesh, fixed_mask=None, iterations=10, lambda_factor=0.5):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    vertex_neighbors = {i: set() for i in range(len(vertices))}
    for tri in triangles:
        for i in range(3):
            vertex_neighbors[tri[i]].update([tri[(i + 1) % 3], tri[(i + 2) % 3]])

    if fixed_mask is None:
        fixed_mask = np.zeros(len(vertices), dtype=bool)
    
    for _ in range(iterations):
        new_vertices = np.copy(vertices)

        for i in range(len(vertices)):
            if fixed_mask[i]:
                continue
            
            neighbors = vertex_neighbors[i]
            if len(neighbors) > 0:
                neighbor_positions = np.array([vertices[j] for j in neighbors])
                centroid = np.mean(neighbor_positions, axis=0)

                laplacian_vector = centroid - vertices[i]
                new_vertices[i] += lambda_factor * laplacian_vector

        vertices = new_vertices

    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh

DATASET_PATH = "Data/hairstyles" # Replace this with the path to your USC-HairSalon Dataset (https://huliwenkidkid.github.io/liwenhu.github.io)
dataset_path = f"{DATASET_PATH}/hairstyles"
save_path = f"{DATASET_PATH}/hairstyles_obj"
os.makedirs(save_path, exist_ok=True)
hair_path_list = sorted(glob(dataset_path + "/*.data")) 

# Convert the USC-HairSalon dataset to OBJ format for visualization
for hair_path in hair_path_list:
    fin = open (hair_path, "rb")
    num_of_strands = struct.unpack('i', fin.read(4))[0]
    hair = []
    print("num_of_strands: ", num_of_strands)
    for i in range(num_of_strands):
        num_of_vertices = struct.unpack('i', fin.read(4))[0]
        strand = []
        for j in range(num_of_vertices):
            vertex_x = struct.unpack('f', fin.read(4))[0]
            vertex_y = struct.unpack('f', fin.read(4))[0]
            vertex_z = struct.unpack('f', fin.read(4))[0]
            vertex = [vertex_x, vertex_y, vertex_z]
            strand.append(np.array(vertex))
        if len(strand) == 100:
            hair.append(np.array(strand))
    fin.close()
    hair = np.array(hair)
    cols = np.concatenate((np.repeat(np.random.rand(hair.shape[0], 3)[:, None], 100, axis=1), np.ones((hair.shape[0], 100, 1))), axis=-1).reshape(-1, 4)        
    trimesh.PointCloud(hair.reshape(-1, 3), colors=cols).export(f"{save_path}/{os.path.basename(hair_path).split('.')[0]}.obj")
    print(os.path.basename(hair_path).split('.')[0])

# Preprocess hair data
dataset_path = save_path
save_path = f"{DATASET_PATH}/strand_root_obj"
save_path_processed = f"{DATASET_PATH}/strand_root_obj_processed"
save_strand_path_processed = f"{DATASET_PATH}/hairstyles_obj_processed"
os.makedirs(save_path, exist_ok=True)
os.makedirs(save_path_processed, exist_ok=True)
os.makedirs(save_strand_path_processed, exist_ok=True)

hair_path_list = sorted(glob(dataset_path + "/*.obj")) 
head_mesh = trimesh.load(f"{DATASET_PATH}/hairstyles/head_model.obj")
device = "cuda:0"
head_verts_tensor = torch.from_numpy(head_mesh.vertices).float().to(device)
head_faces_tensor = torch.from_numpy(head_mesh.faces).long().to(device)
head_normals_tensor = torch.from_numpy(head_mesh.vertex_normals).float().to(device)
head_BVH = cubvh.cuBVH(head_mesh.vertices, head_mesh.faces)

for hair_path in hair_path_list:
    hair_mesh = trimesh.load(hair_path)
    hair = hair_mesh.vertices.reshape(-1, 100, 3)
    strand_root = hair[:, 0, :]

    strand_root_verts_tensor = torch.from_numpy(strand_root).float().to(device)
    map_dist, map_face, map_uvw = head_BVH.signed_distance(strand_root_verts_tensor, return_uvw=True, mode="raystab")
    points = (head_verts_tensor[head_faces_tensor[map_face], :] * map_uvw[:, :, None]).sum(1).cpu().numpy()
    normals = (head_normals_tensor[head_faces_tensor[map_face], :] * map_uvw[:, :, None]).sum(1).cpu().numpy()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)

    size = 0.01
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector([size, size, size])
    )
    print(f"{os.path.basename(hair_path).split('.')[0]} holes number : {count_holes(mesh)}")
    o3d.io.write_triangle_mesh(f"{save_path}/{os.path.basename(hair_path).split('.')[0]}.obj", mesh)

    ms = ml.MeshSet()
    ms.load_new_mesh(f"{save_path}/{os.path.basename(hair_path).split('.')[0]}.obj")
    ms.apply_filter('compute_selection_from_mesh_border')
    v_selection = ms.current_mesh().vertex_selection_array()
    smoothed_mesh = laplacian_smoothing(copy.deepcopy(mesh), v_selection, iterations=10, lambda_factor=0.1)
    o3d.io.write_triangle_mesh(f"{save_path_processed}/{os.path.basename(hair_path).split('.')[0]}.obj", smoothed_mesh)

    hair = hair - np.asarray(mesh.vertices)[:, None] + np.asarray(smoothed_mesh.vertices)[:, None]
    cols = np.concatenate((np.repeat(np.random.rand(hair.shape[0], 3)[:, None], 100, axis=1), np.ones((hair.shape[0], 100, 1))), axis=-1).reshape(-1, 4)        
    trimesh.PointCloud(hair.reshape(-1, 3), colors=cols).export(f"{save_strand_path_processed}/{os.path.basename(hair_path).split('.')[0]}.obj")
    print(os.path.basename(hair_path).split('.')[0])


    




