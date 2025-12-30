import os
import random
from dataclasses import dataclass, field
import smplx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import heapq
import threestudio
from threestudio.models.geometry.base import BaseImplicitGeometry, contract_to_unisphere
from threestudio.models.mesh import Mesh
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.misc import broadcast, get_rank
from threestudio.utils.typing import *
from pysdf import SDF
import scipy.sparse
import cubvh
import pymeshlab as ml
import copy
import trimesh
from pytorch3d.ops import knn_points
import pickle
import yaml
# new
import sys
import json
from pytorch3d.structures import Meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'NeuralHaircut'))
import contextlib
import argparse
import trimesh
import open3d as o3d
from NeuralHaircut.src.hair_networks.optimizable_textured_strands_customization_pretrain import OptimizableTexturedStrands
from threestudio.utils.build_prisms import build_prisms

@contextlib.contextmanager
def freeze_gradients(model):
    is_training = model.training
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    yield
    if is_training:
        model.train()
    for p in model.parameters():
        p.requires_grad_(True)


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

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



def smooth_mesh(mesh, mask=None, iterations=10, lambda_param=0.5):
    vertices = mesh.vertices
    faces = mesh.faces

    for _ in range(iterations):
        vertex_neighbors = {i: set() for i in range(len(vertices))}
        for face in faces:
            for i in range(3):
                vertex_neighbors[face[i]].add(face[(i+1) % 3])
                vertex_neighbors[face[i]].add(face[(i+2) % 3])

        new_vertices = np.copy(vertices)
        if mask is None:
            for i, neighbors in vertex_neighbors.items():
                if len(neighbors) > 0:
                    new_vertices[i] = (1 - lambda_param) * vertices[i] + lambda_param * np.mean(vertices[list(neighbors)], axis=0)
        else:
            for i, neighbors in vertex_neighbors.items():
                if len(neighbors) > 0 and mask[i] == True:
                    new_vertices[i] = (1 - lambda_param) * vertices[i] + lambda_param * np.mean(vertices[list(neighbors)], axis=0)

        vertices = new_vertices

    return trimesh.Trimesh(vertices=vertices, faces=faces)


@threestudio.register("implicit-sdf-strand")
class ImplicitSDF(BaseImplicitGeometry):
    @dataclass
    class Config(BaseImplicitGeometry.Config):
        n_input_dims: int = 3
        n_feature_dims: int = 3
        pos_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "HashGrid",
                "n_levels": 16,
                "n_features_per_level": 2,
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.447269237440378,
            }
        )
        mlp_network_config: dict = field(
            default_factory=lambda: {
                "otype": "VanillaMLP",
                "activation": "ReLU",
                "output_activation": "none",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        normal_type: Optional[
            str
        ] = "finite_difference"  # in ['pred', 'finite_difference', 'finite_difference_laplacian']
        finite_difference_normal_eps: Union[
            float, str
        ] = 0.01  # in [float, "progressive"]
        shape_init: Optional[str] = None
        shape_init_params: Optional[Any] = None
        shape_init_mesh_up: str = "+z"
        shape_init_mesh_front: str = "+x"
        force_shape_init: bool = False
        sdf_bias: Union[float, str] = 0.0
        sdf_bias_params: Optional[Any] = None

        # no need to removal outlier for SDF
        isosurface_remove_outliers: bool = False

        # improve the resolution of DMTET at these steps
        progressive_resolution: dict = field(default_factory=dict)
        # progressive_resolution_steps: Optional[Any] = None

        # new
        test_save_path: str = "./.threestudio_cache"
        gender: str = "neutral"
        flame_path: str = "./load/flame_models"
        start_temp_loss_step: int = 100000
        update_temp_loss_step: int = 100000
        start_collision_loss_step: int = 100000
        NHC_config: dict = field(default_factory=dict)

    cfg: Config

    def configure(self) -> None:
        super().configure()
        self.encoding = get_encoding(
            self.cfg.n_input_dims, self.cfg.pos_encoding_config
        )
        self.sdf_network = get_mlp(
            self.encoding.n_output_dims, 1, self.cfg.mlp_network_config
        )

        if self.cfg.n_feature_dims > 0:
            self.feature_network = get_mlp(
                self.encoding.n_output_dims,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )

        if self.cfg.normal_type == "pred":
            self.normal_network = get_mlp(
                self.encoding.n_output_dims, 3, self.cfg.mlp_network_config
            )
        if self.cfg.isosurface_deformable_grid:
            assert (
                self.cfg.isosurface_method == "mt"
            ), "isosurface_deformable_grid only works with mt"
            self.deformation_network = get_mlp(
                self.encoding.n_output_dims, 3, self.cfg.mlp_network_config
            )

        self.finite_difference_normal_eps: Optional[float] = None
        self.cached_sdf = None

        # new
        self.flame_model = smplx.create(
                self.cfg.flame_path,
                model_type="flame",
                gender=self.cfg.gender,
                use_face_contour=False,
                num_expression_coeffs=10,
                ext="pkl",
            )
        self.temp_mesh = None
        self.temp_mesh_sdf = None
        self.temp_strand = None
        self.head_mesh = None
        self.head_mesh_sdf = None

        self.register_buffer("scale", torch.ones(3).float())
        self.register_buffer("trans", torch.zeros(3).float())


        NHC_config = load_config(self.cfg.NHC_config.config_path)
        NHC_config['textured_strands']["path_to_mesh"] = self.cfg.NHC_config.init_mesh_path
        self.strands_model = OptimizableTexturedStrands(**NHC_config['textured_strands'], diffusion_cfg=NHC_config['diffusion_prior'])
        


    def openmesh2clothmesh(self, init_mesh, scale_factor1=1, scale_factor2=1, hollow_flag=True, smooth_flag=True):
        import trimesh
        os.makedirs(f"{self.cfg.test_save_path}/obj/init_mesh", exist_ok=True)
        init_mesh.export(f"{self.cfg.test_save_path}/obj/init_mesh/init_open_mesh.obj")
        if hollow_flag:
            init_mesh_scale1 = copy.copy(init_mesh)
            init_mesh_scale1.vertices = init_mesh_scale1.vertices + init_mesh_scale1.vertex_normals * (scale_factor1 - 1)
            init_mesh_scale1.export(f"{self.cfg.test_save_path}/obj/init_mesh/init_open_mesh_scale1.obj")
            init_mesh_scale2 = copy.copy(init_mesh)
            init_mesh_scale2.vertices = init_mesh_scale2.vertices + init_mesh_scale2.vertex_normals * (scale_factor2 - 1)
            init_mesh_scale2.export(f"{self.cfg.test_save_path}/obj/init_mesh/init_open_mesh_scale2.obj")
            # ms = ml.MeshSet()
            # ms.load_new_mesh(f"{self.cfg.test_save_path}/obj/init_mesh/init_open_mesh.obj")
            # ms.apply_filter('meshing_close_holes', maxholesize = init_mesh.vertices.shape[0])
            # ms.save_current_mesh(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh.obj")
            ms = ml.MeshSet()
            ms.load_new_mesh(f"{self.cfg.test_save_path}/obj/init_mesh/init_open_mesh.obj")
            ms.apply_filter('compute_selection_from_mesh_border')
            v_selection = ms.current_mesh().vertex_selection_array()
            border_vertices_idx = list(np.unique(np.where(v_selection==True)[0]))
            replace_dict = {}
            for i in border_vertices_idx:
                replace_dict[i + init_mesh.vertices.shape[0]] = i 
            tmp_faces = init_mesh_scale1.faces + init_mesh.vertices.shape[0]
            for k, v in replace_dict.items():
                tmp_faces = np.where(tmp_faces==k, v, tmp_faces)
            obj_str = ""
            for i in range(init_mesh.vertices.shape[0]):
                if i in border_vertices_idx:
                    tmp_vert = (init_mesh_scale1.vertices[i] + init_mesh_scale2.vertices[i]) / 2
                    obj_str += f"v {tmp_vert[0]} {tmp_vert[1]} {tmp_vert[2]}"
                    obj_str += "\n"
                else:
                    obj_str += f"v {init_mesh_scale1.vertices[i][0]} {init_mesh_scale1.vertices[i][1]} {init_mesh_scale1.vertices[i][2]}"
                    obj_str += "\n"
            for i in range(init_mesh.vertices.shape[0]):
                if i in border_vertices_idx:
                    tmp_vert = (init_mesh_scale1.vertices[i] + init_mesh_scale2.vertices[i]) / 2
                    obj_str += f"v {tmp_vert[0]} {tmp_vert[1]} {tmp_vert[2]}"
                    obj_str += "\n"
                else:
                    obj_str += f"v {init_mesh_scale2.vertices[i][0]} {init_mesh_scale2.vertices[i][1]} {init_mesh_scale2.vertices[i][2]}"
                    obj_str += "\n"
            for i in range(init_mesh_scale1.faces.shape[0]):
                obj_str += f"f {init_mesh_scale1.faces[i][0]+1} {init_mesh_scale1.faces[i][2]+1} {init_mesh_scale1.faces[i][1]+1}"
                obj_str += "\n"
            for i in range(tmp_faces.shape[0]):
                obj_str += f"f {tmp_faces[i][0]+1} {tmp_faces[i][1]+1} {tmp_faces[i][2]+1}"
                obj_str += "\n"
            obj_str += "\n"
            with open(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh.obj", "w") as f:
                f.write(obj_str)
            init_close_mesh = trimesh.load(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh.obj")
            init_close_mesh.export(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh.obj")
            for i in range(10):
                ms = ml.MeshSet()
                ms.load_new_mesh(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh.obj")
                ms.apply_filter('compute_selection_from_mesh_border')
                v_selection = ms.current_mesh().vertex_selection_array()
                tmp = np.where(v_selection==True)[0].shape[0]
                if tmp == 0:
                    break
                ms = ml.MeshSet()
                ms.load_new_mesh(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh.obj")
                ms.apply_filter('meshing_close_holes', maxholesize = init_close_mesh.vertices.shape[0])
                trimesh.Trimesh(vertices=ms.current_mesh().vertex_matrix(), faces=ms.current_mesh().face_matrix()).export(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh.obj")
                print(i)
                print(tmp)
            init_close_mesh = trimesh.load(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh.obj")
            if smooth_flag:
                smooth_mask = np.zeros(init_close_mesh.vertices.shape[0])
                smooth_mask[border_vertices_idx] = 1
                smooth_mask = smooth_mask.astype(np.bool_)
                init_close_mesh = smooth_mesh(init_close_mesh, mask=smooth_mask, iterations=10, lambda_param=0.5)
                init_close_mesh.export(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh_smooth.obj")
                init_close_mesh = trimesh.load(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh_smooth.obj")
        else:
            init_mesh_scale = copy.copy(init_mesh)
            init_mesh_scale.vertices = init_mesh_scale.vertices + init_mesh_scale.vertex_normals * (scale_factor1 - 1)
            init_mesh_scale.export(f"{self.cfg.test_save_path}/obj/init_mesh/init_open_mesh_scale.obj")
            ms = ml.MeshSet()
            ms.load_new_mesh(f"{self.cfg.test_save_path}/obj/init_mesh/init_open_mesh_scale.obj")
            ms.apply_filter('meshing_close_holes', maxholesize = init_mesh_scale.vertices.shape[0])
            trimesh.Trimesh(vertices=ms.current_mesh().vertex_matrix(), faces=ms.current_mesh().face_matrix()).export(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh_scale.obj")
            for i in range(10):
                ms = ml.MeshSet()
                ms.load_new_mesh(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh_scale.obj")
                ms.apply_filter('compute_selection_from_mesh_border')
                v_selection = ms.current_mesh().vertex_selection_array()
                tmp = np.where(v_selection==True)[0].shape[0]
                if tmp == 0:
                    break
                ms = ml.MeshSet()
                ms.load_new_mesh(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh_scale.obj")
                ms.apply_filter('meshing_close_holes', maxholesize = init_mesh_scale.vertices.shape[0])
                trimesh.Trimesh(vertices=ms.current_mesh().vertex_matrix(), faces=ms.current_mesh().face_matrix()).export(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh_scale.obj")
                print(i)
                print(tmp)
            init_close_mesh = trimesh.load(f"{self.cfg.test_save_path}/obj/init_mesh/init_close_mesh_scale.obj")

        return init_close_mesh


    def openface2clothface(self, init_mesh, smooth_flag=True):
        import trimesh
        os.makedirs(f"{self.cfg.test_save_path}/obj/init_face", exist_ok=True)
        init_mesh.export(f"{self.cfg.test_save_path}/obj/init_face/init_open_face.obj")
        init_mesh_scale1 = copy.copy(init_mesh)
        init_mesh_scale1.vertices = init_mesh_scale1.vertices
        init_mesh_scale1.export(f"{self.cfg.test_save_path}/obj/init_face/init_open_face_scale1.obj")
        init_mesh_scale2 = copy.copy(init_mesh)
        init_mesh_scale2.vertices[:, 0] = init_mesh_scale2.vertices[:, 0] + 1
        init_mesh_scale2.export(f"{self.cfg.test_save_path}/obj/init_face/init_open_face_scale2.obj")
        ms = ml.MeshSet()
        ms.load_new_mesh(f"{self.cfg.test_save_path}/obj/init_face/init_open_face.obj")
        ms.apply_filter('compute_selection_from_mesh_border')
        v_selection = ms.current_mesh().vertex_selection_array()
        border_vertices_idx = list(np.unique(np.where(v_selection==True)[0]))
        replace_dict = {}
        for i in border_vertices_idx:
            replace_dict[i + init_mesh.vertices.shape[0]] = i 
        tmp_faces = init_mesh_scale1.faces + init_mesh.vertices.shape[0]
        for k, v in replace_dict.items():
            tmp_faces = np.where(tmp_faces==k, v, tmp_faces)
        obj_str = ""
        for i in range(init_mesh.vertices.shape[0]):
            if i in border_vertices_idx:
                tmp_vert = (init_mesh_scale1.vertices[i] + init_mesh_scale2.vertices[i]) / 2
                obj_str += f"v {tmp_vert[0]} {tmp_vert[1]} {tmp_vert[2]}"
                obj_str += "\n"
            else:
                obj_str += f"v {init_mesh_scale1.vertices[i][0]} {init_mesh_scale1.vertices[i][1]} {init_mesh_scale1.vertices[i][2]}"
                obj_str += "\n"
        for i in range(init_mesh.vertices.shape[0]):
            if i in border_vertices_idx:
                tmp_vert = (init_mesh_scale1.vertices[i] + init_mesh_scale2.vertices[i]) / 2
                obj_str += f"v {tmp_vert[0]} {tmp_vert[1]} {tmp_vert[2]}"
                obj_str += "\n"
            else:
                obj_str += f"v {init_mesh_scale2.vertices[i][0]} {init_mesh_scale2.vertices[i][1]} {init_mesh_scale2.vertices[i][2]}"
                obj_str += "\n"
        for i in range(init_mesh_scale1.faces.shape[0]):
            obj_str += f"f {init_mesh_scale1.faces[i][0]+1} {init_mesh_scale1.faces[i][2]+1} {init_mesh_scale1.faces[i][1]+1}"
            obj_str += "\n"
        for i in range(tmp_faces.shape[0]):
            obj_str += f"f {tmp_faces[i][0]+1} {tmp_faces[i][1]+1} {tmp_faces[i][2]+1}"
            obj_str += "\n"
        obj_str += "\n"
        with open(f"{self.cfg.test_save_path}/obj/init_face/init_close_face.obj", "w") as f:
            f.write(obj_str)
        init_close_mesh = trimesh.load(f"{self.cfg.test_save_path}/obj/init_face/init_close_face.obj")
        init_close_mesh.export(f"{self.cfg.test_save_path}/obj/init_face/init_close_face.obj")
        for i in range(10):
            ms = ml.MeshSet()
            ms.load_new_mesh(f"{self.cfg.test_save_path}/obj/init_face/init_close_face.obj")
            ms.apply_filter('compute_selection_from_mesh_border')
            v_selection = ms.current_mesh().vertex_selection_array()
            tmp = np.where(v_selection==True)[0].shape[0]
            if tmp == 0:
                break
            ms = ml.MeshSet()
            ms.load_new_mesh(f"{self.cfg.test_save_path}/obj/init_face/init_close_face.obj")
            ms.apply_filter('meshing_close_holes', maxholesize = init_close_mesh.vertices.shape[0])
            trimesh.Trimesh(vertices=ms.current_mesh().vertex_matrix(), faces=ms.current_mesh().face_matrix()).export(f"{self.cfg.test_save_path}/obj/init_face/init_close_face.obj")
            print(i)
            print(tmp)
        init_close_mesh = trimesh.load(f"{self.cfg.test_save_path}/obj/init_face/init_close_face.obj")
        if smooth_flag:
            smooth_mask = np.zeros(init_close_mesh.vertices.shape[0])
            smooth_mask[border_vertices_idx] = 1
            smooth_mask = smooth_mask.astype(np.bool_)
            init_close_mesh = smooth_mesh(init_close_mesh, mask=smooth_mask, iterations=10, lambda_param=0.5)
            init_close_mesh.export(f"{self.cfg.test_save_path}/obj/init_face/init_close_face_smooth.obj")
            init_close_mesh = trimesh.load(f"{self.cfg.test_save_path}/obj/init_face/init_close_face_smooth.obj")

        return init_close_mesh

    def initialize_shape(self, verts=None, faces=None, system_cfg=None, opt_flame_mesh=None) -> None:
        sdf = None
        import trimesh
        if self.cfg.shape_init is None and not self.cfg.force_shape_init:
            return

        # do not initialize shape if weights are provided
        if self.cfg.weights is not None and not self.cfg.force_shape_init:
            return

        if self.cfg.sdf_bias != 0.0:
            threestudio.warn(
                "shape_init and sdf_bias are both specified, which may lead to unexpected results."
            )

        get_gt_sdf: Callable[[Float[Tensor, "N 3"]], Float[Tensor, "N 1"]]
        assert isinstance(self.cfg.shape_init, str)
        if self.cfg.shape_init == "ellipsoid":
            assert system_cfg is not None
            assert (
                isinstance(self.cfg.shape_init_params, Sized)
                and len(self.cfg.shape_init_params) == 3
            )
            size = torch.as_tensor(self.cfg.shape_init_params).to(self.device)
            bbox_info = system_cfg.cloth.bbox_info
            self.scale = torch.tensor(bbox_info[:3]).float().to(self.device)
            self.trans = torch.tensor(bbox_info[3:]).float().to(self.device)
            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return ((points_rand / size) ** 2).sum(
                    dim=-1, keepdim=True
                ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid

            get_gt_sdf = func
        elif self.cfg.shape_init == "sphere":
            assert system_cfg is not None
            assert isinstance(self.cfg.shape_init_params, float)
            radius = self.cfg.shape_init_params
            bbox_info = system_cfg.cloth.bbox_info
            self.scale = torch.tensor(bbox_info[:3]).float().to(self.device)
            self.trans = torch.tensor(bbox_info[3:]).float().to(self.device)
            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return (points_rand**2).sum(dim=-1, keepdim=True).sqrt() - radius

            get_gt_sdf = func
        elif self.cfg.shape_init.startswith("mesh:"):
            assert system_cfg is not None
            assert isinstance(self.cfg.shape_init_params, float)
            mesh_path = self.cfg.shape_init[5:]
            if not os.path.exists(mesh_path):
                raise ValueError(f"Mesh file {mesh_path} does not exist.")

            scene = trimesh.load(mesh_path)
            if isinstance(scene, trimesh.Trimesh):
                mesh = scene
            elif isinstance(scene, trimesh.scene.Scene):
                mesh = trimesh.Trimesh()
                for obj in scene.geometry.values():
                    mesh = trimesh.util.concatenate([mesh, obj])
            else:
                raise ValueError(f"Unknown mesh type at {mesh_path}.")

            # move to center
            centroid = (mesh.vertices.max(0) + mesh.vertices.min(0)) / 2
            mesh.vertices = mesh.vertices - centroid

            # align to up-z and front-x
            dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
            dir2vec = {
                "+x": np.array([1, 0, 0]),
                "+y": np.array([0, 1, 0]),
                "+z": np.array([0, 0, 1]),
                "-x": np.array([-1, 0, 0]),
                "-y": np.array([0, -1, 0]),
                "-z": np.array([0, 0, -1]),
            }
            if (
                self.cfg.shape_init_mesh_up not in dirs
                or self.cfg.shape_init_mesh_front not in dirs
            ):
                raise ValueError(
                    f"shape_init_mesh_up and shape_init_mesh_front must be one of {dirs}."
                )
            if self.cfg.shape_init_mesh_up[1] == self.cfg.shape_init_mesh_front[1]:
                raise ValueError(
                    "shape_init_mesh_up and shape_init_mesh_front must be orthogonal."
                )
            z_, x_ = (
                dir2vec[self.cfg.shape_init_mesh_up],
                dir2vec[self.cfg.shape_init_mesh_front],
            )
            y_ = np.cross(z_, x_)
            std2mesh = np.stack([x_, y_, z_], axis=0).T
            mesh2std = np.linalg.inv(std2mesh)

            # scaling
            scale = np.abs(mesh.vertices).max()
            mesh.vertices = mesh.vertices / scale * self.cfg.shape_init_params
            mesh.vertices = np.dot(mesh2std, mesh.vertices.T).T

            bbox_info = system_cfg.cloth.bbox_info
            self.scale = torch.tensor(bbox_info[:3]).float().to(self.device)
            self.trans = torch.tensor(bbox_info[3:]).float().to(self.device)

            sdf = SDF(mesh.vertices, mesh.faces)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                # add a negative signed here
                # as in pysdf the inside of the shape has positive signed distance
                return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                    points_rand
                )[..., None]

            get_gt_sdf = func

        elif self.cfg.shape_init.startswith("head:"):
            assert verts is not None 
            assert faces is not None
            assert system_cfg is not None

            if self.cfg.shape_init.startswith("head:opt_flame:"):
                assert opt_flame_mesh is not None
                assert isinstance(self.cfg.shape_init_params, float)
                shape_init_strlist = self.cfg.shape_init.split(":")

                if shape_init_strlist[2].split("_")[0] == "NHC":
                    flame_mesh = trimesh.Trimesh(
                            vertices=opt_flame_mesh.v_pos.detach().cpu().numpy(),
                            faces=opt_flame_mesh.t_pos_idx.detach().cpu().numpy(),
                        )
                    head_mesh = trimesh.Trimesh(
                            vertices=verts,
                            faces=faces,
                        )
                    init_NHC_prompt = shape_init_strlist[2].split("_")[1]
                    with open(self.cfg.NHC_config.init_NHC_dict_path, 'r') as f:
                        init_NHC_dict = json.load(f)
                    init_NHC_path = init_NHC_dict[init_NHC_prompt]
                    self.strands_model.init_NHC(init_NHC_path, head_mesh, self.device)
                    with torch.no_grad():
                        init_NHC_strand = self.strands_model(num_strands=-1)[0].detach()

                    centroid = (init_NHC_strand.reshape(-1, 3).max(0)[0] + init_NHC_strand.reshape(-1, 3).min(0)[0]) / 2 
                    scale = torch.abs(init_NHC_strand.reshape(-1, 3) - centroid).max()
                    self.scale = self.scale * (1 / self.cfg.shape_init_params * scale)
                    self.trans = centroid
                    
                    self.temp_strand = init_NHC_strand
                    self.head_mesh = head_mesh
                    self.head_mesh_sdf = SDF(self.head_mesh.vertices, self.head_mesh.faces)

                    self.head_mesh_center = torch.from_numpy(((self.head_mesh.vertices).max(0) + (self.head_mesh.vertices).min(0)) / 2).float().to(self.device)
                    # init_NHC_strand = self.strands_model(num_strands=-1)[0]
                    # save_hair(init_NHC_strand.detach().cpu().numpy(), os.path.join(obj_save_path, f"GT_strand.obj"))
                    if True:
                        flame_face_mask = np.load(os.path.join(self.cfg.flame_path, "flame", "face_flame_index.npy"))
                        flame_face_faces = np.load(os.path.join(self.cfg.flame_path, "flame", "face_flame_faces.npy"))
                        new_mesh = trimesh.Trimesh(vertices=flame_mesh.vertices[flame_face_mask], faces=flame_face_faces)
                        close_face_mesh = self.openface2clothface(new_mesh)
                        sdf = SDF(close_face_mesh.vertices, close_face_mesh.faces)
                        def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                            # add a negative signed here
                            # as in pysdf the inside of the shape has positive signed distance
                            return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                                points_rand
                            )[..., None]
                        get_gt_sdf = func
                        # Initialize SDF to a given shape when no weights are provided or force_shape_init is True
                        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
                        from tqdm import tqdm

                        for _ in tqdm(
                            range(4000),
                            desc=f"Initializing SDF to a face:",
                            disable=get_rank() != 0,
                        ):  
                            if sdf is None:
                                points_rand = (
                                    torch.rand((40000, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0
                                )
                            else:
                                pt_num = 20000
                                bbox_points = torch.rand((pt_num, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0
                                surface_points = torch.from_numpy(sdf.sample_surface(pt_num) + np.random.normal(0, 0.05, (pt_num, 3))).float().to(self.device)
                                points_rand = torch.cat((bbox_points, surface_points), dim=0)
                                points_rand = points_rand[torch.randperm(points_rand.shape[0])]
                            sdf_gt = get_gt_sdf(points_rand)
                            sdf_pred = self.forward_sdf(points_rand)
                            loss = F.mse_loss(sdf_pred, sdf_gt)
                            optim.zero_grad()
                            loss.backward()
                            optim.step()

                        # explicit broadcast to ensure param consistency across ranks
                        for param in self.parameters():
                            broadcast(param, src=0)

                        tmp_mesh = self.isosurface()
                        trimesh.Trimesh(vertices=tmp_mesh.v_pos.detach().cpu().numpy(), faces=tmp_mesh.t_pos_idx.detach().cpu().numpy()).export(f"{self.cfg.test_save_path}/obj/init_face/init_fitting_face.obj")
                    return 
                else:
                   assert False
        else:   
            raise ValueError(
                f"Unknown shape initialization type: {self.cfg.shape_init}"
            )

        # Initialize SDF to a given shape when no weights are provided or force_shape_init is True
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        from tqdm import tqdm

        for _ in tqdm(
            range(4000),
            desc=f"Initializing SDF to a(n) {self.cfg.shape_init}:",
            disable=get_rank() != 0,
        ):  
            if sdf is None:
                points_rand = (
                    torch.rand((40000, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0
                )
            else:
                pt_num = 20000
                bbox_points = torch.rand((pt_num, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0
                surface_points = torch.from_numpy(sdf.sample_surface(pt_num) + np.random.normal(0, 0.05, (pt_num, 3))).float().to(self.device)
                points_rand = torch.cat((bbox_points, surface_points), dim=0)
                points_rand = points_rand[torch.randperm(points_rand.shape[0])]
            sdf_gt = get_gt_sdf(points_rand)
            sdf_pred = self.forward_sdf(points_rand)
            loss = F.mse_loss(sdf_pred, sdf_gt)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # explicit broadcast to ensure param consistency across ranks
        for param in self.parameters():
            broadcast(param, src=0)

    def get_shifted_sdf(
        self, points: Float[Tensor, "*N Di"], sdf: Float[Tensor, "*N 1"]
    ) -> Float[Tensor, "*N 1"]:
        sdf_bias: Union[float, Float[Tensor, "*N 1"]]
        if self.cfg.sdf_bias == "ellipsoid":
            assert (
                isinstance(self.cfg.sdf_bias_params, Sized)
                and len(self.cfg.sdf_bias_params) == 3
            )
            size = torch.as_tensor(self.cfg.sdf_bias_params).to(points)
            sdf_bias = ((points / size) ** 2).sum(
                dim=-1, keepdim=True
            ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid
        elif self.cfg.sdf_bias == "sphere":
            assert isinstance(self.cfg.sdf_bias_params, float)
            radius = self.cfg.sdf_bias_params
            sdf_bias = (points**2).sum(dim=-1, keepdim=True).sqrt() - radius
        elif isinstance(self.cfg.sdf_bias, float):
            sdf_bias = self.cfg.sdf_bias
        else:
            raise ValueError(f"Unknown sdf bias {self.cfg.sdf_bias}")
        return sdf + sdf_bias

    def get_temp_loss(self):
        pass
        return

    def get_collision_loss(self, strands, head_sdf):
        with freeze_gradients(head_sdf):
            out = head_sdf.forward_sdf(strands.reshape(-1, 3))
        sdf = out.reshape(strands.shape[0], strands.shape[1])
        strand_sdf = sdf[:, 1:]
        epsilon = torch.ones_like(strand_sdf).to(self.device) * 0
        zero_tensor = torch.zeros_like(strand_sdf).to(self.device)
        loss = torch.max(zero_tensor, epsilon - strand_sdf).sum()
        return loss

    def get_face_collision_loss(self, strands):
        with freeze_gradients(self.sdf_network):
            out = self.forward_sdf(strands.reshape(-1, 3))
        sdf = out.reshape(strands.shape[0], strands.shape[1])
        strand_sdf = sdf[:, 1:]
        epsilon = torch.ones_like(strand_sdf).to(self.device) * 0
        zero_tensor = torch.zeros_like(strand_sdf).to(self.device)
        loss = torch.max(zero_tensor, epsilon - strand_sdf).sum()
        return loss
    
    def get_ori_consist_loss(self, strands):
        strands_ori = strands[:, 1:, :] - strands[:, :-1, :]
        strands_ori = strands_ori / (torch.norm(strands_ori, dim=-1)[:, :, None] + 1e-20)
        loss = (1 - F.cosine_similarity(strands_ori[self.strands_model.sampled_strand_root_verts_self_idx].reshape(-1, 3), strands_ori[self.strands_model.sampled_strand_root_verts_neighbors_idx].reshape(-1, 3), dim=-1)).mean()
        return loss
    
    def get_norm_consist_loss(self, strands):
        strands_ori = strands[:, 1:, :] - strands[:, :-1, :]
        strands_norm  = torch.norm(strands_ori, dim=-1)
        loss = F.l1_loss(strands_norm[self.strands_model.sampled_strand_root_verts_self_idx], strands_norm[self.strands_model.sampled_strand_root_verts_neighbors_idx])
        return loss

    def get_curv_reg_loss(self, strands, target_curv):
        strands_ori = strands[:, 1:, :] - strands[:, :-1, :]
        strands_ori = strands_ori / (torch.norm(strands_ori, dim=-1)[:, :, None] + 1e-20)
        strand_cur = strands_ori[:, 1:, :] - strands_ori[:, :-1, :]
        curvature = torch.norm(strand_cur, dim=-1)
        loss = F.l1_loss(curvature, torch.ones_like(curvature).to(strands.device) * target_curv)
        return loss
    
    def get_normal_consist_loss(self, strands, faces):
        meshes = Meshes(verts=[strands[:, i, :] for i in range(0, strands.shape[1], 1)], faces=[faces for i in range(0, strands.shape[1], 1)])
        return mesh_normal_consistency(meshes)

    def get_bbox_loss(self, strands_can):
        # sdf = torch.norm(strands_can, dim=-1) - 1
        # loss = F.relu(sdf).sum()
        points = strands_can.reshape(-1, 3)
        bbox_center = torch.tensor([0.0, 0.0, 0.0]).to(strands_can.device)
        bbox_size = torch.tensor([2.0, 2.0, 2.0]).to(strands_can.device)
        q = torch.abs(points - bbox_center) - bbox_size / 2
        outside_distance = torch.norm(torch.maximum(q, torch.zeros_like(q)), dim=1)
        inside_distance = torch.minimum(torch.maximum(q[:, 0], torch.maximum(q[:, 1], q[:, 2])), torch.tensor(0.0))
        sdf = outside_distance + inside_distance
        loss = F.relu(sdf).sum()
        return loss

    def get_strand(self, num_strands=-1, sample_mode="fixed", run_batch=2000, scale_factor=None, reset_sampled_mesh=False):
        return self.strands_model(num_strands=num_strands, sample_mode=sample_mode, run_batch=run_batch, scale_factor=scale_factor, reset_sampled_mesh=reset_sampled_mesh)
    
    def get_strand_mesh(self, num_strands=-1, sample_mode="fixed", run_batch=2000, scale_factor=None, reset_sampled_mesh=False, radius_scale=1, num_edges=2):
        strand_out = self.strands_model(num_strands=num_strands, sample_mode=sample_mode, run_batch=run_batch, scale_factor=scale_factor, reset_sampled_mesh=reset_sampled_mesh)
        if num_strands == -1:
            num_strands = self.strands_model.num_strands
        strand_verts, strand_faces, indices = build_prisms(strands=strand_out[0], center=self.head_mesh_center, reverse_flag=True, w=np.sqrt(self.strands_model.num_strands/num_strands)*radius_scale*self.strands_model.strand_radius.cpu().item(), num_edges=num_edges)
        strand_mesh = Mesh(v_pos=strand_verts.reshape(-1, 3), t_pos_idx=strand_faces.reshape(-1, 3))
        return strand_mesh, strand_out[0], strand_out[1], strand_out[2]

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        grad_enabled = torch.is_grad_enabled()

        if output_normal and self.cfg.normal_type == "analytic":
            torch.set_grad_enabled(True)
            points.requires_grad_(True)

        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(
            points, self.bbox, self.unbounded
        )  # points normalized to (0, 1)

        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        sdf = self.sdf_network(enc).view(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        output = {"sdf": sdf}

        if self.cfg.n_feature_dims > 0:
            features = self.feature_network(enc).view(
                *points.shape[:-1], self.cfg.n_feature_dims
            )
            output.update({"features": features})

        if output_normal:
            if (
                self.cfg.normal_type == "finite_difference"
                or self.cfg.normal_type == "finite_difference_laplacian"
            ):
                assert self.finite_difference_normal_eps is not None
                eps: float = self.finite_difference_normal_eps
                if self.cfg.normal_type == "finite_difference_laplacian":
                    offsets: Float[Tensor, "6 3"] = torch.as_tensor(
                        [
                            [eps, 0.0, 0.0],
                            [-eps, 0.0, 0.0],
                            [0.0, eps, 0.0],
                            [0.0, -eps, 0.0],
                            [0.0, 0.0, eps],
                            [0.0, 0.0, -eps],
                        ]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 6 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    sdf_offset: Float[Tensor, "... 6 1"] = self.forward_sdf(
                        points_offset
                    )
                    sdf_grad = (
                        0.5
                        * (sdf_offset[..., 0::2, 0] - sdf_offset[..., 1::2, 0])
                        / eps
                    )
                else:
                    offsets: Float[Tensor, "3 3"] = torch.as_tensor(
                        [[eps, 0.0, 0.0], [0.0, eps, 0.0], [0.0, 0.0, eps]]
                    ).to(points_unscaled)
                    points_offset: Float[Tensor, "... 3 3"] = (
                        points_unscaled[..., None, :] + offsets
                    ).clamp(-self.cfg.radius, self.cfg.radius)
                    sdf_offset: Float[Tensor, "... 3 1"] = self.forward_sdf(
                        points_offset
                    )
                    sdf_grad = (sdf_offset[..., 0::1, 0] - sdf) / eps
                normal = F.normalize(sdf_grad, dim=-1)
            elif self.cfg.normal_type == "pred":
                normal = self.normal_network(enc).view(*points.shape[:-1], 3)
                normal = F.normalize(normal, dim=-1)
                sdf_grad = normal
            elif self.cfg.normal_type == "analytic":
                sdf_grad = -torch.autograd.grad(
                    sdf,
                    points_unscaled,
                    grad_outputs=torch.ones_like(sdf),
                    create_graph=True,
                )[0]
                normal = F.normalize(sdf_grad, dim=-1)
                if not grad_enabled:
                    sdf_grad = sdf_grad.detach()
                    normal = normal.detach()
            else:
                raise AttributeError(f"Unknown normal type {self.cfg.normal_type}")
            output.update(
                {"normal": normal, "shading_normal": normal, "sdf_grad": sdf_grad}
            )
        return output

    def forward_sdf(self, points: Float[Tensor, "*N Di"]) -> Float[Tensor, "*N 1"]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)

        sdf = self.sdf_network(
            self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        ).reshape(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        return sdf

    def forward_field(
        self, points: Float[Tensor, "*N Di"]
    ) -> Tuple[Float[Tensor, "*N 1"], Optional[Float[Tensor, "*N 3"]]]:
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        sdf = self.sdf_network(enc).reshape(*points.shape[:-1], 1)
        sdf = self.get_shifted_sdf(points_unscaled, sdf)
        deformation: Optional[Float[Tensor, "*N 3"]] = None
        if self.cfg.isosurface_deformable_grid:
            deformation = self.deformation_network(enc).reshape(*points.shape[:-1], 3)

        sdf_loss: Optional[Float[Tensor, "*N 1"]] = None
        if self.cfg.use_sdf_loss and self.cached_sdf is not None:
            selected_points_idx = torch.LongTensor(random.sample(range(points_unscaled.shape[0]), 100000)).to(self.device)
            gt_sdf = torch.from_numpy(-self.cached_sdf(points_unscaled[selected_points_idx].cpu().numpy())).to(
                    points_unscaled
                )[..., None]
            sdf_loss = F.mse_loss(gt_sdf, sdf[selected_points_idx], reduction='sum')
        return sdf, deformation, sdf_loss

    def forward_level(
        self, field: Float[Tensor, "*N 1"], threshold: float
    ) -> Float[Tensor, "*N 1"]:
        return field - threshold

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.n_feature_dims == 0:
            return out
        points_unscaled = points
        points = contract_to_unisphere(points_unscaled, self.bbox, self.unbounded)
        enc = self.encoding(points.reshape(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        out.update(
            {
                "features": features,
            }
        )
        return out

    def update_step(self, epoch: int, global_step: int, on_load_weights: bool = False):

        if global_step >= (self.cfg.start_sdf_loss_step + 1) and self.cached_sdf is None:
            import trimesh
            mesh_v_pos = np.load(f'{self.cfg.test_save_path}/mesh_v_pos.npy')
            mesh_t_pos_idx = np.load(f'{self.cfg.test_save_path}/mesh_t_pos_idx.npy')
            cached_mesh = trimesh.Trimesh(
                vertices=mesh_v_pos,
                faces=mesh_t_pos_idx,
            )
            self.cached_sdf = SDF(cached_mesh.vertices, cached_mesh.faces)

        if self.temp_mesh is not None:
            if global_step % self.cfg.update_temp_loss_step == 0:
                if global_step == 0:
                    self.temp_mesh_sdf = SDF(self.temp_mesh.vertices, self.temp_mesh.faces)
                    os.makedirs(f"{self.cfg.test_save_path}/obj/temp", exist_ok=True)
                    self.temp_mesh.export(f"{self.cfg.test_save_path}/obj/temp/temp_{global_step}.obj")
                else:
                    if self.cfg.use_sdf_loss:
                        tmp_mesh = self.isosurface()[0]
                    else:
                        tmp_mesh = self.isosurface()
                    import trimesh
                    self.temp_mesh = trimesh.Trimesh(
                            vertices=tmp_mesh.v_pos.detach().cpu().numpy(),
                            faces=tmp_mesh.t_pos_idx.detach().cpu().numpy(),
                        )
                    self.temp_mesh_sdf = SDF(self.temp_mesh.vertices, self.temp_mesh.faces)
                    self.temp_mesh.export(f"{self.cfg.test_save_path}/obj/temp/temp_{global_step}.obj")

        if (
            self.cfg.normal_type == "finite_difference"
            or self.cfg.normal_type == "finite_difference_laplacian"
        ):
            if isinstance(self.cfg.finite_difference_normal_eps, float):
                self.finite_difference_normal_eps = (
                    self.cfg.finite_difference_normal_eps
                )
            elif self.cfg.finite_difference_normal_eps == "progressive":
                # progressive finite difference eps from Neuralangelo
                # https://arxiv.org/abs/2306.03092
                hg_conf: Any = self.cfg.pos_encoding_config
                assert (
                    hg_conf.otype == "ProgressiveBandHashGrid"
                ), "finite_difference_normal_eps=progressive only works with ProgressiveBandHashGrid"
                current_level = min(
                    hg_conf.start_level
                    + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                    hg_conf.n_levels,
                )
                grid_res = hg_conf.base_resolution * hg_conf.per_level_scale ** (
                    current_level - 1
                )
                grid_size = 2 * self.cfg.radius / grid_res
                if grid_size != self.finite_difference_normal_eps:
                    threestudio.info(
                        f"Update finite_difference_normal_eps to {grid_size}"
                    )
                self.finite_difference_normal_eps = grid_size
            else:
                raise ValueError(
                    f"Unknown finite_difference_normal_eps={self.cfg.finite_difference_normal_eps}"
                )
