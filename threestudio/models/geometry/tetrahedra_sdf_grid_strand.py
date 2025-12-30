from dataclasses import dataclass, field

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import threestudio
from threestudio.models.geometry.base import (
    BaseExplicitGeometry,
    BaseGeometry,
    contract_to_unisphere,
)
from threestudio.models.geometry.implicit_sdf_strand import ImplicitSDF
from threestudio.models.geometry.implicit_volume import ImplicitVolume
from threestudio.models.isosurface import MarchingTetrahedraHelper
from threestudio.models.mesh import Mesh
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.ops import scale_tensor
from threestudio.utils.typing import *
from threestudio.utils.misc import broadcast, get_rank
from threestudio.utils.misc import cleanup, get_device, load_module_weights
# new
import sys
import yaml
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

@threestudio.register("tetrahedra-sdf-grid-strand")
class TetrahedraSDFGrid(BaseExplicitGeometry):
    @dataclass
    class Config(BaseExplicitGeometry.Config):
        isosurface_resolution: int = 128
        isosurface_deformable_grid: bool = True
        isosurface_remove_outliers: bool = False
        isosurface_outlier_n_faces_threshold: Union[int, float] = 0.01

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
        freq_encoding_config: dict = field(
            default_factory=lambda: {
                "otype": "Frequency",
                "n_frequencies": 4
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
        shape_init: Optional[str] = None
        shape_init_params: Optional[Any] = None
        force_shape_init: bool = False
        geometry_only: bool = False
        fix_geometry: bool = False

        # new
        test_save_path: str = "./.threestudio_cache"
        NHC_config: dict = field(default_factory=dict)
        input_type: str = "xyz"

    cfg: Config

    def configure(self) -> None:
        super().configure()

        # this should be saved to state_dict, register as buffer
        self.isosurface_bbox: Float[Tensor, "2 3"]
        self.register_buffer("isosurface_bbox", self.bbox.clone())

        self.isosurface_helper = MarchingTetrahedraHelper(
            self.cfg.isosurface_resolution,
            f"load/tets/{self.cfg.isosurface_resolution}_tets.npz",
        )

        self.sdf: Float[Tensor, "Nv 1"]
        self.deformation: Optional[Float[Tensor, "Nv 3"]]

        if not self.cfg.fix_geometry:
            self.register_parameter(
                "sdf",
                nn.Parameter(
                    torch.zeros(
                        (self.isosurface_helper.grid_vertices.shape[0], 1),
                        dtype=torch.float32,
                    )
                ),
            )
            if self.cfg.isosurface_deformable_grid:
                self.register_parameter(
                    "deformation",
                    nn.Parameter(
                        torch.zeros_like(self.isosurface_helper.grid_vertices)
                    ),
                )
            else:
                self.deformation = None
        else:
            self.register_buffer(
                "sdf",
                torch.zeros(
                    (self.isosurface_helper.grid_vertices.shape[0], 1),
                    dtype=torch.float32,
                ),
            )
            if self.cfg.isosurface_deformable_grid:
                self.register_buffer(
                    "deformation",
                    torch.zeros_like(self.isosurface_helper.grid_vertices),
                )
            else:
                self.deformation = None

        self.mesh: Optional[Mesh] = None
        self.register_buffer("scale", torch.ones(3).float())
        self.register_buffer("trans", torch.zeros(3).float())
        NHC_config = load_config(self.cfg.NHC_config.config_path)
        NHC_config['textured_strands']["path_to_mesh"] = self.cfg.NHC_config.init_mesh_path
        self.strands_model = OptimizableTexturedStrands(**NHC_config['textured_strands'], diffusion_cfg=NHC_config['diffusion_prior'])

        if not self.cfg.geometry_only:
            self.encoding = get_encoding(
                self.cfg.n_input_dims, self.cfg.pos_encoding_config
            )
            self.freq_encoding = get_encoding(
                self.cfg.n_input_dims, self.cfg.freq_encoding_config
            )
            input_dims = 0
            if "xyz" in self.cfg.input_type:
                input_dims += self.encoding.n_output_dims
            if "uvh" in self.cfg.input_type:
                input_dims += self.encoding.n_output_dims
            if "ori" in self.cfg.input_type:
                input_dims += self.freq_encoding.n_output_dims
            if "strands_tex" in self.cfg.input_type:
                input_dims += NHC_config["textured_strands"]["appearance_descriptor_size"]
            
            self.feature_network = get_mlp(
                input_dims,
                self.cfg.n_feature_dims,
                self.cfg.mlp_network_config,
            )
            self.input_dims = input_dims



    def initialize_shape(self) -> None:
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
            assert (
                isinstance(self.cfg.shape_init_params, Sized)
                and len(self.cfg.shape_init_params) == 3
            )
            size = torch.as_tensor(self.cfg.shape_init_params).to(self.device)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return ((points_rand / size) ** 2).sum(
                    dim=-1, keepdim=True
                ).sqrt() - 1.0  # pseudo signed distance of an ellipsoid

            get_gt_sdf = func
        elif self.cfg.shape_init == "sphere":
            assert isinstance(self.cfg.shape_init_params, float)
            radius = self.cfg.shape_init_params

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                return (points_rand**2).sum(dim=-1, keepdim=True).sqrt() - radius

            get_gt_sdf = func
        elif self.cfg.shape_init.startswith("mesh:"):
            assert isinstance(self.cfg.shape_init_params, float)
            mesh_path = self.cfg.shape_init[5:]
            if not os.path.exists(mesh_path):
                raise ValueError(f"Mesh file {mesh_path} does not exist.")

            import trimesh

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
            centroid = mesh.vertices.mean(0)
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

            from pysdf import SDF

            sdf = SDF(mesh.vertices, mesh.faces)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                # add a negative signed here
                # as in pysdf the inside of the shape has positive signed distance
                return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                    points_rand
                )[..., None]

            get_gt_sdf = func

        else:
            raise ValueError(
                f"Unknown shape initialization type: {self.cfg.shape_init}"
            )

        # Initialize SDF to a given shape when no weights are provided or force_shape_init is True
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        from tqdm import tqdm

        for _ in tqdm(
            range(1000),
            desc=f"Initializing SDF to a(n) {self.cfg.shape_init}:",
            disable=get_rank() != 0,
        ):
            points_rand = (
                torch.rand((10000, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0
            )
            sdf_gt = get_gt_sdf(points_rand)
            sdf_pred = self.forward_sdf(points_rand)
            loss = F.mse_loss(sdf_pred, sdf_gt)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # explicit broadcast to ensure param consistency across ranks
        for param in self.parameters():
            broadcast(param, src=0)

    def isosurface(self) -> Mesh:
        # return cached mesh if fix_geometry is True to save computation
        if self.cfg.fix_geometry and self.mesh is not None:
            return self.mesh
        mesh = self.isosurface_helper(self.sdf, self.deformation)
        mesh.v_pos = scale_tensor(
            mesh.v_pos, self.isosurface_helper.points_range, self.isosurface_bbox
        )
        if self.cfg.isosurface_remove_outliers:
            mesh = mesh.remove_outlier(self.cfg.isosurface_outlier_n_faces_threshold)
        self.mesh = mesh
        return mesh

    def forward(
        self, points: Float[Tensor, "*N Di"], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        if self.cfg.geometry_only:
            return {}
        assert (
            output_normal == False
        ), f"Normal output is not supported for {self.__class__.__name__}"
        points_unscaled = points  # points in the original scale
        points = contract_to_unisphere(points, self.bbox)  # points normalized to (0, 1)
        enc = self.encoding(points.view(-1, self.cfg.n_input_dims))
        features = self.feature_network(enc).view(
            *points.shape[:-1], self.cfg.n_feature_dims
        )
        return {"features": features}

    @staticmethod
    @torch.no_grad()
    def create_from(
        other: BaseGeometry,
        cfg: Optional[Union[dict, DictConfig]] = None,
        copy_net: bool = True,
        **kwargs,
    ) -> "TetrahedraSDFGrid":
        if isinstance(other, TetrahedraSDFGrid):
            instance = TetrahedraSDFGrid(cfg, **kwargs)
            assert instance.cfg.isosurface_resolution == other.cfg.isosurface_resolution
            instance.isosurface_bbox = other.isosurface_bbox.clone()
            instance.sdf.data = other.sdf.data.clone()
            if (
                instance.cfg.isosurface_deformable_grid
                and other.cfg.isosurface_deformable_grid
            ):
                assert (
                    instance.deformation is not None and other.deformation is not None
                )
                instance.deformation.data = other.deformation.data.clone()
            if (
                not instance.cfg.geometry_only
                and not other.cfg.geometry_only
                and copy_net
            ):
                instance.encoding.load_state_dict(other.encoding.state_dict())
                instance.feature_network.load_state_dict(
                    other.feature_network.state_dict()
                )
                
            instance.scale = other.scale
            instance.trans = other.trans
            return instance
        elif isinstance(other, ImplicitVolume):
            instance = TetrahedraSDFGrid(cfg, **kwargs)
            if other.cfg.isosurface_method != "mt":
                other.cfg.isosurface_method = "mt"
                threestudio.warn(
                    f"Override isosurface_method of the source geometry to 'mt'"
                )
            if other.cfg.isosurface_resolution != instance.cfg.isosurface_resolution:
                other.cfg.isosurface_resolution = instance.cfg.isosurface_resolution
                threestudio.warn(
                    f"Override isosurface_resolution of the source geometry to {instance.cfg.isosurface_resolution}"
                )
            mesh, _ = other.isosurface()
            instance.isosurface_bbox = mesh.extras["bbox"]
            instance.sdf.data = (
                mesh.extras["grid_level"].to(instance.sdf.data).clamp(-1, 1)
            )
            if not instance.cfg.geometry_only and copy_net:
                instance.encoding.load_state_dict(other.encoding.state_dict())
                instance.feature_network.load_state_dict(
                    other.feature_network.state_dict()
                )
            return instance
        elif isinstance(other, ImplicitSDF):
            instance = TetrahedraSDFGrid(cfg, **kwargs)
            assert other.cfg.isosurface_resolution == instance.cfg.isosurface_resolution
            if other.cfg.isosurface_method != "mt":
                other.cfg.isosurface_method = "mt"
                threestudio.warn(
                    f"Override isosurface_method of the source geometry to 'mt'"
                )
            if other.cfg.isosurface_resolution != instance.cfg.isosurface_resolution:
                other.cfg.isosurface_resolution = instance.cfg.isosurface_resolution
                threestudio.warn(
                    f"Override isosurface_resolution of the source geometry to {instance.cfg.isosurface_resolution}"
                )

            mesh = other.isosurface()
            if isinstance(mesh, tuple):
                mesh = mesh[0]

            instance.isosurface_bbox = mesh.extras["bbox"]
            instance.sdf.data = mesh.extras["grid_level"].to(instance.sdf.data)
            if (
                instance.cfg.isosurface_deformable_grid
                and other.cfg.isosurface_deformable_grid
            ):
                assert instance.deformation is not None
                instance.deformation.data = mesh.extras["grid_deformation"].to(
                    instance.deformation.data
                )
            if not instance.cfg.geometry_only and copy_net:
                instance.encoding.load_state_dict(other.encoding.state_dict())
                instance.feature_network.load_state_dict(
                    other.feature_network.state_dict()
                )
            instance.scale = other.scale
            instance.trans = other.trans

            return instance
        else:
            raise TypeError(
                f"Cannot create {TetrahedraSDFGrid.__name__} from {other.__class__.__name__}"
            )

    @staticmethod
    @torch.no_grad()
    def create_strand_from(
        other: BaseGeometry,
        prev_geometry_cfg: Optional[Union[dict, DictConfig]] = None,
        cfg: Optional[Union[dict, DictConfig]] = None,
        init_NHC_path = None, head_mesh = None,
        copy_net_path: str = None,
        **kwargs,
    ) -> "TetrahedraSDFGrid":
        # assert other.cfg.isosurface_resolution == instance.cfg.isosurface_resolution
        # if other.cfg.isosurface_method != "mt":
        #     other.cfg.isosurface_method = "mt"
        #     threestudio.warn(
        #         f"Override isosurface_method of the source geometry to 'mt'"
        #     )
        # if other.cfg.isosurface_resolution != instance.cfg.isosurface_resolution:
        #     other.cfg.isosurface_resolution = instance.cfg.isosurface_resolution
        #     threestudio.warn(
        #         f"Override isosurface_resolution of the source geometry to {instance.cfg.isosurface_resolution}"
        #     )

        # mesh = other.isosurface()
        # if isinstance(mesh, tuple):
        #     mesh = mesh[0]

        # instance.isosurface_bbox = mesh.extras["bbox"]
        # instance.sdf.data = mesh.extras["grid_level"].to(instance.sdf.data)
        # if (
        #     instance.cfg.isosurface_deformable_grid
        #     and other.cfg.isosurface_deformable_grid
        # ):
        #     assert instance.deformation is not None
        #     instance.deformation.data = mesh.extras["grid_deformation"].to(
        #         instance.deformation.data
        #     )
        cfg["NHC_config"] = prev_geometry_cfg.NHC_config
        instance = TetrahedraSDFGrid(cfg, **kwargs)
        instance.strands_model.init_NHC(init_NHC_path, head_mesh, instance.device)
        tmp_state_dict = other.strands_model.state_dict()
        instance.strands_model.register_buffer('fixed_idx', tmp_state_dict["fixed_idx"])
        instance.strands_model.load_state_dict(tmp_state_dict, strict=True)
        if not instance.cfg.geometry_only and copy_net_path is not None:
            copy_net_path = copy_net_path.replace("stage1-geometry", "stage2-coarse-texture")
            instance.encoding.load_state_dict(load_module_weights(copy_net_path, module_name="geometry.encoding", map_location=instance.device)[0], strict=True)
            instance.freq_encoding.load_state_dict(load_module_weights(copy_net_path, module_name="geometry.freq_encoding", map_location=instance.device)[0], strict=True)
            instance.feature_network.load_state_dict(load_module_weights(copy_net_path, module_name="geometry.feature_network", map_location=instance.device)[0], strict=True)
            instance.strands_model.app_texture_decoder.load_state_dict(load_module_weights(copy_net_path, module_name="geometry.strands_model.app_texture_decoder", map_location=instance.device)[0], strict=True)
        instance.scale = other.scale
        instance.trans = other.trans
        instance.head_mesh_center = torch.from_numpy(((head_mesh.vertices).max(0) + (head_mesh.vertices).min(0)) / 2).float().to(instance.device)

        return instance

    def get_strand(self, num_strands=-1, sample_mode="fixed", run_batch=2000, scale_factor=None, reset_sampled_mesh=False):
        return self.strands_model(num_strands=num_strands, sample_mode=sample_mode, run_batch=run_batch, scale_factor=scale_factor, reset_sampled_mesh=reset_sampled_mesh)
    
    def get_strand_mesh(self, num_strands=-1, sample_mode="fixed", run_batch=2000, scale_factor=None, reset_sampled_mesh=False, radius_scale=1, num_edges=2):
        strand_out = self.strands_model(num_strands=num_strands, sample_mode=sample_mode, run_batch=run_batch, scale_factor=scale_factor, reset_sampled_mesh=reset_sampled_mesh)
        if self.mesh is None:
            if num_strands == -1:
                num_strands = self.strands_model.num_strands
            strand_verts, strand_faces, indices = build_prisms(strands=strand_out[0], center=self.head_mesh_center, reverse_flag=True, w=np.sqrt(self.strands_model.num_strands/num_strands)*radius_scale*self.strands_model.strand_radius.cpu().item(), num_edges=num_edges)
            strand_mesh = Mesh(v_pos=strand_verts.reshape(-1, 3).detach(), t_pos_idx=strand_faces.reshape(-1, 3).detach())
            self.mesh = strand_mesh
        if "uvh" in self.cfg.input_type:
            h = torch.linspace(-1, 1, steps=strand_out[0].shape[1])[None, :, None].tile(strand_out[0].shape[0], 1, 1).to(strand_out[0].device)
            uv = self.strands_model.uvs[self.strands_model.fixed_idx][:, None, :].tile(1, strand_out[0].shape[1], 1)
            strand_uvh = torch.cat((uv, h), dim=-1)
            strand_mesh_uvh = strand_uvh[:, :, None, :].tile(1, 1, int(self.mesh.v_pos.shape[0] / strand_uvh.shape[0] / strand_uvh.shape[1]) ,1).reshape(strand_uvh.shape[0], -1, 3)
            if num_edges == 2:
                self.mesh_uvh = strand_mesh_uvh.reshape(-1, 3).detach()
            else:
                self.mesh_uvh = torch.cat((strand_mesh_uvh, strand_uvh[:, 0:1, :], strand_uvh[:, -1:, :]), dim=1).reshape(-1, 3).detach()
        if "ori" in self.cfg.input_type:
            strands_ori = strand_out[0][:, 1:, :].detach() - strand_out[0][:, :-1, :].detach()
            strands_ori = torch.cat((strands_ori[:, :1, :], strands_ori), dim=1)
            strands_ori = strands_ori / (torch.norm(strands_ori, dim=-1)[:, :, None] + 1e-20)
            strand_mesh_ori = strands_ori[:, :, None, :].tile(1, 1, int(self.mesh.v_pos.shape[0] / strands_ori.shape[0] / strands_ori.shape[1]) ,1).reshape(strands_ori.shape[0], -1, 3)
            if num_edges == 2:
                self.mesh_ori = strand_mesh_ori.reshape(-1, 3).detach()
            else:
                self.mesh_ori = torch.cat((strand_mesh_ori, strands_ori[:, 0:1, :], strands_ori[:, -1:, :]), dim=1).reshape(-1, 3).detach()
        if "strands_tex" in self.cfg.input_type:
            strand_ztex = strand_out[2][:, None, :].tile(1, strand_out[0].shape[1], 1)
            strand_mesh_ztex = strand_ztex[:, :, None, :].tile(1, 1, int(self.mesh.v_pos.shape[0] / strand_ztex.shape[0] / strand_ztex.shape[1]) ,1).reshape(strand_ztex.shape[0], -1, strand_ztex.shape[-1])
            if num_edges == 2:
                self.mesh_ztex = strand_mesh_ztex.reshape(-1, strand_ztex.shape[-1])
            else:
                self.mesh_ztex = torch.cat((strand_mesh_ztex, strand_ztex[:, 0:1, :], strand_ztex[:, -1:, :]), dim=1).reshape(-1, strand_ztex.shape[-1])

        return self.mesh, strand_out[0], strand_out[1], strand_out[2]

    def get_strand_texture(
        self, input_dict: Dict[str, Float[Tensor, "..."]], output_normal: bool = False
    ) -> Dict[str, Float[Tensor, "..."]]:
        if self.cfg.geometry_only:
            return {}
        assert (
            output_normal == False
        ), f"Normal output is not supported for {self.__class__.__name__}"
        input_list = []
        if "xyz" in input_dict.keys():
            xyz = contract_to_unisphere(input_dict["xyz"], self.bbox)  # points normalized to (0, 1)
            enc = self.encoding(xyz)
            input_list.append(enc)
        if "uvh" in input_dict.keys():
            uvh = input_dict["uvh"] / 2 + 0.5
            enc = self.encoding(uvh)
            input_list.append(enc)
        if "ori" in input_dict.keys():
            ori = input_dict["ori"]
            enc = self.freq_encoding(ori)
            input_list.append(enc)
        if "strands_tex" in input_dict.keys():
            strands_tex = input_dict["strands_tex"]
            enc = strands_tex
            input_list.append(enc)
        features = self.feature_network(torch.cat(input_list, dim=-1))
        return {"features": features}

    def export(self, points: Float[Tensor, "*N Di"], **kwargs) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.cfg.geometry_only or self.cfg.n_feature_dims == 0:
            return out
        input_list = []
        if "xyz" in self.cfg.input_type:
            xyz = contract_to_unisphere((self.mesh.v_pos.detach() - self.trans) / self.scale, self.bbox)  # points normalized to (0, 1)
            enc = self.encoding(xyz)
            input_list.append(enc)
        if "uvh" in self.cfg.input_type:
            uvh = self.mesh_uvh / 2 + 0.5
            enc = self.encoding(uvh)
            input_list.append(enc)
        if "ori" in self.cfg.input_type:
            ori = self.mesh_ori
            enc = self.freq_encoding(ori)
            input_list.append(enc)
        if "strands_tex" in self.cfg.input_type:
            strands_tex = self.mesh_ztex
            enc = strands_tex
            input_list.append(enc)
        features = self.feature_network(torch.cat(input_list, dim=-1))
        out.update(
            {
                "features": features,
            }
        )
        return out
