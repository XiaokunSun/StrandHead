import os
import random
from dataclasses import dataclass, field
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import smplx
import threestudio
import pymeshlab as ml
from threestudio.models.geometry.base import BaseImplicitGeometry, contract_to_unisphere
from threestudio.models.mesh import Mesh
from threestudio.models.networks import get_encoding, get_mlp
from threestudio.utils.misc import broadcast, get_rank
from threestudio.utils.typing import *
from pysdf import SDF
import trimesh
from threestudio.utils.utils import compute_normal
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)


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

@threestudio.register("implicit-sdf")
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
        progressive_resolution_steps: Optional[int] = None

        # new
        test_save_path: str = "./.threestudio_cache"
        gender: str = "neutral"
        flame_path: str = "./load/flame_models"
        start_flame_loss_step: int = 100000
        update_flame_loss_step: int = 100000
        opt_flame_step_num: int = 100000




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

        if self.cfg.shape_init.startswith("opt_flame"):
            tmp_strlist = self.cfg.shape_init.split(":")
            if len(tmp_strlist[0].split("_")) == 3:
                pass
            else:
                N = 3931
                self.upsample_nums = 0

            self.opt_flame_flag = True
            self.dim_betas = int(tmp_strlist[1])
            self.dim_offsets = int(tmp_strlist[2])
            self.evolving_flag = (tmp_strlist[3] == "True")
            self.flame_model = smplx.create(
                    self.cfg.flame_path,
                    model_type="flame",
                    gender=self.cfg.gender,
                    use_face_contour=False,
                    num_expression_coeffs=self.dim_betas,
                    ext="pkl",
                )
            self.register_parameter(
                'betas', nn.Parameter(torch.zeros(self.dim_betas), requires_grad=True))
            self.register_parameter(
                'v_offsets', nn.Parameter(torch.zeros(N, self.dim_offsets), requires_grad=True))
        else:
            self.opt_flame_flag = False
            self.evolving_flag = False
            self.flame_model = smplx.create(
                    self.cfg.flame_path,
                    model_type="flame",
                    gender=self.cfg.gender,
                    use_face_contour=False,
                    num_expression_coeffs=self.dim_betas,
                    ext="pkl",
                )
        self.flame_mesh = None
        self.flame_mesh_sdf = None
        self.temp_mesh = None
        self.temp_mesh_sdf = None

        self.woeyes_flame_index = np.load(os.path.join(self.cfg.flame_path, "flame", "woeyes_flame_index.npy"))
        self.closed_woeyes_flame_faces = np.load(os.path.join(self.cfg.flame_path, "flame", "closed_woeyes_flame_faces.npy"))

    def initialize_shape(self, verts=None, faces=None, system_cfg=None) -> None:
        sdf = None

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

            # adjust the position of mesh
            if "full_body" in mesh_path:
                mesh.vertices[:,1] = mesh.vertices[:,1] + 0.3
            elif "half_body" in mesh_path:
                mesh.vertices[:,1] = mesh.vertices[:,1] + 0.1
            elif "head_only" in mesh_path:
                mesh.vertices[:,2] = mesh.vertices[:,2] + 0.15
            elif "t-pose" in mesh_path:
                mesh.vertices[:,1] = mesh.vertices[:,1] + 0.4
            elif "pose" in mesh_path:
                mesh.vertices[:,1] = mesh.vertices[:,1] + 0.4

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

            sdf = SDF(mesh.vertices, mesh.faces)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                # add a negative signed here
                # as in pysdf the inside of the shape has positive signed distance
                return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                    points_rand
                )[..., None]

            get_gt_sdf = func
        elif self.cfg.shape_init.startswith("flame"):
            assert isinstance(self.cfg.shape_init_params, float)

            flame_output = self.flame_model(return_verts=True)
            flame_vertices = flame_output.vertices.detach().cpu().numpy()[0][self.woeyes_flame_index]
            flame_faces = self.closed_woeyes_flame_faces

            flame_mesh = trimesh.Trimesh(
                vertices=flame_vertices,
                faces=flame_faces,
            )
            
            # flame_mesh = self.remove_eyes_mesh(flame_mesh)

            centroid = flame_mesh.vertices.mean(0)
            flame_mesh.vertices = flame_mesh.vertices - centroid

            flame_mesh.vertices = flame_mesh.vertices * 3.8349
            flame_mesh.vertices[:,1] = flame_mesh.vertices[:,1] + 0.1
            flame_mesh.vertices[:,2] = flame_mesh.vertices[:,2] + 0.15
            


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
            scale = np.abs(flame_mesh.vertices).max()
            flame_mesh.vertices = flame_mesh.vertices / scale * self.cfg.shape_init_params
            flame_mesh.vertices = np.dot(mesh2std, flame_mesh.vertices.T).T

            sdf = SDF(flame_mesh.vertices, flame_mesh.faces)

            def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
                # add a negative signed here
                # as in pysdf the inside of the shape has positive signed distance
                return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                    points_rand
                )[..., None]

            get_gt_sdf = func

            self.flame_mesh = flame_mesh
            self.temp_mesh = flame_mesh
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

    def initialize_shape_from_flame(self, flame_mesh, global_step):
        sdf = SDF(flame_mesh.vertices, flame_mesh.faces)
        def func(points_rand: Float[Tensor, "N 3"]) -> Float[Tensor, "N 1"]:
            # add a negative signed here
            # as in pysdf the inside of the shape has positive signed distance
            return torch.from_numpy(-sdf(points_rand.cpu().numpy())).to(
                points_rand
            )[..., None]
        get_gt_sdf = func
        self.flame_mesh = flame_mesh
        self.temp_mesh = flame_mesh
        self.flame_mesh_sdf = SDF(self.flame_mesh.vertices, self.flame_mesh.faces)
        self.temp_mesh_sdf = SDF(self.temp_mesh.vertices, self.temp_mesh.faces)
        os.makedirs(f"{self.cfg.test_save_path}/obj/flame", exist_ok=True)
        self.flame_mesh.export(f"{self.cfg.test_save_path}/obj/flame/flame.obj")
        self.temp_mesh.export(f"{self.cfg.test_save_path}/obj/flame/temp_{global_step}.obj")
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

    def close_mesh(self, head_mesh, iters=10):
        head_mesh.export(f"{self.cfg.test_save_path}/obj/flame/flame_close_tmp.obj")
        for i in range(iters):
            ms = ml.MeshSet()
            ms.load_new_mesh(f"{self.cfg.test_save_path}/obj/flame/flame_close_tmp.obj")
            ms.apply_filter('compute_selection_from_mesh_border')
            v_selection = ms.current_mesh().vertex_selection_array()
            tmp = np.where(v_selection==True)[0].shape[0]
            if tmp == 0:
                break
            ms = ml.MeshSet()
            ms.load_new_mesh(f"{self.cfg.test_save_path}/obj/flame/flame_close_tmp.obj")
            ms.apply_filter('meshing_close_holes', maxholesize = 1000)
            trimesh.Trimesh(vertices=ms.current_mesh().vertex_matrix(), faces=ms.current_mesh().face_matrix()).export(f"{self.cfg.test_save_path}/obj/flame/flame_close_tmp.obj")
            print(i + ":" + tmp)
        return trimesh.load(f"{self.cfg.test_save_path}/obj/flame/flame_close_tmp.obj")

    def remove_eyes_mesh(self, head_mesh):
        LBS_weights = self.flame_model.lbs_weights.cpu().detach().numpy()
        flame_verts = head_mesh.vertices
        LBS_argmax = np.argmax(LBS_weights, axis=-1)
        flame_faces = head_mesh.faces
        eyes_mask = np.logical_or(LBS_argmax == 3, LBS_argmax == 4)
        inside_faces = [f for f in flame_faces if np.all(~eyes_mask[f])]
        used_vertices = np.unique(np.hstack(inside_faces))
        new_vertices = flame_verts[used_vertices]
        vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
        remapped_faces = np.array([[vertex_mapping[vtx] for vtx in face] for face in inside_faces])
        new_mesh = trimesh.Trimesh(vertices=new_vertices, faces=remapped_faces)
        return new_mesh

    def get_opt_flame_mesh(self, offsets_flag=True, upsample_nums=0):
        flame_output = self.flame_model(betas=self.betas.unsqueeze(0), return_verts=True)
        flame_vertices = flame_output.vertices[0]
        flame_vertices = flame_vertices[torch.from_numpy(self.woeyes_flame_index).to(flame_vertices.device)]
        flame_faces = self.closed_woeyes_flame_faces

        centroid = flame_vertices.mean(0)
        flame_vertices = flame_vertices - centroid

        flame_vertices = flame_vertices * 3.8349
        flame_vertices[:,1] = flame_vertices[:,1] + 0.1
        flame_vertices[:,2] = flame_vertices[:,2] + 0.15

        # align to up-z and front-x
        dirs = ["+x", "+y", "+z", "-x", "-y", "-z"]
        dir2vec = {
            "+x": torch.tensor([1, 0, 0]).to(self.device).float(),
            "+y": torch.tensor([0, 1, 0]).to(self.device).float(),
            "+z": torch.tensor([0, 0, 1]).to(self.device).float(),
            "-x": torch.tensor([-1, 0, 0]).to(self.device).float(),
            "-y": torch.tensor([0, -1, 0]).to(self.device).float(),
            "-z": torch.tensor([0, 0, -1]).to(self.device).float(),
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
        y_ = torch.cross(z_, x_)
        std2mesh = torch.stack([x_, y_, z_], axis=0).T
        mesh2std = torch.linalg.inv(std2mesh)

        # scaling
        scale = torch.abs(flame_vertices).max().detach()
        flame_vertices = flame_vertices / scale * self.cfg.shape_init_params
        flame_vertices = torch.matmul(mesh2std, flame_vertices.T).T 
        upsample_mesh = trimesh.Trimesh(vertices=flame_vertices.cpu().detach().numpy(), faces=flame_faces)
        if upsample_nums > 0:
            upsample_vertices = torch.from_numpy(self.up_trans_mtx).float().to(self.device) @ flame_vertices
            upsample_faces = self.up_faces
        elif upsample_nums == 0:
            upsample_vertices = flame_vertices
            upsample_faces = upsample_mesh.faces
            
        if offsets_flag:
            if self.v_offsets.shape[1] == 1:
                flame_vertices_normal = compute_normal(upsample_vertices, upsample_faces.astype(np.int64), upsample_vertices.device)
                upsample_vertices += flame_vertices_normal[0] * self.v_offsets
            else:
                upsample_vertices += self.v_offsets

        return Mesh(v_pos=upsample_vertices.contiguous(), t_pos_idx=torch.from_numpy(upsample_faces.astype(np.int64)).long().to(self.device).contiguous())

    def get_flame_loss(self, sample_mode="random+bbox+surface", loss_mode="mse", pt_num=20000):
        assert self.temp_mesh is not None
        assert self.temp_mesh_sdf is not None
        sample_pts = []
        if "random" in sample_mode:
            sample_pts.append(torch.rand((pt_num, 3), dtype=torch.float32).to(self.device) * 2.0 - 1.0)
        if "bbox" in sample_mode:
            center = torch.from_numpy((self.temp_mesh.vertices.max(0) + self.temp_mesh.vertices.min(0)) / 2).float().to(self.device)
            scale = torch.from_numpy((self.temp_mesh.vertices.max(0) - self.temp_mesh.vertices.min(0))).float().to(self.device)
            sample_pts.append((torch.rand((pt_num, 3), dtype=torch.float32).to(self.device) - 0.5) * scale + center)
        if "surface" in sample_mode:
            sample_pts.append(torch.from_numpy(self.temp_mesh_sdf.sample_surface(pt_num) + np.random.normal(0, 0.05, (pt_num, 3))).float().to(self.device))
        sample_pts = torch.cat(sample_pts, dim=0)
        sample_pts = sample_pts[torch.randperm(sample_pts.shape[0])]

        sdf_gt = torch.from_numpy(-self.temp_mesh_sdf(sample_pts.cpu().numpy())).to(sample_pts)[..., None]
        if loss_mode == "mse":
            sdf_pred = self.forward_sdf(sample_pts)
            loss = F.mse_loss(sdf_pred, sdf_gt, reduction='sum')
        elif loss_mode == "in":
            in_flag = torch.where(sdf_gt < 0)[0]
            sample_pts = sample_pts[in_flag, :]
            sdf_pred = self.forward_sdf(sample_pts)
            epsilon = torch.ones_like(sdf_pred).to(self.device) * 0
            zero_tensor = torch.zeros_like(sdf_pred).to(self.device)
            loss = torch.max(zero_tensor, sdf_pred - epsilon).sum()

        return loss

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
            selected_points_idx = torch.LongTensor(random.sample(range(points_unscaled.shape[0]), 100000))
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
            mesh_v_pos = np.load(f'{self.cfg.test_save_path}/mesh_v_pos.npy')
            mesh_t_pos_idx = np.load(f'{self.cfg.test_save_path}/mesh_t_pos_idx.npy')
            cached_mesh = trimesh.Trimesh(
                vertices=mesh_v_pos,
                faces=mesh_t_pos_idx,
            )
            self.cached_sdf = SDF(cached_mesh.vertices, cached_mesh.faces)

        if self.flame_mesh is not None and self.temp_mesh is not None:
            if global_step % self.cfg.update_flame_loss_step == 0:
                if global_step == 0:
                    self.flame_mesh_sdf = SDF(self.flame_mesh.vertices, self.flame_mesh.faces)
                    self.temp_mesh_sdf = SDF(self.temp_mesh.vertices, self.temp_mesh.faces)
                    os.makedirs(f"{self.cfg.test_save_path}/obj/flame", exist_ok=True)
                    self.flame_mesh.export(f"{self.cfg.test_save_path}/obj/flame/flame.obj")
                    self.temp_mesh.export(f"{self.cfg.test_save_path}/obj/flame/temp_{global_step}.obj")
                else:
                    if global_step >= self.cfg.start_flame_loss_step:
                        if self.opt_flame_flag and self.evolving_flag:
                            if self.cfg.use_sdf_loss:
                                tmp_mesh = self.isosurface()[0]
                            else:
                                tmp_mesh = self.isosurface()
                            GT_mesh = Meshes(verts=[tmp_mesh.v_pos.detach()], faces=[tmp_mesh.t_pos_idx.detach()])
                            from tqdm import tqdm
                            w_chamfer = 100.0
                            w_edge = 1.0
                            w_normal = 0.01
                            w_laplacian = 0.1
                            optim = torch.optim.Adam([self.v_offsets], lr=1e-4)
                            for i in tqdm(range(100)):
                                optim.zero_grad()
                                tmp_flame_mesh = self.get_opt_flame_mesh(offsets_flag=True, upsample_nums=self.upsample_nums)
                                flame_mesh = Meshes(verts=[tmp_flame_mesh.v_pos], faces=[tmp_flame_mesh.t_pos_idx])

                                GT_sample_pts = sample_points_from_meshes(GT_mesh, 50000)
                                flame_sample_pts = sample_points_from_meshes(flame_mesh, 50000)
                                
                                # We compare the two sets of pointclouds by computing (a) the chamfer loss
                                loss_chamfer, _ = chamfer_distance(GT_sample_pts, flame_sample_pts)
                                
                                # and (b) the edge length of the predicted mesh
                                loss_edge = mesh_edge_loss(flame_mesh)
                                
                                # mesh normal consistency
                                loss_normal = mesh_normal_consistency(flame_mesh)
                                
                                # mesh laplacian smoothing
                                loss_laplacian = mesh_laplacian_smoothing(flame_mesh, method="uniform")
                                # print(torch.abs(self.v_offsets).mean().item())
                                # Weighted sum of the losses
                                loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian
                                loss.backward()
                                optim.step()
                            optim.zero_grad()
                            final_verts, final_faces = flame_mesh.get_mesh_verts_faces(0)

                            self.temp_mesh = trimesh.Trimesh(
                                    vertices=final_verts.detach().cpu().numpy(),
                                    faces=final_faces.detach().cpu().numpy(),
                                )
                            self.temp_mesh_sdf = SDF(self.temp_mesh.vertices, self.temp_mesh.faces)
                            self.temp_mesh.export(f"{self.cfg.test_save_path}/obj/flame/temp_{global_step}.obj")
                        else:
                            if self.cfg.use_sdf_loss:
                                tmp_mesh = self.isosurface()[0]
                            else:
                                tmp_mesh = self.isosurface()
                            self.temp_mesh = trimesh.Trimesh(
                                    vertices=tmp_mesh.v_pos.detach().cpu().numpy(),
                                    faces=tmp_mesh.t_pos_idx.detach().cpu().numpy(),
                                )
                            self.temp_mesh_sdf = SDF(self.temp_mesh.vertices, self.temp_mesh.faces)
                            self.temp_mesh.export(f"{self.cfg.test_save_path}/obj/flame/temp_{global_step}.obj")

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
