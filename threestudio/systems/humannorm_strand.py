from dataclasses import dataclass, field
import os
import json
import torch
import torch.nn.functional as F
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.misc import cleanup, get_device, load_module_weights
import trimesh
import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.systems.humannorm import HumanNorm
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from PIL import Image
import imageio
import numpy as np
import open3d as o3d
from pysdf import SDF
from threestudio.utils.ops import (
    get_mvp_matrix,
    get_projection_matrix,
    get_ray_directions,
    get_rays,
)
import math

def camera_info_process(batch):
    fovy_list = []
    center_list = []
    position_list = []
    up_list = []

    for b in range(batch['fovy'].shape[0]):
        fovy_list.append(batch['fovy'][b].item())
        center_list.append([batch['center'][b][i].item() for i in range(3)])
        position_list.append([batch['camera_positions'][b][i].item() for i in range(3)])
        up_list.append([batch['up'][b][i].item() for i in range(3)])

    return fovy_list, center_list, position_list, up_list

def convert_proj(K, H, W, near, far):
    return [
        [2 * K[0, 0] / W, -2 * K[0, 1] / W, (W - 2 * K[0, 2]) / W, 0],
        [0, -2 * K[1, 1] / H, (H - 2 * K[1, 2]) / H, 0],
        [0, 0, (-far - near) / (far - near), -2 * far * near / (far - near)],
        [0, 0, -1, 0],
    ]

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

@threestudio.register("humannorm-strand-system")
class HumanNorm_strand(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        texture: bool = False
        remove_img: bool = False
        start_sdf_loss_step: int = 300000

        # new
        test_save_path: str = "./.threestudio_cache"
        asset: dict = field(default_factory=dict)
        strand: dict = field(default_factory=dict)
        RD: dict = field(default_factory=dict)

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        # super().configure()
        self.perceptual_loss = PerceptualLoss().to(get_device())
        self.frames = []
        self.transforms = {
                "camera_model": "OPENCV",
                "orientation_override": "none",
            }
        
        # new
        self.RD_frames = []
        self.RD_transforms = {
                "camera_model": "OPENCV",
                "orientation_override": "none",
            }
        self.asset_geometry_list = torch.nn.ModuleList()
        self.asset_material_list = torch.nn.ModuleList()
        self.asset_background_list = torch.nn.ModuleList()
        self.asset_system_cfg_list = []
        self.asset_num = len(self.cfg.asset.ck_path)
        for i in range(self.asset_num):
            threestudio.info(f"Initializing {self.cfg.asset.name[i]} from a given checkpoint {self.cfg.asset.ck_path[i]}")
            from threestudio.utils.config import load_config, parse_structured

            asset_cfg = load_config(
                os.path.join(
                    os.path.dirname(self.cfg.asset.ck_path[i]),
                    "../configs/parsed.yaml",
                )
            )  # TODO: hard-coded relative path
            if self.cfg.asset.name[i] == "head":
                asset_system_cfg: BaseLift3DSystem.Config = parse_structured(
                    HumanNorm.Config, asset_cfg.system
                )
            else:
                asset_system_cfg: BaseLift3DSystem.Config = parse_structured(
                    self.Config, asset_cfg.system
                )
            self.asset_system_cfg_list.append(asset_system_cfg)
            asset_geometry_cfg = asset_system_cfg.geometry
            # asset_geometry_cfg.update(self.cfg.geometry_convert_override)
            asset_geometry = threestudio.find(asset_system_cfg.geometry_type)(
                asset_geometry_cfg
            )
            state_dict, epoch, global_step = load_module_weights(
                self.cfg.asset.ck_path[i],
                module_name="geometry",
                map_location="cpu",
            )
            asset_geometry.load_state_dict(state_dict, strict=True)
            asset_geometry.do_update_step(epoch, global_step, on_load_weights=True)
            asset_geometry = asset_geometry.to(get_device())
            asset_geometry.requires_grad_(False)
            self.asset_geometry_list.append(asset_geometry)

            asset_material_cfg = asset_system_cfg.material
            # asset_material_cfg.update(self.cfg.material_convert_override)
            asset_material = threestudio.find(asset_system_cfg.material_type)(
                asset_material_cfg
            )
            state_dict, epoch, global_step = load_module_weights(
                self.cfg.asset.ck_path[i],
                module_name="material",
                map_location="cpu",
            )
            asset_material.load_state_dict(state_dict, strict=True)
            asset_material.do_update_step(epoch, global_step, on_load_weights=True)
            asset_material = asset_material.to(get_device())
            asset_material.requires_grad_(False)
            self.asset_material_list.append(asset_material)

            asset_background_cfg = asset_system_cfg.background
            # asset_background_cfg.update(self.cfg.background_convert_override)
            asset_background = threestudio.find(asset_system_cfg.background_type)(
                asset_background_cfg
            )
            state_dict, epoch, global_step = load_module_weights(
                self.cfg.asset.ck_path[i],
                module_name="background",
                map_location="cpu",
            )
            asset_background.load_state_dict(state_dict, strict=True)
            asset_background.do_update_step(epoch, global_step, on_load_weights=True)
            asset_background = asset_background.to(get_device())
            asset_background.requires_grad_(False)
            self.asset_background_list.append(asset_background)

        if (
            self.cfg.geometry_convert_from  # from_coarse must be specified
            and not self.cfg.weights  # not initialized from coarse when weights are specified
            and not self.resumed  # not initialized from coarse when resumed from checkpoints
        ):
            threestudio.info(f"Initializing geometry from a given checkpoint {self.cfg.geometry_convert_from} ...")
            from threestudio.utils.config import load_config, parse_structured

            prev_cfg = load_config(
                os.path.join(
                    os.path.dirname(self.cfg.geometry_convert_from),
                    "../configs/parsed.yaml",
                )
            )  # TODO: hard-coded relative path
            prev_system_cfg: BaseLift3DSystem.Config = parse_structured(
                self.Config, prev_cfg.system
            )
            prev_geometry_cfg = prev_system_cfg.geometry
            prev_geometry_cfg.update(self.cfg.geometry_convert_override)
            prev_geometry = threestudio.find(prev_system_cfg.geometry_type)(
                prev_geometry_cfg
            )
            shape_init_strlist = prev_geometry_cfg.shape_init.split(":") 
            init_NHC_prompt = shape_init_strlist[2].split("_")[1]
            with open(prev_geometry_cfg.NHC_config.init_NHC_dict_path, 'r') as f:
                init_NHC_dict = json.load(f)
            init_NHC_path = init_NHC_dict[init_NHC_prompt]
            head_mesh = self.asset_geometry_list[0].isosurface()
            head_mesh = trimesh.Trimesh(vertices=head_mesh.v_pos.detach().cpu().numpy(), faces=head_mesh.t_pos_idx.detach().cpu().numpy())
            prev_geometry.strands_model.init_NHC(init_NHC_path, head_mesh, prev_geometry.device)
            state_dict, epoch, global_step = load_module_weights(
                self.cfg.geometry_convert_from,
                module_name="geometry",
                map_location=prev_geometry.device,
            )
            if len(state_dict["scale"].shape) == 0:
                state_dict["scale"] = state_dict["scale"] * torch.ones(3).float().to(prev_geometry.device)
            prev_geometry.strands_model.register_buffer('fixed_idx', state_dict["strands_model.fixed_idx"])
            prev_geometry.load_state_dict(state_dict, strict=True)
            # restore step-dependent states
            prev_geometry.do_update_step(epoch, global_step, on_load_weights=True)
            # convert from coarse stage geometry
            prev_geometry = prev_geometry.to(get_device())
            self.geometry = threestudio.find(self.cfg.geometry_type).create_strand_from(
                prev_geometry,
                prev_geometry_cfg,
                self.cfg.geometry,
                init_NHC_path=init_NHC_path, head_mesh=head_mesh,
                copy_net_path=self.cfg.geometry_convert_from if self.cfg.geometry_convert_inherit_texture else None
            )
            del prev_geometry
            cleanup()
        else:
            self.geometry = threestudio.find(self.cfg.geometry_type)(self.cfg.geometry)
        self.material = threestudio.find(self.cfg.material_type)(self.cfg.material)
        self.background = threestudio.find(self.cfg.background_type)(self.cfg.background)

        self.renderer = threestudio.find(self.cfg.renderer_type)(
            self.cfg.renderer,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
            asset_geometry_list=self.asset_geometry_list,
            asset_material_list=self.asset_material_list,
            asset_background_list=self.asset_background_list,
            asset_system_cfg_list=self.asset_system_cfg_list,
            asset_cfg=self.cfg.asset,
            strand_cfg=self.cfg.strand,
        )

        self.renderer_RD = threestudio.find(self.cfg.renderer_RD_type)(
            self.cfg.renderer_RD,
            geometry=self.geometry,
            material=self.material,
            background=self.background,
        )


    def forward(self, batch: Dict[str, Any], render_mode="comp+strand_head+strand_can") -> Dict[str, Any]:
        out = {}
        if "comp" in render_mode and "uncond_HN" in batch.keys():
            HN_render_out = self.renderer(**batch["uncond_HN"], render_rgb=self.cfg.texture, render_mode=render_mode)
            out.update({"HN_render_out": HN_render_out})
        return out

    def on_fit_start(self) -> None:
        super().on_fit_start()

        if self.cfg.guidance_type != "":
            self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                self.cfg.prompt_processor
            )
            self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)
            if self.cfg.prompt_processor.ori_prompt != "":
                self.cfg.prompt_processor.prompt = self.cfg.prompt_processor.ori_prompt
                self.ori_prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
                    self.cfg.prompt_processor
                )
            else:
                self.ori_prompt_processor = None
        else:
            self.prompt_processor = None
            self.guidance = None
            self.ori_prompt_processor = None

        if self.cfg.guidance_type_add != "":
            self.prompt_processor_add = threestudio.find(self.cfg.prompt_processor_type_add)(
                self.cfg.prompt_processor_add
            )
            self.guidance_add = threestudio.find(self.cfg.guidance_type_add)(self.cfg.guidance_add)
            if self.cfg.prompt_processor_add.ori_prompt != "":
                self.cfg.prompt_processor_add.prompt = self.cfg.prompt_processor_add.ori_prompt
                self.ori_prompt_processor_add = threestudio.find(self.cfg.prompt_processor_type_add)(
                    self.cfg.prompt_processor_add
                )
            else:
                self.ori_prompt_processor_add = None
        else:
            self.prompt_processor_add = None
            self.guidance_add = None
            self.ori_prompt_processor_add = None

        os.makedirs(self.cfg.test_save_path, exist_ok=True)
        os.makedirs(os.path.join(self.cfg.test_save_path, "RD"), exist_ok=True)
        self.obj_dir_path = os.path.join(self.get_save_dir(), "obj")

        if not self.cfg.texture:
            # initialize SDF
            # FIXME: what if using other geometry types?
            self.strand_mesh_trimesh_list = []
            self.strand_mesh_bbox_list = []
            self.strand_mesh_sdf_list = []
            for i in range(len(self.asset_system_cfg_list)):
                if i == 0:
                    if self.asset_system_cfg_list[i].renderer.use_sdf_loss:
                        self.head_mesh, _ = self.asset_geometry_list[i].isosurface()
                    else:
                        self.head_mesh = self.asset_geometry_list[i].isosurface()
                    self.head_mesh_trimesh = trimesh.Trimesh(vertices=self.head_mesh.v_pos.cpu().numpy(), faces=self.head_mesh.t_pos_idx.cpu().numpy())
                    self.head_mesh_sdf = SDF(self.head_mesh_trimesh.vertices, self.head_mesh_trimesh.faces)
                else:
                    if self.asset_system_cfg_list[i].renderer.use_sdf_loss:
                        strand_mesh, _ = self.asset_geometry_list[i].isosurface()
                    else:
                        strand_mesh = self.asset_geometry_list[i].isosurface()
                    tmp_vertices = strand_mesh.v_pos.detach() * self.asset_geometry_list[i].scale + self.asset_geometry_list[i].trans
                    self.strand_mesh_trimesh_list.append(trimesh.Trimesh(vertices=tmp_vertices.cpu().numpy(), faces=strand_mesh.t_pos_idx.cpu().numpy()))
                    self.strand_mesh_bbox_list.append([tmp_vertices.min(0)[0], tmp_vertices.max(0)[0]])
                    self.strand_mesh_sdf_list.append(SDF(self.strand_mesh_trimesh_list[-1].vertices, self.strand_mesh_trimesh_list[-1].faces))

            if self.cfg.asset.name[0] == "head" and self.cfg.geometry.shape_init.startswith("head:"):
                if self.cfg.geometry.shape_init.startswith("head:opt_flame"):
                    opt_flame_mesh = self.asset_geometry_list[0].get_opt_flame_mesh(offsets_flag=True)
                    self.geometry.initialize_shape(verts=self.head_mesh.v_pos.detach().cpu().numpy(), faces=self.head_mesh.t_pos_idx.detach().cpu().numpy(), system_cfg=self.cfg, opt_flame_mesh=opt_flame_mesh)
                    os.makedirs(os.path.join(self.obj_dir_path, "init"), exist_ok=True)
                    trimesh.Trimesh(vertices=opt_flame_mesh.v_pos.detach().cpu().numpy(), faces=opt_flame_mesh.t_pos_idx.detach().cpu().numpy()).export(os.path.join(self.obj_dir_path, "init", "flame.obj"))
                    trimesh.Trimesh(vertices=self.head_mesh.v_pos.detach().cpu().numpy(), faces=self.head_mesh.t_pos_idx.detach().cpu().numpy()).export(os.path.join(self.obj_dir_path, "init", "head.obj"))
                    with torch.no_grad():
                        init_NHC_strand_mesh, init_NHC_strand, _, _ = self.geometry.get_strand_mesh(num_strands=self.cfg.strand.num_strands, sample_mode=self.cfg.strand.sample_mode, run_batch=self.cfg.strand.run_batch, radius_scale=self.cfg.strand.radius_scale, reset_sampled_mesh=True, num_edges=self.cfg.strand.num_edges)
                        trimesh_mesh = trimesh.Trimesh(
                            vertices=init_NHC_strand_mesh.v_pos.detach().cpu().numpy(),
                            faces=init_NHC_strand_mesh.t_pos_idx.detach().cpu().numpy(),
                        )
                        trimesh_mesh.export(os.path.join(self.obj_dir_path, "init", f"init_strand_mesh.obj"))
                        save_hair(init_NHC_strand.detach().cpu().numpy(), os.path.join(self.obj_dir_path, "init", "init_strand.obj"))

                        trimesh.Trimesh(init_NHC_strand[:, 0, :].detach().cpu().numpy(), self.geometry.strands_model.sampled_strand_root_scalp_mesh.faces).export(os.path.join(self.obj_dir_path, "init", "init_strand_root_scalp.obj"))

                        # init_NHC_strand_mesh, init_NHC_strand, _, _ = self.geometry.get_strand_mesh(num_strands=-1, sample_mode=self.cfg.strand.sample_mode, run_batch=self.cfg.strand.run_batch, radius_scale=self.cfg.strand.radius_scale, num_edges=self.cfg.strand.num_edges)
                        # trimesh_mesh = trimesh.Trimesh(
                        #     vertices=init_NHC_strand_mesh.v_pos.detach().cpu().numpy(),
                        #     faces=init_NHC_strand_mesh.t_pos_idx.detach().cpu().numpy(),
                        # )
                        # trimesh_mesh.export(os.path.join(self.obj_dir_path, "init", f"init_strand_all_mesh.obj"))
                        # save_hair(init_NHC_strand.detach().cpu().numpy(), os.path.join(self.obj_dir_path, "init", "init_strand_all.obj"))

                        # trimesh.Trimesh(init_NHC_strand[:, 0, :].detach().cpu().numpy(), self.geometry.strands_model.strand_root_faces.detach().cpu().numpy()).export(os.path.join(self.obj_dir_path, "init", "init_strand_all_root_scalp.obj"))
                else:
                    self.geometry.initialize_shape(verts=self.head_mesh.v_pos.detach().cpu().numpy(), faces=self.head_mesh.t_pos_idx.detach().cpu().numpy(), system_cfg=self.cfg)
            else:
                self.geometry.initialize_shape(system_cfg=self.cfg)

    # def on_before_optimizer_step(self, optimizer):
    #     with torch.no_grad():
    #         if not self.cfg.texture:
    #             print(torch.abs(self.geometry.sdf_network.layers[0].weight.grad).mean().item())
    #         else:
    #             print(torch.abs(self.geometry.feature_network.layers[0].weight.grad).mean().item())
    #     return 

    def training_step(self, batch, batch_idx):
        loss = 0.0
        loss_dict = {}
        mesh = None
        if self.C(self.cfg.loss.lambda_l1) > 0 and self.C(self.cfg.loss.lambda_p) > 0:
            out = self(batch, render_mode="comp")
        else:
            out = self(batch, render_mode="comp+strand_head")
        if "HN_render_out" in out.keys():
            HN_out = out["HN_render_out"]
            if not self.cfg.texture: 
                if self.C(self.cfg.loss.lambda_sds) > 0:
                    prompt_utils = self.prompt_processor()
                    if self.ori_prompt_processor is not None:
                        ori_prompt_utils = self.ori_prompt_processor()
                    else:
                        ori_prompt_utils = None
                    guidance_inp = HN_out["comp_render_normal"]
                    guidance_out = self.guidance(
                        guidance_inp, prompt_utils, **batch["uncond_HN"], ori_prompt_utils=ori_prompt_utils
                    ) 
                    loss_dict.update({"loss_sds":guidance_out["loss_sds"]})
                if self.C(self.cfg.loss.lambda_sds_add) > 0:
                    prompt_utils = self.prompt_processor_add()
                    if self.ori_prompt_processor_add is not None:
                        ori_prompt_utils = self.ori_prompt_processor_add()
                    else:
                        ori_prompt_utils = None                
                    guidance_inp = HN_out["comp_render_depth"].repeat(1,1,1,3)
                    guidance_out = self.guidance_add(
                        guidance_inp, prompt_utils, **batch["uncond_HN"], ori_prompt_utils=ori_prompt_utils
                    )
                    loss_dict.update({"loss_sds_add":guidance_out["loss_sds"]})
                mesh = HN_out["strand_head_mesh"]
                strands = HN_out["strand_head"]
                strands_can = HN_out["strand_can"]
            else:
                if self.C(self.cfg.loss.lambda_sds) > 0:
                    prompt_utils = self.prompt_processor()
                    if self.ori_prompt_processor is not None:
                        ori_prompt_utils = self.ori_prompt_processor()
                    else:
                        ori_prompt_utils = None 
                    guidance_inp = HN_out["comp_render_rgb"]
                    cond_inp = HN_out["comp_render_normal"] # conditon for controlnet
                    guidance_out = self.guidance(
                        guidance_inp, cond_inp, prompt_utils, **batch["uncond_HN"], ori_prompt_utils=ori_prompt_utils
                    )
                    loss_dict.update({"loss_sds":guidance_out["loss_sds"]})
                if self.C(self.cfg.loss.lambda_l1) > 0 and self.C(self.cfg.loss.lambda_p) > 0:
                    prompt_utils = self.prompt_processor()
                    if self.ori_prompt_processor is not None:
                        ori_prompt_utils = self.ori_prompt_processor()
                    else:
                        ori_prompt_utils = None 
                    guidance_inp = HN_out["comp_render_rgb"]
                    cond_inp = HN_out["comp_render_normal"] # conditon for controlnet
                    guidance_out = self.guidance(
                        guidance_inp, cond_inp, prompt_utils, **batch["uncond_HN"], ori_prompt_utils=ori_prompt_utils
                    )
                    loss_dict.update({"loss_l1":guidance_out["loss_l1"]})
                    loss_dict.update({"loss_p":guidance_out["loss_p"]})

        if not self.cfg.texture:  # geometry training
            if self.true_global_step == self.cfg.start_sdf_loss_step:
                np.save(f'{self.cfg.test_save_path}/mesh_v_pos.npy', mesh.v_pos.detach().cpu().numpy())
                np.save(f'{self.cfg.test_save_path}/mesh_t_pos_idx.npy', mesh.t_pos_idx.detach().cpu().numpy())

            if self.C(self.cfg.loss.lambda_normal_consistency) > 0:
                loss_normal_consistency = mesh.normal_consistency()
                loss_dict.update({"loss_normal_consistency": loss_normal_consistency})
            
            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                loss_laplacian_smoothness = mesh.laplacian()
                loss_dict.update({"loss_laplacian_smoothness": loss_laplacian_smoothness})

            if self.C(self.cfg.loss.lambda_temp) > 0 and self.true_global_step >= self.cfg.geometry.start_temp_loss_step:
                loss_temp = self.geometry.get_temp_loss()
                loss_dict.update({"loss_temp": loss_temp})

            if self.C(self.cfg.loss.lambda_collision) > 0 and self.true_global_step >= self.cfg.geometry.start_collision_loss_step:
                loss_collision = self.geometry.get_collision_loss(strands, self.asset_geometry_list[0])
                loss_dict.update({"loss_collision": loss_collision})

            if self.C(self.cfg.loss.lambda_ori_consist) > 0:
                loss_ori_consist = self.geometry.get_ori_consist_loss(strands)
                loss_dict.update({"loss_ori_consist": loss_ori_consist})

            if self.C(self.cfg.loss.lambda_norm_consist) > 0:
                loss_norm_consist = self.geometry.get_norm_consist_loss(strands)
                loss_dict.update({"loss_norm_consist": loss_norm_consist})

            if self.C(self.cfg.loss.lambda_curv_reg) > 0:
                loss_curv_reg = self.geometry.get_curv_reg_loss(strands, self.cfg.strand.target_curv)
                loss_dict.update({"loss_curv_reg": loss_curv_reg})

            if self.C(self.cfg.loss.lambda_normal_consist) > 0:
                loss_normal_consist = self.geometry.get_normal_consist_loss(strands, torch.from_numpy(self.geometry.strands_model.sampled_strand_root_scalp_mesh.faces).long().to(strands.device))
                loss_dict.update({"loss_normal_consist": loss_normal_consist})

            if self.C(self.cfg.loss.lambda_bbox) > 0:
                loss_bbox = self.geometry.get_bbox_loss(strands_can)
                loss_dict.update({"loss_bbox": loss_bbox})

            if self.C(self.cfg.loss.lambda_face) > 0:
                loss_face_collision = self.geometry.get_face_collision_loss(strands)
                loss_dict.update({"loss_face": loss_face_collision})

            # if self.C(self.cfg.loss.lambda_collision) > 0:      
            #     trans = (mesh.v_pos.detach().max(0)[0] + mesh.v_pos.detach().min(0)[0]) / 2
            #     scale = (mesh.v_pos.detach().max(0)[0] - mesh.v_pos.detach().min(0)[0])
            #     sample_pts = (((torch.rand((200000, 3), dtype=torch.float32).to(self.device) - 0.5) * scale + trans) * self.geometry.scale + self.geometry.trans)
            #     in_flag = torch.zeros(sample_pts.shape[0], dtype=torch.bool).to(self.device)
            #     for i in range(len(self.asset_geometry_list)):
            #         if i == 0:
            #             sdf_gt = torch.from_numpy(-self.head_mesh_sdf(sample_pts.cpu().numpy())).to(sample_pts)[..., None]
            #             in_flag = torch.logical_or(in_flag, (sdf_gt < 0).squeeze())
            #         else:
            #             sample_pts_min, sample_pts_max = sample_pts.min(0)[0], sample_pts.max(0)[0]
            #             if (sample_pts_max >= self.strand_mesh_bbox_list[i-1][0]).all() and (sample_pts_min <= self.strand_mesh_bbox_list[i-1][1]).all():
            #                 sdf_gt = torch.from_numpy(-self.strand_mesh_sdf_list[i-1](sample_pts.cpu().numpy())).to(sample_pts)[..., None]
            #                 in_flag = torch.logical_or(in_flag, (sdf_gt < 0).squeeze())
            #     sample_pts = sample_pts[in_flag, :]
            #     strand_sdf = self.geometry.forward_sdf(sample_pts)
            #     epsilon = torch.ones_like(strand_sdf).to(self.device) * 0
            #     zero_tensor = torch.zeros_like(strand_sdf).to(self.device)
            #     loss_collision = torch.max(zero_tensor, epsilon - strand_sdf).sum()
            #     loss_dict.update({"loss_collision": loss_collision})

        for name, value in loss_dict.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
                
        # print(loss_dict)

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch, render_mode="strand_head+strand_can+comp")
        HN_render_out = out["HN_render_out"]
        if self.cfg.texture:
            self.save_image_grid(
                f"it{self.true_global_step}-val-color/{batch['uncond_HN']['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": HN_render_out["strand_can_render_rgb"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ] if "strand_can_render_rgb" in HN_render_out else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": HN_render_out["strand_head_render_rgb"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ] if "strand_head_render_rgb" in HN_render_out else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": HN_render_out["comp_render_rgb"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ] if "comp_render_rgb" in HN_render_out else []
                ),
                name="validation_step",
                step=self.true_global_step,
            )
        else:
            self.save_image_grid(
                f"it{self.true_global_step}-val-normal/{batch['uncond_HN']['index'][0]}.png",
                ([
                    {
                        "type": "rgb",
                        "img": HN_render_out["strand_can_render_normal"][0] + (1 - HN_render_out["strand_can_render_mask"][0, :, :, :]),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ] if "strand_can_render_normal" in HN_render_out else [])
                + ([
                    {
                        "type": "rgb",
                        "img": HN_render_out["strand_head_render_normal"][0] + (1 - HN_render_out["strand_head_render_mask"][0, :, :, :]),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ] if "strand_head_render_normal" in HN_render_out else [])
                + [
                    {
                        "type": "rgb",
                        "img": HN_render_out["comp_render_normal"][0] + (1 - HN_render_out["comp_render_mask"][0, :, :, :]),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    },
                ],
                name="validation_step",
                step=self.true_global_step,
            )

    def on_validation_epoch_end(self):
        if self.cfg.texture:
            self.save_img_sequence(
                f"it{self.true_global_step}-val-color",
                f"it{self.true_global_step}-val-color",
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
                name="val",
                step=self.true_global_step,
                remove_img=self.cfg.remove_img
            )
        else:
            self.save_img_sequence(
                f"it{self.true_global_step}-val-normal",
                f"it{self.true_global_step}-val-normal",
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
                name="val",
                step=self.true_global_step,
                remove_img=self.cfg.remove_img
            )

    def test_step(self, batch, batch_idx):
        if 'focal' in batch['uncond_HN'] and self.cfg.texture:
            out = self(batch, render_mode="strand_head+strand_can+asset+comp")
            # out = self(batch, render_mode="comp")
        else:
            out = self(batch, render_mode="strand_head+strand_can+asset+comp")
            # out = self(batch, render_mode="comp")
        HN_render_out = out["HN_render_out"]
        if self.cfg.exporter.save_flag and batch['uncond_HN']['index'][0] == 0:
            os.makedirs(self.obj_dir_path, exist_ok=True)
            for idx, mesh in enumerate(HN_render_out["comp_mesh_list"]):
                os.makedirs(os.path.join(self.obj_dir_path, "result"), exist_ok=True)
                if self.cfg.texture:
                    if idx == 0:
                        tmp_fmt = self.cfg.exporter.fmt
                        self.cfg.exporter.fmt = "obj"
                        self.exporter = threestudio.find(self.cfg.exporter_type)(
                            self.cfg.exporter,
                            geometry=self.asset_geometry_list[idx] if idx < self.asset_num else self.geometry,
                            material=self.asset_material_list[idx] if idx < self.asset_num else self.material,
                            background=self.asset_background_list[idx] if idx < self.asset_num else self.background,
                        )
                        output = self.exporter()
                        self.cfg.exporter.fmt = tmp_fmt
                        self.save_obj(os.path.join("obj", "result", f"comp_{idx}.obj"), **output[0].params)
                    else:
                        self.exporter = threestudio.find(self.cfg.exporter_type)(
                            self.cfg.exporter,
                            geometry=self.asset_geometry_list[idx] if idx < self.asset_num else self.geometry,
                            material=self.asset_material_list[idx] if idx < self.asset_num else self.material,
                            background=self.asset_background_list[idx] if idx < self.asset_num else self.background,
                        )
                        output = self.exporter()
                        scale = self.asset_geometry_list[idx].scale if idx < self.asset_num else self.geometry.scale
                        trans = self.asset_geometry_list[idx].trans if idx < self.asset_num else self.geometry.trans
                        self.save_obj(os.path.join("obj", "result", f"comp_{idx}.obj"), **output[0].params)
                        mesh_color = output[0].params["mesh"].v_rgb.detach().cpu().numpy()
                else:
                    trimesh_mesh = trimesh.Trimesh(
                        vertices=mesh.v_pos.detach().cpu().numpy(),
                        faces=mesh.t_pos_idx.detach().cpu().numpy(),
                    )
                    trimesh_mesh.export(os.path.join(self.obj_dir_path, "result", f"comp_{idx}.obj"))

            if self.cfg.texture:
                init_NHC_strand = self.geometry.get_strand(num_strands=self.cfg.strand.num_strands, sample_mode=self.cfg.strand.sample_mode, run_batch=self.cfg.strand.run_batch)[0]
                strand_color = mesh_color.reshape(init_NHC_strand.shape[0], -1, 3)[:, :self.cfg.strand.num_edges * 100, :]
                strand_color = strand_color.reshape(init_NHC_strand.shape[0], 100, self.cfg.strand.num_edges, 3)[:, :, 0, :]
                save_hair(init_NHC_strand.detach().cpu().numpy(), os.path.join(self.obj_dir_path, "result", "result_strand.obj"), strand_color)

            else:
                init_NHC_strand = self.geometry.get_strand(num_strands=self.cfg.strand.num_strands, sample_mode=self.cfg.strand.sample_mode, run_batch=self.cfg.strand.run_batch)[0]
                save_hair(init_NHC_strand.detach().cpu().numpy(), os.path.join(self.obj_dir_path, "result", "result_strand.obj"))

        if self.cfg.texture:
            self.save_image_grid(
                f"it{self.true_global_step}-test-color/{batch['uncond_HN']['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": HN_render_out["comp_render_rgb"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ] if "comp_render_rgb" in HN_render_out else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": HN_render_out["asset_render_rgb"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ] if "asset_render_rgb" in HN_render_out else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": HN_render_out["strand_head_render_rgb"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ] if "strand_head_render_rgb" in HN_render_out else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": HN_render_out["strand_can_render_rgb"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ] if "strand_can_render_rgb" in HN_render_out else []
                ),
                name="test_step",
                step=self.true_global_step,
            )

            self.save_image_grid(
                f"it{self.true_global_step}-test-normal/{batch['uncond_HN']['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": HN_render_out["comp_render_normal"][0] + (1 - HN_render_out["comp_render_mask"][0, :, :, :]),
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ] if "comp_render_normal" in HN_render_out else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": HN_render_out["asset_render_normal"][0] + (1 - HN_render_out["asset_render_mask"][0, :, :, :]),
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ] if "asset_render_normal" in HN_render_out else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img": HN_render_out["strand_head_render_normal"][0] + (1 - HN_render_out["strand_head_render_mask"][0, :, :, :]),
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ] if "strand_head_render_normal" in HN_render_out else []
                )
                + (
                    [
                        {
                            "type": "rgb",
                            "img":HN_render_out["strand_can_render_normal"][0] + (1 - HN_render_out["strand_can_render_mask"][0, :, :, :]),
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ] if "strand_can_render_normal" in HN_render_out else []
                ),
                name="test_step",
                step=self.true_global_step,
            )
        else:
            pass

        # save camera parameters and views in coarse texture stage for multi-step SDS loss in fine texture stage

        if 'focal' in batch['uncond_HN'] and self.cfg.texture:
            if True:
                if batch_idx == 0:
                    self.prepare_render_data()
                tmp_batch = {
                    "index": batch_idx,
                    "rays_o": self.rays_o[batch_idx][None],
                    "rays_d": self.rays_d[batch_idx][None],
                    "mvp_mtx": self.mvp_mtx[batch_idx][None],
                    "c2w": self.c2w[batch_idx][None],
                    "camera_positions": self.camera_positions[batch_idx][None],
                    "light_positions": self.light_positions[batch_idx][None],
                    "elevation": self.elevation_deg[batch_idx][None],
                    "azimuth": self.azimuth_deg[batch_idx][None],
                    "camera_distances": self.camera_distances[batch_idx][None],
                    "height": self.eval_height,
                    "width": self.eval_width,
                    "focal": self.focal_length[batch_idx][None],
                    "cx": self.cx[batch_idx][None],
                    "cy": self.cy[batch_idx][None],
                    "up": self.up[batch_idx][None],
                    "fovy": self.fovy_deg[batch_idx][None],
                    "center": self.center[batch_idx][None],
                    "n_views": self.n_views
                }
                tmp_render_out = self.renderer(**tmp_batch, render_rgb=self.cfg.texture, render_mode="comp")
                # save camera parameters
                c2w = tmp_batch['c2w'][0].cpu().numpy()
                down_scale = tmp_batch['width'] / 512 # ensure the resolution is set to 512
                frame = {
                        "fl_x": float(tmp_batch['focal'][0].cpu()) / down_scale,
                        "fl_y": float(tmp_batch['focal'][0].cpu()) / down_scale ,
                        "cx": float(tmp_batch['cx'][0].cpu()) / down_scale,
                        "cy": float(tmp_batch['cy'][0].cpu()) / down_scale,
                        "w": int(tmp_batch['width'] / down_scale),
                        "h": int(tmp_batch['height'] / down_scale),
                        "file_path": f"./image/{tmp_batch['index']}.png",
                        "transform_matrix": c2w.tolist(),
                        "elevation": float(tmp_batch['elevation'][0].cpu()),
                        "azimuth": float(tmp_batch['azimuth'][0].cpu()),
                        "camera_distances": float(tmp_batch['camera_distances'][0].cpu()),
                    }
                self.frames.append(frame)
                if tmp_batch['index'] == (tmp_batch['n_views']-1):
                    self.transforms["frames"] = self.frames
                    with open(os.path.join(self.cfg.test_save_path, 'transforms.json'), 'w') as f:
                        f.write(json.dumps(self.transforms, indent=4))
                    # init
                    self.frames.clear()
                save_img = tmp_render_out["comp_render_rgb"]
                save_img = F.interpolate(save_img.permute(0,3,1,2), (512, 512), mode="bilinear", align_corners=False)
                os.makedirs(f"{self.cfg.test_save_path}/image", exist_ok=True)
                imageio.imwrite(f"{self.cfg.test_save_path}/image/{tmp_batch['index']}.png", (save_img.permute(0, 2, 3, 1)[0].detach().cpu().numpy() * 255).astype(np.uint8))
            else:
                # save camera parameters
                c2w = batch['uncond_HN']['c2w'][0].cpu().numpy()
                down_scale = batch['uncond_HN']['width'] / 512 # ensure the resolution is set to 512
                frame = {
                        "fl_x": float(batch['uncond_HN']['focal'][0].cpu()) / down_scale,
                        "fl_y": float(batch['uncond_HN']['focal'][0].cpu()) / down_scale ,
                        "cx": float(batch['uncond_HN']['cx'][0].cpu()) / down_scale,
                        "cy": float(batch['uncond_HN']['cy'][0].cpu()) / down_scale,
                        "w": int(batch['uncond_HN']['width'] / down_scale),
                        "h": int(batch['uncond_HN']['height'] / down_scale),
                        "file_path": f"./image/{batch['uncond_HN']['index'][0]}.png",
                        "transform_matrix": c2w.tolist(),
                        "elevation": float(batch['uncond_HN']['elevation'][0].cpu()),
                        "azimuth": float(batch['uncond_HN']['azimuth'][0].cpu()),
                        "camera_distances": float(batch['uncond_HN']['camera_distances'][0].cpu()),
                    }
                self.frames.append(frame)
                if batch['uncond_HN']['index'][0] == (batch['uncond_HN']['n_views'][0]-1):
                    self.transforms["frames"] = self.frames
                    with open(os.path.join(self.cfg.test_save_path, 'transforms.json'), 'w') as f:
                        f.write(json.dumps(self.transforms, indent=4))
                    # init
                    self.frames.clear()
                save_img = HN_render_out["comp_render_rgb"]
                save_img = F.interpolate(save_img.permute(0,3,1,2), (512, 512), mode="bilinear", align_corners=False)
                os.makedirs(f"{self.cfg.test_save_path}/image", exist_ok=True)
                imageio.imwrite(f"{self.cfg.test_save_path}/image/{batch['uncond_HN']['index'][0]}.png", (save_img.permute(0, 2, 3, 1)[0].detach().cpu().numpy() * 255).astype(np.uint8))

    def prepare_render_data(self):
        n_views = 120
        asset_scale = []
        asset_trans = []
        for i in range(1, len(self.asset_system_cfg_list)):
            asset_scale.append(self.asset_geometry_list[i].scale.detach().cpu())
            asset_trans.append(self.asset_geometry_list[i].trans.detach().cpu())
        asset_scale.append(self.geometry.scale.detach().cpu())
        asset_trans.append(self.geometry.trans.detach().cpu())
        asset_scale = torch.stack(asset_scale, dim=0)
        asset_trans = torch.stack(asset_trans, dim=0)
        body_prop = 0.5
        strand_prop = 1 - body_prop
        n_views_list = [int(body_prop * n_views)] + [int(strand_prop * n_views / asset_scale.shape[0])] * asset_scale.shape[0]
        n_views_list[-1] += (n_views - sum(n_views_list))
        n_views_part = torch.tensor(n_views_list, dtype=torch.int)
        azimuth_deg = torch.cat([torch.linspace(0, 360.0, n_views_part[i]) for i in range(n_views_part.shape[0])], dim=0)
        camera_distance_range = [3, 3]
        elevation_range = [-5, 15]
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

        focal_scale_list = []
        focal_scale_list.append(torch.full([n_views_part[0].item()], 1.0))
        for i in range(asset_trans.shape[0]):
            tmp_focal_scale = 1.0 / asset_scale[i][0].item() * 0.9
            focal_scale_list.append(torch.full([n_views_part[i+1].item()], tmp_focal_scale))

        focal_scale = torch.cat(focal_scale_list, 0)
        focal_length *= focal_scale

        cx = torch.full_like(focal_length, eval_width / 2)
        cy = torch.full_like(focal_length, eval_height / 2)

        center[:n_views_part[0].item(), 2] += 0
        tmp_start = n_views_part[0].item()
        tmp_end = n_views_part[0].item() + n_views_part[1].item()
        for i in range(asset_trans.shape[0]):            
            center[tmp_start:tmp_end, :] += asset_trans[i]
            tmp_start += n_views_part[i+1].item()
            tmp_end += n_views_part[i+1].item()
        
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

        self.rays_o, self.rays_d = rays_o.to(self.device), rays_d.to(self.device)
        self.mvp_mtx = mvp_mtx.to(self.device)
        self.c2w = c2w.to(self.device)
        self.camera_positions = camera_positions.to(self.device)
        self.light_positions = light_positions.to(self.device)
        self.elevation, self.azimuth = elevation, azimuth.to(self.device)
        self.elevation_deg, self.azimuth_deg = elevation_deg.to(self.device), azimuth_deg.to(self.device)
        self.camera_distances = camera_distances.to(self.device)
        self.focal_length = focal_length.to(self.device)
        self.cx = cx.to(self.device)
        self.cy = cy.to(self.device)

        self.up = up.to(self.device)
        self.fovy_deg = fovy_deg.to(self.device)
        self.center = center.to(self.device)
        self.eval_height = eval_height
        self.eval_width = eval_width
        self.n_views = n_views

    def on_test_epoch_end(self):
        if self.cfg.texture:
            self.save_img_sequence(
                f"it{self.true_global_step}-test-color",
                f"it{self.true_global_step}-test-color",
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
                name="test",
                step=self.true_global_step
            )
            self.save_img_sequence(
                f"it{self.true_global_step}-test-normal",
                f"it{self.true_global_step}-test-normal",
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
                name="test",
                step=self.true_global_step
            )

        # if self.cfg.texture:
        #     self.save_img_sequence(
        #         f"it{self.true_global_step}-test-color",
        #         f"it{self.true_global_step}-test-color",
        #         "(\d+)\.png",
        #         save_format="mp4",
        #         fps=30,
        #         name="test",
        #         step=self.true_global_step,
        #         remove_img=self.cfg.remove_img
        #     )

        # self.save_img_sequence(
        #     f"it{self.true_global_step}-test-normal",
        #     f"it{self.true_global_step}-test-normal",
        #     "(\d+)\.png",
        #     save_format="mp4",
        #     fps=30,
        #     name="test",
        #     step=self.true_global_step,
        #     remove_img=self.cfg.remove_img
        # )
