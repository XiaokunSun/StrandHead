from dataclasses import dataclass, field

import os
import json
import torch
import torch.nn.functional as F
from threestudio.utils.perceptual import PerceptualLoss
from threestudio.utils.misc import cleanup, get_device
import trimesh
import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from PIL import Image
import imageio
import numpy as np



@threestudio.register("humannorm-system")
class HumanNorm(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        texture: bool = False
        remove_img: bool = False
        start_sdf_loss_step: int = 300000
        test_save_path: str = "./.threestudio_cache"

    cfg: Config

    def configure(self):
        # create geometry, material, background, renderer
        super().configure()
        self.perceptual_loss = PerceptualLoss().to(self.device)
        self.frames = []
        self.transforms = {
                "camera_model": "OPENCV",
                "orientation_override": "none",
            }


    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        render_out = self.renderer(**batch, render_rgb=self.cfg.texture)
        return {
            **render_out,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        # addtional prompt processor and guiance such as depth stable diffusion model
        self.prompt_processor_add = None
        self.guidance_add = None
        if len(self.cfg.prompt_processor_type_add) > 0:
            self.prompt_processor_add = threestudio.find(self.cfg.prompt_processor_type_add)(
                self.cfg.prompt_processor_add
            )

        if len(self.cfg.guidance_type_add) > 0:
            self.guidance_add = threestudio.find(self.cfg.guidance_type_add)(self.cfg.guidance_add)

        self.obj_dir_path = os.path.join(self.get_save_dir(), "obj")
        os.makedirs(self.cfg.test_save_path, exist_ok=True)

        if not self.cfg.texture:
            # initialize SDF
            # FIXME: what if using other geometry types?
            if self.geometry.opt_flame_flag:
                os.makedirs(os.path.join(self.obj_dir_path, "opt_flame"), exist_ok=True)
            else:
                self.geometry.initialize_shape()
                if self.cfg.renderer.use_sdf_loss:
                    mesh, _ = self.geometry.isosurface()
                else:
                    mesh = self.geometry.isosurface()
                mesh = trimesh.Trimesh(
                        vertices=mesh.v_pos.detach().cpu().numpy(),
                        faces=mesh.t_pos_idx.detach().cpu().numpy(),
                    )
                os.makedirs(os.path.join(self.obj_dir_path, "init_mesh"), exist_ok=True)
                mesh.export(os.path.join(self.obj_dir_path, "init_mesh", f"init_mesh.obj"))


    def on_before_optimizer_step(self, optimizer):
        with torch.no_grad():
            if not self.cfg.texture:
                if self.geometry.opt_flame_flag and self.true_global_step <= self.geometry.cfg.opt_flame_step_num:
                    # print(torch.abs(self.geometry.betas.grad).mean().item())
                    # print(torch.abs(self.geometry.betas).mean().item())
                    # print(torch.abs(self.geometry.v_offsets.grad).mean().item())
                    pass
                else:
                    # print(torch.abs(self.geometry.sdf_network.layers[0].weight.grad).mean().item())
                    pass
        return 

    def training_step(self, batch, batch_idx):
        if not self.cfg.texture:
            if self.geometry.opt_flame_flag and self.true_global_step <= self.geometry.cfg.opt_flame_step_num:   
                loss = 0.0
                mesh = self.geometry.get_opt_flame_mesh(offsets_flag = False)
                render_out = self.renderer.render_mesh(mesh, **batch, render_rgb=self.cfg.texture)
                prompt_utils = self.prompt_processor()
                # normal SDS loss
                guidance_inp = render_out["comp_normal"]
                guidance_out = self.guidance(
                    guidance_inp, prompt_utils, **batch
                )

                # depth SDS loss
                if self.prompt_processor_add is not None:
                    prompt_utils = self.prompt_processor_add()

                if self.guidance_add is not None and self.C(self.cfg.loss.lambda_sds_add) > 0:
                    guidance_inp = render_out["comp_depth"].repeat(1,1,1,3)
                    guidance_out_add = self.guidance_add(
                        guidance_inp, prompt_utils, **batch
                    )
                    guidance_out.update({"loss_sds_add":guidance_out_add["loss_sds"]})
                else:
                    guidance_out.update({"loss_sds_add":0})


                loss_normal_consistency = render_out["mesh"].normal_consistency()
                self.log("train/loss_normal_consistency", loss_normal_consistency)
                loss += loss_normal_consistency * self.C(
                    self.cfg.loss.lambda_normal_consistency
                )
                
                if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                    loss_laplacian_smoothness = render_out["mesh"].laplacian()
                    self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                    loss += loss_laplacian_smoothness * self.C(
                        self.cfg.loss.lambda_laplacian_smoothness
                    )
                for name, value in guidance_out.items():
                    self.log(f"train/{name}", value)
                    if name.startswith("loss_"):
                        loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
                        # print(name, value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])) # for debugging

                for name, value in self.cfg.loss.items():
                    self.log(f"train_params/{name}", self.C(value))

                if self.true_global_step % 100 == 0:
                    tri_mesh = trimesh.Trimesh(
                            vertices=mesh.v_pos.detach().cpu().numpy(),
                            faces=mesh.t_pos_idx.detach().cpu().numpy(),
                        )
                    tri_mesh.export(os.path.join(self.obj_dir_path, "opt_flame", f"flame_{self.true_global_step}.obj"))

                if self.true_global_step == self.geometry.cfg.opt_flame_step_num:
                    self.geometry.initialize_shape_from_flame(tri_mesh, self.true_global_step)
                    if self.cfg.renderer.use_sdf_loss:
                        mesh, _ = self.geometry.isosurface()
                    else:
                        mesh = self.geometry.isosurface()
                    mesh = trimesh.Trimesh(
                            vertices=mesh.v_pos.detach().cpu().numpy(),
                            faces=mesh.t_pos_idx.detach().cpu().numpy(),
                        )
                    os.makedirs(os.path.join(self.obj_dir_path, "init_mesh"), exist_ok=True)
                    mesh.export(os.path.join(self.obj_dir_path, "init_mesh", f"init_mesh.obj"))

                return {"loss": loss}

        loss = 0.0

        out = self(batch)
        prompt_utils = self.prompt_processor()

        if self.true_global_step == self.cfg.start_sdf_loss_step:
            np.save(f'{self.cfg.test_save_path}/mesh_v_pos.npy', out['mesh'].v_pos.detach().cpu().numpy())
            np.save(f'{self.cfg.test_save_path}/mesh_t_pos_idx.npy', out['mesh'].t_pos_idx.detach().cpu().numpy())

        if not self.cfg.texture:  # geometry training

            # normal SDS loss
            guidance_inp = out["comp_normal"]
            guidance_out = self.guidance(
                guidance_inp, prompt_utils, **batch
            )

            # depth SDS loss
            if self.prompt_processor_add is not None:
                prompt_utils = self.prompt_processor_add()

            if self.guidance_add is not None and self.C(self.cfg.loss.lambda_sds_add) > 0:
                guidance_inp = out["comp_depth"].repeat(1,1,1,3)
                guidance_out_add = self.guidance_add(
                    guidance_inp, prompt_utils, **batch
                )
                guidance_out.update({"loss_sds_add":guidance_out_add["loss_sds"]})
            else:
                guidance_out.update({"loss_sds_add":0})

            # SDF loss
            if out['sdf_loss'] is not None:
                guidance_out.update({"loss_sdf": out['sdf_loss']})

            loss_normal_consistency = out["mesh"].normal_consistency()
            self.log("train/loss_normal_consistency", loss_normal_consistency)
            loss += loss_normal_consistency * self.C(
                self.cfg.loss.lambda_normal_consistency
            )
            
            if self.C(self.cfg.loss.lambda_laplacian_smoothness) > 0:
                loss_laplacian_smoothness = out["mesh"].laplacian()
                self.log("train/loss_laplacian_smoothness", loss_laplacian_smoothness)
                loss += loss_laplacian_smoothness * self.C(
                    self.cfg.loss.lambda_laplacian_smoothness
                )
                # print('loss_lap', loss_laplacian_smoothness * self.C(self.cfg.loss.lambda_laplacian_smoothness)) # for debugging

            if self.C(self.cfg.loss.lambda_flame) > 0 and self.true_global_step >= self.cfg.geometry.start_flame_loss_step:
                loss_flame = self.geometry.get_flame_loss()
                self.log("train/loss_flame", loss_flame)
                loss += loss_flame * self.C(
                    self.cfg.loss.lambda_flame
                )

        else:  # texture training
            guidance_inp = out["comp_rgb"]
            if isinstance(
                self.guidance,
                (
                    threestudio.models.guidance.controlnet_guidance.ControlNetGuidance,
                    threestudio.models.guidance.controlnet_vsd_guidance.ControlNetVSDGuidance,
                    threestudio.models.guidance.sds_du_controlnet_guidance.SDSDUControlNetGuidance,
                ),
            ):
                cond_inp = out["comp_normal"] # conditon for controlnet
                guidance_out = self.guidance(
                    guidance_inp, cond_inp, prompt_utils, **batch
                )
            else:
                guidance_out = self.guidance(
                    guidance_inp, prompt_utils, **batch
                )

        for name, value in guidance_out.items():
            self.log(f"train/{name}", value)
            if name.startswith("loss_"):
                loss += value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])
                # print(name, value * self.C(self.cfg.loss[name.replace("loss_", "lambda_")])) # for debugging

        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        if self.cfg.texture:
            self.save_image_grid(
                f"it{self.true_global_step}-val-color/{batch['index'][0]}.jpg",
                (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_rgb"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if self.cfg.texture
                    else []
                ),
                name="validation_step",
                step=self.true_global_step,
            )

        if not self.cfg.texture:
            self.save_image_grid(
                f"it{self.true_global_step}-val-normal/{batch['index'][0]}.jpg",
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0] + (1 - out["opacity"][0, :, :, :]),
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
                "(\d+)\.jpg",
                save_format="mp4",
                fps=30,
                name="val",
                step=self.true_global_step,
                remove_img=self.cfg.remove_img
            )
        if not self.cfg.texture:
            self.save_img_sequence(
                f"it{self.true_global_step}-val-normal",
                f"it{self.true_global_step}-val-normal",
                "(\d+)\.jpg",
                save_format="mp4",
                fps=30,
                name="val",
                step=self.true_global_step,
                remove_img=self.cfg.remove_img
            )

    def test_step(self, batch, batch_idx):
        out = self(batch)
        if self.cfg.exporter.save_flag and batch['index'][0] == 0:
            if not self.cfg.texture:
                obj_dir_path = os.path.join(self.get_save_dir(), "obj", "result")
                os.makedirs(obj_dir_path, exist_ok=True)
                mesh = trimesh.Trimesh(
                    vertices=out["mesh"].v_pos.detach().cpu().numpy(),
                    faces=out["mesh"].t_pos_idx.detach().cpu().numpy(),
                )
                mesh.export(os.path.join(obj_dir_path, f"head.obj"))

            if self.cfg.texture:
                obj_dir_path = os.path.join(self.get_save_dir(), "obj", "result")
                os.makedirs(obj_dir_path, exist_ok=True)
                tmp_fmt = self.cfg.exporter.fmt
                self.cfg.exporter.fmt = "obj"
                self.exporter = threestudio.find(self.cfg.exporter_type)(
                    self.cfg.exporter,
                    geometry=self.geometry,
                    material=self.material,
                    background=self.background,
                )
                output = self.exporter()
                self.cfg.exporter.fmt = tmp_fmt
                self.save_obj(os.path.join("obj", "result", "head.obj"), **output[0].params)

        if self.cfg.texture:
            self.save_image_grid(
                f"it{self.true_global_step}-test-color/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        },
                    ]
                    if "comp_rgb" in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )

        self.save_image_grid(
            f"it{self.true_global_step}-test-normal/{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0] + (1 - out["opacity"][0, :, :, :]),
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="test_step",
            step=self.true_global_step,
        )

        # save camera parameters and views in coarse texture stage for multi-step SDS loss in fine texture stage
        if 'focal' in batch and self.cfg.texture:
            # save camera parameters
            c2w = batch['c2w'][0].cpu().numpy()

            down_scale = batch['width'] / 512 # ensure the resolution is set to 512

            frame = {
                    "fl_x": float(batch['focal'][0].cpu()) / down_scale,
                    "fl_y": float(batch['focal'][0].cpu()) / down_scale ,
                    "cx": float(batch['cx'][0].cpu()) / down_scale,
                    "cy": float(batch['cy'][0].cpu()) / down_scale,
                    "w": int(batch['width'] / down_scale),
                    "h": int(batch['height'] / down_scale),
                    "file_path": f"./image/{batch['index'][0]}.png",
                    "transform_matrix": c2w.tolist(),
                    "elevation": float(batch['elevation'][0].cpu()),
                    "azimuth": float(batch['azimuth'][0].cpu()),
                    "camera_distances": float(batch['camera_distances'][0].cpu()),
                }
            self.frames.append(frame)

            if batch['index'][0] == (batch['n_views'][0]-1):
                self.transforms["frames"] = self.frames
                with open(os.path.join(self.cfg.test_save_path, 'transforms.json'), 'w') as f:
                    f.write(json.dumps(self.transforms, indent=4))

                # init
                self.frames.clear()

            save_img = out["comp_rgb"]
            save_img = F.interpolate(save_img.permute(0,3,1,2), (512, 512), mode="bilinear", align_corners=False)
            os.makedirs(f"{self.cfg.test_save_path}/image", exist_ok=True)
            imageio.imwrite(f"{self.cfg.test_save_path}/image/{batch['index'][0]}.png", (save_img.permute(0, 2, 3, 1)[0].detach().cpu().numpy() * 255).astype(np.uint8))


    def on_test_epoch_end(self):
        if self.cfg.texture:
            self.save_img_sequence(
                f"it{self.true_global_step}-test-color",
                f"it{self.true_global_step}-test-color",
                "(\d+)\.png",
                save_format="mp4",
                fps=30,
                name="test",
                step=self.true_global_step,
            )

        self.save_img_sequence(
            f"it{self.true_global_step}-test-normal",
            f"it{self.true_global_step}-test-normal",
            "(\d+)\.png",
            save_format="mp4",
            fps=30,
            name="test",
            step=self.true_global_step,
        )