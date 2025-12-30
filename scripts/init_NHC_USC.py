import numpy as np
import torch
import os
import smplx
import trimesh
from glob import glob
import cubvh
import yaml

import sys
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), 'NeuralHaircut'))
from tqdm import tqdm
import argparse
import trimesh
from NeuralHaircut.src.hair_networks.optimizable_textured_strands_customization import OptimizableTexturedStrands
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import json
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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


def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', default="./outputs/init_NHC256", type=str)
    parser.add_argument('--config_path', default="./configs/NHC_strand_fitting_256.yaml", type=str)
    parser.add_argument('--w_cur', default=1, type=float)
    parser.add_argument('--w_ori', default=0.05, type=float)
    parser.add_argument('--iter_num', default=10000, type=int)
    parser.add_argument('--save_num', default=1000, type=int)
    parser.add_argument('--gpu_idx', default=0, type=int)
    args = parser.parse_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = ""

    device = "cuda:0"
    USC_head_obj = trimesh.load("./load/strandhead/USC_head.obj")
    flame_head_obj = trimesh.load("./load/strandhead/haar_head.obj")
    scalp_verts_idx = torch.load("./load/flame_models/flame/NHC_scalp_vertex_idx.pth").numpy()
    scalp_faces = torch.load("./load/flame_models/flame/NHC_scalp_faces.pth").numpy()
    scalp_uv_tensor = torch.load("./load/flame_models/flame/NHC_scalp_uvcoords.pth").to(device)
    flame_scalp_obj = trimesh.Trimesh(vertices=flame_head_obj.vertices[scalp_verts_idx], faces=scalp_faces)

    flame_head_BVH = cubvh.cuBVH(flame_head_obj.vertices, flame_head_obj.faces)
    flame_scalp_BVH = cubvh.cuBVH(flame_scalp_obj.vertices, flame_scalp_obj.faces)
    strand_obj_dirpath = "Data/hairstyles/hairstyles_obj_processed" # The path to the preprocessed USC-HairSalon dataset, please see preprocess_USC.py for more details
    scalp_obj_dirpath = "Data/hairstyles/strand_root_obj_processed" # The path to the preprocessed USC-HairSalon dataset, please see preprocess_USC.py for more details
    strand_obj_path_list = sorted(glob(os.path.join(strand_obj_dirpath, "strands*")))
    scalp_obj_path_list = sorted(glob(os.path.join(scalp_obj_dirpath, "strands*")))
    choosen_strand_dict = {"strands00357": "A side-swept haircut",
                           "strands00052": "A side-swept hairstyle",
                           "strands00065": "A slicked-back hairstyle",
                           "strands00205": "A slicked-back haircut",
                           "strands00358": "A mohawk haircut",
                           "strands00372": "A mohawk hairstyle",
                           "strands00150": "A short haircut",
                           "strands00055": "A short wavy hairstyle",
                           "strands00166": "A short hairstyle",
                           "strands00384": "A short spiky haircut",
                           "strands00136": "A short bob haircut",
                           "strands00062": "A short bob haircut with bangs",
                           "strands00098": "A straight bob haircut",
                           "strands00137": "A wavy bob haircut",
                           "strands00171": "A curly bob haircut",
                           "strands00080": "A wavy haircut with bangs",
                           "strands00249": "A wavy haircut with bangs",
                           "strands00059": "A medium-length wavy hairstyle",
                           "strands00045": "A medium-length curly hairstyle",
                           "strands00082": "A medium-length straight hairstyle",
                           "strands00022": "A long wavy hairstyle",
                           "strands00110": "A long wavy haircut",
                           "strands00118": "A long wavy haircut with bangs",
                           "strands00160": "A long straight haircut"
                           }
    init_NHC_dict = {}

    flame_verts_tensor = torch.from_numpy(flame_head_obj.vertices).float().to(device)
    flame_faces_tensor = torch.from_numpy(flame_head_obj.faces).long().to(device)
    scalp_verts_tensor = torch.from_numpy(flame_scalp_obj.vertices).float().to(device)
    scalp_faces_tensor = torch.from_numpy(flame_scalp_obj.faces).long().to(device)

    NHC_config = load_config(args.config_path)
    NHC_config['textured_strands']["path_to_mesh"] = "./load/strandhead/haar_head.obj"
    save_path = args.save_path
    w_ori = args.w_ori
    w_cur = args.w_cur

    for hair_idx, strand_obj_path in tqdm(enumerate(strand_obj_path_list)):
        if os.path.basename(strand_obj_path).split(".")[0] not in choosen_strand_dict.keys():
            continue
        tmp_save_path = os.path.join(save_path, os.path.basename(strand_obj_path).split('.')[0])

        init_NHC_dict[choosen_strand_dict[os.path.basename(strand_obj_path).split(".")[0]]] = tmp_save_path

        obj_save_path = os.path.join(tmp_save_path, "obj")
        os.makedirs(obj_save_path, exist_ok=True)
        tensorboard_save_path = os.path.join(tmp_save_path, "tensorboard")
        os.makedirs(tensorboard_save_path, exist_ok=True)
        model_save_path = os.path.join(tmp_save_path, "model")
        os.makedirs(model_save_path, exist_ok=True)

        writer = SummaryWriter(log_dir=tensorboard_save_path)

        strand_obj = trimesh.load(strand_obj_path)
        
        scalp_obj = trimesh.load(scalp_obj_path_list[hair_idx])

        if os.path.basename(strand_obj_path).split(".")[0] == "strands00357":
            mask = np.ones(scalp_obj.vertices.shape[0]).astype(np.bool_)
            mask[6031] = False
            mask[4240] = False
            inside_faces = [f for f in scalp_obj.faces if np.all(mask[f])]
            used_vertices = np.unique(np.hstack(inside_faces))
            new_vertices = scalp_obj.vertices[used_vertices]
            vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
            remapped_faces = np.array([[vertex_mapping[vtx] for vtx in face] for face in inside_faces])
            scalp_obj = trimesh.Trimesh(vertices=new_vertices, faces=remapped_faces)
            strand_obj.vertices = strand_obj.vertices.reshape(-1, 100, 3)[mask]
        elif os.path.basename(strand_obj_path).split(".")[0] == "strands00384":
            mask = (np.ones(scalp_obj.vertices.shape[0]).astype(np.bool_) & (scalp_obj.vertices[:, 2] > 0.05))
            mask = mask & (scalp_obj.vertices[:, 1] < 1.838)
            mask = ~mask
            inside_faces = [f for f in scalp_obj.faces if np.all(mask[f])]
            used_vertices = np.unique(np.hstack(inside_faces))
            new_vertices = scalp_obj.vertices[used_vertices]
            vertex_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertices)}
            remapped_faces = np.array([[vertex_mapping[vtx] for vtx in face] for face in inside_faces])
            scalp_obj = trimesh.Trimesh(vertices=new_vertices, faces=remapped_faces)
            strand_obj.vertices = strand_obj.vertices.reshape(-1, 100, 3)[mask]

        strand_verts_tensor = torch.from_numpy(strand_obj.vertices.reshape(-1, 100, 3)).float().to(device)
        strand_root_verts_tensor = strand_verts_tensor[:, 0, :]

        map_dist, map_face, map_uvw = flame_head_BVH.signed_distance(strand_root_verts_tensor, return_uvw=True, mode="raystab")
        USC2flame_verts_tensor = (flame_verts_tensor[flame_faces_tensor[map_face], :] * map_uvw[:, :, None]).sum(1)
        GT_strand_verts_tensor = strand_verts_tensor - strand_root_verts_tensor[:, None, :] + USC2flame_verts_tensor[:, None, :]

        _, scalp_map_face, scalp_map_uvw = flame_scalp_BVH.signed_distance(GT_strand_verts_tensor[:, 0, :], return_uvw=True, mode="raystab")
        idx = torch.where(scalp_map_uvw < -1e-4)
        mask = torch.ones(scalp_map_uvw.shape[0]).bool().to(device)
        if idx[0].shape[0] > 0:
            # mask[idx[0]] = False
            print(f"warning! {idx[0].shape[0]} strands are not in the scalp")

        GT_strand_verts_tensor = GT_strand_verts_tensor[mask]
        USC2scalp_uv_tensor = (scalp_uv_tensor[scalp_faces_tensor[scalp_map_face], :] * scalp_map_uvw[:, :, None]).sum(1)[mask]
        GT_strand_root_tensor = GT_strand_verts_tensor[:, 0, :]
        GT_strand_faces_idx_tensor = scalp_map_face[mask]

        GT_strand_ori_tensor = GT_strand_verts_tensor[:, 1:, :] - GT_strand_verts_tensor[:, :-1, :]
        GT_strand_ori_tensor = GT_strand_ori_tensor / (torch.norm(GT_strand_ori_tensor, dim=-1)[:, :, None] + 1e-20)
        GT_strand_cur_tensor = torch.norm(GT_strand_ori_tensor[:, 1:, :] - GT_strand_ori_tensor[:, :-1, :], dim=-1)

        strand_num = GT_strand_verts_tensor.shape[0]

        strands_model = OptimizableTexturedStrands(**NHC_config['textured_strands'], diffusion_cfg=NHC_config['diffusion_prior']).to(device)
        strands_model.scale_decoder = 1
        optimizer = torch.optim.Adam(strands_model.parameters(), NHC_config['general']['lr'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=NHC_config['general']['milestones'], gamma=NHC_config['general']['gamma'])

        save_hair(GT_strand_verts_tensor.detach().cpu().numpy(), os.path.join(obj_save_path, f"GT_strand.obj"))

        for iter in tqdm(range(args.iter_num)):
            strands_out = strands_model.forward_customization(GT_strand_root_tensor, USC2scalp_uv_tensor, GT_strand_faces_idx_tensor, run_batch=10000, scale_factor=1)

            optimizer.zero_grad()

            pos_loss = torch.norm(strands_out[0] - GT_strand_verts_tensor, dim=-1).mean()

            pre_ori = strands_out[0][:, 1:, :] - strands_out[0][:, :-1, :]
            pre_ori = pre_ori / (torch.norm(pre_ori, dim=-1)[:, :, None] + 1e-20)

            pre_cur = torch.norm(pre_ori[:, 1:, :] - pre_ori[:, :-1, :], dim=-1)

            # ori_loss = F.mse_loss(pre_ori, GT_strand_ori_tensor)
            ori_loss = 1 - F.cosine_similarity(pre_ori.reshape(-1, 3), GT_strand_ori_tensor.reshape(-1, 3), dim=-1).mean()

            cur_loss = F.l1_loss(pre_cur, GT_strand_cur_tensor)

            total_loss = pos_loss + ori_loss * w_ori + cur_loss * w_cur
            # total_loss = pos_loss
            total_loss.backward()
            # print(pos_loss.item())

            writer.add_scalar('Total Loss', total_loss.item(), iter)
            writer.add_scalar('Pos Loss', pos_loss.item(), iter)
            writer.add_scalar('Ori Loss', ori_loss.item(), iter)
            writer.add_scalar('Cur Loss', cur_loss.item(), iter)

            for param in optimizer.param_groups[0]['params']:
                if param.grad is not None and param.grad.isnan().any():
                    optimizer.zero_grad()
                    print('NaN during backprop was found, skipping iteration...')

            optimizer.step()
            scheduler.step()

            if iter % args.save_num == 0:
                save_hair(strands_out[0].detach().cpu().numpy(), os.path.join(obj_save_path, f"pre_strand_10k_{iter}.obj"))

        save_hair(strands_out[0].detach().cpu().numpy(), os.path.join(obj_save_path, f"pre_strand_10k_{iter+1}.obj"))

        new_scalp_obj = trimesh.Trimesh(vertices=strands_out[0][:, 0, :].detach().cpu().numpy(), faces=scalp_obj.faces)
        new_scalp_obj.export(os.path.join(obj_save_path, "scalp.obj"))

        writer.close()

        torch.save(strands_model.state_dict(), os.path.join(model_save_path, f"model_{iter+1}.pth"))
        torch.save(USC2scalp_uv_tensor, os.path.join(model_save_path, f"map_uv_tensor.pth"))
        torch.save(scalp_map_face[mask], os.path.join(model_save_path, f"map_face_tensor.pth"))
        torch.save(scalp_map_uvw[mask], os.path.join(model_save_path, f"map_face_uvw_tensor.pth"))
        torch.save(torch.randperm(USC2scalp_uv_tensor.shape[0]), os.path.join(model_save_path, f"random_idx.pth"))

    if os.path.exists(f'{save_path}/init_NHC_dict.json'):
        with open(f'{save_path}/init_NHC_dict.json', 'r') as f:
            existing_data = json.load(f)
    else:
        existing_data = {}
    existing_data.update(init_NHC_dict)
    with open(f'{save_path}/init_NHC_dict.json', 'w') as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)
    
        

        
        












    

