import os 
import subprocess
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dict_path', default="./load/strandhead/data_dict.json", type=str)
parser.add_argument('--bald_head_exp_root_dir', default="./outputs/bald_head", type=str)
parser.add_argument('--hair_head_exp_root_dir', default="./outputs/strand_head", type=str)
parser.add_argument('--idx', default="0:1:1", type=str)
parser.add_argument('--gpu_idx', default=0, type=int)
args = parser.parse_args()

with open(args.dict_path, 'r') as f:
    dict_data = json.load(f)

bald_head_exp_root_dir=args.bald_head_exp_root_dir
hair_head_exp_root_dir=args.hair_head_exp_root_dir
exp_name1="stage1-geometry"
exp_name2="stage2-coarse-texture"
exp_name3="stage3-fine-texture"
gpu_idx=args.gpu_idx

idx_list = args.idx.split(":")
idx_list = list(range(int(idx_list[0]), int(idx_list[1]), int(idx_list[2])))

for i in idx_list:
    bald_head_prompt = dict_data[i]["bald_head_prompt"]
    hair_head_prompt = dict_data[i]["hair_head_prompt"]
    hair_prompt = dict_data[i]["hair_prompt"]
    init_prompt = dict_data[i]["init_NHC"]

    if "afro" in hair_prompt:
        init_curv = 0.2
    else:
        init_curv = 0.05

    if "straight" in hair_prompt:
        init_curv = 0.02
    elif "curly" in hair_prompt:
        init_curv = 0.2
    elif "wavy" in hair_prompt:
        init_curv = 0.1

    HN_prompt = hair_head_prompt
    HN_neg_prompt = ""
    shape_init = f"head:opt_flame:NHC_{init_prompt}"

    asset_ckpt_list = [
        [],
        [],
    ]
    asset_name_list = ["head"]
    hair_head_tag = HN_prompt.replace(" ", "_").replace(".", "").replace(",", "") 
    bald_head_tag = bald_head_prompt.replace(" ", "_").replace(".", "").replace(",", "")
    asset_ckpt_list[0].append(f"{bald_head_exp_root_dir}/{exp_name1}/{bald_head_tag}/ckpts/last.ckpt")
    asset_ckpt_list[1].append(f"{bald_head_exp_root_dir}/{exp_name3}/{bald_head_tag}/ckpts/last.ckpt")
    asset_ckpt_list[0].append(f"{hair_head_exp_root_dir}/{exp_name1}/{hair_head_tag}/ckpts/last.ckpt")
    asset_ckpt_list[1].append(f"{hair_head_exp_root_dir}/{exp_name2}/{hair_head_tag}/ckpts/last.ckpt")
    init_mesh_path = f"{bald_head_exp_root_dir}/{exp_name1}/{bald_head_tag}/save/cache/obj/flame/temp_300.obj"

    cache_path1 = f"{hair_head_exp_root_dir}/{exp_name1}/{hair_head_tag}/save/cache"
    cache_path2 = f"{hair_head_exp_root_dir}/{exp_name2}/{hair_head_tag}/save/cache"

    command1 = ["python", "launch.py", "--config", "configs/strand-geometry.yaml", "--gpu", f"{gpu_idx}",
                "--train", f"tag={hair_head_tag}", f"name={exp_name1}", f"exp_root_dir={hair_head_exp_root_dir}",
                f"system.test_save_path={cache_path1}", f"system.prompt_processor.prompt={HN_prompt}, black background, normal map", 
                f"system.prompt_processor_add.prompt={HN_prompt}, black background, depth map",
                f"system.geometry.NHC_config.init_mesh_path={init_mesh_path}", f"system.strand.target_curv={init_curv}",
                f"system.geometry.shape_init={shape_init}", "system.geometry.shape_init_params=0.9", f"system.asset.ck_path={asset_ckpt_list[0][:1]}", 
                f"system.asset.name={asset_name_list}"]
    command2 = ["python", "launch.py", "--config", "configs/strand-texture-coarse.yaml", "--gpu", f"{gpu_idx}",
                "--train", f"tag={hair_head_tag}", f"name={exp_name2}", f"exp_root_dir={hair_head_exp_root_dir}",
                f"system.geometry_convert_from={asset_ckpt_list[0][1]}",
                f"system.test_save_path={cache_path2}", f"system.prompt_processor.prompt={HN_prompt}",
                f"system.asset.ck_path={asset_ckpt_list[1][:1]}", 
                f"system.asset.name={asset_name_list}"]
    command3 = ["python", "launch.py", "--config", "configs/strand-texture-fine.yaml", "--gpu", f"{gpu_idx}",
                "--train", f"tag={hair_head_tag}", f"name={exp_name3}", f"exp_root_dir={hair_head_exp_root_dir}",
                f"data.dataroot={cache_path2}",  f"system.geometry_convert_from={asset_ckpt_list[0][1]}",
                f"system.test_save_path={cache_path2}", f"system.prompt_processor.prompt={HN_prompt}",
                f"system.asset.ck_path={asset_ckpt_list[1][:1]}", 
                f"system.asset.name={asset_name_list}", "system.exporter.save_flag=true"]

    if True:
        command1.append("trainer.max_steps=5000")
        command2.append("trainer.max_steps=2000")
        command3.append("trainer.max_steps=5000")

    tmp_str = ""
    for c in command1:
        tmp_str += c + " "
    print(tmp_str)
    subprocess.run(["rm", "-rf", f"{hair_head_exp_root_dir}/{exp_name1}/{hair_head_tag}"])
    subprocess.run(command1)

    tmp_str = ""
    for c in command2:
        tmp_str += c + " "
    print(tmp_str)
    subprocess.run(["rm", "-rf", f"{hair_head_exp_root_dir}/{exp_name2}/{hair_head_tag}"])
    subprocess.run(command2)

    tmp_str = ""
    for c in command3:
        tmp_str += c + " "
    print(tmp_str)
    subprocess.run(["rm", "-rf", f"{hair_head_exp_root_dir}/{exp_name3}/{hair_head_tag}"])
    subprocess.run(command3)