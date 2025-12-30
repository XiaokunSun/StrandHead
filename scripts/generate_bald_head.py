import os
import subprocess
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dict_path', default="./load/strandhead/data_dict.json", type=str)
parser.add_argument('--bald_head_exp_root_dir', default="./outputs/bald_head", type=str)
parser.add_argument('--bald_head_idx', default="0:1:1", type=str)
parser.add_argument('--gpu_idx', default=0, type=int)
args = parser.parse_args()

with open(args.dict_path, 'r') as f:
    dict_data = json.load(f)

bald_head_exp_root_dir=args.bald_head_exp_root_dir
exp_name1="stage1-geometry"
exp_name2="stage2-coarse-texture"
exp_name3="stage3-fine-texture"
gpu_idx=args.gpu_idx
bald_head_idx_list = args.bald_head_idx.split(":")
bald_head_idx_list = list(range(int(bald_head_idx_list[0]), int(bald_head_idx_list[1]), int(bald_head_idx_list[2])))
HN_neg_prompt = "hairs, earrings, glasses, necklaces, studs"

for i in bald_head_idx_list:
    bald_head_prompt = dict_data[i]["bald_head_prompt"]
    bald_head_tag = bald_head_prompt.replace(" ", "_").replace(".", "").replace(",", "")

    test_save_path=f"{bald_head_exp_root_dir}/{exp_name1}/{bald_head_tag}/save/cache"
    geometry_convert_from=f"{bald_head_exp_root_dir}/{exp_name1}/{bald_head_tag}/ckpts/last.ckpt" 
    resume=f"{bald_head_exp_root_dir}/{exp_name2}/{bald_head_tag}/ckpts/last.ckpt" 

    command1 = ["python", "launch.py", "--config", "configs/head-geometry.yaml", "--gpu", f"{gpu_idx}",
                "--train", f"tag={bald_head_tag}", f"name={exp_name1}", f"exp_root_dir={bald_head_exp_root_dir}",
                f"system.test_save_path={test_save_path}", f"system.prompt_processor.prompt={bald_head_prompt}, black background, normal map", 
                f"system.prompt_processor_add.prompt={bald_head_prompt}, black background, depth map", 
                "system.geometry.shape_init=opt_flame:10:3:True", "system.geometry.start_flame_loss_step=0",
                "system.geometry.update_flame_loss_step=1001", "system.geometry.opt_flame_step_num=300", "system.loss.lambda_flame=1000"]
    
    command2 = ["python", "launch.py", "--config", "configs/head-texture-coarse.yaml", "--gpu", f"{gpu_idx}",
                "--train", f"tag={bald_head_tag}", f"name={exp_name2}", f"exp_root_dir={bald_head_exp_root_dir}", 
                f"system.geometry_convert_from={geometry_convert_from}",
                f"system.test_save_path={test_save_path}", f"system.prompt_processor.prompt={bald_head_prompt}"]

    command3 = ["python", "launch.py", "--config", "configs/head-texture-fine.yaml", "--gpu", f"{gpu_idx}",
                "--train", f"tag={bald_head_tag}", f"name={exp_name3}", f"exp_root_dir={bald_head_exp_root_dir}",
                f"system.geometry_convert_from={geometry_convert_from}",
                f"system.test_save_path={test_save_path}", f"system.prompt_processor.prompt={bald_head_prompt}", 
                f"data.dataroot={test_save_path}", f"resume={resume}", "system.exporter.save_flag=true"]

    if True:
        command1.append(f"system.prompt_processor.negative_prompt={HN_neg_prompt}")
        command1.append(f"system.prompt_processor_add.negative_prompt={HN_neg_prompt}")

    if True:
        command1.append("trainer.max_steps=10000")
        command2.append("trainer.max_steps=2000")
        command3.append("trainer.max_steps=10000")

    tmp_str = ""
    for c in command1:
        tmp_str += c + " "
    print(tmp_str)
    subprocess.run(["rm", "-rf", f"{bald_head_exp_root_dir}/{exp_name1}/{bald_head_tag}"])
    subprocess.run(command1)

    tmp_str = ""
    for c in command2:
        tmp_str += c + " "
    print(tmp_str)
    subprocess.run(["rm", "-rf", f"{bald_head_exp_root_dir}/{exp_name2}/{bald_head_tag}"])
    subprocess.run(command2)

    tmp_str = ""
    for c in command3:
        tmp_str += c + " "
    print(tmp_str)
    subprocess.run(["rm", "-rf", f"{bald_head_exp_root_dir}/{exp_name3}/{bald_head_tag}"])
    subprocess.run(command3)