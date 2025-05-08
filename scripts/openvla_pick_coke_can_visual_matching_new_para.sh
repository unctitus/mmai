#!/usr/bin/env bash
gpu_id=0


env_name=GraspSingleOpenedCokeCanInScene-v0 # change number trials

scene_name=dummy_tabletop1
rgb_overlay_path=./ManiSkill2_real2sim/data/real_inpainting/table_env1.png
#scene_name=dummy_wood_tabletop 
#rgb_overlay_path=/work/users/t/i/tis/mmai/ManiSkill2_real2sim/data/real_inpainting/wood_tabletop2.png

declare -a arr=("openvla/openvla-7b")
declare -a coke_can_options_arr=("lr_switch=True" "upright=True" "laid_vertically=True")
declare -a urdf_version_arr=(None "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" "recolor_cabinet_visual_matching_1")

# build an array of full commands
commands=()
for urdf_version in "${urdf_version_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    for coke_can_option in "${coke_can_options_arr[@]}"; do
      commands+=(
        "CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference2.py --policy-model openvla --ckpt-path ${ckpt_path} \
          --robot google_robot_static \
          --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
          --env-name ${env_name} --scene-name ${scene_name} \
          --rgb-overlay-path ${rgb_overlay_path} \
          --robot-init-x 0.35 0.35 1 --robot-init-y 0.20 0.20 1 --obj-init-x -0.35 -0.12 5 --obj-init-y -0.02 0.42 5 \
          --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
          --additional-env-build-kwargs ${coke_can_option} urdf_version=${urdf_version}"
      )
    done
  done
done

# decide how many jobs to run in parallel (not more than 8 for L40)
printf "%s\n" "${commands[@]}" | parallel -j2