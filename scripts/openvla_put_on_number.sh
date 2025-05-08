#!/usr/bin/env bash
gpu_id=0

env_name=PutOnNumberGoogleInScene-v0
env_name=PutOnNumberSmallestGoogleInScene-v0
env_name=PutOnNumberSumGoogleInScene-v0
env_name=PutOnNumberMultGoogleInScene-v0

scene_name=dummy_reasoning_numbers
rgb_overlay_path=/work/users/t/i/tis/mmai/new_envs2/resized/reasoning_numbers.png

declare -a arr=("openvla/openvla-7b")

declare -a urdf_version_arr=(None "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" "recolor_cabinet_visual_matching_1")

# build an array of full commands
commands=()
for urdf_version in "${urdf_version_arr[@]}"; do
  for ckpt_path in "${arr[@]}"; do
    commands+=(
      "CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference2.py \
        --policy-model openvla --ckpt-path ${ckpt_path} \
        --robot google_robot_static \
        --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
        --env-name ${env_name} --scene-name ${scene_name} \
        --rgb-overlay-path ${rgb_overlay_path} \
        --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 \
        --obj-variation-mode episode --obj-episode-range 0 8 \
        --robot-init-rot-quat-center 0 0 0 1 \
        --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
        --additional-env-build-kwargs urdf_version=${urdf_version} \
        --additional-env-save-tags baked_except_bpb_orange"
    )
  done
done

# decide how many jobs to run in parallel (not more than 8 for L40)
printf "%s\n" "${commands[@]}" | parallel -j2