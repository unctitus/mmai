#!/usr/bin/env bash
gpu_id=0
env_name=MoveNearGoogleBakedTexInSceneNew-v0
#scene_name=dummy_tabletop1
#rgb_overlay_path=./ManiSkill2_real2sim/data/real_inpainting/table_env1.png
scene_name=dummy_wood_tabletop
rgb_overlay_path=/work/users/t/i/tis/mmai/ManiSkill2_real2sim/data/real_inpainting/wood_tabletop2.png


policy_models=( "octo-base" )
urdf_version_arr=( None "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" "recolor_cabinet_visual_matching_1")

# build an array of full commands
commands=()
for pm in "${policy_models[@]}"; do
  for urdf_version in "${urdf_version_arr[@]}"; do
    commands+=(
      "CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py \
        --policy-model ${pm} --ckpt-path None \
        --robot google_robot_static \
        --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
        --env-name ${env_name} --scene-name ${scene_name} \
        --rgb-overlay-path ${rgb_overlay_path} \
        --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 \
        --obj-variation-mode episode --obj-episode-range 0 60 \
        --robot-init-rot-quat-center 0 0 0 1 \
        --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
        --additional-env-build-kwargs urdf_version=${urdf_version} \
        --additional-env-save-tags baked_except_bpb_orange"
    )
  done
done

# decide how many jobs to run in parallel (not more than 8 for L40)
printf "%s\n" "${commands[@]}" | parallel -j4