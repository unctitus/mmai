gpu_id=0

declare -a policy_models=( "octo-base" )
env_name=PutOnNumberGoogleInScene-v0
scene_name=dummy_reasoning_numbers
rgb_overlay_path=/work/users/t/i/tis/mmai/new_envs2/resized/reasoning_numbers.png

# URDF variations
declare -a urdf_version_arr=(None "recolor_tabletop_visual_matching_1" "recolor_tabletop_visual_matching_2" "recolor_cabinet_visual_matching_1")

for policy_model in "${policy_models[@]}"; do echo "$policy_model"; done

for urdf_version in "${urdf_version_arr[@]}";

do for policy_model in "${policy_models[@]}";

do CUDA_VISIBLE_DEVICES=${gpu_id} python simpler_env/main_inference.py --policy-model ${policy_model} --ckpt-path None \
  --robot google_robot_static \
  --control-freq 3 --sim-freq 513 --max-episode-steps 80 \
  --env-name ${env_name} --scene-name ${scene_name} \
  --rgb-overlay-path ${rgb_overlay_path} \
  --robot-init-x 0.35 0.35 1 --robot-init-y 0.21 0.21 1 --obj-variation-mode episode --obj-episode-range 0 24 \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 -0.09 -0.09 1 \
  --additional-env-build-kwargs urdf_version=${urdf_version} \
  --additional-env-save-tags baked_except_bpb_orange; 

done

done
