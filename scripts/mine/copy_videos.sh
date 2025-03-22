#!/bin/bash

# Check if the remote server address was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <remote_server>"
    exit 1
fi

# Remote server connection info (e.g., tis@longleaf.unc.edu)
REMOTE="$1"
# Local destination directory
DEST="/Users/titus/Desktop/simpler_videos"
# Directory to store the converted GIFs
GIF_DIR="${DEST}/gif"
# Frame rate to use for GIF conversion (adjust if needed)
FPS=30

# List of remote MP4 files to copy
FILES=(
"/nas/longleaf/home/tis/mmai/results/rt_1_x_tf_trained_for_002272480_step/dummy_drawer/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/CloseBottomDrawerCustomInScene-v0_shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_disable_bad_material_True_urdf_version_None/rob_0.765_0.222_rot_0.000_-0.000_3.117_rgb_overlay_open_drawer_c1/success_obj_0.0_0.0_qpos_0.000.mp4"
"/nas/longleaf/home/tis/mmai/results/rt_1_x_tf_trained_for_002272480_step/dummy_drawer/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/CloseMiddleDrawerCustomInScene-v0_shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_disable_bad_material_True_urdf_version_None/rob_0.652_0.009_rot_0.000_-0.000_3.142_rgb_overlay_open_drawer_b0/success_obj_0.0_0.0_qpos_0.000.mp4"
"/nas/longleaf/home/tis/mmai/results/rt_1_x_tf_trained_for_002272480_step/dummy_drawer/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/CloseTopDrawerCustomInScene-v0_shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_disable_bad_material_True_urdf_version_recolor_tabletop_visual_matching_1/rob_0.765_-0.182_rot_0.000_-0.000_3.122_rgb_overlay_open_drawer_a1/success_obj_0.0_0.0_qpos_0.026.mp4"
"/nas/longleaf/home/tis/mmai/results/rt_1_x_tf_trained_for_002272480_step/dummy_drawer/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/OpenBottomDrawerCustomInScene-v0_shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_disable_bad_material_True_urdf_version_recolor_tabletop_visual_matching_2/rob_0.752_0.009_rot_0.000_-0.000_3.142_rgb_overlay_open_drawer_b1/success_obj_0.0_0.0_qpos_0.287.mp4"
"/nas/longleaf/home/tis/mmai/results/rt_1_x_tf_trained_for_002272480_step/dummy_drawer/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/OpenMiddleDrawerCustomInScene-v0_shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_disable_bad_material_True_urdf_version_None/rob_0.851_0.035_rot_0.000_-0.000_3.142_rgb_overlay_open_drawer_b2/success_obj_0.0_0.0_qpos_0.354.mp4"
"/nas/longleaf/home/tis/mmai/results/rt_1_x_tf_trained_for_002272480_step/dummy_drawer/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/OpenTopDrawerCustomInScene-v0_shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_disable_bad_material_True_urdf_version_recolor_tabletop_visual_matching_1/rob_0.665_0.224_rot_0.000_-0.000_3.142_rgb_overlay_open_drawer_c0/success_obj_0.0_0.0_qpos_0.167.mp4"
"/nas/longleaf/home/tis/mmai/results/rt_1_x_tf_trained_for_002272480_step/dummy_drawer/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/PlaceIntoClosedTopDrawerCustomInScene-v0_shader_dir_rt_station_name_mk_station_recolor_light_mode_simple_disable_bad_material_True_urdf_version_None_model_ids_baked_apple_v2/rob_0.644_-0.179_rot_0.000_-0.000_3.112_rgb_overlay_open_drawer_a0/success_obj_-0.02_0.030000000000000002_qpos_0.20030100643634796_is_drawer_open_True_has_contact_3.mp4"
"/nas/longleaf/home/tis/mmai/results/rt_1_x_tf_trained_for_002272480_step/google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_laid_vertically_True_urdf_version_None/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1/success_obj_-0.12_0.2_n_lift_significant_6_consec_grasp_True_grasped_True.mp4"
"/nas/longleaf/home/tis/mmai/results/rt_1_x_tf_trained_for_002272480_step/google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_lr_switch_True_urdf_version_recolor_cabinet_visual_matching_1/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1/success_obj_-0.235_0.2_n_lift_significant_4_consec_grasp_True_grasped_True.mp4"
"/nas/longleaf/home/tis/mmai/results/rt_1_x_tf_trained_for_002272480_step/google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/GraspSingleOpenedCokeCanInScene-v0_upright_True_urdf_version_recolor_cabinet_visual_matching_1/rob_0.35_0.2_rot_0.000_-0.000_3.142_rgb_overlay_google_coke_can_real_eval_1/success_obj_-0.35_0.42_n_lift_significant_3_consec_grasp_True_grasped_True.mp4"
"/nas/longleaf/home/tis/mmai/results/rt_1_x_tf_trained_for_002272480_step/google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleBakedTexInScene-v0_urdf_version_None_baked_except_bpb_orange/rob_0.35_0.21_rot_0.000_-0.000_3.052_rgb_overlay_google_move_near_real_eval_1/success_obj_episode_22_all_obj_keep_height_True_moved_correct_obj_True_moved_wrong_obj_False_near_tgt_obj_True_is_closest_to_tgt_True.mp4"
"/nas/longleaf/home/tis/mmai/results/rt_1_x_tf_trained_for_002272480_step/google_pick_coke_can_1_v4/arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner/MoveNearGoogleBakedTexInScene-v0_urdf_version_recolor_tabletop_visual_matching_1_baked_except_bpb_orange/rob_0.35_0.21_rot_0.000_-0.000_3.052_rgb_overlay_google_move_near_real_eval_1/success_obj_episode_28_all_obj_keep_height_True_moved_correct_obj_True_moved_wrong_obj_False_near_tgt_obj_True_is_closest_to_tgt_True.mp4"
)

# Create destination directories if they don't exist
mkdir -p "${DEST}"
mkdir -p "${GIF_DIR}"

# Loop through each file and copy it from the remote server to the local destination
for FILE in "${FILES[@]}"; do
    echo "Copying ${FILE} from ${REMOTE} to ${DEST}"
    scp "${REMOTE}:${FILE}" "${DEST}/"
done

echo "All files have been copied."

# Convert each MP4 video in the destination directory to a high-quality GIF
echo "Starting conversion of MP4 files to high-quality GIFs..."
for video in "${DEST}"/*.mp4; do
    filename=$(basename "${video}" .mp4)
    gif="${GIF_DIR}/${filename}.gif"
    palette="/tmp/${filename}_palette.png"
    
    echo "Generating palette for ${video}"
    ffmpeg -y -i "${video}" -vf "fps=${FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2,palettegen" "${palette}"
    
    echo "Converting ${video} to ${gif} with high quality"
    ffmpeg -i "${video}" -i "${palette}" -filter_complex "fps=${FPS},scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,paletteuse" "${gif}"
    
    # Remove temporary palette file
    rm "${palette}"
done

echo "Conversion complete. All GIFs are saved in ${GIF_DIR}"
