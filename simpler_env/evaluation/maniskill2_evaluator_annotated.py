"""
Evaluate a model on ManiSkill2 environment.
"""

import os
import numpy as np
from transforms3d.euler import quat2euler

# Import utilities for building the environment, processing observations, and visualization.
from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
from simpler_env.utils.visualization import write_video


def run_maniskill2_eval_single_episode(
    model,
    ckpt_path,
    robot_name,
    env_name,
    scene_name,
    robot_init_x,
    robot_init_y,
    robot_init_quat,
    control_mode,
    obj_init_x=None,
    obj_init_y=None,
    obj_episode_id=None,
    additional_env_build_kwargs=None,
    rgb_overlay_path=None,
    obs_camera_name=None,
    control_freq=3,
    sim_freq=513,
    max_episode_steps=80,
    instruction=None,
    enable_raytracing=False,
    additional_env_save_tags=None,
    logging_dir="./results",
):
    # If no additional environment build keyword arguments are provided, initialize an empty dictionary.
    if additional_env_build_kwargs is None:
        additional_env_build_kwargs = {}

    # Set up base keyword arguments for creating the ManiSkill2 environment.
    kwargs = dict(
        obs_mode="rgbd",  # Use RGB-D observations.
        robot=robot_name,
        sim_freq=sim_freq,
        control_mode=control_mode,
        control_freq=control_freq,
        max_episode_steps=max_episode_steps,
        scene_name=scene_name,
        camera_cfgs={"add_segmentation": True},  # Enable segmentation in camera configurations.
        rgb_overlay_path=rgb_overlay_path,
    )
    # If raytracing is enabled, update the environment kwargs accordingly.
    if enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        # Ensure raytracing keys come first for compatibility with result naming and metric calculation.
        additional_env_build_kwargs = ray_tracing_dict

    # Build the ManiSkill2 environment using the provided parameters.
    env = build_maniskill2_env(
        env_name,
        **additional_env_build_kwargs,
        **kwargs,
    )

    # Initialize environment reset options with robot initialization parameters.
    env_reset_options = {
        "robot_init_options": {
            "init_xy": np.array([robot_init_x, robot_init_y]),
            "init_rot_quat": robot_init_quat,
        }
    }
    # Set up object initialization options depending on the provided parameters.
    if obj_init_x is not None:
        assert obj_init_y is not None
        obj_variation_mode = "xy"
        env_reset_options["obj_init_options"] = {
            "init_xy": np.array([obj_init_x, obj_init_y]),
        }
    else:
        assert obj_episode_id is not None
        obj_variation_mode = "episode"
        env_reset_options["obj_init_options"] = {
            "episode_id": obj_episode_id,
        }
    # Reset the environment with the defined options to get the initial observation.
    obs, _ = env.reset(options=env_reset_options)
    # For long-horizon tasks, check if the current subtask is the final one.
    is_final_subtask = env.is_final_subtask()

    # Obtain the language instruction for the task.
    if instruction is not None:
        task_description = instruction
    else:
        # Get the default language instruction from the environment.
        task_description = env.get_language_instruction()
    print(task_description)

    # Initialize logging by capturing the initial observation image.
    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
    images = [image]  # List to store images for video generation.
    predicted_actions = []  # List to store raw model actions.
    predicted_terminated, done, truncated = False, False, False  # Flags to control episode termination.

    # Reset the model with the task description to initialize any internal states.
    model.reset(task_description)

    timestep = 0
    success = "failure"  # Default status is failure unless the episode completes successfully.

    # Main simulation loop: continue until the model signals termination or the episode is truncated.
    while not (predicted_terminated or truncated):
        # Obtain the model's raw and processed actions given the current image and task description.
        raw_action, action = model.step(image, task_description)
        predicted_actions.append(raw_action)
        # Check if the model's processed action indicates termination.
        predicted_terminated = bool(action["terminate_episode"][0] > 0)
        if predicted_terminated:
            # If the subtask is not the final one, advance to the next subtask instead of terminating the episode.
            if not is_final_subtask:
                predicted_terminated = False
                env.advance_to_next_subtask()

        # Execute the action in the environment.
        # The action is constructed by concatenating the world vector, rotation (axis-angle), and gripper control.
        obs, reward, done, truncated, info = env.step(
            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
        )
        
        # Update the success flag if the episode is done.
        success = "success" if done else "failure"
        # Check if the language instruction has been updated (e.g., when advancing subtasks).
        new_task_description = env.get_language_instruction()
        if new_task_description != task_description:
            task_description = new_task_description
            print(task_description)
        # Update the flag for the final subtask.
        is_final_subtask = env.is_final_subtask()

        # Print the current timestep and any additional environment information for logging.
        print(timestep, info)

        # Capture the new image observation and add it to the image list.
        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=obs_camera_name)
        images.append(image)
        timestep += 1

    # Extract any episode statistics provided by the environment.
    episode_stats = info.get("episode_stats", {})

    # ------------------------------
    # Video Saving Section
    # ------------------------------
    # Construct a name for the environment save directory based on environment name and additional kwargs.
    env_save_name = env_name
    for k, v in additional_env_build_kwargs.items():
        env_save_name = env_save_name + f"_{k}_{v}"
    if additional_env_save_tags is not None:
        env_save_name = env_save_name + f"_{additional_env_save_tags}"
    # Process the checkpoint path to create a basename for saving.
    ckpt_path_basename = ckpt_path if ckpt_path[-1] != "/" else ckpt_path[:-1]
    ckpt_path_basename = ckpt_path_basename.split("/")[-1]
    # Generate a video file name based on object variation mode and initialization parameters.
    if obj_variation_mode == "xy":
        video_name = f"{success}_obj_{obj_init_x}_{obj_init_y}"
    elif obj_variation_mode == "episode":
        video_name = f"{success}_obj_episode_{obj_episode_id}"
    for k, v in episode_stats.items():
        video_name = video_name + f"_{k}_{v}"
    video_name = video_name + ".mp4"
    # Process the RGB overlay path if provided.
    if rgb_overlay_path is not None:
        rgb_overlay_path_str = os.path.splitext(os.path.basename(rgb_overlay_path))[0]
    else:
        rgb_overlay_path_str = "None"
    # Convert robot initial quaternion to Euler angles for the file naming.
    r, p, y = quat2euler(robot_init_quat)
    video_path = f"{ckpt_path_basename}/{scene_name}/{control_mode}/{env_save_name}/rob_{robot_init_x}_{robot_init_y}_rot_{r:.3f}_{p:.3f}_{y:.3f}_rgb_overlay_{rgb_overlay_path_str}/{video_name}"
    video_path = os.path.join(logging_dir, video_path)
    # Write the video file using the collected images.
    write_video(video_path, images, fps=5)

    # ------------------------------
    # Action Trajectory Saving Section
    # ------------------------------
    # Create a file path for saving a visualization of the action trajectory.
    action_path = video_path.replace(".mp4", ".png")
    action_root = os.path.dirname(action_path) + "/actions/"
    os.makedirs(action_root, exist_ok=True)
    action_path = action_root + os.path.basename(action_path)
    # Use the model's visualization method to save the action trajectory overlaying the images.
    model.visualize_epoch(predicted_actions, images, save_path=action_path)

    # Return True if the episode was successful, otherwise False.
    return success == "success"


def maniskill2_evaluator(model, args):
    # Determine the control mode based on the robot and policy model.
    control_mode = get_robot_control_mode(args.robot, args.policy_model)
    success_arr = []  # List to store success outcomes for each evaluated episode.

    # Loop over combinations of robot initial positions and orientations.
    for robot_init_x in args.robot_init_xs:
        for robot_init_y in args.robot_init_ys:
            for robot_init_quat in args.robot_init_quats:
                # Construct keyword arguments for a single episode evaluation.
                kwargs = dict(
                    model=model,
                    ckpt_path=args.ckpt_path,
                    robot_name=args.robot,
                    env_name=args.env_name,
                    scene_name=args.scene_name,
                    robot_init_x=robot_init_x,
                    robot_init_y=robot_init_y,
                    robot_init_quat=robot_init_quat,
                    control_mode=control_mode,
                    additional_env_build_kwargs=args.additional_env_build_kwargs,
                    rgb_overlay_path=args.rgb_overlay_path,
                    control_freq=args.control_freq,
                    sim_freq=args.sim_freq,
                    max_episode_steps=args.max_episode_steps,
                    enable_raytracing=args.enable_raytracing,
                    additional_env_save_tags=args.additional_env_save_tags,
                    obs_camera_name=args.obs_camera_name,
                    logging_dir=args.logging_dir,
                )
                # Evaluate episodes based on the object variation mode.
                if args.obj_variation_mode == "xy":
                    # Iterate over provided object x and y initialization positions.
                    for obj_init_x in args.obj_init_xs:
                        for obj_init_y in args.obj_init_ys:
                            success_arr.append(
                                run_maniskill2_eval_single_episode(
                                    obj_init_x=obj_init_x,
                                    obj_init_y=obj_init_y,
                                    **kwargs,
                                )
                            )
                elif args.obj_variation_mode == "episode":
                    # Iterate over a range of episode IDs for object initialization variation.
                    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
                        success_arr.append(run_maniskill2_eval_single_episode(obj_episode_id=obj_episode_id, **kwargs))
                else:
                    # Raise an error if an unknown object variation mode is specified.
                    raise NotImplementedError()

    # Return the list of success statuses for all episodes.
    return success_arr