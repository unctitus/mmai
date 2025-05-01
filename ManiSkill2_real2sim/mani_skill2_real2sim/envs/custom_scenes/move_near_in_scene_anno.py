from collections import OrderedDict
from typing import List, Optional

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat

from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.common import random_choice
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.utils.sapien_utils import vectorize_pose

from .base_env import CustomSceneEnv, CustomOtherObjectsInSceneEnv


class MoveNearInSceneEnv(CustomSceneEnv):
    """
    Environment where the goal is to move one object near another within a scene,
    supporting various lighting options and prepackaged evaluation configurations.
    """

    # Class-level defaults for asset and scene roots and model JSON config
    DEFAULT_ASSET_ROOT: str
    DEFAULT_SCENE_ROOT: str
    DEFAULT_MODEL_JSON: str

    def __init__(
        self,
        original_lighting: bool = False,
        slightly_darker_lighting: bool = False,
        slightly_brighter_lighting: bool = False,
        ambient_only_lighting: bool = False,
        prepackaged_config: bool = False,
        **kwargs,
    ):
        """
        Initialize the environment, preparing episode variables and lighting options.

        Args:
            original_lighting (bool): If True, use the scene's original light setup.
            slightly_darker_lighting (bool): If True, reduce directional light intensity.
            slightly_brighter_lighting (bool): If True, increase directional light intensity.
            ambient_only_lighting (bool): If True, remove directional lights, use only ambient.
            prepackaged_config (bool): If True, apply built-in evaluation settings.
            **kwargs: Passed to the base CustomSceneEnv constructor.
        """
        # Prepare containers for three objects tracked in each episode
        self.episode_objs = [None] * 3                          # holds actor references
        self.episode_model_ids = [None] * 3                     # holds model identifiers
        self.episode_model_scales = [None] * 3                  # holds scale factors
        self.episode_model_bbox_sizes = [None] * 3              # holds world bbox dims
        self.episode_model_init_xyzs = [None] * 3               # record initial xyz if needed
        self.episode_obj_heights_after_settle = [None] * 3      # record heights after physics settle
        self.episode_source_obj = None                          # reference to source object
        self.episode_target_obj = None                          # reference to target object
        self.episode_source_obj_bbox_world = None               # world-frame bbox extents for source
        self.episode_target_obj_bbox_world = None               # world-frame bbox extents for target
        self.episode_obj_xyzs_after_settle = [None] * 3         # record settled xyz positions
        self.episode_source_obj_xyz_after_settle = None         # source object settled position
        self.episode_target_obj_xyz_after_settle = None         # target object settled position
        self.episode_stats = None                               # track metrics like success/failure flags

        # Store any object-init options provided at reset time
        self.obj_init_options = {}

        # Lighting flags set by the user
        self.original_lighting = original_lighting
        self.slightly_darker_lighting = slightly_darker_lighting
        self.slightly_brighter_lighting = slightly_brighter_lighting
        self.ambient_only_lighting = ambient_only_lighting

        # If requested, apply default evaluation settings immediately
        self.prepackaged_config = prepackaged_config
        if self.prepackaged_config:
            # Merge returned dict into kwargs for base init
            kwargs.update(self._setup_prepackaged_env_init_config())

        # Call parent constructor to complete environment initialization
        super().__init__(**kwargs)


    # This will just add a few default keyword argument to the init call of the parent class
    def _setup_prepackaged_env_init_config(self):
        """
        Build a dictionary of default settings for visual matching evaluation.

        Returns:
            dict: keys include robot type, frequencies, control mode, scene name,
                  and RGB overlay settings.
        """
        ret = {}                                              # container for settings
        ret["robot"] = "google_robot_static"               # fixed robot used for evaluation
        ret["control_freq"] = 3                              # control loop frequency (Hz)
        ret["sim_freq"] = 513                                # physics sim frequency (Hz)
        ret["control_mode"] = (
            "arm_pd_ee_delta_pose_align_interpolate_by_planner_"
            "gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        )                                                      # complex control law identifier
        ret["scene_name"] = "google_pick_coke_can_1_v4"    # specific scene asset
        ret["camera_cfgs"] = {"add_segmentation": True}     # enable segmentation overlay
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/google_move_near_real_eval_1.png"
        )                                                     # path to real image overlay
        ret["rgb_overlay_cameras"] = ["overhead_camera"]     # apply overlay to this camera
        print("Loading Overlay Image from:", ret["rgb_overlay_path"])  # debug print
        return ret                                           # return config map

    def _get_default_scene_config(self):
        """
        Retrieve the base class's default scene configuration, then adjust contact offset.

        Returns:
            SceneConfig: modified configuration with tighter contact offset.
        """
        scene_config = super()._get_default_scene_config()      # get default settings
        scene_config.contact_offset = 0.005                     # reduce contact threshold
        return scene_config

    def _setup_lighting(self):
        """
        Configure the scene's lighting based on flags set at init.

        Applies directional lights, ambient light, and shadow settings.
        """
        # If a custom background is set, skip lighting adjustments entirely
        if self.bg_name is not None:
            return

        shadow = self.enable_shadow                            # whether shadows are enabled
        # Original scene lighting
        if self.original_lighting:
            self._scene.set_ambient_light([0.3, 0.3, 0.3])      # moderate ambient base
            self._scene.add_directional_light(
                [1, 1, -1], [1, 1, 1], shadow=shadow, scale=5, shadow_map_size=2048
            )                                                  # strong primary light
            self._scene.add_directional_light([0, 0, -1], [1, 1, 1])  # fill light
        # Slightly darker variant
        elif self.slightly_darker_lighting:
            self._scene.set_ambient_light([0.3, 0.3, 0.3])
            self._scene.add_directional_light(
                [1, 1, -1], [0.8, 0.8, 0.8], shadow=shadow, scale=5, shadow_map_size=2048
            )
            self._scene.add_directional_light([0, 0, -1], [0.8, 0.8, 0.8])
        # Slightly brighter variant
        elif self.slightly_brighter_lighting:
            self._scene.set_ambient_light([0.3, 0.3, 0.3])
            self._scene.add_directional_light(
                [0, 0, -1], [3.6, 3.6, 3.6], shadow=shadow, scale=5, shadow_map_size=2048
            )
            self._scene.add_directional_light([-1, -0.5, -1], [1.3, 1.3, 1.3])
            self._scene.add_directional_light([1, 1, -1], [1.3, 1.3, 1.3])
        # Ambient light only (no directional)
        elif self.ambient_only_lighting:
            self._scene.set_ambient_light([1.0, 1.0, 1.0])      # full white ambient
        # Default fallback lighting
        else:
            self._scene.set_ambient_light([0.3, 0.3, 0.3])
            self._scene.add_directional_light(
                [0, 0, -1], [2.2, 2.2, 2.2], shadow=shadow, scale=5, shadow_map_size=2048
            )
            self._scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
            self._scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])

    def _load_actors(self):
        """
        Load the arena and object actors into the scene, then set physics damping.
        """
        self._load_arena_helper()                              # load static scene geometry
        self._load_model()                                     # load dynamic object models
        for obj in self.episode_objs:
            obj.set_damping(0.1, 0.1)                          # reduce jitter via damping

    def _load_model(self):
        """
        Abstract method stub for loading object models;
        requires subclass implementation.
        """
        raise NotImplementedError                            # enforce override by subclasses

    def reset(self, seed=None, options=None):
        """
        Reset environment state for a new episode.

        Chooses model IDs/scales, applies config resets, and returns initial observation.
        """
        if options is None:
            options = {}                                      # ensure options is a dict
        options = options.copy()                              # copy to avoid mutating user input

        self.obj_init_options = options.get("obj_init_options", {})  # record init positions

        self.set_episode_rng(seed)                           # reseed RNG for reproducibility
        # Get user-provided or random model IDs and scales
        model_scales = options.get("model_scales", None)
        model_ids = options.get("model_ids", None)
        reconfigure = options.get("reconfigure", False)     # track if we need to reload actors
        _reconfigure = self._set_model(model_ids, model_scales)
        reconfigure = reconfigure or _reconfigure             # update flag if model changed

        if self.prepackaged_config:
            # apply any extra prepackaged resets (e.g. URDF randomization)
            _reconfigure = self._additional_prepackaged_config_reset(options)
            reconfigure = reconfigure or _reconfigure         # update flag

        options["reconfigure"] = reconfigure                # pass flag to parent reset

        self._initialize_episode_stats()                      # zero out episode metrics

        obs, info = super().reset(seed=self._episode_seed, options=options)  # base reset
        # Add extra info fields about chosen models and initial poses
        info.update({
            "episode_model_ids": self.episode_model_ids,
            "episode_model_scales": self.episode_model_scales,
            "episode_source_obj_name": self.episode_source_obj.name,
            "episode_target_obj_name": self.episode_target_obj.name,
            "episode_source_obj_init_pose_wrt_robot_base":
                self.agent.robot.pose.inv() * self.episode_source_obj.pose,
            "episode_target_obj_init_pose_wrt_robot_base":
                self.agent.robot.pose.inv() * self.episode_target_obj.pose,
        })
        return obs, info                                    # return starting observation

    def _additional_prepackaged_config_reset(self, options):
        """
        Apply extra adjustments when using prepackaged configs:
        positions the robot and randomly recolors URDFs.

        Returns:
            bool: True if any reconfiguration occurred.
        """
        options["robot_init_options"] = {
            "init_xy": [0.35, 0.21],                          # fixed robot xy initial location
            "init_rot_quat": (
                sapien.Pose(q=euler2quat(0, 0, -0.09)) * sapien.Pose(q=[0, 0, 0, 1])
            ).q,                                               # initial orientation quaternion
        }
        # Randomly choose a recoloring variant
        new_urdf_version = self._episode_rng.choice([
            "", "recolor_tabletop_visual_matching_1",
            "recolor_tabletop_visual_matching_2",
            "recolor_cabinet_visual_matching_1",
        ])
        if new_urdf_version != self.urdf_version:
            self.urdf_version = new_urdf_version             # set new URDF variant
            self._configure_agent()                          # reload robot with new URDF
            return True                                      # indicate change
        return False                                         # no change

    def _initialize_episode_stats(self):
        """
        Create the OrderedDict structure to track per-episode metrics.
        """
        self.episode_stats = OrderedDict(
            all_obj_keep_height=False,                        # did all objects stay upright?
            moved_correct_obj=False,                          # was the source moved?
            moved_wrong_obj=False,                            # did any other object move?
            near_tgt_obj=False,                               # is source near target?
            is_closest_to_tgt=False,                          # is source the closest to target?
        )

    @staticmethod
    def _list_equal(l1, l2):
        """
        Compare two lists element-wise for exact equality.

        Returns:
            bool: True if both lists match in length and contents.
        """
        if len(l1) != len(l2):
            return False
        for i in range(len(l1)):
            if l1[i] != l2[i]:
                return False
        return True

    def _set_model(self, model_ids, model_scales):
        """
        Determine and set the model IDs/scales for this episode,
        randomly sampling if not user-provided.

        Returns:
            bool: True if any ID or scale changed, triggering a reload.
        """
        reconfigure = False                                  # track if models changed

        # Choose model IDs if unspecified
        if model_ids is None:
            model_ids = [random_choice(self.model_ids, self._episode_rng) for _ in range(3)]
        if not self._list_equal(model_ids, self.episode_model_ids):
            self.episode_model_ids = model_ids                # update stored IDs
            reconfigure = True

        # Choose model scales if unspecified
        if model_scales is None:
            model_scales = []                                 # build list of scales
            for m_id in self.episode_model_ids:
                scales = self.model_db[m_id].get("scales", None)
                if scales is None:
                    model_scales.append(1.0)                  # default scale
                else:
                    model_scales.append(random_choice(scales, self._episode_rng))
        if not self._list_equal(model_scales, self.episode_model_scales):
            self.episode_model_scales = model_scales          # update stored scales
            reconfigure = True

        # Compute bounding boxes in world frame for each object
        model_bbox_sizes = []                                # extents list
        for m_id, m_scale in zip(self.episode_model_ids, self.episode_model_scales):
            model_info = self.model_db[m_id]
            if "bbox" in model_info:
                bbox = model_info["bbox"]                   # dict with min/max
                size = np.array(bbox["max"]) - np.array(bbox["min"])
                model_bbox_sizes.append(size * m_scale)
            else:
                raise ValueError(f"Model {m_id} lacks bbox info.")
        self.episode_model_bbox_sizes = model_bbox_sizes

        return reconfigure

    def _initialize_actors(self):
        """
        Place all objects in the scene based on init options,
        let them fall to settle, then record their final poses.
        """
        # Extract specified source/target indices
        source_obj_id = self.obj_init_options.get("source_obj_id")
        target_obj_id = self.obj_init_options.get("target_obj_id")
        assert source_obj_id is not None and target_obj_id is not None, "Source and target must be set"
        # Map indices to object references
        self.episode_source_obj = self.episode_objs[source_obj_id]
        self.episode_target_obj = self.episode_objs[target_obj_id]
        # Store world bbox extents for collision checks
        self.episode_source_obj_bbox_world = self.episode_model_bbox_sizes[source_obj_id]
        self.episode_target_obj_bbox_world = self.episode_model_bbox_sizes[target_obj_id]

        # Get XY placements for all objects
        init_xys = np.array(self.obj_init_options.get("init_xys"))
        assert init_xys.shape == (len(self.episode_objs), 2), "init_xys must match object count"
        # Compute Z so objects start above table
        z0 = self.obj_init_options.get("init_z", self.scene_table_height) + 0.5

        # Rotation quaternions for orientation; default is identity
        rot_quats = self.obj_init_options.get("init_rot_quats")
        if rot_quats is not None:
            rot_quats = np.array(rot_quats)
            assert rot_quats.shape == (len(self.episode_objs), 4)
        else:
            rot_quats = np.tile([1.0, 0.0, 0.0, 0.0], (len(self.episode_objs), 1))

        # Place and lock each object
        for i, obj in enumerate(self.episode_objs):
            pos = np.hstack([init_xys[i], z0])               # [x, y, z]
            quat = rot_quats[i]                              # [w, x, y, z]
            obj.set_pose(sapien.Pose(pos, quat))             # apply initial pose
            obj.lock_motion(0, 0, 0, 1, 1, 0)                # lock rotations along x/y

        # Move robot base out of view during settling
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        self._settle(0.5)                                   # let physics settle
        # Unlock motions and reset velocities
        for obj in self.episode_objs:
            obj.lock_motion(0, 0, 0, 0, 0, 0)                # allow full motion
            obj.set_pose(obj.pose)                          # wake actor
            obj.set_velocity(np.zeros(3))                   # clear linear velocity
            obj.set_angular_velocity(np.zeros(3))            # clear angular velocity
        self._settle(0.5)

        # If still moving, settle longer
        lin_vel = sum(np.linalg.norm(o.velocity) for o in self.episode_objs)
        ang_vel = sum(np.linalg.norm(o.angular_velocity) for o in self.episode_objs)
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(1.5)

        # Record final poses after settling
        self.episode_obj_xyzs_after_settle = [obj.pose.p for obj in self.episode_objs]
        self.episode_source_obj_xyz_after_settle = self.episode_obj_xyzs_after_settle[source_obj_id]
        self.episode_target_obj_xyz_after_settle = self.episode_obj_xyzs_after_settle[target_obj_id]
        # Rotate bounding boxes into final orientation
        self.episode_source_obj_bbox_world = quat2mat(self.episode_source_obj.pose.q) @ self.episode_source_obj_bbox_world
        self.episode_target_obj_bbox_world = quat2mat(self.episode_target_obj.pose.q) @ self.episode_target_obj_bbox_world

    @property
    def source_obj_pose(self):
        """
        Compute and return the source object's center-of-mass pose in world frame.
        """
        # Transform local COM pose by actor world transform
        return self.episode_source_obj.pose.transform(
            self.episode_source_obj.cmass_local_pose
        )

    @property
    def target_obj_pose(self):
        """
        Compute and return the target object's center-of-mass pose in world frame.
        """
        return self.episode_target_obj.pose.transform(
            self.episode_target_obj.cmass_local_pose
        )

    def _get_obs_extra(self) -> OrderedDict:
        """
        Generate additional observation entries when using state-based modes.

        Outputs include TCP pose and relative object positions.
        """
        obs = OrderedDict()
        # Always include robot end-effector pose
        obs['tcp_pose'] = vectorize_pose(self.tcp.pose)
        if self._obs_mode in ['state', 'state_dict']:
            # Include object COM poses and relative vector
            obs['source_obj_pose'] = vectorize_pose(self.source_obj_pose)
            obs['target_obj_pose'] = vectorize_pose(self.target_obj_pose)
            obs['tcp_to_source_obj_pos'] = (self.source_obj_pose.p - self.tcp.pose.p)
        return obs

    def evaluate(self, **kwargs):
        """
        Assess the current scene for success/failure metrics.

        Checks height changes, object displacement, proximity, and ranking,
        then aggregates into a boolean success flag.

        Returns:
            dict: Detailed metrics and episode_stats.
        """
        # Fetch current COM poses for source and target
        src_pose = self.source_obj_pose
        tgt_pose = self.target_obj_pose

        # Identify other objects not involved in the task
        other_ids = [i for i, obj in enumerate(self.episode_objs)
                     if obj.name not in [self.episode_source_obj.name, self.episode_target_obj.name]]
        # Measure height differences to detect falls
        other_heights = [self.episode_objs[i].pose.p[2] for i in other_ids]
        settle_heights = [self.episode_obj_xyzs_after_settle[i][2] for i in other_ids]
        height_diffs = [h - h0 for h, h0 in zip(other_heights, settle_heights)]
        other_obj_keep_height = all(d > -0.02 for d in height_diffs)
        # Check source and target remain above table
        src_diff = src_pose.p[2] - self.episode_source_obj_xyz_after_settle[2]
        tgt_diff = tgt_pose.p[2] - self.episode_target_obj_xyz_after_settle[2]
        all_obj_keep_height = other_obj_keep_height and (src_diff > -0.15) and (tgt_diff > -0.15)

        # Distance moved by source vs others to check correct movement
        src_move = np.linalg.norm(self.episode_source_obj_xyz_after_settle[:2] - src_pose.p[:2])
        other_moves = [np.linalg.norm(self.episode_obj_xyzs_after_settle[i][:2] - self.episode_objs[i].pose.p[:2])
                       for i in other_ids]
        moved_correct_obj = (src_move > 0.03) and all(m < src_move for m in other_moves)
        moved_wrong_obj = any(m > 0.03 for m in other_moves) and any(m > src_move for m in other_moves)

        # Proximity test: source near target within bounding box extents + margin
        dist_to_tgt = np.linalg.norm(src_pose.p[:2] - tgt_pose.p[:2])
        tgt_bbox = np.linalg.norm(self.episode_target_obj_bbox_world[:2]) / 2
        src_bbox = np.linalg.norm(self.episode_source_obj_bbox_world[:2]) / 2
        near_tgt_obj = dist_to_tgt < (tgt_bbox + src_bbox + 0.10)

        # Check if source is closest to target compared to others
        other_distances = [np.linalg.norm(src_pose.p[:2] - self.episode_objs[i].pose.p[:2])
                           for i in other_ids]
        is_closest_to_tgt = all(dist_to_tgt < d + 0.01 for d in other_distances)

        # Aggregate metrics into success
        success = all_obj_keep_height and moved_correct_obj and near_tgt_obj and is_closest_to_tgt

        # Build return dict
        ret = dict(
            all_obj_keep_height=all_obj_keep_height,
            moved_correct_obj=moved_correct_obj,
            moved_wrong_obj=moved_wrong_obj,
            near_tgt_obj=near_tgt_obj,
            is_closest_to_tgt=is_closest_to_tgt,
            success=success,
        )
        # Update episode_stats to reflect these booleans
        for k in self.episode_stats:
            self.episode_stats[k] = ret[k]
        ret['episode_stats'] = self.episode_stats
        return ret

    def compute_dense_reward(self, info, **kwargs):
        """
        Compute the reward based on success: 1.0 if successful, else 0.0.

        Args:
            info (dict): metrics from evaluate().
        Returns:
            float: dense reward.
        """
        return 1.0 if info.get('success', False) else 0.0

    def compute_normalized_dense_reward(self, **kwargs):
        """
        Return the dense reward normalized by maximum possible (1.0).

        Returns:
            float: normalized reward in [0, 1].
        """
        # Since max reward is 1.0, normalization is identity
        return self.compute_dense_reward(**kwargs)

    def get_language_instruction(self, **kwargs):
        """
        Generate a simple textual instruction for the current task instance.

        Returns:
            str: e.g., "move apple near bottle".
        """
        # Get human-readable names via helper method
        src_name = self._get_instruction_obj_name(self.episode_source_obj.name)
        tgt_name = self._get_instruction_obj_name(self.episode_target_obj.name)
        # Format and return command string
        return f"move {src_name} near {tgt_name}"




@register_env("MoveNearGoogleInScene-v0", max_episode_steps=80)
class MoveNearGoogleInSceneEnv(MoveNearInSceneEnv, CustomOtherObjectsInSceneEnv):
    """
    Environment where a robot must move near specified target objects in a scene, optionally among distractors.

    This class sets up object triplets (source, target, distractor), their initial positions, orientations, and
    densities, handles episode initialization by selecting randomized configurations, and loads physical
    object models into the SAPIEN scene.
    """
    def __init__(self, no_distractor=False, **kwargs):
        """
        Initialize the MoveNearGoogleInScene environment.

        Args:
            no_distractor (bool): If True, only include source and target objects (no distractors).
            **kwargs: Additional options passed to parent environment constructors.
        """
        # Whether to exclude distractor objects from the scene
        self.no_distractor = no_distractor
        # Prepare object configuration data (triplets, poses, densities)
        self._setup_obj_configs()
        # Initialize parent classes with any remaining keyword arguments
        super().__init__(**kwargs)

    def _setup_obj_configs(self):
        """
        Define the available object triplets, their initial XY placements, orientations, and special densities.

        This config drives how each episode samples source, target, and optional distractor objects.
        """
        # List of object name triplets: (distractor, source, target)
        self.triplets = [
            ("blue_plastic_bottle", "opened_pepsi_can", "orange"),
            ("opened_7up_can", "apple", "sponge"),
            ("opened_coke_can", "opened_redbull_can", "apple"),
            ("sponge", "blue_plastic_bottle", "opened_7up_can"),
            ("orange", "opened_pepsi_can", "opened_redbull_can"),
        ]
        # Flattened lists of indices for source and target selections within each triplet
        self._source_obj_ids, self._target_obj_ids = [], []
        for i in range(3):
            for j in range(3):
                if i != j:
                    self._source_obj_ids.append(i)
                    self._target_obj_ids.append(j)
        # Predefined XY coordinates for the three objects in each triplet
        self._xy_config_per_triplet = [
            ([-0.33, 0.04], [-0.33, 0.34], [-0.13, 0.19]),
            ([-0.13, 0.04], [-0.33, 0.19], [-0.13, 0.34]),
        ]
        # Map of object names to quaternions for initial orientation
        self.obj_init_quat_dict = {
            "blue_plastic_bottle": euler2quat(np.pi / 2, 0, np.pi / 2),
            "opened_pepsi_can":     euler2quat(np.pi / 2, 0, 0),
            "orange":               euler2quat(0, 0, np.pi / 2),
            "opened_7up_can":       euler2quat(np.pi / 2, 0, 0),
            "apple":                [1.0, 0.0, 0.0, 0.0],            # Identity quaternion
            "sponge":               euler2quat(0, 0, np.pi / 2),
            "opened_coke_can":      euler2quat(np.pi / 2, 0, 0),
            "opened_redbull_can":   euler2quat(np.pi / 2, 0, 0),
        }
        # Override densities for specific objects (others use default from asset database)
        self.special_density_dict = {
            "apple": 200,   # Toy apple density
            "orange": 200,  # Toy orange density
            # Default densities: opened cans=50, plastic bottle=50, sponge=150
        }

    def reset(self, seed=None, options=None):
        """
        Reset the environment for a new episode.

        Randomly selects a triplet, source/target indices, and position/orientation configs. Optionally
        removes distractor if no_distractor=True. Passes these options to the parent reset method.

        Args:
            seed (int): Seed for episode RNG.
            options (dict): Optional override for initialization parameters.

        Returns:
            obs: Initial observation.
            info (dict): Episode metadata including 'episode_id'.
        """
        # Prepare options dict and episode RNG
        if options is None:
            options = {}
        options = options.copy()
        self.set_episode_rng(seed)

        # Extract or initialize object init options
        obj_init_options = options.get("obj_init_options", {}).copy()

        # Compute total possible episodes
        num_triplets = len(self.triplets)
        num_pairs = len(self._source_obj_ids)
        num_xy = len(self._xy_config_per_triplet)
        _num_episodes = num_triplets * num_pairs * num_xy

        # Determine episode_id either from options or randomly
        episode_id = obj_init_options.get(
            "episode_id", self._episode_rng.randint(_num_episodes)
        )
        # Decode triplet and source/target indices
        triplet = self.triplets[
            episode_id // (num_pairs * num_xy)
        ]
        source_obj_id = self._source_obj_ids[episode_id % num_pairs]
        target_obj_id = self._target_obj_ids[episode_id % num_pairs]
        # Select XY coords and quaternions for this episode
        xy_config_triplet = self._xy_config_per_triplet[
            (episode_id % (num_pairs * num_xy)) // num_pairs
        ]
        quat_config_triplet = [
            self.obj_init_quat_dict[name] for name in triplet
        ]

        # If no distractors, trim to only source and target
        if self.no_distractor:
            triplet = [triplet[source_obj_id], triplet[target_obj_id]]
            xy_config_triplet = [xy_config_triplet[source_obj_id], xy_config_triplet[target_obj_id]]
            quat_config_triplet = [quat_config_triplet[source_obj_id], quat_config_triplet[target_obj_id]]
            source_obj_id = 0
            target_obj_id = 1

        # Populate options for parent reset
        options["model_ids"] = triplet
        obj_init_options.update({
            "source_obj_id": source_obj_id,
            "target_obj_id": target_obj_id,
            "init_xys": xy_config_triplet,
            "init_rot_quats": quat_config_triplet,
        })
        options["obj_init_options"] = obj_init_options

        # Call parent reset and append episode metadata
        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update({"episode_id": episode_id})
        return obs, info

    def _load_model(self):
        """
        Load and instantiate physical object models into the SAPIEN scene for the current episode.

        Iterates over selected model IDs and scales, applies special densities if specified,
        and builds actors with physical materials.
        """
        self.episode_objs = []  # Container for created object actors
        for model_id, model_scale in zip(self.episode_model_ids, self.episode_model_scales):
            # Determine object density (override if in special dict)
            density = self.special_density_dict.get(
                model_id,
                self.model_db[model_id].get("density", 1000)
            )
            # Build the physical actor in the scene
            obj = self._build_actor_helper(
                model_id,
                self._scene,
                scale=model_scale,
                density=density,
                physical_material=self._scene.create_physical_material(
                    static_friction=self.obj_static_friction,
                    dynamic_friction=self.obj_dynamic_friction,
                    restitution=0.0,
                ),
                root_dir=self.asset_root,
            )
            obj.name = model_id  # Tag actor with its model name
            self.episode_objs.append(obj)


@register_env("MoveNearGoogleBakedTexInScene-v0", max_episode_steps=80)
class MoveNearGoogleBakedTexInSceneEnv(MoveNearGoogleInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_baked_tex_v0.json"

    def _setup_obj_configs(self):
        # Note: the cans are "opened" here to match the real evaluation; we'll remove "open" when getting language instruction
        self.triplets = [
            ("blue_plastic_bottle", "baked_opened_pepsi_can", "orange"),
            ("baked_opened_7up_can", "baked_apple", "baked_sponge"),
            ("baked_opened_coke_can", "baked_opened_redbull_can", "baked_apple"),
            ("baked_sponge", "blue_plastic_bottle", "baked_opened_7up_can"),
            ("orange", "baked_opened_pepsi_can", "baked_opened_redbull_can"),
        ]
        self._source_obj_ids, self._target_obj_ids = [], []
        for i in range(3):
            for j in range(3):
                if i != j:
                    self._source_obj_ids.append(i)
                    self._target_obj_ids.append(j)
        self._xy_config_per_triplet = [
            ([-0.33, 0.04], [-0.33, 0.34], [-0.13, 0.19]),
            ([-0.13, 0.04], [-0.33, 0.19], [-0.13, 0.34]),
        ]
        self.obj_init_quat_dict = {
            "blue_plastic_bottle": euler2quat(np.pi / 2, 0, np.pi / 2),
            "baked_opened_pepsi_can": euler2quat(np.pi / 2, 0, 0),
            "orange": euler2quat(0, 0, np.pi / 2),
            "baked_opened_7up_can": euler2quat(np.pi / 2, 0, 0),
            "baked_apple": [1.0, 0.0, 0.0, 0.0],
            "baked_sponge": euler2quat(0, 0, np.pi / 2),
            "baked_opened_coke_can": euler2quat(np.pi / 2, 0, 0),
            "baked_opened_redbull_can": euler2quat(np.pi / 2, 0, 0),
        }
        self.special_density_dict = {"baked_apple": 200, "orange": 200}


@register_env("MoveNearGoogleBakedTexInScene-v1", max_episode_steps=80)
class MoveNearGoogleBakedTexInSceneEnvV1(MoveNearGoogleInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_baked_tex_v1.json"

    def __init__(self, light_mode=None, **kwargs):
        self.light_mode = light_mode
        super().__init__(**kwargs)

    def _setup_lighting(self):
        if (self.light_mode is None) or ("simple" not in self.light_mode):
            super()._setup_lighting()
        elif self.light_mode == "simple":
            self._scene.set_ambient_light([1.0] * 3)
        elif self.light_mode == "simple2":
            self._scene.set_ambient_light([1.0] * 3)
            angle = 90
            self._scene.add_directional_light(
                [-np.cos(np.deg2rad(angle)), 0, -np.sin(np.deg2rad(angle))], [0.5] * 3
            )

    def _setup_obj_configs(self):
        # Note: the cans are "opened" here to match the real evaluation; we'll remove "open" when getting language instruction
        self.triplets = [
            ("blue_plastic_bottle", "baked_opened_pepsi_can_v2", "orange"),
            ("baked_opened_7up_can_v2", "baked_apple_v2", "baked_sponge_v2"),
            (
                "baked_opened_coke_can_v2",
                "baked_opened_redbull_can_v2",
                "baked_apple_v2",
            ),
            ("baked_sponge_v2", "blue_plastic_bottle", "baked_opened_7up_can_v2"),
            ("orange", "baked_opened_pepsi_can_v2", "baked_opened_redbull_can_v2"),
        ]
        self._source_obj_ids, self._target_obj_ids = [], []
        for i in range(3):
            for j in range(3):
                if i != j:
                    self._source_obj_ids.append(i)
                    self._target_obj_ids.append(j)
        self._xy_config_per_triplet = [
            ([-0.33, 0.04], [-0.33, 0.34], [-0.13, 0.19]),
            ([-0.13, 0.04], [-0.33, 0.19], [-0.13, 0.34]),
        ]
        self.obj_init_quat_dict = {
            "blue_plastic_bottle": euler2quat(np.pi / 2, 0, np.pi / 2),
            "baked_opened_pepsi_can_v2": euler2quat(np.pi / 2, 0, 0),
            "orange": euler2quat(0, 0, np.pi / 2),
            "baked_opened_7up_can_v2": euler2quat(np.pi / 2, 0, 0),
            "baked_apple_v2": [1.0, 0.0, 0.0, 0.0],
            "baked_sponge_v2": euler2quat(0, 0, np.pi / 2),
            "baked_opened_coke_can_v2": euler2quat(np.pi / 2, 0, 0),
            "baked_opened_redbull_can_v2": euler2quat(np.pi / 2, 0, 0),
        }
        self.special_density_dict = {"baked_apple_v2": 200, "orange": 200}

    def _load_model(self):
        super()._load_model()
        for obj in self.episode_objs:
            for visual in obj.get_visual_bodies():
                for rs in visual.get_render_shapes():
                    mtl = rs.material
                    mtl.set_roughness(1.0)
                    mtl.set_metallic(0.0)
                    mtl.set_specular(0.0)
                    rs.set_material(mtl)


@register_env("MoveNearAltGoogleCameraInScene-v0", max_episode_steps=80)
class MoveNearAltGoogleCameraInSceneEnv(MoveNearGoogleInSceneEnv):
    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        if "robot_init_options" not in options:
            options["robot_init_options"] = {}
        options["robot_init_options"]["qpos"] = np.array(
            [
                -0.2639457174606611,
                0.0831913360274175,
                0.5017611504652179,
                1.156859026208673,
                0.028583671314766423,
                1.592598203487462,
                -1.080652960128774,
                0,
                0,
                -0.00285961,
                0.9351361,
            ]
        )

        return super().reset(seed=seed, options=options)


@register_env("MoveNearAltGoogleCamera2InScene-v0", max_episode_steps=80)
class MoveNearAltGoogleCamera2InSceneEnv(MoveNearGoogleInSceneEnv):
    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        if "robot_init_options" not in options:
            options["robot_init_options"] = {}
        options["robot_init_options"]["qpos"] = np.array(
            [
                -0.2639457174606611,
                0.0831913360274175,
                0.5017611504652179,
                1.156859026208673,
                0.028583671314766423,
                1.592598203487462,
                -1.080652960128774,
                0,
                0,
                -0.00285961,
                0.6651361,
            ]
        )

        return super().reset(seed=seed, options=options)

@register_env("NewScene", max_episode_steps=80)
class MoveNearWithCustomOverlayEnv(MoveNearGoogleInSceneEnv):
    def __init__(self, **kwargs):
        # force use of the “prepackaged” lookup (so _setup_prepackaged_env_init_config is applied)
        kwargs.setdefault("prepackaged_config", True)
        super().__init__(**kwargs)

    def _setup_prepackaged_env_init_config(self):
        # grab the default config dict…
        cfg = super()._setup_prepackaged_env_init_config()
        # then swap in your new overlay image
        cfg["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting" / "table_env1.png"
        )
        # (optionally you can adjust cfg["rgb_overlay_cameras"] as well)
        print("Loading Overlay Image from:", cfg["rgb_overlay_path"])
        return cfg