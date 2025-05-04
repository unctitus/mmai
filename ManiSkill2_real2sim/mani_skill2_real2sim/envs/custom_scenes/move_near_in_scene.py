from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np
import sapien.core as sapien
from transforms3d.euler import euler2quat, quat2euler
from transforms3d.quaternions import quat2mat

from mani_skill2_real2sim import ASSET_DIR
from mani_skill2_real2sim.utils.common import random_choice
from mani_skill2_real2sim.utils.registration import register_env
from mani_skill2_real2sim.utils.sapien_utils import vectorize_pose


from .base_env import CustomSceneEnv, CustomOtherObjectsInSceneEnv

class MoveNearInSceneEnv(CustomSceneEnv):
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
        # Stores a reference to the actual actor objects
        # _load_model
        self.episode_objs = [None] * 3
        
        # Reference to the source and target actor object
        # _initialize_actors via obj_init_options["source_obj_id"]
        # "source_obj_id" is a pointer into self.episode_objs
        self.episode_source_obj = None
        self.episode_target_obj = None

        # Name of the objects like "blue_plastic_bottle"
        # _set_model
        self.episode_model_ids = [None] * 3

        # Determines how much to scale the object
        # _set_model
        self.episode_model_scales = [None] * 3
        
        # box size vectors [w, d, h] (full_extent) in the scaled local frame for all objects
        # _set_model 
        self.episode_model_bbox_sizes = [None] * 3

        # bbox size vector [w, d, h] (full_extent) in the world frame for source and target object 
        # _initialize_actors
        self.episode_source_obj_bbox_world = None
        self.episode_target_obj_bbox_world = None

        # Initial xyz position of the objects
        # _initialize_actors
        self.episode_model_init_xyzs = [None] * 3
        
        # Position of objects in the world frame after they have fallen on the table
        # _initialize_actors
        #self.episode_obj_heights_after_settle = [None] * 3 #unused
        self.episode_obj_xyzs_after_settle = [None] * 3
        self.episode_source_obj_xyz_after_settle = None
        self.episode_target_obj_xyz_after_settle = None
        
        # _initialize_episode_stats
        self.episode_stats = None

        # reset
        self.obj_init_options = {}

        self.original_lighting = original_lighting
        self.slightly_darker_lighting = slightly_darker_lighting
        self.slightly_brighter_lighting = slightly_brighter_lighting
        self.ambient_only_lighting = ambient_only_lighting

        self.prepackaged_config = prepackaged_config
        if self.prepackaged_config:
            # use prepackaged evaluation configs (visual matching)
            kwargs.update(self._setup_prepackaged_env_init_config())

        super().__init__(**kwargs)
    
    #No Change Required, just injets a bunch of kwargs
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "google_pick_coke_can_1_v4"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(ASSET_DIR / "real_inpainting/google_move_near_real_eval_1.png")
        #ret["rgb_overlay_path"] = str(ASSET_DIR / "real_inpainting/table_env1.png")

        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        print("Loading Overlay Image from:", ret["rgb_overlay_path"])
        return ret

    # No Change Required
    def _get_default_scene_config(self):
        scene_config = super()._get_default_scene_config()
        scene_config.contact_offset = (
            0.005
        )  # important to avoid "false-positive" collisions with other objects
        return scene_config

    # No Change Required
    def _setup_lighting(self):
        if self.bg_name is not None:
            return

        shadow = self.enable_shadow
        if self.original_lighting:
            self._scene.set_ambient_light([0.3, 0.3, 0.3])
            self._scene.add_directional_light(
                [1, 1, -1], [1, 1, 1], shadow=shadow, scale=5, shadow_map_size=2048
            )
            self._scene.add_directional_light([0, 0, -1], [1, 1, 1])
        elif self.slightly_darker_lighting:
            self._scene.set_ambient_light([0.3, 0.3, 0.3])
            self._scene.add_directional_light(
                [1, 1, -1],
                [0.8, 0.8, 0.8],
                shadow=shadow,
                scale=5,
                shadow_map_size=2048,
            )
            self._scene.add_directional_light([0, 0, -1], [0.8, 0.8, 0.8])
        elif self.slightly_brighter_lighting:
            self._scene.set_ambient_light([0.3, 0.3, 0.3])
            self._scene.add_directional_light(
                [0, 0, -1],
                [3.6, 3.6, 3.6],
                shadow=shadow,
                scale=5,
                shadow_map_size=2048,
            )
            self._scene.add_directional_light([-1, -0.5, -1], [1.3, 1.3, 1.3])
            self._scene.add_directional_light([1, 1, -1], [1.3, 1.3, 1.3])
        elif self.ambient_only_lighting:
            self._scene.set_ambient_light([1.0, 1.0, 1.0])
        else:
            # Default lighting
            self._scene.set_ambient_light([0.3, 0.3, 0.3])
            self._scene.add_directional_light(
                [0, 0, -1],
                [2.2, 2.2, 2.2],
                shadow=shadow,
                scale=5,
                shadow_map_size=2048,
            )
            self._scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
            self._scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])

    # No Change Required, Loads Arena and Models via _load_model method
    def _load_actors(self):
        self._load_arena_helper()
        self._load_model()
        for obj in self.episode_objs:
            obj.set_damping(0.1, 0.1)

    ##### Needs to be implemented by child class
    def _load_model(self):
        """Load the target object."""
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.obj_init_options = options.get("obj_init_options", {})

        # Sets the seed for the number generator of the current episode 
        self.set_episode_rng(seed)

        #Extracts the model ids and scales from the options and call _set_model method
        model_scales = options.get("model_scales", None)
        model_ids = options.get("model_ids", None)
        reconfigure = options.get("reconfigure", False)
        # This loads the 
        _reconfigure = self._set_model(model_ids, model_scales)
        reconfigure = _reconfigure or reconfigure

        # Apply any extra prepackaged resets (e.g. URDF randomization)
        if self.prepackaged_config:
            _reconfigure = self._additional_prepackaged_config_reset(options)
            reconfigure = reconfigure or _reconfigure
        options["reconfigure"] = reconfigure

        # Reset the epsidode metrics
        self._initialize_episode_stats()

        # Call the reset method of the parent class
        obs, info = super().reset(seed=self._episode_seed, options=options)
        
        # Update the episode info 
        info.update(
            {
                "episode_model_ids": self.episode_model_ids,
                "episode_model_scales": self.episode_model_scales,
                "episode_source_obj_name": self.episode_source_obj.name,
                "episode_target_obj_name": self.episode_target_obj.name,
                "episode_source_obj_init_pose_wrt_robot_base": self.agent.robot.pose.inv()
                * self.episode_source_obj.pose,
                "episode_target_obj_init_pose_wrt_robot_base": self.agent.robot.pose.inv()
                * self.episode_target_obj.pose,
            }
        )
        return obs, info

    def _additional_prepackaged_config_reset(self, options):
        # use prepackaged robot evaluation configs under visual matching setup
        options["robot_init_options"] = {
            "init_xy": [0.35, 0.21],
            "init_rot_quat": (
                sapien.Pose(q=euler2quat(0, 0, -0.09)) * sapien.Pose(q=[0, 0, 0, 1])
            ).q,
        }
        new_urdf_version = self._episode_rng.choice(
            [
                "",
                "recolor_tabletop_visual_matching_1",
                "recolor_tabletop_visual_matching_2",
                "recolor_cabinet_visual_matching_1",
            ]
        )
        if new_urdf_version != self.urdf_version:
            self.urdf_version = new_urdf_version
            self._configure_agent()
            return True
        return False

    # No Change Required Resets the epsisode stats
    def _initialize_episode_stats(self):
        self.episode_stats = OrderedDict(
            all_obj_keep_height=False,
            moved_correct_obj=False,
            moved_wrong_obj=False,
            near_tgt_obj=False,
            is_closest_to_tgt=False,
        )

    # No Change Required
    @staticmethod
    def _list_equal(l1, l2):
        if len(l1) != len(l2):
            return False
        for i in range(len(l1)):
            if l1[i] != l2[i]:
                return False
        return True

    ##### Change Required
    # Methods set self.episode_model_ids, self.episode_model_scales, and self.episode_model_bbox_sizes
    def _set_model(self, model_ids, model_scales):
        """Set the model id and scale. If not provided, choose a triplet randomly from self.model_ids."""
        reconfigure = False

        # model ids
        if model_ids is None:
            model_ids = []
            for _ in range(3):
                model_ids.append(random_choice(self.model_ids, self._episode_rng))
        if not self._list_equal(model_ids, self.episode_model_ids):
            self.episode_model_ids = model_ids
            reconfigure = True

        # model scales
        if model_scales is None:
            model_scales = []
            for model_id in self.episode_model_ids:
                this_available_model_scales = self.model_db[model_id].get(
                    "scales", None
                )
                if this_available_model_scales is None:
                    model_scales.append(1.0)
                else:
                    model_scales.append(
                        random_choice(this_available_model_scales, self._episode_rng)
                    )
        if not self._list_equal(model_scales, self.episode_model_scales):
            self.episode_model_scales = model_scales
            reconfigure = True

        # model bbox sizes
        model_bbox_sizes = []
        for model_id, model_scale in zip(
            self.episode_model_ids, self.episode_model_scales
        ):
            model_info = self.model_db[model_id]
            if "bbox" in model_info:
                bbox = model_info["bbox"]
                bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
                model_bbox_sizes.append(bbox_size * model_scale)
            else:
                raise ValueError(f"Model {model_id} does not have bbox info.")
        self.episode_model_bbox_sizes = model_bbox_sizes

        return reconfigure

    # No Change Required
    def _initialize_actors(self):
        source_obj_id: int = self.obj_init_options.get("source_obj_id", None)
        target_obj_id: int = self.obj_init_options.get("target_obj_id", None)
        assert source_obj_id is not None and target_obj_id is not None
        self.episode_source_obj = self.episode_objs[source_obj_id]
        self.episode_target_obj = self.episode_objs[target_obj_id]

        # Initalized with the scaled bbox vector from the local frame 
        self.episode_source_obj_bbox_world = self.episode_model_bbox_sizes[source_obj_id]  
        self.episode_target_obj_bbox_world = self.episode_model_bbox_sizes[target_obj_id]

        # Get initial x and y positions of the objects
        obj_init_xys = self.obj_init_options.get("init_xys", None)
        assert obj_init_xys is not None
        obj_init_xys = np.array(obj_init_xys)  # [n_objects, 2]
        assert obj_init_xys.shape == (len(self.episode_objs), 2)

        # Place all objects half a meter over the table    
        obj_init_z = self.obj_init_options.get("init_z", self.scene_table_height)
        obj_init_z = obj_init_z + 0.5 # let object fall onto the table

        # Load the quaternions of describing the rotation of the objects
        obj_init_rot_quats = self.obj_init_options.get("init_rot_quats", None)
        if obj_init_rot_quats is not None:
            obj_init_rot_quats = np.array(obj_init_rot_quats)
            assert obj_init_rot_quats.shape == (len(self.episode_objs), 4)
        else:
            obj_init_rot_quats = np.zeros((len(self.episode_objs), 4))
            obj_init_rot_quats[:, 0] = 1.0

        for i, obj in enumerate(self.episode_objs):
            p = np.hstack([obj_init_xys[i], obj_init_z])
            q = obj_init_rot_quats[i]
            obj.set_pose(sapien.Pose(p, q))
            # Lock rotation around x and y
            obj.lock_motion(0, 0, 0, 1, 1, 0)

        # Move the robot far away to avoid collision
        # The robot should be initialized later in _initialize_agent (in base_env.py)
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        self._settle(0.5)
        
        # Unlock motion
        for obj in self.episode_objs:
            obj.lock_motion(0, 0, 0, 0, 0, 0)
            # NOTE(jigu): Explicit set pose to ensure the actor does not sleep
            obj.set_pose(obj.pose)
            obj.set_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        self._settle(0.5)

        # Some objects need longer time to settle
        lin_vel, ang_vel = 0.0, 0.0
        for obj in self.episode_objs:
            lin_vel += np.linalg.norm(obj.velocity)
            ang_vel += np.linalg.norm(obj.angular_velocity)
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(1.5)

        # Get the position of the objects after they have settled
        self.episode_obj_xyzs_after_settle = []
        for obj in self.episode_objs:
            self.episode_obj_xyzs_after_settle.append(obj.pose.p)
        self.episode_source_obj_xyz_after_settle = self.episode_obj_xyzs_after_settle[source_obj_id]
        self.episode_target_obj_xyz_after_settle = self.episode_obj_xyzs_after_settle[target_obj_id]
        
        # gives size vector [w, d, h] in world frame
        self.episode_source_obj_bbox_world = (
            quat2mat(self.episode_source_obj.pose.q)
            @ self.episode_source_obj_bbox_world
        )
        self.episode_target_obj_bbox_world = (
            quat2mat(self.episode_target_obj.pose.q)
            @ self.episode_target_obj_bbox_world
        )

    # No Change Required
    @property
    def source_obj_pose(self):
        """Get the center of mass (COM) pose."""
        return self.episode_source_obj.pose.transform(
            self.episode_source_obj.cmass_local_pose
        )

    # No Change Required
    @property
    def target_obj_pose(self):
        """Get the center of mass (COM) pose."""
        return self.episode_target_obj.pose.transform(
            self.episode_target_obj.cmass_local_pose
        )

    # No Change Required this method returns a dict with the tcp, src_obj, and tgt_obj poses
    # and the tcp to source object pose as a numpy array [x, y, z, qw, qx, qy, qz]
    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(tcp_pose=vectorize_pose(self.tcp.pose))
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                source_obj_pose=vectorize_pose(self.source_obj_pose),
                target_obj_pose=vectorize_pose(self.target_obj_pose),
                tcp_to_source_obj_pos=self.source_obj_pose.p - self.tcp.pose.p,
            )
        return obs

    ##### Change Required
    def evaluate(self, **kwargs):
        source_obj_pose = self.source_obj_pose
        target_obj_pose = self.target_obj_pose

        # Check if objects are knocked down or knocked off table
        other_obj_ids = [
            i
            for (i, obj) in enumerate(self.episode_objs)
            if (obj.name != self.episode_source_obj.name)
            and (obj.name != self.episode_target_obj.name)
        ]
        other_obj_heights = [self.episode_objs[i].pose.p[2] for i in other_obj_ids]
        other_obj_heights_after_settle = [
            self.episode_obj_xyzs_after_settle[i][2] for i in other_obj_ids
        ]
        other_obj_diff_heights = [
            x - y for (x, y) in zip(other_obj_heights, other_obj_heights_after_settle)
        ]
        other_obj_keep_height = all(
            [x > -0.02 for x in other_obj_diff_heights]
        )  # require other objects to not be knocked down on the table
        source_obj_diff_height = (
            source_obj_pose.p[2] - self.episode_source_obj_xyz_after_settle[2]
        )  # source object should not be knocked off the table
        target_obj_diff_height = (
            target_obj_pose.p[2] - self.episode_target_obj_xyz_after_settle[2]
        )  # target object should not be knocked off the table
        all_obj_keep_height = (
            other_obj_keep_height
            and (source_obj_diff_height > -0.15)
            and (target_obj_diff_height > -0.15)
        )

        # Check if moving the correct source object
        source_obj_xy_move_dist = np.linalg.norm(
            self.episode_source_obj_xyz_after_settle[:2]
            - self.episode_source_obj.pose.p[:2]
        )
        other_obj_xy_move_dist = []
        for obj, obj_xyz_after_settle in zip(
            self.episode_objs, self.episode_obj_xyzs_after_settle
        ):
            if obj.name == self.episode_source_obj.name:
                continue
            other_obj_xy_move_dist.append(
                np.linalg.norm(obj_xyz_after_settle[:2] - obj.pose.p[:2])
            )
        moved_correct_obj = (source_obj_xy_move_dist > 0.03) and (
            all([x < source_obj_xy_move_dist for x in other_obj_xy_move_dist])
        )
        moved_wrong_obj = any([x > 0.03 for x in other_obj_xy_move_dist]) and any(
            [x > source_obj_xy_move_dist for x in other_obj_xy_move_dist]
        )

        # Check if the source object is near the target object
        dist_to_tgt_obj = np.linalg.norm(source_obj_pose.p[:2] - target_obj_pose.p[:2])
        tgt_obj_bbox_xy_dist = (
            np.linalg.norm(self.episode_target_obj_bbox_world[:2]) / 2
        )  # get half-length of bbox xy diagonol distance in the world frame at timestep=0
        src_obj_bbox_xy_dist = (
            np.linalg.norm(self.episode_source_obj_bbox_world[:2]) / 2
        )
        # print(dist_to_tgt_obj, tgt_obj_bbox_xy_dist, src_obj_bbox_xy_dist)
        near_tgt_obj = (
            dist_to_tgt_obj < tgt_obj_bbox_xy_dist + src_obj_bbox_xy_dist + 0.10
        )

        # Check if the source object is closest to the target object
        dist_to_other_objs = []
        for obj in self.episode_objs:
            if obj.name == self.episode_source_obj.name:
                continue
            dist_to_other_objs.append(
                np.linalg.norm(source_obj_pose.p[:2] - obj.pose.p[:2])
            )
        is_closest_to_tgt = all(
            [dist_to_tgt_obj < x + 0.01 for x in dist_to_other_objs]
        )

        success = (
            all_obj_keep_height
            and moved_correct_obj
            and near_tgt_obj
            and is_closest_to_tgt
        )

        ret_info = dict(
            all_obj_keep_height=all_obj_keep_height,
            moved_correct_obj=moved_correct_obj,
            moved_wrong_obj=moved_wrong_obj,
            near_tgt_obj=near_tgt_obj,
            is_closest_to_tgt=is_closest_to_tgt,
            success=success,
        )
        for k in self.episode_stats:
            self.episode_stats[k] = ret_info[
                k
            ]  # for this environment, episode stats equal to the current step stats
        ret_info["episode_stats"] = self.episode_stats

        return ret_info

    # No Change Required 
    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0
        if info["success"]:
            reward = 1.0
        return reward

    # No Change Required
    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 1.0

    ##### Change Required
    def get_language_instruction(self, **kwargs):
        src_name = self._get_instruction_obj_name(self.episode_source_obj.name)
        tgt_name = self._get_instruction_obj_name(self.episode_target_obj.name)
        return f"move {src_name} near {tgt_name}"



@register_env("MoveNearGoogleInScene-v0", max_episode_steps=80)
class MoveNearGoogleInSceneEnv(MoveNearInSceneEnv, CustomOtherObjectsInSceneEnv):
    def __init__(self, no_distractor=False, **kwargs):
        self.no_distractor = no_distractor
        self._setup_obj_configs()
        super().__init__(**kwargs)

    def _setup_obj_configs(self):
        # Note: the cans are "opened" here to match the real evaluation; we'll remove "open" when getting language instruction
        self.triplets = [
            ("blue_plastic_bottle", "opened_pepsi_can", "orange"),
            ("opened_7up_can", "apple", "sponge"),
            ("opened_coke_can", "opened_redbull_can", "apple"),
            ("sponge", "blue_plastic_bottle", "opened_7up_can"),
            ("orange", "opened_pepsi_can", "opened_redbull_can"),
        ]
        self._source_obj_ids, self._target_obj_ids = [], []
        for i in range(3):
            for j in range(3):
                if i != j:
                    self._source_obj_ids.append(i)
                    self._target_obj_ids.append(j)
        
        # z will be set to table 
        self._xy_config_per_triplet = [
            ([-0.33, 0.04], [-0.33, 0.34], [-0.13, 0.19]),
            ([-0.13, 0.04], [-0.33, 0.19], [-0.13, 0.34]),
        ]
        self.obj_init_quat_dict = {
            "blue_plastic_bottle": euler2quat(np.pi / 2, 0, np.pi / 2),
            "opened_pepsi_can": euler2quat(np.pi / 2, 0, 0),
            "orange": euler2quat(0, 0, np.pi / 2),
            "opened_7up_can": euler2quat(np.pi / 2, 0, 0),
            "apple": [1.0, 0.0, 0.0, 0.0],
            "sponge": euler2quat(0, 0, np.pi / 2),
            "opened_coke_can": euler2quat(np.pi / 2, 0, 0),
            "opened_redbull_can": euler2quat(np.pi / 2, 0, 0),
        }
        self.special_density_dict = {
            "apple": 200,  # toy apple as in real eval
            "orange": 200
            # by default, opened cans have density 50; blue plastic bottle has density 50; sponge has density 150
        }

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.set_episode_rng(seed)

        obj_init_options = options.get("obj_init_options", {})
        obj_init_options = obj_init_options.copy()

        _num_episodes = (
            len(self.triplets)
            * len(self._source_obj_ids)
            * len(self._xy_config_per_triplet)
        )
        episode_id = obj_init_options.get(
            "episode_id", self._episode_rng.randint(_num_episodes)
        )
        triplet = self.triplets[
            episode_id // (len(self._source_obj_ids) * len(self._xy_config_per_triplet))
        ]
        source_obj_id = self._source_obj_ids[episode_id % len(self._source_obj_ids)]
        target_obj_id = self._target_obj_ids[episode_id % len(self._target_obj_ids)]
        xy_config_triplet = self._xy_config_per_triplet[
            (
                episode_id
                % (len(self._source_obj_ids) * len(self._xy_config_per_triplet))
            )
            // len(self._source_obj_ids)
        ]
        quat_config_triplet = [
            self.obj_init_quat_dict[model_id] for model_id in triplet
        ]
        if self.no_distractor:
            triplet = [triplet[source_obj_id], triplet[target_obj_id]]
            xy_config_triplet = [
                xy_config_triplet[source_obj_id],
                xy_config_triplet[target_obj_id],
            ]
            quat_config_triplet = [
                quat_config_triplet[source_obj_id],
                quat_config_triplet[target_obj_id],
            ]
            source_obj_id = 0
            target_obj_id = 1

        options["model_ids"] = triplet
        obj_init_options["source_obj_id"] = source_obj_id
        obj_init_options["target_obj_id"] = target_obj_id
        obj_init_options["init_xys"] = xy_config_triplet
        obj_init_options["init_rot_quats"] = quat_config_triplet
        options["obj_init_options"] = obj_init_options

        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update({"episode_id": episode_id})
        return obs, info

    def _load_model(self):
        self.episode_objs = []
        for (model_id, model_scale) in zip(
            self.episode_model_ids, self.episode_model_scales
        ):
            if model_id in self.special_density_dict:
                density = self.special_density_dict[model_id]
            else:
                density = self.model_db[model_id].get("density", 1000)

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
            obj.name = model_id
            self.episode_objs.append(obj)


@register_env("PutOnNumberGoogleInScene-v0", max_episode_steps=80)
class PutOnNumberGoogleInSceneEnv(MoveNearInSceneEnv, CustomOtherObjectsInSceneEnv):
    DEFAULT_MODEL_JSON = "info_pick_custom_baked_tex_v0.json"

    def __init__(self, **kwargs):
        self._setup_obj_configs()
        super().__init__(**kwargs)

    
    def _setup_obj_configs(self):
        # Note: the cans are "opened" here to match the real evaluation; we'll remove "open" when getting language instruction
        self.source_objects_ids = [
            "blue_plastic_bottle", "baked_opened_pepsi_can", "orange",
            "baked_opened_7up_can", "baked_apple", "baked_sponge",
            "baked_opened_coke_can", "baked_opened_redbull_can", "baked_apple",
        ]

        self.source_object_xys = [[-0.33, 0.34],]

        thickness = 0.001
        self.target_objects_info = {
            "number6": {
                "xy":   [-0.065,  0.127],
                "quat":  euler2quat(0, 0, 0.6204),
                "size": [0.21,    0.261,   thickness],
            },
            "number3": {
                "xy":   [-0.483, -0.080],
                "quat":  euler2quat(0, 0, 0.5584),
                "size": [0.323,   0.324,   thickness],
            },
            "number2": {
                "xy":   [-0.679,  0.378],
                "quat":  euler2quat(0, 0, 0.4054),
                "size": [0.307,   0.341,   thickness],
            },
        }
        self.target_objects_ids = list(self.target_objects_info.keys())
        self.target_objects_xys = [v["xy"]   for v in self.target_objects_info.values()]
        self.target_objects_quats = [v["quat"]  for v in self.target_objects_info.values()]
        
        self.target_objects_size = {
            name: info["size"]
            for name, info in self.target_objects_info.items()
        }

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

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()
        obj_init_options = options.get("obj_init_options", {})
        obj_init_options = obj_init_options.copy()
        
        self.set_episode_rng(seed)
        
        target_num  = len(self.target_objects_ids)
        source_num = len(self.source_objects_ids)
        source_xys_num = len(self.source_object_xys)
        num_episodes = target_num * source_num * source_xys_num
        episode_id = obj_init_options.get("episode_id", self._episode_rng.randint(num_episodes))
        target_obj_id, remainder = divmod(episode_id, (source_num * source_xys_num))
        source_obj_id, source_obj_xys_id = divmod(remainder, source_xys_num)
        source_object = self.source_objects_ids[source_obj_id]

        # First 3 objects are the surfaces and the last one must be the source object
        options["model_ids"] = self.target_objects_ids + [source_object]
        # Assumes that 4 objects are loaded and last one is source object
        obj_init_options["source_obj_id"] = 3
        obj_init_options["target_obj_id"] = target_obj_id
        obj_init_options["init_xys"] = self.target_objects_xys + [self.source_object_xys[source_obj_xys_id]]
        obj_init_options["init_rot_quats"] = self.target_objects_quats + [self.obj_init_quat_dict[source_object]]
        options["obj_init_options"] = obj_init_options

        #otherwise sheets might not be aligned
        options["robot_init_options"] = {
            "init_xy": [0.35, 0.21],
            "init_rot_quat": (
                sapien.Pose(q=euler2quat(0, 0, -0.09)) * sapien.Pose(q=[0, 0, 0, 1])
            ).q,
        }

        obs, info = super().reset(seed=self._episode_seed, options=options)
        info.update({"episode_id": episode_id})
        return obs, info

    # Methods set self.episode_model_ids, self.episode_model_scales, and self.episode_model_bbox_sizes
    def _set_model(self, model_ids, model_scales):
       
        reconfigure = False

        # set new episode_model_ids and a set reconfigure to true if changed
        if not self._list_equal(model_ids, self.episode_model_ids):
            self.episode_model_ids = model_ids
            reconfigure = True

        # set all model scales to 1 (currently the model json files only contain scale 1)
        self.episode_model_scales = [1.0] * len(self.episode_model_ids)

        # model bbox sizes
        model_bbox_sizes = []
        for model_id, model_scale in zip(self.episode_model_ids, self.episode_model_scales):
            if model_id in self.target_objects_ids:
                 size = self.target_objects_size[model_id]
                 model_bbox_sizes.append(np.array(size, dtype=np.float32))
            else:
                model_info = self.model_db[model_id]
                if "bbox" in model_info:
                    bbox = model_info["bbox"]
                    bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
                    model_bbox_sizes.append(bbox_size * model_scale)
                else:
                    raise ValueError(f"Model {model_id} does not have bbox info.")
        self.episode_model_bbox_sizes = model_bbox_sizes

        return reconfigure

    # Builds all actors and stores a reference to them in self.episode_objs
    # Expects that (self.episode_model_ids, self.episode_model_scales) are set
    def _load_model(self):
        self.episode_objs = []
        
        # model_id are object names 
        for (model_id, model_scale) in zip(self.episode_model_ids, self.episode_model_scales):
            
            if model_id in self.target_objects_ids:
               
               # Load the target object (boxes)
               obj = self._build_box_actor(
                        scene=self._scene,
                        size=self.target_objects_size[model_id],   # full box extents in meters
                        #xyz=self.target_objects_xys[model_id] + [self.scene_table_height], # x, y, z 
                        #quat=self.target_objects_quats[model_id],   # qw, qx, qy, qz 
                        rgba=(0., 1., 0., 1.),          # base_color RGBA (green)
                        material_props=(0.0, 0.3, 0.8), # (metallic, roughness, specular)
                        density=1200.0,                  # in kg/m³
                        physical_material=self._scene.create_physical_material(
                            static_friction=self.obj_static_friction,
                            dynamic_friction=self.obj_dynamic_friction,
                            restitution=0.0,
                        ),
                    )
            else:
                
                # Set the object density
                if model_id in self.special_density_dict:
                    density = self.special_density_dict[model_id]
                else:
                    density = self.model_db[model_id].get("density", 1000)
               
                # Load the source object
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

            obj.name = model_id
            self.episode_objs.append(obj)





# This method is used
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

@register_env("MoveNearGoogleBakedTexInSceneNew-v0", max_episode_steps=80)
class MoveNearGoogleBakedTexInSceneEnvNew(MoveNearGoogleBakedTexInSceneEnv):
    def __init__(self, **kwargs):
        # force use of the “prepackaged” lookup (so _setup_prepackaged_env_init_config is applied)
        #kwargs.setdefault("prepackaged_config", True)
        super().__init__(**kwargs)

    def _setup_prepackaged_env_init_config(self):
        # grab the default config dict…
        cfg = super()._setup_prepackaged_env_init_config()
        # then swap in your new overlay image
        cfg["rgb_overlay_path"] = str(ASSET_DIR / "real_inpainting" / "table_env1.png")
        cfg["scene_name"] = "dummy_tabletop1"
        print("Loading Overlay Image from:", cfg["rgb_overlay_path"])
        return cfg


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


@register_env("MoveNearWithCustomOverlayEnv", max_episode_steps=80)
class MoveNearWithCustomOverlayEnv(MoveNearGoogleInSceneEnv):
    def __init__(self, **kwargs):    
        # storage for surface actors and their params
        self.surface_actors: dict[str, sapien.pysapien.ActorStatic] = {}
        self.surface_params: dict[str, dict] = {}
        # store default base pose and size
        q = euler2quat(0, 0, np.deg2rad(28))
        self.default_base_pose = sapien.Pose(p=[-0.42, 0, 0.017 + 0.865 / 2], q=q)
        self.default_base_half_size = np.array([0.95, 3.2, 0.865]) / 2
        super().__init__(**kwargs)

    def get_surface_actor(self, name: str) -> sapien.pysapien.ActorStatic:
        """
        Retrieve a surface actor by its registered name.
        """
        actor = self.surface_actors.get(name)
        if actor is None:
            raise KeyError(f"No surface found with name '{name}'")
        return actor
    
    def get_all_surface_actors(self) -> list[sapien.pysapien.Actor]:
        """
        Retrieve all surface actors.
        """
        return list(self.surface_actors.values())
    
    def add_surface(
        self,
        base_pose: sapien.Pose = None,
        base_half_size: np.ndarray = None,
        *,
        size_xy: tuple[float, float] = (0.2, 0.2),
        thickness: float = 1e-3,
        offset_xy: tuple[float, float] = (-0.35, 0.15),
        color: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 1.0),
        metallic: float = 0.0,
        roughness: float = 0.3,
        specular: float = 0.8,
        name: str = "surface_sheet",
    ) -> sapien.pysapien.ActorStatic:
        """
        Create a thin box “sheet” on top of another box, with both visual and collision geometry.
        Supports non-uniform size.
        """
        # check if name already exists
        if name in self.surface_actors:
            return self.surface_actors[name]
        # defaults
        if base_pose is None:
            base_pose = self.default_base_pose
        if base_half_size is None:
            base_half_size = self.default_base_half_size

        # store params for future resizing
        params = dict(
            base_pose=base_pose,
            base_half_size=base_half_size,
            size_xy=size_xy,
            thickness=thickness,
            offset_xy=offset_xy,
            color=color,
            metallic=metallic,
            roughness=roughness,
            specular=specular,
            name=name,
        )
        self.surface_params[name] = params

        # build actor
        actor = self._build_surface_actor(params)
        return actor

    def _build_surface_actor(self, params: dict) -> sapien.pysapien.ActorStatic:
        # unpack params
        base_pose = params['base_pose']
        base_half_size = params['base_half_size']
        size_xy = params['size_xy']
        thickness = params['thickness']
        offset_xy = params['offset_xy']
        color = params['color']
        metallic = params['metallic']
        roughness = params['roughness']
        specular = params['specular']
        name = params['name']

        scene = self._scene
        renderer = self._renderer
        builder = scene.create_actor_builder()

        # compute half-size and pose
        half = np.array([size_xy[0], size_xy[1], thickness], dtype=np.float32) / 2
        p = base_pose.p.copy()
        p[0] += offset_xy[0]
        p[1] += offset_xy[1]
        p[2] += base_half_size[2] + thickness / 2
        sheet_pose = sapien.Pose(p=p, q=base_pose.q)

        # material
        m = renderer.create_material()
        m.base_color = np.array(color, dtype=np.float32)
        m.metallic = metallic
        m.roughness = roughness
        m.specular = specular

        # add collision and visual
        builder.add_box_collision(pose=sheet_pose, half_size=half)
        builder.add_box_visual(pose=sheet_pose, half_size=half, material=m)
        #actor = builder.build_static(name=name)
        actor = builder.build(name=name)
        self.surface_actors[name] = actor
        return actor
    
    def build_surface_actor_from_pose(
        self, 
        pose: Tuple[float, float, float, float, float, float],
        size: Tuple[float, float, float] = [0.2, 0.2, 1e-3],
        color: Tuple[float, float, float, float] = [0.0, 1.0, 0.0, 1.0],
        material: Tuple[float, float, float] = [0.0, 0.3, 0.8],
        name: str = "surface_sheet"):
      
        scene = self._scene
        renderer = self._renderer
        builder = scene.create_actor_builder()

        half_size = np.array(size, dtype=np.float32) / 2
        sheet_pose = sapien.Pose(p=pose[:3], q=euler2quat(pose[3], pose[4], pose[5]))

        # material
        m = renderer.create_material()
        m.base_color = np.array(color, dtype=np.float32)
        m.metallic, m.roughness, m.specular = material

        # add collision and visual
        builder.add_box_collision(pose=sheet_pose, half_size=half_size)
        builder.add_box_visual(pose=sheet_pose, half_size=half_size, material=m)

        actor = builder.build(name=name)

        # store params for future resizing
        params = dict(
            base_pose=sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, 0)),
            base_half_size= np.array([0, 0, 0]),
            size_xy=size[:2],
            thickness=size[2],
            offset_xy=(0, 0),
            color=color,
            metallic=0.0,
            roughness=0.3,
            specular=0.8,
            name=name,
        )
        
        self.surface_params[name] = params
        self.surface_actors[name] = actor
    

    def set_surface_pose(self, name: str, pose: sapien.Pose) -> None:
        actor = self.surface_actors.get(name)
        if actor is None:
            raise KeyError(f"No surface found with name '{name}'")
        actor.set_pose(pose)

    def set_surface_pose_from_values(
        self,
        name: str,
        x: float,
        y: float,
        z: float,
        roll: float = 0.0,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> None:
        quat = euler2quat(roll, pitch, yaw)
        self.set_surface_pose(name, sapien.Pose(p=[x, y, z], q=quat))

    def set_surface_size(
        self,
        name: str,
        size_xy: tuple[float, float],
        thickness: float = 1e-3
    ) -> None:
        """
        Resize both visual and collision geometry of a previously added surface.
        """
        params = self.surface_params.get(name)
        if params is None:
            raise KeyError(f"No surface found with name '{name}'")
        # update size
        params['size_xy'] = size_xy
        params['thickness'] = thickness
        # release old actor

        old = self.surface_actors.pop(name, None)
        if old is not None:
            self._scene.remove_actor(old)

        # rebuild with new size
        actor = self._build_surface_actor(params)
        self.surface_actors[name] = actor
    
    def get_surface_actor_pose_values(self, name: str) -> tuple[float, float, float, float, float, float]:
        """
        Return (x, y, z, roll, pitch, yaw) of the named surface actor.
        """
        actor = self.surface_actors.get(name)
        if actor is None:
            raise KeyError(f"No surface found with name '{name}'")

        pose = actor.get_pose()
        x, y, z = pose.p
        qx, qy, qz, qw = pose.q  # sapien: [x, y, z, w]

        # transforms3d expects [w, x, y, z]
        roll, pitch, yaw = quat2euler([qw, qx, qy, qz], axes='sxyz')
        return x, y, z, roll, pitch, yaw     
    
    def get_surface_size(self, name: str) -> tuple[float, float]:
        """
        Get the size of a previously added surface.
        """
        params = self.surface_params.get(name)
        if params is None:
            raise KeyError(f"No surface found with name '{name}'")
        return params['size_xy'], params['thickness']

    def _load_model(self):
        self.build_surface_actor_from_pose(pose=[-0.065, 0.127, 0.882, 0.0, 0.0, 0.6204], 
                                            size=[0.21, 0.261, 1e-3], 
                                            name = "number6") 
        self.build_surface_actor_from_pose(pose=[-0.483, -0.08, 0.882, 0.0, 0.0, 0.5584], 
                                            size=[0.323, 0.324, 1e-3], 
                                            name = "number3")  
        self.build_surface_actor_from_pose(pose=[-0.679, 0.378, 0.882,  0.0, 0.0, 0.4054], 
                                            size=[0.307, 0.341, 1e-3], 
                                            name = "number2")                                                
 
        super()._load_model()


@register_env("MoveNearWithCustomOverlayEnv2", max_episode_steps=80)
class MoveNearWithCustomOverlayEnv(MoveNearGoogleInSceneEnv):
    def __init__(self, **kwargs):
        # --- same as before: build your overlay-sheets storage + defaults ---
        self.surface_actors: dict[str, sapien.pysapien.ActorStatic] = {}
        self.surface_params: dict[str, dict] = {}
        q = euler2quat(0, 0, np.deg2rad(28))
        self.default_base_pose = sapien.Pose(p=[-0.42, 0, 0.017 + 0.865 / 2], q=q)
        self.default_base_half_size = np.array([0.95, 3.2, 0.865]) / 2
        
        # for our “put-on” logic
        #self.consecutive_grasp = 0
        #self.episode_stats: OrderedDict = OrderedDict()
        #self.episode_source_obj = None
        #self.source_obj_pose = None
        #self.episode_source_obj_bbox_world = None
        #self.episode_target_surface = None
        #self.target_surface_name = None
        #self.episode_target_surface_bbox_world = None

        super().__init__(**kwargs)

    def _initialize_episode_stats(self):
        # track exactly the same five boolean signals
        self.episode_stats = OrderedDict(
            moved_correct_obj=False,
            moved_wrong_obj=False,
            is_src_obj_grasped=False,
            consecutive_grasp=False,
            src_on_target=False,
        )

    def reset(self, *args, **kwargs):
        # reset our counters & stats, then let the base env sample everything
        self.consecutive_grasp = 0
        self._initialize_episode_stats()
        
        obs, info = super().reset(*args, **kwargs)

        self.episode_source_obj = random_choice(self.episode_objs)
        self._init_source_pose = self.episode_source_obj.pose
        shape = self.episode_source_obj.get_collision_shapes()[0]
        half = shape.geometry.half_size
        self._init_source_bbox = half * 2

        # pick one of the three numbered sheets as the target
        self.target_surface_name = random_choice(["number2", "number3", "number6"])
        self.episode_target_surface = self.surface_actors[self.target_surface_name]
        shape_t = self.episode_target_surface.get_collision_shapes()[0]
        half_t = shape_t.geometry.half_size
        self.episode_target_surface_bbox_world = half_t * 2

        return obs, info
     

        return obs

    def build_surface_actor_from_pose(
        self, 
        pose: Tuple[float, float, float, float, float, float],
        size: Tuple[float, float, float] = [0.2, 0.2, 1e-3],
        color: Tuple[float, float, float, float] = [0.0, 1.0, 0.0, 1.0],
        material: Tuple[float, float, float] = [0.0, 0.3, 0.8],
        name: str = "surface_sheet"):
      
        scene = self._scene
        renderer = self._renderer
        builder = scene.create_actor_builder()

        half_size = np.array(size, dtype=np.float32) / 2
        sheet_pose = sapien.Pose(p=pose[:3], q=euler2quat(pose[3], pose[4], pose[5]))

        # material
        m = renderer.create_material()
        m.base_color = np.array(color, dtype=np.float32)
        m.metallic, m.roughness, m.specular = material

        # add collision and visual
        builder.add_box_collision(pose=sheet_pose, half_size=half_size)
        builder.add_box_visual(pose=sheet_pose, half_size=half_size, material=m)

           # store params for future resizing
        params = dict(
            base_pose=sapien.Pose(p=[0, 0, 0], q=euler2quat(0, 0, 0)),
            base_half_size= np.array([0, 0, 0]),
            size_xy=size[:2],
            thickness=size[2],
            offset_xy=(0, 0),
            color=color,
            metallic=0.0,
            roughness=0.3,
            specular=0.8,
            name=name,
        )
        
        actor = builder.build(name=name)

        self.surface_params[name] = params
        self.surface_actors[name] = actor
    def _load_model(self):
        # 1) build the three “number” surfaces
        self.build_surface_actor_from_pose(
            pose=[-0.065,  0.127, 0.882, 0.0, 0.0, 0.6204],
            size=[0.21,   0.261, 1e-3],
            name="number6",
        )
        self.build_surface_actor_from_pose(
            pose=[-0.483, -0.08,  0.882, 0.0, 0.0, 0.5584],
            size=[0.323,  0.324, 1e-3],
            name="number3",
        )
        self.build_surface_actor_from_pose(
            pose=[-0.679,  0.378, 0.882, 0.0, 0.0, 0.4054],
            size=[0.307,  0.341, 1e-3],
            name="number2",
        )

        # 2) now let the parent load whatever object models it would normally load
        super()._load_model()

    def evaluate(
        self,
        success_require_src_completely_on_target: bool = True,
        z_flag_required_offset: float = 0.02,
        **kwargs
    ):
        # Copy–paste (and adapt) from PutOnInSceneEnv.evaluate, swapping in our
        # surface actor as `target_obj`
        src = self.episode_source_obj
        tgt = self.episode_target_surface
        # --- compute displacements in XY for all scene objects ---
        src_xy = np.linalg.norm(
            self.episode_source_obj.get_pose().p[:2]
            - self.source_obj_pose.p[:2]
        )
        other_dists = []
        for o in self.episode_objs:
            if o is src:
                continue
            other_dists.append(
                np.linalg.norm(
                    o.get_pose().p[:2] - o.get_pose().p[:2]  # or initial vs final if stored
                )
            )
        moved_correct = (src_xy > 0.03) and all(x < src_xy for x in other_dists)
        moved_wrong   = any(x > 0.03 for x in other_dists) and any(x > src_xy for x in other_dists)

        # --- grasp logic (5 consecutive steps) ---
        is_grasped = self.agent.check_grasp(src)
        if is_grasped:
            self.consecutive_grasp += 1
        else:
            self.consecutive_grasp = 0
        cons = (self.consecutive_grasp >= 5)

        # --- geometric “on‐top” test relative to our surface actor ---
        pos_src = src.get_pose().p
        pos_tgt = tgt.get_pose().p
        off = pos_src - pos_tgt

        xy_flag = (
            np.linalg.norm(off[:2])
            <= np.linalg.norm(self.episode_target_surface_bbox_world[:2]) / 2 + 0.003
        )
        z_flag = (off[2] > 0) and (
            off[2]
            - self.episode_target_surface_bbox_world[2] / 2
            - self.episode_source_obj_bbox_world[2] / 2
            <= z_flag_required_offset
        )
        on_top = xy_flag and z_flag

        # --- optional contact check ---
        if success_require_src_completely_on_target:
            contacts = self._scene.get_contacts()
            robot_links = [l.name for l in self.agent.robot.get_links()]
            ignore = [self.target_surface_name] + robot_links
            for c in contacts:
                a0, a1 = c.actor0.name, c.actor1.name
                other = None
                if a0 == src.name:
                    other = a1
                elif a1 == src.name:
                    other = a0
                if other and (other not in ignore):
                    imp = np.sum([pt.impulse for pt in c.points], axis=0)
                    if np.linalg.norm(imp) > 1e-6:
                        on_top = False
                        break

        success = on_top

        # --- update episode_stats ---
        self.episode_stats["moved_correct_obj"]  |= moved_correct
        self.episode_stats["moved_wrong_obj"]    |= moved_wrong
        self.episode_stats["is_src_obj_grasped"] |= is_grasped
        self.episode_stats["consecutive_grasp"]  |= cons
        self.episode_stats["src_on_target"]      |= on_top

        return dict(
            moved_correct_obj=moved_correct,
            moved_wrong_obj=moved_wrong,
            is_src_obj_grasped=is_grasped,
            consecutive_grasp=cons,
            src_on_target=on_top,
            episode_stats=self.episode_stats,
            success=success,
        )

    def get_language_instruction(self, **kwargs):
        # “put <object> on numberX”
        src_name = self._get_instruction_obj_name(self.episode_source_obj.name)
        return f"put {src_name} on {self.target_surface_name}"
    

class MoveNearInSceneEnvNew(CustomSceneEnv):
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
        self.episode_source_obj = None
        self.episode_target_obj = None
        self.episode_distractor_obj = None

        self.episode_source_obj_id = None
        self.episode_target_obj_id = None
        self.episode_distractor_obj_id = None

        
        #self.episode_source_obj_scale = None
        #self.episode_distractor_obj_scale = None
        
        #self.episode_source_obj_bbox_world = None
        #self.episode_target_obj_bbox_world = None
        #self.episode_distractor_obj_bbox_world = None

        # Won't settle target 
        #self.episode_source_obj_xyz_after_settle = None
        #self.episode_distractor_obj_xyz_after_settle = None

        #self.episode_source_obj_xyz_after_settle = None
        #self.episode_distractor_obj_xyz_after_settle = None


        self.consecutive_grasp = 0
        self.episode_stats = None
        self.obj_init_options = {}

       
        # Everywhere the same from here 
        self.original_lighting = original_lighting
        self.slightly_darker_lighting = slightly_darker_lighting
        self.slightly_brighter_lighting = slightly_brighter_lighting
        self.ambient_only_lighting = ambient_only_lighting

        self.prepackaged_config = prepackaged_config
        if self.prepackaged_config:
            # use prepackaged evaluation configs (visual matching)
            kwargs.update(self._setup_prepackaged_env_init_config())

        super().__init__(**kwargs)
    
    #No Change Required, just injets a bunch of kwargs
    def _setup_prepackaged_env_init_config(self):
        ret = {}
        ret["robot"] = "google_robot_static"
        ret["control_freq"] = 3
        ret["sim_freq"] = 513
        ret[
            "control_mode"
        ] = "arm_pd_ee_delta_pose_align_interpolate_by_planner_gripper_pd_joint_target_delta_pos_interpolate_by_planner"
        ret["scene_name"] = "google_pick_coke_can_1_v4"
        ret["camera_cfgs"] = {"add_segmentation": True}
        ret["rgb_overlay_path"] = str(
            ASSET_DIR / "real_inpainting/google_move_near_real_eval_1.png"
        )
        ret["rgb_overlay_cameras"] = ["overhead_camera"]
        print("Loading Overlay Image from:", ret["rgb_overlay_path"])
        return ret

    # No Change Required
    def _get_default_scene_config(self):
        scene_config = super()._get_default_scene_config()
        scene_config.contact_offset = (
            0.005
        )  # important to avoid "false-positive" collisions with other objects
        return scene_config

    # No Change Required
    def _setup_lighting(self):
        if self.bg_name is not None:
            return

        shadow = self.enable_shadow
        if self.original_lighting:
            self._scene.set_ambient_light([0.3, 0.3, 0.3])
            self._scene.add_directional_light(
                [1, 1, -1], [1, 1, 1], shadow=shadow, scale=5, shadow_map_size=2048
            )
            self._scene.add_directional_light([0, 0, -1], [1, 1, 1])
        elif self.slightly_darker_lighting:
            self._scene.set_ambient_light([0.3, 0.3, 0.3])
            self._scene.add_directional_light(
                [1, 1, -1],
                [0.8, 0.8, 0.8],
                shadow=shadow,
                scale=5,
                shadow_map_size=2048,
            )
            self._scene.add_directional_light([0, 0, -1], [0.8, 0.8, 0.8])
        elif self.slightly_brighter_lighting:
            self._scene.set_ambient_light([0.3, 0.3, 0.3])
            self._scene.add_directional_light(
                [0, 0, -1],
                [3.6, 3.6, 3.6],
                shadow=shadow,
                scale=5,
                shadow_map_size=2048,
            )
            self._scene.add_directional_light([-1, -0.5, -1], [1.3, 1.3, 1.3])
            self._scene.add_directional_light([1, 1, -1], [1.3, 1.3, 1.3])
        elif self.ambient_only_lighting:
            self._scene.set_ambient_light([1.0, 1.0, 1.0])
        else:
            # Default lighting
            self._scene.set_ambient_light([0.3, 0.3, 0.3])
            self._scene.add_directional_light(
                [0, 0, -1],
                [2.2, 2.2, 2.2],
                shadow=shadow,
                scale=5,
                shadow_map_size=2048,
            )
            self._scene.add_directional_light([-1, -0.5, -1], [0.7, 0.7, 0.7])
            self._scene.add_directional_light([1, 1, -1], [0.7, 0.7, 0.7])

    # No Change Required, Loads Arena and Models via _load_model method
    def _load_actors(self):
        self._load_arena_helper()
        self._load_model()
        # Target is static
        self.episode_source_obj.set_damping(0.1, 0.1)
        self.episode_distractor_obj.set_damping(0.1, 0.1)

    ##### Needs to be implemented by child class
    # This class should also build the static surfaces 
    def _load_model(self):
        """Load the target object."""
        raise NotImplementedError

    def reset(self, seed=None, options=None):
        if options is None:
            options = dict()
        options = options.copy()

        self.obj_init_options = options.get("obj_init_options", {})

        # Sets the seed for the number generator of the current episode 
        self.set_episode_rng(seed)

        #Extracts the model ids and scales from the options and call _set_model method
        model_scales = options.get("model_scales", None)
        model_ids = options.get("model_ids", None)
        reconfigure = options.get("reconfigure", False)
        # This loads the 
        _reconfigure = self._set_model(model_ids, model_scales)
        reconfigure = _reconfigure or reconfigure

        # Apply any extra prepackaged resets (e.g. URDF randomization)
        if self.prepackaged_config:
            _reconfigure = self._additional_prepackaged_config_reset(options)
            reconfigure = reconfigure or _reconfigure
        options["reconfigure"] = reconfigure

        # Reset the epsidode metrics
        self._initialize_episode_stats()

        # Call the reset method of the parent class
        obs, info = super().reset(seed=self._episode_seed, options=options)
        
        # Update the episode info 
        info.update(
            {
                "episode_model_ids": self.episode_model_ids,
                "episode_model_scales": self.episode_model_scales,
                "episode_source_obj_name": self.episode_source_obj.name,
                "episode_target_obj_name": self.episode_target_obj.name,
                "episode_source_obj_init_pose_wrt_robot_base": self.agent.robot.pose.inv()
                * self.episode_source_obj.pose,
                "episode_target_obj_init_pose_wrt_robot_base": self.agent.robot.pose.inv()
                * self.episode_target_obj.pose,
            }
        )
        return obs, info

    def _additional_prepackaged_config_reset(self, options):
        # use prepackaged robot evaluation configs under visual matching setup
        options["robot_init_options"] = {
            "init_xy": [0.35, 0.21],
            "init_rot_quat": (
                sapien.Pose(q=euler2quat(0, 0, -0.09)) * sapien.Pose(q=[0, 0, 0, 1])
            ).q,
        }
        new_urdf_version = self._episode_rng.choice(
            [
                "",
                "recolor_tabletop_visual_matching_1",
                "recolor_tabletop_visual_matching_2",
                "recolor_cabinet_visual_matching_1",
            ]
        )
        if new_urdf_version != self.urdf_version:
            self.urdf_version = new_urdf_version
            self._configure_agent()
            return True
        return False

    # No Change Required Resets the epsisode stats
    def _initialize_episode_stats(self):
        self.episode_stats = OrderedDict(
            all_obj_keep_height=False,
            moved_correct_obj=False,
            moved_wrong_obj=False,
            near_tgt_obj=False,
            is_closest_to_tgt=False,
        )

    # No Change Required
    @staticmethod
    def _list_equal(l1, l2):
        if len(l1) != len(l2):
            return False
        for i in range(len(l1)):
            if l1[i] != l2[i]:
                return False
        return True

    ##### Change Required
    # Methods set self.episode_model_ids, self.episode_model_scales, and self.episode_model_bbox_sizes
    def _set_model(self, model_ids, model_scales):
        """Set the model id and scale. If not provided, choose a triplet randomly from self.model_ids."""
        reconfigure = False

        # model ids
        if model_ids is None:
            model_ids = [
                random_choice(self.model_ids, self._episode_rng)
                for _ in range(3)
            ]
        if not self._list_equal(model_ids, self.episode_model_ids):
            self.episode_model_ids = model_ids
            reconfigure = True

        # model scales
        if model_scales is None:
            model_scales = []
            for model_id in self.episode_model_ids:
                if "static_surface" in model_id:
                    # Always use scale 1.0 for static_surface models
                    model_scales.append(1.0)
                else:
                    this_available_model_scales = self.model_db[model_id].get("scales", None)
                    if this_available_model_scales is None:
                        model_scales.append(1.0)
                    else:
                        model_scales.append(
                            random_choice(this_available_model_scales, self._episode_rng)
                        )
        if not self._list_equal(model_scales, self.episode_model_scales):
            self.episode_model_scales = model_scales
            reconfigure = True

        # model bbox sizes
        # computes the size vector [w, d, h] in the model local frame and scales it
        model_bbox_sizes = []
        for model_id, model_scale in zip(
            self.episode_model_ids, self.episode_model_scales
        ):
            model_info = self.model_db[model_id]
            if "bbox" in model_info:
                if "static_surface" in model_id:
                    # TODO: compute bbox_size differently for static_surface models
                    bbox_size = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                else:
                    bbox = model_info["bbox"]
                    bbox_size = (
                        np.array(bbox["max"], dtype=np.float32)
                        - np.array(bbox["min"], dtype=np.float32)
                    )
                model_bbox_sizes.append(bbox_size * model_scale)
            else:
                raise ValueError(f"Model {model_id} does not have bbox info.")
        self.episode_model_bbox_sizes = model_bbox_sizes

        return reconfigure

    def _set_model_old(self, model_ids, model_scales):
        """Set the model id and scale. If not provided, choose a triplet randomly from self.model_ids."""
        reconfigure = False

        # model ids
        if model_ids is None:
            model_ids = []
            for _ in range(3):
                model_ids.append(random_choice(self.model_ids, self._episode_rng))
        if not self._list_equal(model_ids, self.episode_model_ids):
            self.episode_model_ids = model_ids
            reconfigure = True

        # model scales
        if model_scales is None:
            model_scales = []
            for model_id in self.episode_model_ids:
                this_available_model_scales = self.model_db[model_id].get(
                    "scales", None
                )
                if this_available_model_scales is None:
                    model_scales.append(1.0)
                else:
                    model_scales.append(
                        random_choice(this_available_model_scales, self._episode_rng)
                    )
        if not self._list_equal(model_scales, self.episode_model_scales):
            self.episode_model_scales = model_scales
            reconfigure = True

        # model bbox sizes
        # computes the size vector [w, d, h] in the model local frame and scales it
        model_bbox_sizes = []
        for model_id, model_scale in zip(
            self.episode_model_ids, self.episode_model_scales
        ):
            model_info = self.model_db[model_id]
            if "bbox" in model_info:
                bbox = model_info["bbox"]
                bbox_size = np.array(bbox["max"]) - np.array(bbox["min"])
                model_bbox_sizes.append(bbox_size * model_scale)
            else:
                raise ValueError(f"Model {model_id} does not have bbox info.")
        self.episode_model_bbox_sizes = model_bbox_sizes

        return reconfigure

    # This method gets called at the end of the reset function of BaseEnv
    # BaseEnv reset function calles initialize_episode which calls _initialize_actors

    # Sets the pose and location for all objects based on the options provided through the reset function
    def _initialize_actors(self):

        # Set source and target object ids
        source_obj_id: int = self.obj_init_options.get("source_obj_id", None)
        target_obj_id: int = self.obj_init_options.get("target_obj_id", None)
        assert source_obj_id is not None and target_obj_id is not None
        
        self.episode_source_obj = self.episode_objs[source_obj_id]
        self.episode_target_obj = self.episode_objs[target_obj_id]

        # scaled bbox size vectors [w, d, h] in the model's local frame
        self.episode_source_obj_bbox_world = self.episode_model_bbox_sizes[
            source_obj_id
        ]  
        self.episode_target_obj_bbox_world = self.episode_model_bbox_sizes[
            target_obj_id
        ]

        # Objects will fall from a certain initial height onto the table
        obj_init_xys = self.obj_init_options.get("init_xys", None)
        assert obj_init_xys is not None
        obj_init_xys = np.array(obj_init_xys)  # [n_objects, 2]
        assert obj_init_xys.shape == (len(self.episode_objs), 2)

        obj_init_z = self.obj_init_options.get("init_z", self.scene_table_height)
        obj_init_z = obj_init_z + 0.5 # let object fall onto the table

        obj_init_rot_quats = self.obj_init_options.get("init_rot_quats", None)
        if obj_init_rot_quats is not None:
            obj_init_rot_quats = np.array(obj_init_rot_quats)
            assert obj_init_rot_quats.shape == (len(self.episode_objs), 4)
        else:
            obj_init_rot_quats = np.zeros((len(self.episode_objs), 4))
            obj_init_rot_quats[:, 0] = 1.0

        for i, obj in enumerate(self.episode_objs):
            p = np.hstack([obj_init_xys[i], obj_init_z])
            q = obj_init_rot_quats[i]
            obj.set_pose(sapien.Pose(p, q))
            # Lock rotation around x and y
            obj.lock_motion(0, 0, 0, 1, 1, 0)

        # Move the robot far away to avoid collision
        # The robot should be initialized later in _initialize_agent (in base_env.py)
        self.agent.robot.set_pose(sapien.Pose([-10, 0, 0]))

        self._settle(0.5)
        
        # Unlock motion
        for obj in self.episode_objs:
            obj.lock_motion(0, 0, 0, 0, 0, 0)
            # NOTE(jigu): Explicit set pose to ensure the actor does not sleep
            obj.set_pose(obj.pose)
            obj.set_velocity(np.zeros(3))
            obj.set_angular_velocity(np.zeros(3))
        self._settle(0.5)

        # Some objects need longer time to settle
        lin_vel, ang_vel = 0.0, 0.0
        for obj in self.episode_objs:
            lin_vel += np.linalg.norm(obj.velocity)
            ang_vel += np.linalg.norm(obj.angular_velocity)
        if lin_vel > 1e-3 or ang_vel > 1e-2:
            self._settle(1.5)

        # Get the xyz positions of the objects after settling
        self.episode_obj_xyzs_after_settle = []
        for obj in self.episode_objs:
            self.episode_obj_xyzs_after_settle.append(obj.pose.p)
        self.episode_source_obj_xyz_after_settle = self.episode_obj_xyzs_after_settle[
            source_obj_id
        ]
        self.episode_target_obj_xyz_after_settle = self.episode_obj_xyzs_after_settle[
            target_obj_id
        ]

        # Takes the size vector [w, d, h] in the objects local frame, 
        # and rotates it into the world frame by the objects rotation 
        self.episode_source_obj_bbox_world = (
            quat2mat(self.episode_source_obj.pose.q)
            @ self.episode_source_obj_bbox_world
        )

        self.episode_target_obj_bbox_world = (
            quat2mat(self.episode_target_obj.pose.q)
            @ self.episode_target_obj_bbox_world
        )

    # No Change Required
    @property
    def source_obj_pose(self):
        """Get the center of mass (COM) pose."""
        return self.episode_source_obj.pose.transform(
            self.episode_source_obj.cmass_local_pose
        )

    # No Change Required
    @property
    def target_obj_pose(self):
        """Get the center of mass (COM) pose."""
        return self.episode_target_obj.pose.transform(
            self.episode_target_obj.cmass_local_pose
        )

    # No Change Required this method returns a dict with the tcp, src_obj, and tgt_obj poses
    # and the tcp to source object pose as a numpy array [x, y, z, qw, qx, qy, qz]
    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(tcp_pose=vectorize_pose(self.tcp.pose))
        if self._obs_mode in ["state", "state_dict"]:
            obs.update(
                source_obj_pose=vectorize_pose(self.source_obj_pose),
                target_obj_pose=vectorize_pose(self.target_obj_pose),
                tcp_to_source_obj_pos=self.source_obj_pose.p - self.tcp.pose.p,
            )
        return obs

    ##### Change Required
    def evaluate(self, **kwargs):
        source_obj_pose = self.source_obj_pose
        target_obj_pose = self.target_obj_pose

        # Check if objects are knocked down or knocked off table
        other_obj_ids = [
            i
            for (i, obj) in enumerate(self.episode_objs)
            if (obj.name != self.episode_source_obj.name)
            and (obj.name != self.episode_target_obj.name)
        ]
        other_obj_heights = [self.episode_objs[i].pose.p[2] for i in other_obj_ids]
        other_obj_heights_after_settle = [
            self.episode_obj_xyzs_after_settle[i][2] for i in other_obj_ids
        ]
        other_obj_diff_heights = [
            x - y for (x, y) in zip(other_obj_heights, other_obj_heights_after_settle)
        ]
        other_obj_keep_height = all(
            [x > -0.02 for x in other_obj_diff_heights]
        )  # require other objects to not be knocked down on the table
        source_obj_diff_height = (
            source_obj_pose.p[2] - self.episode_source_obj_xyz_after_settle[2]
        )  # source object should not be knocked off the table
        target_obj_diff_height = (
            target_obj_pose.p[2] - self.episode_target_obj_xyz_after_settle[2]
        )  # target object should not be knocked off the table
        all_obj_keep_height = (
            other_obj_keep_height
            and (source_obj_diff_height > -0.15)
            and (target_obj_diff_height > -0.15)
        )

        # Check if moving the correct source object
        source_obj_xy_move_dist = np.linalg.norm(
            self.episode_source_obj_xyz_after_settle[:2]
            - self.episode_source_obj.pose.p[:2]
        )
        other_obj_xy_move_dist = []
        for obj, obj_xyz_after_settle in zip(
            self.episode_objs, self.episode_obj_xyzs_after_settle
        ):
            if obj.name == self.episode_source_obj.name:
                continue
            other_obj_xy_move_dist.append(
                np.linalg.norm(obj_xyz_after_settle[:2] - obj.pose.p[:2])
            )
        moved_correct_obj = (source_obj_xy_move_dist > 0.03) and (
            all([x < source_obj_xy_move_dist for x in other_obj_xy_move_dist])
        )
        moved_wrong_obj = any([x > 0.03 for x in other_obj_xy_move_dist]) and any(
            [x > source_obj_xy_move_dist for x in other_obj_xy_move_dist]
        )

        # Check if the source object is near the target object
        dist_to_tgt_obj = np.linalg.norm(source_obj_pose.p[:2] - target_obj_pose.p[:2])
        tgt_obj_bbox_xy_dist = (
            np.linalg.norm(self.episode_target_obj_bbox_world[:2]) / 2
        )  # get half-length of bbox xy diagonol distance in the world frame at timestep=0
        src_obj_bbox_xy_dist = (
            np.linalg.norm(self.episode_source_obj_bbox_world[:2]) / 2
        )
        # print(dist_to_tgt_obj, tgt_obj_bbox_xy_dist, src_obj_bbox_xy_dist)
        near_tgt_obj = (
            dist_to_tgt_obj < tgt_obj_bbox_xy_dist + src_obj_bbox_xy_dist + 0.10
        )

        # Check if the source object is closest to the target object
        dist_to_other_objs = []
        for obj in self.episode_objs:
            if obj.name == self.episode_source_obj.name:
                continue
            dist_to_other_objs.append(
                np.linalg.norm(source_obj_pose.p[:2] - obj.pose.p[:2])
            )
        is_closest_to_tgt = all(
            [dist_to_tgt_obj < x + 0.01 for x in dist_to_other_objs]
        )

        success = (
            all_obj_keep_height
            and moved_correct_obj
            and near_tgt_obj
            and is_closest_to_tgt
        )

        ret_info = dict(
            all_obj_keep_height=all_obj_keep_height,
            moved_correct_obj=moved_correct_obj,
            moved_wrong_obj=moved_wrong_obj,
            near_tgt_obj=near_tgt_obj,
            is_closest_to_tgt=is_closest_to_tgt,
            success=success,
        )
        for k in self.episode_stats:
            self.episode_stats[k] = ret_info[
                k
            ]  # for this environment, episode stats equal to the current step stats
        ret_info["episode_stats"] = self.episode_stats

        return ret_info

    # No Change Required 
    def compute_dense_reward(self, info, **kwargs):
        reward = 0.0
        if info["success"]:
            reward = 1.0
        return reward

    # No Change Required
    def compute_normalized_dense_reward(self, **kwargs):
        return self.compute_dense_reward(**kwargs) / 1.0

    ##### Change Required
    def get_language_instruction(self, **kwargs):
        src_name = self._get_instruction_obj_name(self.episode_source_obj.name)
        tgt_name = self._get_instruction_obj_name(self.episode_target_obj.name)
        return f"move {src_name} near {tgt_name}"

