from pathlib import Path
from typing import Dict, List, Optional, Type, Union

import numpy as np
import os
import sapien.core as sapien
from sapien.core import Pose
import cv2

from mani_skill2_real2sim import ASSET_DIR, format_path
from mani_skill2_real2sim.utils.io_utils import load_json
from mani_skill2_real2sim.agents.base_agent import BaseAgent
from mani_skill2_real2sim.agents.robots.googlerobot import (
    GoogleRobotStaticBase,
    GoogleRobotStaticBaseWorseControl1, GoogleRobotStaticBaseWorseControl2, GoogleRobotStaticBaseWorseControl3,
    GoogleRobotStaticBaseHalfFingerFriction, GoogleRobotStaticBaseQuarterFingerFriction, GoogleRobotStaticBaseOneEighthFingerFriction, GoogleRobotStaticBaseTwiceFingerFriction
)
from mani_skill2_real2sim.agents.robots.widowx import WidowX, WidowXBridgeDatasetCameraSetup, WidowXSinkCameraSetup
from mani_skill2_real2sim.agents.robots.panda import Panda
from mani_skill2_real2sim.envs.sapien_env import BaseEnv
from mani_skill2_real2sim.sensors.camera import CameraConfig
from mani_skill2_real2sim.utils.sapien_utils import (
    get_entity_by_name,
    look_at,
    set_articulation_render_material,
    vectorize_pose,
)


class CustomSceneEnv(BaseEnv):
    """
    CustomSceneEnv extends BaseEnv to set up a simulation environment with
    configurable robot, scene assets, camera overlays, and reward computation.
    """
    # Mapping from string keys to robot agent classes
    SUPPORTED_ROBOTS = {
        "google_robot_static": GoogleRobotStaticBase,
        "widowx": WidowX,
        "widowx_bridge_dataset_camera_setup": WidowXBridgeDatasetCameraSetup,
        "widowx_sink_camera_setup": WidowXSinkCameraSetup,
        "panda": Panda,
        # Ablation and friction variants
        "google_robot_static_worse_control1": GoogleRobotStaticBaseWorseControl1,
        "google_robot_static_worse_control2": GoogleRobotStaticBaseWorseControl2,
        "google_robot_static_worse_control3": GoogleRobotStaticBaseWorseControl3,
        "google_robot_static_half_finger_friction": GoogleRobotStaticBaseHalfFingerFriction,
        "google_robot_static_quarter_finger_friction": GoogleRobotStaticBaseQuarterFingerFriction,
        "google_robot_static_one_eighth_finger_friction": GoogleRobotStaticBaseOneEighthFingerFriction,
        "google_robot_static_twice_finger_friction": GoogleRobotStaticBaseTwiceFingerFriction,
    }
    agent: Union[GoogleRobotStaticBase, WidowX, Panda]
    DEFAULT_ASSET_ROOT: str
    DEFAULT_SCENE_ROOT: str
    DEFAULT_MODEL_JSON: str

    def __init__(
        self,
        robot: str = "google_robot_static",
        rgb_overlay_path: Optional[str] = None,
        rgb_overlay_cameras: List[str] = [],
        rgb_overlay_mode: str = 'background',
        rgb_always_overlay_objects: List[str] = [],
        disable_bad_material: bool = False,
        asset_root: Optional[str] = None,
        scene_root: Optional[str] = None,
        scene_name: Optional[str] = None,
        scene_offset: Optional[List[float]] = None,
        scene_pose: Optional[List[float]] = None,
        scene_table_height: float = 0.87,
        model_json: Optional[str] = None,
        model_ids: List[str] = (),
        model_db_override: Dict[str, Dict] = {},
        urdf_version: str = "",
        **kwargs
    ):
        """
        Initialize the scene environment.

        Args:
            robot: key for SUPPORTED_ROBOTS to choose an agent class.
            rgb_overlay_path: path to overlay image (greenscreen) for background.
            rgb_overlay_cameras: list of camera names to apply overlay.
            rgb_overlay_mode: 'background', 'object', 'debug', or combos for blending.
            rgb_always_overlay_objects: object names to always overlay regardless of mode.
            disable_bad_material: if True, skip applying enhanced render materials.
            asset_root: directory for asset files (overrides DEFAULT_ASSET_ROOT).
            scene_root: directory for scene files (overrides DEFAULT_SCENE_ROOT).
            scene_name: name of specific scene GLB file (without extension).
            scene_offset: translation offset for placing scene geometry.
            scene_pose: quaternion [x,y,z,w] for scene orientation.
            scene_table_height: height of table surface in meters.
            model_json: name or path of JSON describing object models.
            model_ids: list of model IDs to include in scene; defaults to all.
            model_db_override: dict to override entries in loaded model DB.
            urdf_version: suffix for robot URDF filename.
            **kwargs: passed to BaseEnv.
        """
        # Determine asset and scene roots
        if asset_root is None:
            asset_root = self.DEFAULT_ASSET_ROOT
        self.asset_root = Path(format_path(asset_root))

        if scene_root is None:
            scene_root = self.DEFAULT_SCENE_ROOT
        self.scene_root = Path(format_path(scene_root))
        self.scene_name = scene_name
        self.scene_offset = scene_offset
        self.scene_pose = scene_pose
        self.scene_table_height = scene_table_height

        # Load model database JSON
        if model_json is None:
            model_json = self.DEFAULT_MODEL_JSON
        model_json = self.asset_root / format_path(model_json)
        if not model_json.exists():
            raise FileNotFoundError(
                f"{model_json} not found. Set MS2_REAL2SIM_ASSET_DIR to your asset directory."
            )
        self.model_db: Dict[str, Dict] = load_json(model_json)
        self.model_db.update(model_db_override)

        # Normalize model_ids input
        if isinstance(model_ids, str):
            model_ids = [model_ids]
        if len(model_ids) == 0:
            model_ids = sorted(self.model_db.keys())
        assert model_ids, f"No models found in {model_json}"
        self.model_ids = model_ids
        self._check_assets()

        # Load overlay image if provided
        if rgb_overlay_path:
            if not os.path.exists(rgb_overlay_path):
                raise FileNotFoundError(f"Overlay image {rgb_overlay_path} not found.")
            img = cv2.imread(rgb_overlay_path)
            self.rgb_overlay_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        else:
            self.rgb_overlay_img = None

        # Ensure cameras list type
        if not isinstance(rgb_overlay_cameras, list):
            rgb_overlay_cameras = [rgb_overlay_cameras]
        self.rgb_overlay_path = rgb_overlay_path
        self.rgb_overlay_cameras = rgb_overlay_cameras
        self.rgb_overlay_mode = rgb_overlay_mode
        self.rgb_always_overlay_objects = rgb_always_overlay_objects
        assert 'background' in rgb_overlay_mode or 'debug' in rgb_overlay_mode, f"Invalid rgb_overlay_mode: {rgb_overlay_mode}"

        # Robot configuration
        self.arena = None
        self.robot_uid = robot
        self.urdf_version = urdf_version or ""
        self.disable_bad_material = disable_bad_material

        super().__init__(**kwargs)

    def _check_assets(self) -> None:
        """
        Verify that required asset files for selected model_ids exist on disk.

        Raises:
            FileNotFoundError: if any model directory or files are missing.
        """
        # Implementation empty; override in subclasses if needed
        pass

    def _load_arena_helper(self, add_collision: bool = True) -> None:
        """
        Build the static scene geometry (arena) around the robot.

        Args:
            add_collision: if True, add collision meshes along with visuals.
        """
        builder = self._scene.create_actor_builder()
        # Determine scene GLB path
        if self.scene_name is None:
            # Default scenes based on robot type
            if 'google_robot_static' in self.robot_uid:
                glb = "stages/google_pick_coke_can_1_v4.glb"
            elif 'widowx' in self.robot_uid:
                glb = "stages/bridge_table_1_v1.glb"
            else:
                raise NotImplementedError(f"No default scene for {self.robot_uid}")
            scene_path = self.scene_root / glb
        elif 'dummy' in self.scene_name:
            scene_path = None
        else:
            scene_path = self.scene_root / 'stages' / f'{self.scene_name}.glb'

        # Compute offsets and poses
        if self.scene_offset is None:
            default_offset = {
                'google_robot_static': np.array([-1.6616, -3.0337, 0.0]),
                'widowx': np.array([-2.0634, -2.8313, 0.0])
            }
            scene_offset = default_offset.get(self.robot_uid.split('_')[0])
        else:
            scene_offset = np.array(self.scene_offset)
        scene_pose = self.scene_pose and Pose(q=self.scene_pose) or Pose(q=[0.707, 0.707, 0, 0])

        # Load or build arena geometry
        if scene_path:
            if add_collision:
                builder.add_nonconvex_collision_from_file(str(scene_path), scene_pose)
            builder.add_visual_from_file(str(scene_path), scene_pose)
        else:
            # Dummy builds for simple scenes
            builder.add_box_visual(half_size=np.array([10,10,0.0085]))

        self.arena = builder.build_static(name="arena")
        # Position arena relative to robot
        self.arena.set_pose(Pose(-scene_offset))

    def _settle(self, t: float) -> None:
        """
        Step the physics simulation for t seconds to let objects settle.

        Args:
            t: time in seconds to step simulation.
        """
        steps = int(self.sim_freq * t)
        for _ in range(steps):
            self._scene.step()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None):
        """
        Reset environment and return initial observation and info.

        Args:
            seed: random seed for reproducibility.
            options: dict with possible 'robot_init_options'.

        Returns:
            obs: observation dict after reset.
            info: additional info dict including scene parameters.
        """
        self.robot_init_options = options.get("robot_init_options", {}) if options else {}
        obs, info = super().reset(seed=seed, options=options)
        info.update({
            'scene_name': self.scene_name,
            'scene_offset': self.scene_offset,
            'scene_pose': self.scene_pose,
            'scene_table_height': self.scene_table_height,
            'urdf_version': self.urdf_version,
            'rgb_overlay_path': self.rgb_overlay_path,
            'rgb_overlay_cameras': self.rgb_overlay_cameras,
            'rgb_overlay_mode': self.rgb_overlay_mode,
            'disable_bad_material': self.disable_bad_material,
        })
        return obs, info

    def _configure_agent(self) -> None:
        """
        Configure agent class and URDF path based on robot_uid and urdf_version.
        """
        agent_cls: Type[BaseAgent] = self.SUPPORTED_ROBOTS[self.robot_uid]
        self._agent_cfg = agent_cls.get_default_config()
        if self.urdf_version:
            self._agent_cfg.urdf_path = self._agent_cfg.urdf_path.replace(
                ".urdf", f"_{self.urdf_version}.urdf"
            )

    def _load_agent(self) -> None:
        """
        Instantiate the robot agent and set up the TCP link and render materials.
        """
        agent_cls: Type[GoogleRobotStaticBase] = self.SUPPORTED_ROBOTS[self.robot_uid]
        self.agent = agent_cls(
            self._scene, self._control_freq, self._control_mode, config=self._agent_cfg
        )
        self.tcp: sapien.Link = get_entity_by_name(
            self.agent.robot.get_links(), self.agent.config.ee_link_name
        )
        if not self.disable_bad_material:
            set_articulation_render_material(
                self.agent.robot, specular=0.9, roughness=0.3
            )

    def _initialize_agent(self) -> None:
        """
        Set initial joint positions and base pose for the robot.

        Uses either default poses for each robot type or overrides from robot_init_options.
        """
        # Default qpos and pose based on robot type
        if 'google_robot_static' in self.robot_uid:
            default_qpos = np.array([...])  # omitted for brevity
            init_height = 0.07905
            init_quat = [0, 0, 0, 1]
        elif 'widowx' in self.robot_uid:
            default_qpos = np.array([...])
            init_height = 0.87 if 'bridge' in self.robot_uid else 0.85
            init_quat = [0, 0, 0, 1]
        else:
            raise NotImplementedError(self.robot_uid)
        # Apply overrides
        qpos = self.robot_init_options.get('qpos', default_qpos)
        self.agent.reset(qpos)
        # Set base pose
        height = self.robot_init_options.get('init_height', init_height)
        quat = self.robot_init_options.get('init_rot_quat', init_quat)
        xy = self.robot_init_options.get('init_xy')
        if xy:
            xyz = [xy[0], xy[1], height]
        else:
            xyz = [0.3, 0.1, height]
        self.agent.robot.set_pose(Pose(xyz, quat))

    def _register_cameras(self) -> CameraConfig:
        """
        Register the default agent-mounted sensor camera.

        Returns:
            CameraConfig: configuration for base observation camera.
        """
        pose = look_at([0.3, 0, 0.6], [-0.1, 0, 0.1])
        return CameraConfig(
            name="base_camera",
            position=pose.p, orientation=pose.q,
            width=128, height=128, fov=np.pi/2,
            near=0.01, far=10
        )

    def _register_render_cameras(self) -> CameraConfig:
        """
        Register a debug render camera for visualization.

        Returns:
            CameraConfig: configuration for render camera.
        """
        pose = look_at([0.5,0.5,1.0], [0,0,0.5])
        return CameraConfig("render_camera", pose.p, pose.q, 512, 512, 1, 0.01, 10)

    def _setup_viewer(self) -> None:
        """
        Configure the GUI viewer camera orientation.
        """
        super()._setup_viewer()
        self._viewer.set_camera_xyz(0.8, 0, 1.0)
        self._viewer.set_camera_rpy(0, -0.5, 3.14)

    def _get_obs_agent(self) -> Dict:
        """
        Get proprioceptive observation from the agent.

        Returns:
            dict: contains joint states and base pose.
        """
        obs = self.agent.get_proprioception()
        obs['base_pose'] = vectorize_pose(self.agent.robot.pose)
        return obs

    def get_obs(self):
        """
        Extend BaseEnv.get_obs() to apply a greenscreen overlay of the real-world image
        onto the simulated background, preserving simulated robot and object visuals.

        Returns:
            obs (dict): Observation dictionary with 'image' entries modified.
        """
        # 1. Retrieve the default observation from the parent environment
        obs = super().get_obs()

        # 2. Only apply overlay when in image mode and overlay image is provided
        if self._obs_mode == "image" and self.rgb_overlay_img is not None:
            # 2a. Identify static scene objects to treat as foreground (not overlaid)
            fg_exclude = ['ground', 'goal_site', '', 'arena'] + self.rgb_always_overlay_objects
            target_object_actor_ids = [
                x.id for x in self.get_actors()
                if x.name not in fg_exclude
            ]
            target_object_actor_ids = np.array(target_object_actor_ids, dtype=np.int32)

            # 2b. Collect robot link IDs (always foreground)
            robot_link_ids = np.array(
                [link.id for link in self.agent.robot.get_links()],
                dtype=np.int32
            )

            # 2c. Gather link IDs from other articulated objects (foreground)
            other_link_ids = []
            for art_obj in self._scene.get_all_articulations():
                # skip the main robot and any always-overlay objects
                if art_obj is self.agent.robot or art_obj.name in self.rgb_always_overlay_objects:
                    continue
                other_link_ids.extend([link.id for link in art_obj.get_links()])
            other_link_ids = np.array(other_link_ids, dtype=np.int32)

            # 3. Process each specified camera view
            for cam in self.rgb_overlay_cameras:
                # 3a. Ensure segmentation mask is present
                assert 'Segmentation' in obs['image'][cam], \
                    'Image overlay requires segment info in observation!'
                seg = obs['image'][cam]['Segmentation']  # shape (H, W, 4)
                actor_seg = seg[..., 1]                # actor-level IDs, shape (H, W)

                # 3b. Build overlay mask: 1 = use real image, 0 = keep simulation
                mask = np.ones_like(actor_seg, dtype=np.float32)
                # If mode allows background or debug overlay
                if 'background' in self.rgb_overlay_mode or 'debug' in self.rgb_overlay_mode:
                    # Zero-out mask where actor_seg matches any foreground IDs
                    fg_ids = np.concatenate([robot_link_ids, target_object_actor_ids, other_link_ids])
                    mask[np.isin(actor_seg, fg_ids)] = 0.0
                else:
                    # Unsupported mode combination
                    raise NotImplementedError(f"Invalid rgb_overlay_mode: {self.rgb_overlay_mode}")
                # Expand to (H, W, 1) for broadcasting over RGB
                mask = mask[..., np.newaxis]

                # 4. Resize real-world overlay image to match camera resolution
                color_img = obs['image'][cam]['Color']  # shape (H, W, 3)
                h, w = color_img.shape[:2]
                overlay_resized = cv2.resize(self.rgb_overlay_img, (w, h))

                # 5. Composite: simulation * (1-mask) + overlay * mask
                if 'debug' not in self.rgb_overlay_mode:
                    obs['image'][cam]['Color'][..., :3] = (
                        color_img * (1 - mask)
                        + overlay_resized * mask
                    )
                else:
                    # Debug: simple 50/50 blend for visual inspection
                    obs['image'][cam]['Color'][..., :3] = (
                        color_img * 0.5
                        + overlay_resized * 0.5
                    )

        # 6. Return the potentially modified observation
        return obs

    def compute_dense_reward(self, info: Dict, **kwargs) -> float:
        """
        Compute sparse reward based on success flag.

        Args:
            info: dictionary containing 'success' boolean.

        Returns:
            float: 1.0 if succeeded, else 0.0.
        """
        return 1.0 if info.get('success', False) else 0.0

    def compute_normalized_dense_reward(self, **kwargs) -> float:
        """
        Normalize dense reward. Currently identity scaling.

        Returns:
            float: normalized reward.
        """
        return self.compute_dense_reward(**kwargs)

    @staticmethod
    def _get_instruction_obj_name(s: str) -> str:
        """
        Clean object model ID to human-readable name for instructions.

        Args:
            s: original model ID string.

        Returns:
            str: cleaned instruction name.
        """
        parts = s.split('_')
        filters = {'opened','light','generated','modified','objaverse','bridge','baked','v2'}
        cleaned = [w for w in parts if w not in filters and not w.endswith('cm')]
        return ' '.join(cleaned)

    def advance_to_next_subtask(self) -> None:
        """
        Advance environment to next subtask. Implement multi-stage tasks.

        Raises:
            NotImplementedError: as default.
        """
        raise NotImplementedError("advance_to_next_subtask not implemented")

    def is_final_subtask(self) -> bool:
        """
        Check if current subtask is the final one.

        Returns:
            bool: True if final.
        """
        return True


class CustomOtherObjectsInSceneEnv(CustomSceneEnv):
    """
    Custom scene environment that uses a different model JSON and
    adds utility to build arbitrary object actors.
    """
    DEFAULT_ASSET_ROOT = f"{ASSET_DIR}/custom"
    DEFAULT_SCENE_ROOT = f"{ASSET_DIR}/hab2_bench_assets"
    DEFAULT_MODEL_JSON = "info_pick_custom_v0.json"
    obj_static_friction = 0.5
    obj_dynamic_friction = 0.5

    def _check_assets(self) -> None:
        """
        Ensure all custom object model directories and collision files exist.

        Raises:
            FileNotFoundError: if any model or collision.obj is missing.
        """
        models_dir = self.asset_root / "models"
        for mid in self.model_ids:
            mdir = models_dir / mid
            if not mdir.exists():
                raise FileNotFoundError(f"Model dir {mdir} not found.")
            col = mdir / "collision.obj"
            if not col.exists():
                raise FileNotFoundError("convex.obj has been renamed to collision.obj.")

    @staticmethod
    def _build_actor_helper(
        model_id: str,
        scene: sapien.Scene,
        scale: float = 1.0,
        physical_material: sapien.PhysicalMaterial = None,
        density: float = 1000.0,
        root_dir: str = ASSET_DIR / "custom",
    ) -> sapien.Actor:
        """
        Build an actor from custom model files.

        Args:
            model_id: ID of the model directory.
            scene: Sapien scene to attach actor to.
            scale: uniform scale factor.
            physical_material: physical material for collisions.
            density: density for collision meshes.
            root_dir: base path for custom assets.

        Returns:
            sapien.Actor: built actor instance.
        """
        builder = scene.create_actor_builder()
        model_dir = Path(root_dir) / "models" / model_id

        # Load collision mesh
        col_file = str(model_dir / "collision.obj")
        builder.add_multiple_collisions_from_file(
            filename=col_file,
            scale=[scale]*3,
            material=physical_material,
            density=density,
        )

        # Load visual mesh (try .obj, .dae, .glb)
        for ext in ['textured.obj','textured.dae','textured.glb']:
            vf = model_dir / ext
            if vf.exists():
                builder.add_visual_from_file(str(vf), scale=[scale]*3)
                break

        return builder.build()
