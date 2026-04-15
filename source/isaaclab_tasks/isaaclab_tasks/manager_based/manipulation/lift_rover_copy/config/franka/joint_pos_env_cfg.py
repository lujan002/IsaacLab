# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift_rover_copy.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets.robots.articulated_arm_rev2 import ARTICULATED_ARM_REV2_CFG  # isort: skip


@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Articulated Arm Rev2 as robot
        self.scene.robot = ARTICULATED_ARM_REV2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Required for contact-based rewards/terminations (e.g., table collision penalty).
        # Obviously the real robot does not have contact sensors, but simulating is as such should
        # allow the robot to learn to avoid collisions with the table.
        self.scene.robot.spawn.activate_contact_sensors = True
    

        # Set actions for the specific robot type (franka)
        # self.actions.arm_action = mdp.JointPositionActionCfg(
        #     asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        # )
        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["panda_finger.*"],
        #     open_command_expr={"panda_finger_.*": 0.04},
        #     close_command_expr={"panda_finger_.*": 0.0},
        # )

        # Set rover actions
        # Original:
        # self.actions.arm_action = mdp.JointPositionActionCfg(
        #     asset_name="robot", joint_names=["ax[1-6]_.*"], scale=0.5, use_default_offset=True
        # )

        # Continuous arm actions
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["ax[1-6]_.*"], scale=0.5, use_default_offset=False
        )

        # Binary gripper actions
        # self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=["ax7_gripper_right", "ax7_gripper_left"],
        #     open_command_expr={
        #         "ax7_gripper_right": -0.11,
        #         "ax7_gripper_left": 0.11,
        #     },
        #     close_command_expr={
        #         "ax7_gripper_right": 0.0,
        #         "ax7_gripper_left": 0.0,
        #     },
        # )
        # Continuous gripper actions
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["ax7_gripper_right", "ax7_gripper_left"],
            scale={
                "ax7_gripper_right": -0.055,
                "ax7_gripper_left": 0.055,
            },
            offset={
                "ax7_gripper_right": -0.055,
                "ax7_gripper_left": 0.055,
            },
            use_default_offset=False,
        )


        # Set the body name for the end effector
        # In your USD, ee_frame is an Xform, while wrist_roll is the link.
        self.commands.object_pose.body_name = "wrist_roll"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0, -0.65, 0.01], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                activate_contact_sensors=True, # Added this for table-block collision reward
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            # FrameTransformer requires rigid bodies. `base_frame` is an Xform helper, so use the base link instead.
            prim_path="{ENV_REGEX_NS}/Robot/chassis_base",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    # Track the wrist link, and apply the EE helper-frame offset.
                    prim_path="{ENV_REGEX_NS}/Robot/wrist_roll",
                    name="end_effector",
                    offset=OffsetCfg(
                        # From URDF: ee_frame_joint origin xyz="0 ${ee_offset} 0", ee_offset=0.15
                        pos=[0.0, 0.15, 0.0],
                        # From URDF: rpy="${pi} 0 ${pi/2}"
                        # (w, x, y, z) = (0, 0.7071, 0.7071, 0)
                        rot=(0.0, 0.70710678, 0.70710678, 0.0),
                    ),
                ),
            ],
        )


@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
