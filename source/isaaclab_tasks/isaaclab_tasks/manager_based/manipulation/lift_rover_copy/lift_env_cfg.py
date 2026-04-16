# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # Contact sensor for robot links (filtered to table contacts).
    # contact_forces = None
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Table/.*"],
    )

    # Contact sensor on the block, filtered to table contacts.
    # object_contact_forces = ContactSensorCfg(
    #     prim_path="{ENV_REGEX_NS}/Object",
    #     history_length=3,
    #     filter_prim_paths_expr=["{ENV_REGEX_NS}/Table/.*"],
    # )

    #  Contact sensor for EE / wrist (and optionally fingers) vs block only.
    #  This is used for the ee_object_contact reward term.
    ee_object_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )

    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, -1.8, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/ThorlabsTable/table_instanceable.usd", scale=(2.0, 2.0, 1.0)),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,  # will be set by agent env cfg
        resampling_time_range=(5.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    # gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg | None = None


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # Original (relative-to-default) terms:
        # joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        # joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # Use absolute terms to avoid NaNs when imported USD default-joint buffers are invalid.
        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        # Original (robot-root-frame object position), can produce NaNs if robot root pose is invalid:
        # object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # Use object position in world/env frame for robustness with imported robot USDs.
        object_position = ObsTerm(func=mdp.root_pos_w, params={"asset_cfg": SceneEntityCfg("object")})
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # Disable contact between chassis_base and shoulder (keep all other collisions enabled).
    # This runs once after the simulation starts (prims exist and physics is initialized).
    filter_chassis_shoulder_contact = EventTerm(
        func=mdp.filter_collisions_between_link_collision_subtrees,
        mode="startup",
        params={
            # Match the "collisions" subtree roots; helper traverses descendants for CollisionAPI prims.
            "link_a_collisions_prim_path_expr": "{ENV_REGEX_NS}/Robot/chassis_base/collisions",
            "link_b_collisions_prim_path_expr": "{ENV_REGEX_NS}/Robot/shoulder/collisions",
            "group_a_prim_path": "/World/CollisionGroups/chassis_no_shoulder",
            "group_b_prim_path": "/World/CollisionGroups/shoulder_no_chassis",
        },
    )

    # object_physics_material = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("object", body_names=".*"),
    #         "static_friction_range": (0.25, 1.00),
    #         "dynamic_friction_range": (0.2, 0.80),
    #         "restitution_range": (0.0, 0.0),
    #         "num_buckets": 16,
    #     },
    # )
    # Table uses AssetBaseCfg (XformPrimView), not RigidObject/Articulation — unsupported by
    # randomize_rigid_body_material. Block randomization still varies block-table friction via PhysX combine.

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Keep object resets in a safer tabletop region to reduce startup collision kicks.
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.2, 0.2), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="Object"),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.2}, weight=0.4)

    # Reward for contact between the end-effector and the object
    # ee_object_contact = RewTerm(
    #     func=mdp.ee_object_contact,
    #     params={
    #         "sensor_cfg": SceneEntityCfg(
    #             "ee_object_contact",
    #             body_names=["wrist_roll", "finger1", "finger2"],
    #         ),
    #         "threshold": 1.0,
    #         "force_shaping_scale": 25.0,
    #     },
    #     weight=1.0,
    # )

    # Reward for keeping gripper half open near mid-range, turns off while touching the object
    # gripper_midrange_open = RewTerm(
    #     func=mdp.gripper_midrange_open_no_contact,
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "joint_names": ["ax7_gripper_right", "ax7_gripper_left"],
    #         "target_sum_abs": 0.11,
    #         "std": 0.03,
    #         "contact_sensor_cfg": SceneEntityCfg("ee_object_contact", body_names=["finger1", "finger2"]),
    #         "contact_threshold": 1.0,
    #         "disable_when_lifted": True,
    #         "minimal_height": 0.04,
    #         "object_cfg": SceneEntityCfg("object"),
    #     },
    #     weight=0.6,
    # )

    # table_collision = None
    # table_collision = RewTerm(
    #     func=mdp.undesired_contacts,
    #     params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "threshold": 1.0},
    #     weight=-0.1,
    # )

    # Discourage dragging/sliding: penalize block-table contact force.
    # block_table_contact = RewTerm(
    #     func=mdp.undesired_contacts,
    #     params={"sensor_cfg": SceneEntityCfg("object_contact_forces", body_names=".*"), "threshold": 1.0},
    #     weight=-0.05,
    # )


    # Static lift reward
    # lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.02}, weight=100.0)
    
    # Static lift reward gated by end-effector proximity.
    lifting_object = RewTerm(
        func=mdp.object_is_lifted_when_ee_near,
        params={"minimal_height": 0.02, "ee_distance_threshold": 0.12},
        weight=15.0,
    )

    # Ramp up the reward for lifting the object as it gets higher
    # lifting_object = RewTerm(
    #     func=mdp.object_lift_height_progress,
    #     params={"minimal_height": 0.02, "max_height": 0.10},
    #     weight=25.0,
    # )

    # Ramp up the reward for lifting the object as it gets higher when the end-effector is near
    # lifting_object_when_ee_near = RewTerm(
    #     func=mdp.object_lift_height_progress_when_ee_near,
    #     params={"minimal_height": 0.02, "max_height": 0.10, "ee_distance_threshold": 0.12},
    #     weight=100.0,
    # )
    #     

    object_goal_tracking_breadcrumb = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.02, "command_name": "object_pose"},
        weight=4.0,
    )

    object_goal_tracking = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.3, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=16.0,
    )
    
    object_goal_tracking_fine_grained = RewTerm(
        func=mdp.object_goal_distance,
        params={"std": 0.05, "minimal_height": 0.04, "command_name": "object_pose"},
        weight=5.0,
    )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1e-4,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )

    # End episode early when the end-effector gets close enough to the block.
    # ee_near_object = DoneTerm(func=mdp.ee_near_object, params={"threshold": 0.5})


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # reaching_object = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "reaching_object", "weight": 2.0, "num_steps": 12000}
    # )

    # increase the weight of the ee_object_contact reward term 1.0 -> 3.0
    # ee_object_contact = CurrTerm(
    #     func=mdp.modify_reward_weight, params={"term_name": "ee_object_contact", "weight": 3.0, "num_steps": 12000}
    # )

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    )

    joint_vel_10k = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-3, "num_steps": 10000}
    )

    joint_vel_20k = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 20000}
    )

    joint_vel_40k = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 40000}
    )
##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # if self.scene.object_contact_forces is not None:
        #     self.scene.object_contact_forces.update_period = self.sim.dt
        if self.scene.ee_object_contact is not None:
            self.scene.ee_object_contact.update_period = self.sim.dt
