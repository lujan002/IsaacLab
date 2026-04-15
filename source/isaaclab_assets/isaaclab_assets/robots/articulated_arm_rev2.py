# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Articulated Arm Rev2 robot."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

# TODO: Consider converting this URDF to a USD and referencing the USD from a portable location.
URDF_PATH = "/home/ubuntu-22/Documents/arm_ik_updated/arm_ik/urdf/Articulated_Arm_Rev2.urdf"
# USD_PATH = f"{ISAACLAB_NUCLEUS_DIR}/Robots/FrankaEmika/rover_instanceable.usd"
USD_PATH = "/home/ubuntu-22/Documents/arm_ik_updated/arm_ik/urdf/Articulated_Arm_Rev2/Articulated_Arm_Rev2.usd"

ARTICULATED_ARM_REV2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # Should be False for the arm to be able to move freely, the rover arm geometry is not compatible with self-collisions
            enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={"ax[1-6]_.*": 0.0,
        "ax7_gripper_right": 0.0,
        "ax7_gripper_left": 0.0,
        },
        # Add gripper joint positions
        joint_vel={"ax[1-6]_.*": 0.0,
        "ax7_gripper_right": 0.0,
        "ax7_gripper_left": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["ax[1-6]_.*"],
            # Override imported joint limits in case URDF/USD carried zeros.
            effort_limit_sim=200.0,
            # stiffness=400.0,
            # damping=40.0,
            velocity_limit_sim=10.0,
            stiffness=120.0,
            damping=12.0,
        ),
        "gripper_finger": ImplicitActuatorCfg(
            joint_names_expr=["ax7_gripper_right", "ax7_gripper_left"],
            effort_limit_sim=40.0,      # override URDF effort=0
            velocity_limit_sim=0.5,     # override URDF velocity=0
            stiffness=300.0,            # start here; increase if fingers feel weak
            damping=30.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# A stiffer version of the same robot configuration (useful for IK tracking).
ARTICULATED_ARM_REV2_HIGH_PD_CFG = ARTICULATED_ARM_REV2_CFG.copy()
ARTICULATED_ARM_REV2_HIGH_PD_CFG.actuators["arm"].stiffness = 800.0
ARTICULATED_ARM_REV2_HIGH_PD_CFG.actuators["arm"].damping = 80.0


