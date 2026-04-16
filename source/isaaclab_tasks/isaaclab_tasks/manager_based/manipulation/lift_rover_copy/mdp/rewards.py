# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_is_lifted_when_ee_near(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    ee_distance_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Binary lift reward gated by object-to-EE proximity."""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)

    is_lifted = object_pos_w[:, 2] > minimal_height
    ee_is_near = object_ee_distance <= ee_distance_threshold
    return (is_lifted & ee_is_near).float()


def object_lift_height_progress(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    max_height: float = 0.10,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Linear lift progress reward between a threshold and a ceiling height.

    Returns:
        0.0 when z <= minimal_height
        linearly increasing in (minimal_height, max_height)
        1.0 when z >= max_height
    """
    object: RigidObject = env.scene[object_cfg.name]
    object_z = object.data.root_pos_w[:, 2]
    height_span = max(max_height - minimal_height, 1.0e-6)
    return torch.clamp((object_z - minimal_height) / height_span, min=0.0, max=1.0)


def object_lift_height_progress_when_ee_near(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    ee_distance_threshold: float,
    max_height: float = 0.10,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Lift progress reward gated by object-to-EE proximity.

    Returns zero unless the object is within ``ee_distance_threshold`` of the
    configured end-effector frame (including any FrameTransformer offset).
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    object_pos_w = object.data.root_pos_w
    ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    object_ee_distance = torch.norm(object_pos_w - ee_pos_w, dim=1)
    ee_is_near = object_ee_distance <= ee_distance_threshold

    object_z = object_pos_w[:, 2]
    height_span = max(max_height - minimal_height, 1.0e-6)
    lift_progress = torch.clamp((object_z - minimal_height) / height_span, min=0.0, max=1.0)
    return lift_progress * ee_is_near.float()


def object_lift_height_progress_when_ee_contact(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    contact_force_threshold: float,
    max_height: float = 0.10,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_object_contact"),
) -> torch.Tensor:
    """Lift progress reward gated by EE-object contact force.

    Returns zero unless selected robot bodies have net contact force above
    ``contact_force_threshold`` against the object.
    """
    object: RigidObject = env.scene[object_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_forces = contact_sensor.data.net_forces_w_history
    force_norm = torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1)
    max_force = torch.max(force_norm, dim=1)[0].max(dim=1)[0]
    has_contact = max_force > contact_force_threshold

    object_z = object.data.root_pos_w[:, 2]
    height_span = max(max_height - minimal_height, 1.0e-6)
    lift_progress = torch.clamp((object_z - minimal_height) / height_span, min=0.0, max=1.0)
    return lift_progress * has_contact.float()


def object_is_lifted_when_ee_contact(
    env: ManagerBasedRLEnv,
    minimal_height: float,
    contact_force_threshold: float,
    ee_distance_threshold: float | None = None,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_object_contact"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Binary lift reward gated by EE-object contact force.

    Optionally also gates on object-to-EE distance when ``ee_distance_threshold`` is provided.
    """
    object: RigidObject = env.scene[object_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    net_forces = contact_sensor.data.net_forces_w_history
    force_norm = torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1)
    max_force = torch.max(force_norm, dim=1)[0].max(dim=1)[0]
    has_contact = max_force > contact_force_threshold

    is_lifted = object.data.root_pos_w[:, 2] > minimal_height
    reward_mask = is_lifted & has_contact

    if ee_distance_threshold is not None:
        ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
        ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
        object_ee_distance = torch.norm(object.data.root_pos_w - ee_pos_w, dim=1)
        ee_is_near = object_ee_distance <= ee_distance_threshold
        reward_mask = reward_mask & ee_is_near

    return reward_mask.float()


class object_is_lifted_when_ee_contact_sustained(ManagerTermBase):
    """Binary lift reward with finger contact gated by consecutive contact steps.

    Increments a per-env counter while finger-object contact force exceeds the threshold; resets
    when contact is lost. Lift credit is given only when the object is lifted, optional EE proximity
    holds, and the contact counter has reached ``min_consecutive_steps``.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._contact_step_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._contact_step_count[:] = 0
        else:
            self._contact_step_count[env_ids] = 0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        minimal_height: float,
        contact_force_threshold: float,
        min_consecutive_steps: int,
        object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("ee_object_contact"),
        ee_distance_threshold: float | None = None,
        ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ) -> torch.Tensor:
        object: RigidObject = env.scene[object_cfg.name]
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

        net_forces = contact_sensor.data.net_forces_w_history
        force_norm = torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1)
        max_force = torch.max(force_norm, dim=1)[0].max(dim=1)[0]
        has_contact = max_force > contact_force_threshold

        self._contact_step_count = torch.where(
            has_contact, self._contact_step_count + 1, torch.zeros_like(self._contact_step_count)
        )
        sustained = self._contact_step_count >= min_consecutive_steps

        is_lifted = object.data.root_pos_w[:, 2] > minimal_height
        reward_mask = is_lifted & sustained & has_contact

        if ee_distance_threshold is not None:
            ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
            ee_pos_w = ee_frame.data.target_pos_w[..., 0, :]
            object_ee_distance = torch.norm(object.data.root_pos_w - ee_pos_w, dim=1)
            ee_is_near = object_ee_distance <= ee_distance_threshold
            reward_mask = reward_mask & ee_is_near

        return reward_mask.float()


def object_root_lin_vel_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Penalize rigid object root linear velocity in world frame (L2 squared)."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_w), dim=1)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float, # reward only when the object is lifted above this height
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))


def ee_object_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.2,
    force_shaping_scale: float = 25.0,
) -> torch.Tensor:
    """Reward contact between selected robot bodies (e.g. wrist / EE) and the manipulated object.

    Expects a :class:`~isaaclab.sensors.contact_sensor.contact_sensor.ContactSensor` with
    ``filter_prim_paths_expr`` including the object so only object contact is reported.

    Args:
        sensor_cfg: Sensor entity name and ``body_names`` for the link(s) to treat as the EE.
        threshold: Report zero reward if max net contact force norm (N) is at or below this.
        force_shaping_scale: Larger values reduce sensitivity to force magnitude (``tanh`` input).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_forces = contact_sensor.data.net_forces_w_history
    force_norm = torch.norm(net_forces[:, :, sensor_cfg.body_ids], dim=-1)
    max_force = torch.max(force_norm, dim=1)[0].max(dim=1)[0]
    shaped = torch.tanh(max_force / force_shaping_scale)
    return torch.where(max_force > threshold, shaped, torch.zeros_like(max_force))


def gripper_midrange_open_no_contact(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    joint_names: list[str],
    target_sum_abs: float = 0.11,
    std: float = 0.03,
    contact_sensor_cfg: SceneEntityCfg | None = None,
    contact_threshold: float = 1.0,
    disable_when_lifted: bool = True,
    minimal_height: float = 0.04,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward keeping gripper opening near midrange, without using object-distance features.

    This term uses only proprioception (gripper joint positions), optional object-contact gating,
    and optional lifted-state gating. It does not consume EE-object distance.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    joint_ids, _ = robot.find_joints(joint_names, preserve_order=True)
    q = robot.data.joint_pos[:, joint_ids]

    # For symmetric prismatic fingers, abs-sum is 0.0 at full-close and ~0.22 at full-open.
    # Midrange is around ~0.11 and can be rewarded with a smooth Gaussian-like kernel.
    sum_abs = torch.sum(torch.abs(q), dim=1)
    midrange_reward = torch.exp(-torch.square(sum_abs - target_sum_abs) / (std * std))

    if contact_sensor_cfg is not None:
        contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]
        net_forces = contact_sensor.data.net_forces_w_history
        force_norm = torch.norm(net_forces[:, :, contact_sensor_cfg.body_ids], dim=-1)
        max_force = torch.max(force_norm, dim=1)[0].max(dim=1)[0]
        no_object_contact = max_force <= contact_threshold
        midrange_reward = midrange_reward * no_object_contact.float()

    if disable_when_lifted:
        object: RigidObject = env.scene[object_cfg.name]
        not_lifted = object.data.root_pos_w[:, 2] <= minimal_height
        midrange_reward = midrange_reward * not_lifted.float()

    return midrange_reward
