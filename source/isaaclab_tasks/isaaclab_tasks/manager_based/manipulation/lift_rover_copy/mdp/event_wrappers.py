# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Event helpers that wrap class-based Isaac Lab events for robust EventManager invocation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.events import randomize_rigid_body_material as RandomizeRigidBodyMaterialCls
from isaaclab.managers import EventTermCfg, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_rigid_body_material(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
    num_buckets: int,
    asset_cfg: SceneEntityCfg,
    make_consistent: bool = False,
) -> None:
    """Apply rigid-body material randomization (friction / restitution buckets).

    ``EventManager`` invokes ``func(env, env_ids, **params)``. Class-based terms sometimes still
    reference the uninstantiated `ManagerTermBase` subclass; wrapping avoids calling ``__init__``
    with the wrong signature.
    """
    term_cfg = EventTermCfg(
        func=RandomizeRigidBodyMaterialCls,
        mode="startup",
        params={
            "asset_cfg": asset_cfg,
            "static_friction_range": static_friction_range,
            "dynamic_friction_range": dynamic_friction_range,
            "restitution_range": restitution_range,
            "num_buckets": num_buckets,
            "make_consistent": make_consistent,
        },
    )
    term = RandomizeRigidBodyMaterialCls(cfg=term_cfg, env=env)
    term(
        env,
        env_ids,
        static_friction_range,
        dynamic_friction_range,
        restitution_range,
        num_buckets,
        asset_cfg,
        make_consistent,
    )
