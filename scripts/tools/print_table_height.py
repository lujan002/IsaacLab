#!/usr/bin/env python3
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""
Print table height and debug object world-z in headless IsaacLab runs.

Usage:
    # 1) table geometry check (existing behavior)
    ./isaaclab.sh -p scripts/tools/print_table_height.py --headless

    # 2) object z debugger in actual RL env
    ./isaaclab.sh -p scripts/tools/print_table_height.py --mode object-z --headless
"""

import argparse

from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser(description="Print table top height from spawned USD bounds.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument(
    "--mode",
    type=str,
    default="table",
    choices=["table", "object-z"],
    help="Run table AABB check or object world-z debugger.",
)
parser.add_argument(
    "--table-type",
    type=str,
    default="thorlabs",
    choices=["thorlabs", "seattle"],
    help="Which Isaac table asset to inspect.",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Lift-Cube-ArticulatedArmRev2-v0",
    help="Gym task id used when mode=object-z.",
)
parser.add_argument(
    "--num-envs",
    type=int,
    default=64,
    help="Number of envs for object-z debug mode.",
)
parser.add_argument(
    "--steps",
    type=int,
    default=50,
    help="How many simulation steps to print in object-z debug mode.",
)
args_cli = parser.parse_args()

# Launch simulator app first.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import torch
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from pxr import Usd, UsdGeom


def print_table_height() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device="cpu"))

    table_rel_path = (
        "Props/Mounts/ThorlabsTable/table_instanceable.usd"
        if args_cli.table_type == "thorlabs"
        else "Props/Mounts/SeattleLabTable/table_instanceable.usd"
    )
    table_usd_path = f"{ISAAC_NUCLEUS_DIR}/{table_rel_path}"

    # Match your lift_rover_copy table transform defaults.
    table_translation = (0.0, -1.8, 0.0)
    table_scale = (2.0, 2.0, 1.0) if args_cli.table_type == "thorlabs" else (1.0, 1.0, 1.0)

    table_cfg = sim_utils.UsdFileCfg(usd_path=table_usd_path, scale=table_scale)
    table_cfg.func("/World/Table", table_cfg, translation=table_translation)

    # Finalize stage and ensure transforms are updated.
    sim.reset()
    sim.step(render=False)

    stage: Usd.Stage = sim.stage
    table_prim = stage.GetPrimAtPath("/World/Table")
    if not table_prim.IsValid():
        raise RuntimeError("Table prim /World/Table was not created.")

    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    world_bound = bbox_cache.ComputeWorldBound(table_prim)
    aligned = world_bound.ComputeAlignedBox()

    z_min = float(aligned.GetMin()[2])
    z_max = float(aligned.GetMax()[2])

    print(f"table_type: {args_cli.table_type}")
    print(f"usd_path: {table_usd_path}")
    print(f"spawn_translation_xyz: {table_translation}")
    print(f"spawn_scale_xyz: {table_scale}")
    print(f"world_aabb_z_min: {z_min:.6f}")
    print(f"world_aabb_z_max: {z_max:.6f}")
    print(f"table_top_height_world_z: {z_max:.6f}")

    sim.clear_all_callbacks()


def debug_object_world_z() -> None:
    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    minimal_height = 0.04
    lifting_term = getattr(env_cfg.rewards, "lifting_object", None)
    if lifting_term is not None and getattr(lifting_term, "params", None) is not None:
        minimal_height = float(lifting_term.params.get("minimal_height", minimal_height))
    env = gym.make(args_cli.task, cfg=env_cfg)
    obs, _ = env.reset()
    del obs  # unused; reset is needed to initialize buffers

    print(f"mode: object-z")
    print(f"task: {args_cli.task}")
    print(f"num_envs: {args_cli.num_envs}")
    print(f"lift_minimal_height: {minimal_height:.6f}")
    print("printing object.data.root_pos_w[:, 2] stats each step")
    reported_step1_crossings = False

    for step in range(args_cli.steps):
        with torch.no_grad():
            object_data = env.unwrapped.scene["object"].data
            object_pos_w = object_data.root_pos_w
            object_lin_vel_w = object_data.root_lin_vel_w
            object_z = object_pos_w[:, 2]
            z_min = float(torch.min(object_z))
            z_mean = float(torch.mean(object_z))
            z_max = float(torch.max(object_z))
            above_mask = object_z > minimal_height
            above_fraction = float(torch.mean(above_mask.float()))
        print(
            f"step={step:02d} z_min={z_min:.6f} z_mean={z_mean:.6f} "
            f"z_max={z_max:.6f} frac_above_min_height={above_fraction:.6f}"
        )

        # Print exact envs crossing threshold right after the first physics step.
        if step == 1 and not reported_step1_crossings:
            crossing_ids = torch.nonzero(above_mask, as_tuple=False).squeeze(-1)
            if crossing_ids.numel() == 0:
                print("step=01 crossing_envs: none")
            else:
                print(f"step=01 crossing_env_count: {int(crossing_ids.numel())}")
                for env_id in crossing_ids.tolist():
                    px, py, pz = object_pos_w[env_id].tolist()
                    vx, vy, vz = object_lin_vel_w[env_id].tolist()
                    print(
                        f"  env={env_id:03d} pos_w=({px:.6f}, {py:.6f}, {pz:.6f}) "
                        f"lin_vel_w=({vx:.6f}, {vy:.6f}, {vz:.6f})"
                    )
            reported_step1_crossings = True

        # Zero action keeps debugger simple; use manager dimension to avoid wrapper-space shape ambiguity.
        action_dim = env.unwrapped.action_manager.total_action_dim
        action = torch.zeros((env.unwrapped.num_envs, action_dim), device=env.unwrapped.device)
        env.step(action)

    env.close()


if __name__ == "__main__":
    if args_cli.mode == "table":
        print_table_height()
    else:
        debug_object_world_z()
    simulation_app.close()
