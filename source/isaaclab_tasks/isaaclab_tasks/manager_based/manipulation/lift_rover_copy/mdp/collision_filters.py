# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

LOGGER = logging.getLogger(__name__)


def filter_collisions_between_link_collision_subtrees(
    env: ManagerBasedEnv,
    env_ids,
    link_a_collisions_prim_path_expr: str,
    link_b_collisions_prim_path_expr: str,
    group_a_prim_path: str = "/World/CollisionGroups/no_contact_a",
    group_b_prim_path: str = "/World/CollisionGroups/no_contact_b",
) -> None:
    """Disable collisions between two sets of collider prims.

    This is implemented using USD Physics collision groups (filtered against each other).

    Notes:
    - The input prim path expressions may contain the IsaacLab env regex namespace placeholder
      ``{ENV_REGEX_NS}``. It will be expanded by the InteractiveScene before calling this function
      if used as an EventTerm (recommended).
    - The expressions should match collider prims (i.e., prims with CollisionAPI), not just Xforms.
    """
    # Import pxr only after the sim app is running (this function is called during env startup).
    from pxr import Usd, UsdPhysics

    stage = env.sim.stage

    # Expand IsaacLab env namespace placeholder if present.
    if "{ENV_REGEX_NS}" in link_a_collisions_prim_path_expr:
        link_a_collisions_prim_path_expr = link_a_collisions_prim_path_expr.format(ENV_REGEX_NS=env.scene.env_regex_ns)
    if "{ENV_REGEX_NS}" in link_b_collisions_prim_path_expr:
        link_b_collisions_prim_path_expr = link_b_collisions_prim_path_expr.format(ENV_REGEX_NS=env.scene.env_regex_ns)

    # Resolve subtree roots from regex, then collect all collider prims under them.
    # Some URDF->USD conversions place colliders under ".../collision" (singular), and some
    # put them directly under the link prim. Fall back to the link root when needed.
    a_root_prims = _resolve_collision_roots(link_a_collisions_prim_path_expr)
    b_root_prims = _resolve_collision_roots(link_b_collisions_prim_path_expr)
    a_paths = _collect_collision_prims_under(stage, a_root_prims)
    b_paths = _collect_collision_prims_under(stage, b_root_prims)
    if len(a_paths) == 0:
        LOGGER.warning(
            "Skipping collision filter: no collider prims matched for A expr=%r",
            link_a_collisions_prim_path_expr,
        )
        return
    if len(b_paths) == 0:
        LOGGER.warning(
            "Skipping collision filter: no collider prims matched for B expr=%r",
            link_b_collisions_prim_path_expr,
        )
        return

    # Define collision groups (idempotent).
    group_a = UsdPhysics.CollisionGroup.Define(stage, group_a_prim_path)
    group_b = UsdPhysics.CollisionGroup.Define(stage, group_b_prim_path)

    # Filter groups against each other.
    group_a.CreateFilteredGroupsRel().AddTarget(group_b.GetPath())
    group_b.CreateFilteredGroupsRel().AddTarget(group_a.GetPath())

    # Add colliders to the groups.
    # API compatibility across USD/Isaac Sim versions:
    # 1) CollisionGroup.CreateCollidersCollectionAPI
    # 2) Usd.CollectionAPI("colliders") on the group prim
    # 3) Legacy includes rel directly on schema
    includes_a = None
    includes_b = None
    if hasattr(group_a, "CreateCollidersCollectionAPI"):
        includes_a = group_a.CreateCollidersCollectionAPI().GetIncludesRel()
        includes_b = group_b.CreateCollidersCollectionAPI().GetIncludesRel()
    if includes_a is None or includes_b is None:
        colliders_a = Usd.CollectionAPI.Apply(group_a.GetPrim(), "colliders")
        colliders_b = Usd.CollectionAPI.Apply(group_b.GetPrim(), "colliders")
        includes_a = colliders_a.GetIncludesRel()
        includes_b = colliders_b.GetIncludesRel()
    if includes_a is None or includes_b is None:
        includes_a = group_a.GetIncludesRel() if hasattr(group_a, "GetIncludesRel") else None
        includes_b = group_b.GetIncludesRel() if hasattr(group_b, "GetIncludesRel") else None
    if includes_a is None or includes_b is None:
        raise RuntimeError("Unable to resolve includes relationship for collision groups on this USD schema version.")
    for p in a_paths:
        includes_a.AddTarget(p)
    for p in b_paths:
        includes_b.AddTarget(p)


def _collect_collision_prims_under(stage, root_prims) -> list[str]:
    """Collect prim paths under roots that have UsdPhysics.CollisionAPI."""
    from pxr import Usd, UsdPhysics

    out: list[str] = []
    for root in root_prims:
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                out.append(str(prim.GetPath()))
    # remove duplicates while preserving order
    out = list(dict.fromkeys(out))
    return out


def _resolve_collision_roots(collisions_prim_path_expr: str):
    """Resolve collision subtree roots with fallbacks for different USD layouts."""
    candidate_exprs = [collisions_prim_path_expr]

    # Fallback 1: some imports include an extra model layer under /Robot.
    if "/Robot/" in collisions_prim_path_expr:
        candidate_exprs.append(collisions_prim_path_expr.replace("/Robot/", "/Robot/.*/", 1))

    # Fallback 2: collisions -> collision
    singular_exprs: list[str] = []
    for expr in candidate_exprs:
        singular_expr = re.sub(r"/collisions$", "/collision", expr)
        if singular_expr != expr:
            singular_exprs.append(singular_expr)
    candidate_exprs.extend(singular_exprs)

    # Fallback 3: use link roots (colliders may be nested directly under the link).
    link_root_exprs = [re.sub(r"/collisions$|/collision$", "", expr) for expr in candidate_exprs]
    candidate_exprs.extend(link_root_exprs)

    # De-duplicate and return first successful match.
    for expr in dict.fromkeys(candidate_exprs):
        roots = sim_utils.find_matching_prims(expr)
        if len(roots) > 0:
            return roots

    return []

