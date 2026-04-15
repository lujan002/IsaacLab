import torch
import gymnasium as gym

from isaaclab.app import AppLauncher

# Launch Isaac Sim app first so pxr and simulation extensions are available.
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# Import registers all tasks (after app launch).
import isaaclab_tasks  # noqa: E402,F401


def _get_policy(obs):
    if isinstance(obs, dict):
        return obs.get("policy", None)
    return obs


def main():
    task = "Isaac-Lift-Cube-ArticulatedArmRev2-v0"

    env = gym.make(task, num_envs=32, headless=True)
    obs, info = env.reset()
    policy = _get_policy(obs)
    print("policy obs shape:", tuple(policy.shape) if policy is not None else None)
    print("reset nan:", torch.isnan(policy).sum().item(), "inf:", torch.isinf(policy).sum().item())
    print("\nPer-term NaN/Inf at reset:")
    _print_term_nan_stats(env)

    device = env.unwrapped.device
    action_dim = env.action_space.shape[0]
    action = torch.zeros((env.num_envs, action_dim), device=device)

    for i in range(10):
        obs, rew, done, trunc, info = env.step(action)
        policy = _get_policy(obs)
        print(
            f"step {i:02d} | obs nan {torch.isnan(policy).sum().item()} inf {torch.isinf(policy).sum().item()} "
            f"| rew nan {torch.isnan(rew).sum().item()} inf {torch.isinf(rew).sum().item()}"
        )
        if i == 0:
            print("\nPer-term NaN/Inf after first step:")
            _print_term_nan_stats(env)

    env.close()


def _print_term_nan_stats(env):
    obs_mgr = env.unwrapped.observation_manager
    group = "policy"
    names = obs_mgr._group_obs_term_names[group]
    cfgs = obs_mgr._group_obs_term_cfgs[group]
    for name, cfg in zip(names, cfgs):
        term = cfg.func(env.unwrapped, **cfg.params)
        n_nan = torch.isnan(term).sum().item()
        n_inf = torch.isinf(term).sum().item()
        shape = tuple(term.shape)
        print(f"  - {name:24s} shape={shape} nan={n_nan} inf={n_inf}")


if __name__ == "__main__":
    main()
    simulation_app.close()

