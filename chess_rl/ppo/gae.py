import torch as T

from chess_rl.config import DEVICE, GAMMA, LAMBDA


def get_advantages(rewards, values, dones, num_envs=4):
    total_samples = rewards.size(0)
    steps_per_env = total_samples // num_envs

    # Reshape rewards and dones to (num_envs, steps_per_env)
    rewards_2d = rewards.view(num_envs, steps_per_env)
    dones_2d = dones.view(num_envs, steps_per_env)

    # values_2d has shape (num_envs, steps_per_env + 1)
    values_2d = values.view(num_envs, steps_per_env + 1)

    returns_2d = T.zeros_like(rewards_2d, device=DEVICE)

    for env_idx in range(num_envs):
        gae = 0
        r = rewards_2d[env_idx]
        d = dones_2d[env_idx]
        v = values_2d[env_idx]

        not_dones = 1.0 - d.float()

        for t in reversed(range(steps_per_env)):
            # In self-play, next state value is negated (-v[t + 1])
            delta = r[t] + GAMMA * (-v[t + 1]) * not_dones[t] - v[t]
            gae = delta + GAMMA * LAMBDA * not_dones[t] * gae
            returns_2d[env_idx, t] = gae + v[t]

    returns = returns_2d.view(-1)
    v_current = values_2d[:, :-1].reshape(-1)

    adv = returns - v_current
    adv_normalized = (adv - adv.mean()) / (adv.std() + 1e-10)

    return returns, adv_normalized
