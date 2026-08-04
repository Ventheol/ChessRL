import numpy as np
import torch as T
from torch.amp import autocast

from chess_rl.config import DEVICE, ROLLOUT_STEPS, SIMULATIONS
from chess_rl.environment.chess_env import ChessEnv
from chess_rl.mcts.node import Node
from chess_rl.mcts.search import backpropagate, dirichlet_noise, expand, select
from chess_rl.utility import apply_mask


def run_parallel_rollout(agent, num_envs=4):
    """Vectorized self-play rollout across parallel environments using the primary network."""
    agent.network.eval()
    envs = [ChessEnv() for _ in range(num_envs)]
    obss = [env.reset() for env in envs]

    env_transitions = [[] for _ in range(num_envs)]
    steps_per_env = ROLLOUT_STEPS // num_envs

    for _ in range(steps_per_env):
        roots = [Node(env.board.copy()) for env in envs]

        # 1. MCTS Search
        for _ in range(SIMULATIONS):
            leaves = [select(root) for root in roots]

            non_terminal_indices = []
            terminal_vals = {}

            for idx, leaf in enumerate(leaves):
                if leaf.board.is_game_over():
                    val = -1.0 if leaf.board.is_checkmate() else 0.0
                    terminal_vals[idx] = val
                else:
                    non_terminal_indices.append(idx)

            if non_terminal_indices:
                obs_batch = [
                    envs[i].observation(leaves[i].board) for i in non_terminal_indices
                ]
                mask_batch = [
                    envs[i].legal_move(leaves[i].board) for i in non_terminal_indices
                ]

                states_t = (
                    T.from_numpy(np.stack(obs_batch))
                    .permute(0, 3, 1, 2)
                    .float()
                    .to(DEVICE)
                )
                masks_t = T.from_numpy(np.stack(mask_batch)).to(DEVICE)

                with T.no_grad(), autocast(DEVICE.type):
                    logits, values = agent.network(states_t)
                    logits = apply_mask(logits, masks_t)
                    probs = T.softmax(logits, dim=-1).cpu().numpy()
                    values = values.squeeze(-1).cpu().numpy()

                for k, idx in enumerate(non_terminal_indices):
                    leaf = leaves[idx]
                    leaf_board = leaf.board
                    priors = {
                        m: float(
                            probs[k, envs[idx].move_to_action(m, board=leaf_board)]
                        )
                        for m in leaf_board.legal_moves
                    }
                    expand(leaf, priors)
                    if leaf == roots[idx]:
                        dirichlet_noise(roots[idx])
                    backpropagate(leaf, float(values[k]))

            for idx, val in terminal_vals.items():
                backpropagate(leaves[idx], val)

        # 2. Extract action and step environment
        for i, env in enumerate(envs):
            root = roots[i]
            action_probs = np.zeros(16384, dtype=np.float32)
            for move, child in root.children.items():
                action_probs[env.move_to_action(move, board=env.board)] = (
                    child.visit_count
                )

            sum_visits = np.sum(action_probs)
            if sum_visits > 0:
                action_probs /= sum_visits

            probs_sum = np.sum(action_probs)
            action = (
                int(np.random.choice(len(action_probs), p=action_probs))
                if probs_sum > 0
                else 0
            )

            mask = env.legal_move(env.board)
            next_obs, reward, done, info = env.step(action)

            env_transitions[i].append(
                (obss[i], action, action_probs, reward, done, mask, root.value)
            )

            obss[i] = env.reset() if done else next_obs

    # 3. Bootstrap values for remaining states
    with T.no_grad():
        last_states = T.stack(
            [T.from_numpy(obss[i]).permute(2, 0, 1).float() for i in range(num_envs)]
        ).to(DEVICE)
        _, last_values = agent.network(last_states)
        last_values = last_values.squeeze(-1).cpu().tolist()

    return env_transitions, last_values
