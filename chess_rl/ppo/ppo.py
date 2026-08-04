# ppo.py

import copy
from datetime import datetime

import numpy as np
import torch as T
import torch.nn.functional as F
from torch.distributions import Categorical

from chess_rl.config import (
    BUFFER_CAPACITY,
    CHECKPOINT_DIR,
    COLOR_TO_PLAY,
    DEVICE,
    LEARNING_RATE,
    MAX_GRAD_NORM,
    MINIBATCH_SIZE,
    NETWORK_HIDDEN,
    NETWORK_INPUT,
    NETWORK_OUTPUT,
    PPO_EPOCHS,
    RENDER_MODE,
    SELF_PLAY,
)
from chess_rl.environment import chess_env
from chess_rl.logger import Logger
from chess_rl.mcts.evaluator import Evaluator
from chess_rl.mcts.mcts import MCTS
from chess_rl.mcts.parallel_rollout import run_parallel_rollout
from chess_rl.models import network
from chess_rl.ppo.buffer import RolloutBuffer
from chess_rl.ppo.gae import get_advantages
from chess_rl.utility import apply_mask, preprocess


class Agent:
    def __init__(self):
        self.env = chess_env.ChessEnv(self_play=SELF_PLAY, color_to_play=COLOR_TO_PLAY)
        self.network = network.ChessAgent(
            NETWORK_INPUT, NETWORK_HIDDEN, NETWORK_OUTPUT
        ).to(DEVICE)
        self.optimizer = T.optim.Adam(self.network.parameters(), lr=LEARNING_RATE)
        self.buffer = RolloutBuffer(BUFFER_CAPACITY)
        self.target_network = copy.deepcopy(self.network).to(DEVICE)
        self.target_network.eval()
        for param in self.target_network.parameters():
            param.requires_grad = False
        self.logger = Logger()
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self.losses = []
        self.actor_losses = []
        self.critic_losses = []
        self.entropies = []
        self.episode_returns = []
        self.current_ep_return = 0.0
        self.mcts = MCTS(Evaluator(self.network, self.env))

    def loss(self, target_pi, logits, values, returns):
        log_probs = F.log_softmax(logits, dim=-1)
        safe_log_probs = T.where(target_pi > 0, log_probs, T.zeros_like(log_probs))
        actor_loss = -(target_pi * safe_log_probs).sum(dim=-1).mean()

        critic_loss = F.mse_loss(values, returns)
        total_loss = actor_loss + critic_loss
        return total_loss, actor_loss, critic_loss

    def learn(self):
        self.network.train()
        data = self.buffer.get()
        if data["states"].size(0) == 0:
            return

        returns, advantages = get_advantages(
            rewards=data["rewards"],
            values=data["values"],
            dones=data["dones"],
            num_envs=4,
        )

        for _ in range(PPO_EPOCHS):
            for batch_idx, batch in self.buffer.minibatch(data, MINIBATCH_SIZE):
                logits, values = self.network(batch["states"])
                values = values.squeeze(-1)
                policy_logits = apply_mask(logits, batch["masks"])

                loss, actor_loss, critic_loss = self.loss(
                    batch["pis"], policy_logits, values, returns[batch_idx]
                )
                probs = F.softmax(policy_logits, dim=-1)
                log_probs = F.log_softmax(policy_logits, dim=-1)
                safe_log = T.where(probs > 0, log_probs, T.zeros_like(log_probs))
                entropy = -(probs * safe_log).sum(dim=-1).mean()

                self.optimizer.zero_grad()
                loss.backward()
                T.nn.utils.clip_grad_norm_(self.network.parameters(), MAX_GRAD_NORM)
                self.optimizer.step()

                self.losses.append(loss.item())
                self.actor_losses.append(actor_loss.item())
                self.critic_losses.append(critic_loss.item())
                self.entropies.append(entropy.item())

        self.buffer.clear()

    def _play_eval_game(self, net_white, net_black, max_steps=100, sims=25):
        """Play 1 evaluation game between net_white and net_black using greedy MCTS search."""
        eval_env = chess_env.ChessEnv()
        obs = eval_env.reset()
        done = False
        step = 0

        mcts_white = MCTS(Evaluator(net_white, eval_env), simulations=sims, c_puct=1.0)
        mcts_black = MCTS(Evaluator(net_black, eval_env), simulations=sims, c_puct=1.0)

        while not done and step < max_steps:
            current_mcts = mcts_white if eval_env.board.turn else mcts_black
            action, _, _ = current_mcts.run(
                eval_env.board, temperture=0.0
            )  # Greedy choice
            obs, reward, done, info = eval_env.step(action)
            step += 1

        board = eval_env.board
        if board.is_checkmate():
            return 1.0 if board.turn == False else 0.0  # Winner gets 1.0 point
        return 0.5  # Draw

    def check_and_update_target(self, update, sync_every_n=50, num_eval_games=10):
        """Every N updates, run a benchmark tournament. Update target if win_rate >= 55%."""
        if update % sync_every_n != 0:
            return

        print(f"\n=== [UPDATE {update}] GATED TOURNAMENT EVALUATION ===")
        self.network.eval()

        main_points = 0.0
        half_games = num_eval_games // 2

        # 1. Main Network plays as White
        for g in range(half_games):
            res = self._play_eval_game(self.network, self.target_network)
            main_points += res  # 1.0 = White win, 0.5 = Draw, 0.0 = Black win

        # 2. Main Network plays as Black
        for g in range(half_games):
            res = self._play_eval_game(self.target_network, self.network)
            main_points += 1.0 - res  # Reverse: 1.0 = Black win, 0.5 = Draw

        win_rate = main_points / num_eval_games
        print(
            f"Tournament Score: {main_points:.1f}/{num_eval_games} | Win Rate: {win_rate * 100:.1f}%"
        )

        if win_rate >= 0.55:
            print(
                ">> Target Passed Threshold (>= 55%)! Updating Frozen Target Network."
            )
            self.target_network.load_state_dict(
                copy.deepcopy(self.network.state_dict())
            )
            self.target_network.eval()
            for p in self.target_network.parameters():
                p.requires_grad = False
        else:
            print(">> Target Failed Threshold (< 55%). Target Network kept frozen.")
        print("====================================================\n")

    def log(self, update):
        loss = np.mean(self.losses) if self.losses else 0.0
        actor_loss = np.mean(self.actor_losses) if self.actor_losses else 0.0
        critic_loss = np.mean(self.critic_losses) if self.critic_losses else 0.0
        entropy = np.mean(self.entropies) if self.entropies else 0.0
        avg_return = np.mean(self.episode_returns) if self.episode_returns else 0.0

        self.logger.log_training(
            update, loss, actor_loss, critic_loss, entropy, avg_return
        )
        print(
            f"[{update:6d}] "
            f"Loss {loss:.4f} | "
            f"Actor {actor_loss:.4f} | "
            f"Critic {critic_loss:.4f} | "
            f"Entropy {entropy:.3f} | "
            f"Reward {avg_return:.3f}"
        )
        self.losses.clear()
        self.actor_losses.clear()
        self.critic_losses.clear()
        self.entropies.clear()
        self.episode_returns.clear()

    def rollout_step(self):
        self.network.eval()
        env_transitions, last_values = run_parallel_rollout(self, num_envs=4)

        for env_idx, transitions in enumerate(env_transitions):
            for obs, action, action_probs, reward, done, mask, val in transitions:
                self.current_ep_return += reward

                self.buffer.add(
                    states=preprocess(obs),
                    pis=action_probs,
                    rewards=reward,
                    dones=done,
                    values=T.tensor(val, dtype=T.float32, device=DEVICE),
                    masks=T.from_numpy(mask).to(DEVICE),
                )
                if done:
                    self.episode_returns.append(self.current_ep_return)
                    self.current_ep_return = 0.0

            last_val_tensor = T.tensor(
                last_values[env_idx], dtype=T.float32, device=DEVICE
            )
            self.buffer.add_last_value(last_val_tensor)

    def act(self, state, mask):

        state = state.to(DEVICE).unsqueeze(0)
        mask = mask.unsqueeze(0)
        with T.no_grad():
            logits, values = self.network(state)
        values = values.squeeze(-1)

        policy_logits = apply_mask(logits, mask)

        dist = Categorical(logits=policy_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.item(), log_prob.squeeze(0), values.squeeze()

    def evaluate(self, states, actions, masks):

        logits, values = self.network(states)

        values = values.squeeze(-1)
        policy_logits = apply_mask(logits, masks)

        dist = Categorical(logits=policy_logits)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()

        return (log_prob, values, entropy)

    def save(self, update=None, update_step="policy"):
        is_policy = True
        if update_step == "pretrained":
            update = 0
            is_policy = False
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint = {
            "model": self.network.state_dict(),
            "target_model": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "timestamp": timestamp,
            "update": update,
        }
        path = (
            CHECKPOINT_DIR
            / f"chess_ppo_{'policy' if is_policy else 'pre_trained'}_{timestamp}.pt"
        )
        T.save(checkpoint, path)
        print(f"Saved Checkpoint: {path}")

    def load(self, file_loc=None):
        if file_loc is None:
            files = sorted(CHECKPOINT_DIR.glob("chess_ppo_*.pt"))
            if not files:
                raise FileNotFoundError("No Checkpoints found.")
            file_loc = files[-1]

        print(f"Loading Checkpoint : {file_loc}")
        checkpoint = T.load(file_loc, map_location=DEVICE)

        self.network.load_state_dict(checkpoint["model"])
        if "target_model" in checkpoint:
            self.target_network.load_state_dict(checkpoint["target_model"])
        else:
            self.target_network.load_state_dict(checkpoint["model"])

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.network.eval()
        self.target_network.eval()
        print("Loaded Checkpoint Successfully.")
        return checkpoint, checkpoint["timestamp"], checkpoint["update"]

    @T.no_grad()
    def select_eval_action(self, obs, mask):
        state = preprocess(obs).unsqueeze(0).to(DEVICE)
        mask_tensor = T.from_numpy(mask).to(DEVICE).unsqueeze(0)

        logits, _ = self.network(state)
        policy_logits = apply_mask(logits, mask_tensor)
        action = T.argmax(policy_logits, dim=-1).item()
        return action

    def render(self, update, fps=12, max_steps=200, vs_target=True):
        print(f"Rendering Match (Main vs {'Target' if vs_target else 'Self'})...")
        self.network.eval()
        self.target_network.eval()

        eval_env = chess_env.ChessEnv(render_mode=RENDER_MODE)
        obs = eval_env.reset()
        done = False
        step = 0

        evaluator_main = Evaluator(self.network, eval_env)
        mcts_main = MCTS(evaluator_main, simulations=50, c_puct=1.0)

        if vs_target:
            evaluator_target = Evaluator(self.target_network, eval_env)
            mcts_target = MCTS(evaluator_target, simulations=50, c_puct=1.0)

        while not done and step < max_steps:
            # White turn = Main Network; Black turn = Target Network (or Main if vs_target=False)
            if eval_env.board.turn:
                current_mcts = mcts_main
            else:
                current_mcts = mcts_target if vs_target else mcts_main

            # Greedy choice (temperature = 0.0)
            action, _, _ = current_mcts.run(eval_env.board, temperture=0.0)
            obs, reward, done, info = eval_env.step(action)
            eval_env.render(fps=fps)
            step += 1

        board = eval_env.board
        outcome = board.outcome()

        if outcome is not None:
            reason = outcome.termination.name
        else:
            reason = f"Hit max step limit ({max_steps})"

        if board.is_fifty_moves():
            print("Draw Reason: 50-Move Rule")
        elif board.is_stalemate():
            print("Draw Reason: Stalemate")
        elif board.is_repetition():
            print("Draw Reason: Threefold Repetition")
        elif board.is_insufficient_material():
            print("Draw Reason: Insufficient Material")

        eval_env.save(update - 1)
        result = info.get("result", "Game Limit Reached")
        print(f"Game Finished Result: {result} | Reason: {reason}\n")
        eval_env.close()
