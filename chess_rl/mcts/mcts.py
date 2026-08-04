import numpy as np

from chess_rl.config import C_PUCT, SIMULATIONS
from chess_rl.mcts.evaluator import Evaluator
from chess_rl.mcts.node import Node
from chess_rl.mcts.search import search


class MCTS:
    def __init__(self, evaluator, simulations=SIMULATIONS, c_puct=C_PUCT):
        self.evaluator: Evaluator = evaluator
        self.simulations = simulations
        self.c_puct = c_puct

    def run(self, board, temperture=1.0):
        root = Node(board.copy())

        for _ in range(self.simulations):
            search(root, self.evaluator.evaluate)
        action_probs = np.zeros(64 * 64 * 4, dtype=np.float32)
        for move, child in root.children.items():
            action_idx = self.evaluator.env.move_to_action(move)
            action_probs[action_idx] = child.visit_count
        sum_visits = np.sum(action_probs)
        if sum_visits > 0:
            action_probs /= sum_visits
        if temperture == 0:
            action = int(np.argmax(action_probs))
        else:
            probs = action_probs ** (1.0 / temperture)
            probs_sum = np.sum(probs)
            if probs_sum > 0:
                probs /= probs_sum
            else:
                probs = action_probs
            action = int(np.random.choice(len(probs), p=probs))
        return action, action_probs, root.value
