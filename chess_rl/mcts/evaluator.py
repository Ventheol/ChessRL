import torch as T
from torch.amp import autocast

from chess_rl.config import DEVICE
from chess_rl.mcts.node import Node
from chess_rl.models.network import ChessAgent
from chess_rl.utility import apply_mask, preprocess


class Evaluator:
    def __init__(self, network: ChessAgent, env):
        self.network = network

        self.env = env

    @T.no_grad()
    def evaluate(self, node: Node):
        self.network.eval()
        obs = self.env.observation(node.board)
        board = preprocess(obs).unsqueeze(0)
        mask = self.env.legal_move(node.board)
        mask_tensor = T.from_numpy(mask).to(DEVICE).unsqueeze(0)
        with autocast(DEVICE.type):
            logits, value = self.network(board)
            logits = apply_mask(logits, mask_tensor)

        policy = T.softmax(logits, dim=-1)
        priors = {}
        for move in node.board.legal_moves:
            idx = self.env.move_to_action(move)
            priors[move] = policy[0, idx].item()
        value = value.squeeze()
        return value.item(), priors
