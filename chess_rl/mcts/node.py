import math

import chess

from chess_rl.config import C_PUCT


class Node:
    def __init__(self, board, move=None, parent=None, prior=0.0, c_puct=C_PUCT):
        self._board: chess.Board = board
        self.move: chess.Move = move
        self.parent = parent
        self.children = {}

        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.c_puct = c_puct

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    @property
    def board(self):
        if self._board is None:
            self._board = self.parent.board.copy()
            self._board.push(self.move)
        return self._board

    def __repr__(self):
        return (
            f"Node(move={self.move}, visits={self.visit_count}, value={self.value:.3f})"
        )

    def puct_score(self, c_puct=C_PUCT):
        parent_visits = 1 if self.parent is None else self.parent.visit_count
        Q = -self.value if self.visit_count > 0 else 0.0
        U = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)

        return Q + U

    def select_child(self):
        best_child = max(
            self.children.values(), key=lambda child: child.puct_score(self.c_puct)
        )
        return best_child


if __name__ == "__main__":
    board = chess.Board()
    root = Node(board)
    policy_probability = 0.0
    for move in board.legal_moves:
        child_board = board.copy()
        child_board.push(move)

        child = Node(
            board=child_board,
            move=move,
            parent=root,
            prior=policy_probability,
        )

        root.children[move] = child
