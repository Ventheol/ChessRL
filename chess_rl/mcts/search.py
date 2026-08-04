import numpy as np

from chess_rl.mcts.node import Node


def backpropagate(leaf: Node, value):
    node = leaf
    while node is not None:
        node.visit_count += 1
        node.value_sum += value
        value = -value
        node = node.parent


def select(node: Node, c_puct=1.4):
    while node.children:
        node = max(node.children.values(), key=lambda child: child.puct_score(c_puct))
    return node


def expand(node: Node, priors):
    if node.children:
        return
    for move in node.board.legal_moves:
        child = Node(board=None, move=move, parent=node, prior=priors.get(move, 0.0))
        node.children[move] = child


def dirichlet_noise(root):
    moves = list(root.children.values())
    if not moves:
        return
    priors = np.array([c.prior for c in moves], dtype=np.float32)
    # Fixed: len(priors) instead of passing the array object
    noise = np.random.dirichlet([0.3] * len(priors))
    epsilon = 0.25
    priors = (1 - epsilon) * priors + epsilon * noise
    for child, p in zip(moves, priors):
        child.prior = float(p)


def search(root, evaluate):
    leaf = select(root)

    # Check for game end before running network evaluation
    if leaf.board.is_game_over():
        if leaf.board.is_checkmate():
            value = -1.0  # Player to move at leaf has been checkmated
        else:
            value = 0.0  # Draw
    else:
        value, priors = evaluate(leaf)
        expand(leaf, priors)
        if leaf == root:
            dirichlet_noise(root)

    backpropagate(leaf, value)
    return leaf
