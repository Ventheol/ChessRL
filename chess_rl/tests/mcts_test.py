from chess_rl.config import DEVICE, NETWORK_HIDDEN, NETWORK_INPUT, NETWORK_OUTPUT
from chess_rl.env.chess_env import ChessEnv
from chess_rl.mcts.evaluator import Evaluator
from chess_rl.mcts.mcts import MCTS
from chess_rl.models.network import ChessAgent

env = ChessEnv()

network = ChessAgent(NETWORK_INPUT, NETWORK_HIDDEN, NETWORK_OUTPUT).to(DEVICE)

evaluator = Evaluator(network, env)

mcts = MCTS(evaluator, simulations=50)
env.reset()
board = env.board
move, root = mcts.run(board)

print("Best move:", move)
for child in root.children.values():
    print(
        child.move,
        child.visit_count,
        round(child.value, 3),
        round(child.prior, 3),
    )
