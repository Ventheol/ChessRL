from pathlib import Path

import chess
import torch as T

# Initialization
SEED = 3007
CHECKPOINT_DIR = Path("checkpoints")
GAME_VID_DIR = Path("game_matches")
DEVICE = T.device("cuda" if T.cuda.is_available() else "cpu")
RENDER = False
# environment
SELF_PLAY = True
COLOR_TO_PLAY = chess.WHITE
RENDER_MODE = "human"
MAX_STEP = 500
# Rollout
BUFFER_CAPACITY = 256
ROLLOUT_STEPS = BUFFER_CAPACITY

# ppo
LEARNING_RATE = 3e-4
PPO_EPOCHS = 4
MINIBATCH_SIZE = 64
CLIP_PARAM = 0.2
GAMMA = 0.99
LAMBDA = 0.95
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
MAX_GRAD_NORM = 0.5

# network
NETWORK_INPUT = 18
NETWORK_HIDDEN = 64
NETWORK_OUTPUT = 64 * 64 * 4
SYNC_EVERY_N = 50
# training
LOG_EVERY = 25
NUM_UPDATES = 5000
SAVE_EVERY = 100
RENDER_EVERY = 10
LOG_DIR = Path("logs")

# MCTS
C_PUCT = 1.4
SIMULATIONS = 50
