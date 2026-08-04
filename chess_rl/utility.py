import random

import numpy as np
import torch as T

from chess_rl.config import DEVICE, SEED


def preprocess(state):
    state = T.from_numpy(state).permute(2, 0, 1).float().to(DEVICE)
    return state


def seed():
    random.seed(SEED)
    np.random.seed(SEED)
    T.manual_seed(SEED)
    if DEVICE.type == "cuda":
        T.cuda.manual_seed_all(SEED)


def apply_mask(logits, mask):
    mask = mask.bool()
    # Use -1e9 instead of -inf to keep backprop gradients finite and clean
    return T.where(
        mask, logits, T.tensor(-1e9, device=logits.device, dtype=logits.dtype)
    )
