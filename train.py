from tqdm import tqdm

from chess_rl.config import (
    DEVICE,
    LOG_EVERY,
    MAX_STEP,
    NUM_UPDATES,
    RENDER,
    RENDER_EVERY,
    SAVE_EVERY,
    SYNC_EVERY_N,
)
from chess_rl.ppo.ppo import Agent
from chess_rl.utility import seed


def train():
    print("Initializing Seed...")
    seed()
    print("Creating Agent...")
    agent = Agent()
    print(f"Training Device: {DEVICE}")
    print("Starting Training...")
    progress_bar = tqdm(
        range(1, NUM_UPDATES + 1), desc="Training Updates", unit="update"
    )

    for update in progress_bar:
        agent.rollout_step()
        agent.learn()
        agent.check_and_update_target(
            update, sync_every_n=SYNC_EVERY_N, num_eval_games=10
        )
        if update % SAVE_EVERY == 0:
            agent.save(update)
        if update % LOG_EVERY == 0:
            agent.log(update)

        if RENDER and (update % RENDER_EVERY) == 0:
            agent.render(update, max_steps=MAX_STEP)


if __name__ == "__main__":
    train()
