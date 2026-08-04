import glob
import os

import chess
import chess.pgn as pgn
import torch as T

from chess_rl.environment.chess_env import ChessEnv


def process_pgn_files(pgn_folder="data", min_elo=2000, max_samples=200_000):
    env = ChessEnv()

    states = []
    actions = []
    masks = []
    values = []

    pgn_files = glob.glob(os.path.join(pgn_folder, "*.pgn"))
    print(f"Found {len(pgn_files)} PGN in '{pgn_folder}' directory.")
    total_moves_extracted = 0

    for pgn_path in pgn_files:
        print(f"Processing: {pgn_path}")
        with open(pgn_path, encoding="utf-8", errors="ignore") as f:
            while True:
                game = pgn.read_game(f)
                if game is None:
                    break
                result = game.headers.get("Result", "*")
                if result not in ["1-0", "0-1"]:
                    continue

                wining_color = chess.WHITE if result == "1-0" else chess.BLACK

                board = game.board()

                for move in game.mainline_moves():
                    if board.turn == wining_color:
                        env.board = board.copy()
                        obs = env.observation()
                        obs = obs.transpose(2, 0, 1)
                        mask = env.legal_move(board)
                        try:
                            action_idx = env.move_to_action(move, board=board)

                            states.append(obs)
                            actions.append(action_idx)
                            masks.append(mask)
                            values.append(1.0)
                            total_moves_extracted += 1
                        except ValueError:
                            pass
                    board.push(move)

                    if total_moves_extracted >= max_samples:
                        break
                if total_moves_extracted >= max_samples:
                    print("Reached Maximum requested Samples")
                    break
    print(f"Successfuly Extracted {total_moves_extracted} (State,Action,Value) tuples")

    states_t = T.tensor(states, dtype=T.float32)
    actions_t = T.tensor(actions, dtype=T.long)
    masks_t = T.tensor(masks, dtype=T.bool)
    values_t = T.tensor(values, dtype=T.float32).unsqueeze(1)

    os.makedirs("data", exist_ok=True)
    save_path = "data/pgn_dataset.pt"

    T.save(
        {
            "states": states_t,
            "actions": actions_t,
            "masks": masks_t,
            "values": values_t,
        },
        save_path,
    )
    print(
        f"Dataset saved to '{save_path}' ({states_t.element_size() * states_t.nelement() / 1e6:.1f} MB)"
    )


if __name__ == "__main__":
    process_pgn_files(pgn_folder="data", max_samples=300000)
