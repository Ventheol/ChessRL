import chess
import imageio.v2 as imio
import numpy as np
import pygame as pyg

from chess_rl.config import GAME_VID_DIR, MAX_STEP

PROMOTION_MAP = {
    1: chess.KNIGHT,
    2: chess.BISHOP,
    3: chess.ROOK,
}

REV_PROMOTION_MAP = {
    chess.KNIGHT: 1,
    chess.BISHOP: 2,
    chess.ROOK: 3,
}


class ChessEnv:
    def __init__(
        self,
        self_play=True,
        color_to_play=chess.WHITE,
        render_mode="human",
        square_size=80,
        material_scale=0.01,
        step_penalty=-0.001,
    ):
        self.board = chess.Board()
        self.done = False
        self.pieces_idx = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.QUEEN: 3,
            chess.ROOK: 4,
            chess.KING: 5,
        }

        self.action_space_size = 64 * 64 * 4
        self.self_play = self_play
        self.color_to_play = color_to_play
        self.render_mode = render_mode

        self.square_size = square_size
        self.board_size = self.square_size * 8
        self.window = None
        self.clock = None

        self.current_step = 0
        self.material_scale = material_scale
        self.step_penalty = step_penalty
        self.piece_values = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.0,
        }
        self.frames = []
        GAME_VID_DIR.mkdir(parents=True, exist_ok=True)
        self.max_steps = MAX_STEP

    def reset(self):
        self.board = chess.Board()
        self.done = False
        self.current_step = 0
        return self.observation()

    def _get_material_balance(self, color):
        my_score = sum(
            len(self.board.pieces(pt, color)) * val
            for pt, val in self.piece_values.items()
        )
        opp_score = sum(
            len(self.board.pieces(pt, not color)) * val
            for pt, val in self.piece_values.items()
        )
        return my_score - opp_score

    def step(self, action_idx):
        if self.done:
            raise RuntimeError(
                "Environment has terminated. Call reset() before stepping."
            )

        self.current_step += 1
        move = self.action_to_move(action_idx)

        if move not in self.board.legal_moves:
            self.done = True
            return self.observation(), -1.0, True, {"reason": "illegal_move"}

        player = self.current_player(self.board)
        mat_before = self._get_material_balance(player)
        self.board.push(move)
        mat_after = self._get_material_balance(player)
        material_reward = (mat_after - mat_before) * self.material_scale
        reward = material_reward + self.step_penalty
        if self.board.is_repetition(2):
            reward -= 0.15
        result = None
        outcome = self.board.outcome(claim_draw=True)
        if outcome is not None:
            self.done = True
            result = outcome.result()
            if outcome.winner == player:
                reward += 1.0
            elif outcome.winner is not None:
                reward += -1.0
            else:
                if outcome.termination == chess.Termination.THREEFOLD_REPETITION:
                    reward += -0.2
                else:
                    reward += -0.1
        elif self.current_step >= self.max_steps:
            self.done = True
            reward += -0.1  # Treat hitting step limit as a draw penalty
            result = "1/2-1/2"
        return self.observation(), reward, self.done, {"result": result}

    def legal_move(self, board=None):
        board = board or self.board
        mask = np.zeros(self.action_space_size, dtype=np.int8)
        for move in board.legal_moves:
            mask[self.move_to_action(move)] = 1
        return mask

    def current_player(self, board):
        if self.self_play:
            my_color = board.turn
        else:
            my_color = self.color_to_play
        return my_color

    def observation(self, board=None):
        board = board or self.board
        state = np.zeros((8, 8, 18), dtype=np.float32)
        my_color = self.current_player(board)

        for square, piece in board.piece_map().items():
            if my_color == chess.BLACK:
                square = square ^ 56
            rank = chess.square_rank(square)
            file = chess.square_file(square)

            offset = 0 if piece.color == my_color else 6

            channel = self.pieces_idx[piece.piece_type] + offset

            state[rank, file, channel] = 1.0
        self.castling(state, board)
        self.en_passant(state, board)
        if board.is_repetition(2):
            state[:, :, 17] = 1.0
        elif board.is_repetition(1):
            state[:, :, 17] = 0.5
        return state

    def action_to_move(self, action_idx, board=None):
        board = board or self.board
        plane = action_idx // 4096
        rem = action_idx % 4096
        from_square = rem // 64
        to_square = rem % 64
        if self.current_player(board) == chess.BLACK:
            from_square = from_square ^ 56
            to_square = to_square ^ 56
        move = chess.Move(from_square, to_square)

        piece = board.piece_at(from_square)
        if piece and piece.piece_type == chess.PAWN:
            white_promo = (
                piece.color == chess.WHITE and chess.square_rank(to_square) == 7
            )
            black_promo = (
                piece.color == chess.BLACK and chess.square_rank(to_square) == 0
            )
            if white_promo or black_promo:
                move.promotion = PROMOTION_MAP.get(plane, chess.QUEEN)
        return move

    def move_to_action(self, move, board=None):
        board = board or self.board
        from_sq = move.from_square
        to_sq = move.to_square
        if self.current_player(board) == chess.BLACK:
            from_sq = from_sq ^ 56
            to_sq = to_sq ^ 56
        plane = REV_PROMOTION_MAP.get(move.promotion, 0)
        return plane * 4096 + (from_sq * 64 + to_sq)

    def castling(self, state, board):
        my_turn = self.current_player(board)
        opp = not my_turn

        state[:, :, 12] = board.has_kingside_castling_rights(my_turn)
        state[:, :, 13] = board.has_queenside_castling_rights(my_turn)
        state[:, :, 14] = board.has_kingside_castling_rights(opp)
        state[:, :, 15] = board.has_queenside_castling_rights(opp)

    def en_passant(self, state, board):
        if board.ep_square is None:
            return
        sq = board.ep_square
        if self.current_player(board) == chess.BLACK:
            sq = sq ^ 56
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)

        state[rank, file, 16] = 1.0

    def render(self, mode=None, fps=10):
        mode = mode or self.render_mode

        # Text console mode
        if mode == "ansi":
            print(self.board)
            print("-" * 20)
            return

            # Graphical pyg Window Mode
        if mode == "human":
            if self.window is None:
                pyg.init()
                pyg.display.init()
                pyg.display.set_caption("Chess RL Environment")
                self.window = pyg.display.set_mode((self.board_size, self.board_size))
                self.clock = pyg.time.Clock()
                try:
                    pyg.display.set_allow_screensaver(False)
                except Exception:
                    pass

            # Pump window events to keep the GUI responsive
            for event in pyg.event.get():
                if event.type == pyg.QUIT:
                    self.close()
                    return

            # Visual Theme Colors
            LIGHT_SQUARE = (238, 238, 210)
            DARK_SQUARE = (118, 150, 86)
            WHITE_PIECE_BG = (245, 245, 245)
            BLACK_PIECE_BG = (40, 40, 40)
            WHITE_PIECE_FG = (20, 20, 20)
            BLACK_PIECE_FG = (240, 240, 240)

            # 1. Draw Board Grid
            for rank in range(8):
                for file in range(8):
                    color = LIGHT_SQUARE if (rank + file) % 2 == 0 else DARK_SQUARE
                    # pyg (0,0) is top-left, so rank 7 (8th rank in chess) is drawn at y=0
                    rect = pyg.Rect(
                        file * self.square_size,
                        (7 - rank) * self.square_size,
                        self.square_size,
                        self.square_size,
                    )
                    pyg.draw.rect(self.window, color, rect)

            # 2. Draw Pieces
            font = pyg.font.SysFont("Arial", int(self.square_size * 0.35), bold=True)
            piece_symbols = {
                chess.PAWN: "P",
                chess.KNIGHT: "N",
                chess.BISHOP: "B",
                chess.ROOK: "R",
                chess.QUEEN: "Q",
                chess.KING: "K",
            }

            for square, piece in self.board.piece_map().items():
                file_idx = chess.square_file(square)
                rank_idx = chess.square_rank(square)

                center_x = int((file_idx + 0.5) * self.square_size)
                center_y = int((7 - rank_idx + 0.5) * self.square_size)
                radius = int(self.square_size * 0.38)

                bg_color = (
                    WHITE_PIECE_BG if piece.color == chess.WHITE else BLACK_PIECE_BG
                )
                fg_color = (
                    WHITE_PIECE_FG if piece.color == chess.WHITE else BLACK_PIECE_FG
                )

                # Draw piece circular badge
                pyg.draw.circle(self.window, bg_color, (center_x, center_y), radius)
                pyg.draw.circle(
                    self.window, (100, 100, 100), (center_x, center_y), radius, 2
                )

                # Draw piece letter label
                symbol = piece_symbols[piece.piece_type]
                text_surface = font.render(symbol, True, fg_color)
                text_rect = text_surface.get_rect(center=(center_x, center_y))
                self.window.blit(text_surface, text_rect)
            frame = pyg.surfarray.array3d(self.window)
            frame = np.transpose(frame, (1, 0, 2))
            frame = np.flipud(frame)
            self.frames.append(frame.copy())
            pyg.display.flip()
            self.clock.tick(fps)

    def save(self, update):
        if not self.frames:
            return
        path = GAME_VID_DIR / f"Game_{update:06d}_.mp4"
        writer = imio.get_writer(path, fps=12)
        for frame in self.frames:
            writer.append_data(frame)
        writer.close()
        self.frames.clear()

    def close(self):
        if self.window is not None:
            pyg.quit()
            self.window = None


if __name__ == "__main__":
    env = ChessEnv(render_mode="human")

    # Run 3 test matches with GUI rendering enabled
    for game in range(3):
        obs = env.reset()
        done = False
        print(f"--- Starting Game {game + 1} ---")

        while not done:
            mask = env.legal_move()
            legal_indices = np.where(mask == 1)[0]

            if len(legal_indices) == 0:
                break

            random_action = np.random.choice(legal_indices)
            next_obs, reward, done, info = env.step(random_action)

            # Render the GUI frame (fps=15 slows it down so you can watch)
            env.render(fps=15)

        print(f"Game ended! Result: {info.get('result', 'Terminated')}")

    env.close()
