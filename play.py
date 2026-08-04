import sys

import chess
import pygame as pyg

from chess_rl.mcts.evaluator import Evaluator
from chess_rl.mcts.mcts import MCTS
from chess_rl.ppo.ppo import Agent


class ChessGUI:
    def __init__(self, agent_simulations=100, board_size=600):
        pyg.init()
        pyg.font.init()

        self.board_size = board_size
        self.square_size = board_size // 8
        self.simulations = agent_simulations

        # Load AI Agent
        print("Loading Trained Chess Agent...")
        self.agent = Agent()
        try:
            self.agent.load()
        except FileNotFoundError:
            print("No checkpoint found! Make sure you have trained a model first.")
            sys.exit(1)

        self.env = self.agent.env
        self.env.reset()
        self.board = self.env.board

        self.evaluator = Evaluator(self.agent.network, self.env)
        self.mcts = MCTS(self.evaluator, simulations=self.simulations, c_puct=1.0)

        # Initialize Pygame Window
        pyg.display.set_caption("Chess RL - Human (White) vs AI Agent (Black)")
        self.window = pyg.display.set_mode((self.board_size, self.board_size))
        self.clock = pyg.time.Clock()

        try:
            pyg.display.set_allow_screensaver(False)
        except Exception:
            pass

        # Interactive state
        self.selected_square = None
        self.legal_destinations = []

    def get_square_from_pos(self, pos):
        """Converts pixel (x, y) on window to chess square (0..63)."""
        x, y = pos
        file_idx = x // self.square_size
        rank_idx = 7 - (y // self.square_size)
        return chess.square(file_idx, rank_idx)

    def draw_board(self):
        # Visual Theme Colors
        LIGHT_SQUARE = (238, 238, 210)
        DARK_SQUARE = (118, 150, 86)
        HIGHLIGHT_SRC = (246, 246, 105)
        HIGHLIGHT_DEST = (186, 202, 68)
        WHITE_PIECE_BG = (245, 245, 245)
        BLACK_PIECE_BG = (40, 40, 40)
        WHITE_PIECE_FG = (20, 20, 20)
        BLACK_PIECE_FG = (240, 240, 240)

        # 1. Draw Board Grid & Highlights
        for rank in range(8):
            for file in range(8):
                sq = chess.square(file, rank)

                # Check highlights
                if sq == self.selected_square:
                    color = HIGHLIGHT_SRC
                elif sq in self.legal_destinations:
                    color = HIGHLIGHT_DEST
                else:
                    color = LIGHT_SQUARE if (rank + file) % 2 == 0 else DARK_SQUARE

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

            bg_color = WHITE_PIECE_BG if piece.color == chess.WHITE else BLACK_PIECE_BG
            fg_color = WHITE_PIECE_FG if piece.color == chess.WHITE else BLACK_PIECE_FG

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

        pyg.display.flip()

    def handle_human_click(self, pos):
        """Processes human user clicking on the Pygame board."""
        clicked_sq = self.get_square_from_pos(pos)

        # 1. If clicking on own piece, select it
        piece = self.board.piece_at(clicked_sq)
        if piece and piece.color == chess.WHITE:
            self.selected_square = clicked_sq
            self.legal_destinations = [
                m.to_square
                for m in self.board.legal_moves
                if m.from_square == clicked_sq
            ]
            return

        # 2. If a piece is selected and valid destination clicked, execute move
        if self.selected_square is not None:
            move = chess.Move(self.selected_square, clicked_sq)

            # Handle Auto-Queen Promotion for pawns hitting 8th rank
            pawn_promote = chess.Move(
                self.selected_square, clicked_sq, promotion=chess.QUEEN
            )

            if move in self.board.legal_moves:
                self.board.push(move)
            elif pawn_promote in self.board.legal_moves:
                self.board.push(pawn_promote)

            self.selected_square = None
            self.legal_destinations = []

    def run(self):
        running = True

        while running and not self.board.is_game_over():
            self.draw_board()

            # --- AI MOVE TURN ---
            if self.board.turn == chess.BLACK:
                pyg.display.set_caption(
                    f"Chess RL - AI is thinking ({self.simulations} sims)..."
                )

                # Execute MCTS Search greedy move
                action, _, _ = self.mcts.run(self.board, temperture=0.0)
                agent_move = self.env.action_to_move(action, board=self.board)
                self.board.push(agent_move)

                pyg.display.set_caption("Chess RL - Your Turn (White)")
                continue

            # --- HUMAN EVENT LOOP ---
            for event in pyg.event.get():
                if event.type == pyg.QUIT:
                    running = False
                    break
                elif event.type == pyg.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_human_click(event.pos)

            self.clock.tick(30)

        # Game Over Screen
        self.draw_board()
        outcome = self.board.outcome()
        if outcome:
            reason = outcome.termination.name
            result = self.board.result()
            print(f"\nGame Over! Result: {result} | Reason: {reason}")
            pyg.display.set_caption(f"GAME OVER - {result} ({reason})")
            if (reason == chess.Termination.CHECKMATE.name) or (
                reason == chess.Termination.STALEMATE.name
            ):
                running = False

        # Keep window open until user closes it
        while running:
            for event in pyg.event.get():
                if event.type == pyg.QUIT:
                    running = False

        pyg.quit()


if __name__ == "__main__":
    gui = ChessGUI(agent_simulations=500)
    gui.run()
