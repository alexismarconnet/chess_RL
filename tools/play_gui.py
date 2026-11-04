import sys
import os
import tkinter as tk
from tkinter import messagebox
import chess

# Ajouter la racine du projet au path pour pouvoir importer environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environment import ChessEnvironment

PIECE_UNICODE = {
    'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
    'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
}

class ChessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Chess RL - Mode Local (Humain vs Humain)")
        
        self.env = ChessEnvironment()
        self.env.reset()

        self.selected_sq = None
        self.legal_destinations = []

        self.create_widgets()
        self.update_board_display()

    def create_widgets(self):
        self.info_label = tk.Label(self.root, text="Tour : Blancs", font=("Arial", 14, "bold"))
        self.info_label.pack(pady=10)

        self.board_frame = tk.Frame(self.root, bg="#2b2b2b", bd=4)
        self.board_frame.pack(padx=10, pady=10)

        self.buttons = [[None for _ in range(8)] for _ in range(8)]
        
        for r in range(8):
            for c in range(8):
                bg_color = "#f0d9b5" if (r + c) % 2 == 0 else "#b58863"
                btn = tk.Button(
                    self.board_frame, text="", font=("Arial", 28), width=2, height=1,
                    bg=bg_color, activebackground="#aaa",
                    command=lambda row=r, col=c: self.on_square_click(row, col)
                )
                btn.grid(row=r, column=c, sticky="nsew")
                self.buttons[r][c] = btn

        self.reset_btn = tk.Button(self.root, text="Nouvelle Partie", font=("Arial", 12), command=self.reset_game)
        self.reset_btn.pack(pady=10)

    def on_square_click(self, r: int, c: int):
        rank = 7 - r
        file = c
        clicked_sq = chess.square(file, rank)
        board = self.env.board

        if self.selected_sq is None:
            piece = board.piece_at(clicked_sq)
            if piece is not None and piece.color == board.turn:
                self.selected_sq = clicked_sq
                self.legal_destinations = [
                    m.to_square for m in board.legal_moves if m.from_square == clicked_sq
                ]
                self.highlight_board()
        else:
            from_sq = self.selected_sq
            to_sq = clicked_sq

            move = None
            for m in board.legal_moves:
                if m.from_square == from_sq and m.to_square == to_sq:
                    move = m
                    break

            self.selected_sq = None
            self.legal_destinations = []

            if move is not None:
                try:
                    _, _, done, info = self.env.step(move)
                    self.update_board_display()
                    if done:
                        messagebox.showinfo("Fin de partie", f"Résultat : {info.get('result', 'Terminée')}")
                except Exception as e:
                    messagebox.showerror("Erreur", str(e))
            else:
                piece = board.piece_at(clicked_sq)
                if piece is not None and piece.color == board.turn:
                    self.selected_sq = clicked_sq
                    self.legal_destinations = [
                        m.to_square for m in board.legal_moves if m.from_square == clicked_sq
                    ]
                    self.highlight_board()
                    return

            self.update_board_display()

    def highlight_board(self):
        for r in range(8):
            for c in range(8):
                rank = 7 - r
                file = c
                sq = chess.square(file, rank)
                base_color = "#f0d9b5" if (r + c) % 2 == 0 else "#b58863"
                if sq == self.selected_sq:
                    base_color = "#baca44"
                elif sq in self.legal_destinations:
                    base_color = "#f7ec7d"
                self.buttons[r][c].config(bg=base_color)

    def update_board_display(self):
        board = self.env.board
        for r in range(8):
            for c in range(8):
                rank = 7 - r
                file = c
                sq = chess.square(file, rank)
                piece = board.piece_at(sq)
                symbol = PIECE_UNICODE[piece.symbol()] if piece else ""
                base_color = "#f0d9b5" if (r + c) % 2 == 0 else "#b58863"
                self.buttons[r][c].config(text=symbol, bg=base_color)

        turn_text = "Tour : Blanches (♔)" if board.turn == chess.WHITE else "Tour : Noires (♚)"
        if board.is_check():
            turn_text += " - ÉCHEC !"
        self.info_label.config(text=turn_text)

    def reset_game(self):
        self.env.reset()
        self.selected_sq = None
        self.legal_destinations = []
        self.update_board_display()

def main():
    root = tk.Tk()
    app = ChessGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
