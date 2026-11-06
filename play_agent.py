import argparse
import os
import sys
import tkinter as tk
from tkinter import messagebox
import chess
from PIL import Image, ImageTk

# Assurer l'accès aux modules du projet
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from environment import ChessEnvironment
from agent import GreedyAgent, EpsilonGreedyAgent, UCBAgent, MCTSAgent

DEFAULT_MODEL_PATH = "saved_models/chess_agent.pt"

PIECE_FILES = {
    'P': 'w_pawn.png', 'N': 'w_knight.png', 'B': 'w_bishop.png', 'R': 'w_rook.png', 'Q': 'w_queen.png', 'K': 'w_king.png',
    'p': 'b_pawn.png', 'n': 'b_knight.png', 'b': 'b_bishop.png', 'r': 'b_rook.png', 'q': 'b_queen.png', 'k': 'b_king.png'
}

PIECE_LABELS = {
    'P': 'P', 'N': 'N', 'B': 'B', 'R': 'R', 'Q': 'Q', 'K': 'K',
    'p': 'p', 'n': 'n', 'b': 'b', 'r': 'r', 'q': 'q', 'k': 'k'
}

class ChessAgentGUI:
    def __init__(self, root, agent, human_color):
        self.root = root
        self.agent = agent
        self.human_color = human_color
        
        title_color = "Blancs" if human_color == chess.WHITE else "Noirs"
        self.root.title(f"Chess RL - Humain ({title_color}) vs Agent AI")
        self.root.geometry("600x720")
        self.root.minsize(450, 500)

        self.env = ChessEnvironment()
        self.env.reset()

        self.selected_sq = None
        self.legal_destinations = []
        
        self.images = {}
        self.loaded_photos = {}
        self.load_piece_images()

        self.create_widgets()
        
        # Si l'humain est Noir, l'IA commence
        if self.env.board.turn != self.human_color:
            self.root.after(500, self.ai_move)

    def load_piece_images(self):
        assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "pieces")
        for symbol, filename in PIECE_FILES.items():
            path = os.path.join(assets_dir, filename)
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert("RGBA")
                    self.images[symbol] = img
                except Exception as e:
                    print(f"Erreur chargement image {filename}: {e}")

    def create_widgets(self):
        self.info_label = tk.Label(self.root, text="Tour : Blancs", font=("Segoe UI", 16, "bold"), fg="#2c3e50", bg="#f8f9fa")
        self.info_label.pack(side=tk.TOP, fill=tk.X, pady=10)

        canvas_frame = tk.Frame(self.root, bg="#1e272e")
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.canvas = tk.Canvas(canvas_frame, bg="#1e272e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Configure>", lambda e: self.draw_board())
        self.canvas.bind("<Button-1>", self.on_canvas_click)

        footer = tk.Frame(self.root, bg="#f8f9fa")
        footer.pack(side=tk.BOTTOM, fill=tk.X, pady=10)

        tk.Button(
            footer, text="Nouvelle Partie", font=("Segoe UI", 12, "bold"),
            bg="#2980b9", fg="white", relief=tk.FLAT, padx=15, pady=8,
            command=self.reset_game
        ).pack()

        self.root.config(bg="#f8f9fa")

    def get_cell_size(self):
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        return max(40, min(w, h) // 8)

    def draw_board(self):
        self.canvas.delete("all")
        cell_size = self.get_cell_size()
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        start_x, start_y = (w - cell_size * 8) // 2, (h - cell_size * 8) // 2
        board = self.env.board

        light, dark = "#ebecd0", "#739552"
        self.loaded_photos.clear()

        for r in range(8):
            for c in range(8):
                rank, file = 7 - r, c
                sq = chess.square(file, rank)
                x1, y1 = start_x + c * cell_size, start_y + r * cell_size
                x2, y2 = x1 + cell_size, y1 + cell_size

                color = light if (r + c) % 2 == 0 else dark
                if sq == self.selected_sq:
                    color = "#baca2b"
                elif sq in self.legal_destinations:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
                    dot = cell_size // 5
                    self.canvas.create_oval(
                        x1+cell_size//2-dot//2, y1+cell_size//2-dot//2,
                        x1+cell_size//2+dot//2, y1+cell_size//2+dot//2,
                        fill="#46622b", outline=""
                    )
                    continue

                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")

                piece = board.piece_at(sq)
                if piece:
                    symbol = piece.symbol()
                    if symbol in self.images:
                        img_size = int(cell_size * 0.82)
                        resized_pil = self.images[symbol].resize((img_size, img_size), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(resized_pil)
                        self.loaded_photos[sq] = photo
                        self.canvas.create_image(x1 + cell_size // 2, y1 + cell_size // 2, image=photo, anchor=tk.CENTER)
                    else:
                        letter = PIECE_LABELS[symbol]
                        pad = int(cell_size * 0.15)
                        bx1, by1 = x1 + pad, y1 + pad
                        bx2, by2 = x2 - pad, y2 - pad
                        is_w = (piece.color == chess.WHITE)
                        b_fill = "#ffffff" if is_w else "#2f3640"
                        b_out = "#dcdde1" if is_w else "#718093"
                        t_col = "#2f3640" if is_w else "#f5f6fa"
                        self.canvas.create_oval(bx1+2, by1+2, bx2+2, by2+2, fill="#000000", outline="")
                        self.canvas.create_oval(bx1, by1, bx2, by2, fill=b_fill, outline=b_out, width=2)
                        self.canvas.create_text(
                            x1 + cell_size // 2, y1 + cell_size // 2,
                            text=letter, font=("Segoe UI", int(cell_size * 0.38), "bold"), fill=t_col
                        )

        turn_text = "Tour : Blancs (♔)" if board.turn == chess.WHITE else "Tour : Noires (♚)"
        if board.is_check(): turn_text += " — ⚠️ ÉCHEC !"
        if board.is_game_over():
            turn_text = f"Partie terminée : {board.result()}"
        self.info_label.config(text=turn_text)

    def on_canvas_click(self, event):
        board = self.env.board
        if board.is_game_over() or board.turn != self.human_color:
            return

        cell_size = self.get_cell_size()
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        start_x, start_y = (w - cell_size * 8) // 2, (h - cell_size * 8) // 2

        if not (start_x <= event.x < start_x + cell_size * 8 and start_y <= event.y < start_y + cell_size * 8):
            return

        c = (event.x - start_x) // cell_size
        r = (event.y - start_y) // cell_size
        clicked_sq = chess.square(c, 7 - r)

        if self.selected_sq is None:
            piece = board.piece_at(clicked_sq)
            if piece and piece.color == board.turn:
                self.selected_sq = clicked_sq
                self.legal_destinations = [m.to_square for m in board.legal_moves if m.from_square == clicked_sq]
                self.draw_board()
        else:
            move = next((m for m in board.legal_moves if m.from_square == self.selected_sq and m.to_square == clicked_sq), None)
            self.selected_sq, self.legal_destinations = None, []

            if move:
                _, _, done, info = self.env.step(move)
                self.draw_board()
                if done:
                    messagebox.showinfo("Fin de partie", f"Résultat : {info.get('result', board.result())}")
                else:
                    # Tour de l'IA
                    self.root.after(300, self.ai_move)
            else:
                piece = board.piece_at(clicked_sq)
                if piece and piece.color == board.turn:
                    self.selected_sq = clicked_sq
                    self.legal_destinations = [m.to_square for m in board.legal_moves if m.from_square == clicked_sq]
                    self.draw_board()
                    return
            self.draw_board()

    def ai_move(self):
        board = self.env.board
        if board.is_game_over() or board.turn == self.human_color:
            return

        self.info_label.config(text="L'agent RL réfléchit...")
        self.root.update()

        move = self.agent.select_action(board)
        if move and move in board.legal_moves:
            _, _, done, info = self.env.step(move)
            self.draw_board()
            if done:
                messagebox.showinfo("Fin de partie", f"Résultat : {info.get('result', board.result())}")

    def reset_game(self):
        self.env.reset()
        self.selected_sq = None
        self.legal_destinations = []
        self.draw_board()
        if self.env.board.turn != self.human_color:
            self.root.after(500, self.ai_move)

def main():
    parser = argparse.ArgumentParser(description="Jouer contre l'agent RL avec interface graphique.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Chemin du modèle.")
    parser.add_argument("--num_simulations", type=int, default=30, help="Nombre de simulations MCTS.")
    parser.add_argument("--strategy", type=str, choices=["greedy", "epsilon", "ucb", "mcts"], default="mcts", help="Stratégie.")
    parser.add_argument("--color", type=str, choices=["white", "black"], default="white", help="Votre couleur.")
    args = parser.parse_args()

    if args.strategy == "greedy":
        agent = GreedyAgent()
    elif args.strategy == "epsilon":
        agent = EpsilonGreedyAgent(epsilon=0.0)
    elif args.strategy == "ucb":
        agent = UCBAgent(c_param=1.41)
    else:
        agent = MCTSAgent(num_simulations=args.num_simulations)

    if os.path.exists(args.model_path):
        if agent.load(args.model_path):
            print(f"Modèle chargé depuis '{args.model_path}'.")
    else:
        print(f"Attention: Aucun modèle trouvé à '{args.model_path}'. Poids aléatoires.")

    human_color = chess.WHITE if args.color.lower() == "white" else chess.BLACK

    root = tk.Tk()
    ChessAgentGUI(root, agent, human_color)
    root.mainloop()

if __name__ == "__main__":
    main()
