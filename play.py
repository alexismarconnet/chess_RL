import argparse
import os
import chess
from agent import GreedyAgent
from tools.play_gui import ChessGUI
import tkinter as tk

DEFAULT_MODEL_PATH = "saved_models/chess_agent.pt"

def main():
    parser = argparse.ArgumentParser(description="Jouer contre l'agent RL aux échecs.")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH, help="Chemin du modèle pré-entraîné.")
    parser.add_argument("--mode", type=str, choices=["gui", "cli"], default="gui", help="Mode de jeu (gui/cli).")
    args = parser.parse_args()

    agent = GreedyAgent()
    if os.path.exists(args.model_path):
        if agent.load(args.model_path):
            print(f"Modèle chargé depuis '{args.model_path}'.")
    else:
        print(f"Attention: Aucun modèle trouvé à '{args.model_path}'. L'agent jouera avec des poids aléatoires.")

    if args.mode == "gui":
        print("Lancement de l'interface graphique Tkinter...")
        root = tk.Tk()
        app = ChessGUI(root)
        root.mainloop()
    else:
        print("Mode Console non implémenté. Utilisez le mode GUI : python play.py --mode gui")

if __name__ == "__main__":
    main()
