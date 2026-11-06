import argparse
import time
import os
import torch
from environment import ChessEnvironment
from agent import GreedyAgent, EpsilonGreedyAgent, UCBAgent
from training import ReplayBuffer, Trainer

DEFAULT_MODEL_PATH = "saved_models/chess_agent.pt"

def main():
    parser = argparse.ArgumentParser(description="Entraînement de l'agent RL aux échecs.")
    parser.add_argument("--num_games", type=int, default=20, help="Nombre de parties autonomes (self-play).")
    parser.add_argument("--batch_size", type=int, default=32, help="Taille du mini-lot d'entraînement.")
    parser.add_argument("--agent_type", type=str, choices=["greedy", "epsilon", "ucb"], default="epsilon", help="Type d'agent.")
    parser.add_argument("--epsilon", type=float, default=0.2, help="Epsilon pour EpsilonGreedyAgent.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Taux d'apprentissage.")
    parser.add_argument("--save_path", type=str, default=DEFAULT_MODEL_PATH, help="Chemin de sauvegarde du modèle.")
    parser.add_argument("--resume", action="store_true", help="Reprendre l'entraînement à partir du modèle sauvegardé.")
    args = parser.parse_args()

    print("=== DÉMARRAGE DE L'ENTRAÎNEMENT RL ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Calculateur : {device.upper()}")
    print(f"Agent : {args.agent_type.upper()} | Parties : {args.num_games}")

    # Initialisation de l'agent
    if args.agent_type == "greedy":
        agent = GreedyAgent(device=device)
    elif args.agent_type == "epsilon":
        agent = EpsilonGreedyAgent(epsilon=args.epsilon, device=device)
    else:
        agent = UCBAgent(c_param=1.41, device=device)

    if args.resume and agent.load(args.save_path):
        print(f"Modèle chargé depuis '{args.save_path}'.")

    env = ChessEnvironment()
    buffer = ReplayBuffer(capacity=10000)
    trainer = Trainer(agent, lr=args.lr)

    start_time = time.time()

    for episode in range(1, args.num_games + 1):
        # 1. Partie d'auto-apprentissage (self-play)
        steps, reward, result = trainer.play_self_game(env, buffer, max_steps=100)

        # 2. Rétropropagation / Descente de gradient si assez d'expériences
        p_loss, v_loss = 0.0, 0.0
        if len(buffer) >= args.batch_size:
            for _ in range(4):
                pl, vl = trainer.train_step(buffer, batch_size=args.batch_size)
                p_loss += pl
                v_loss += vl
            p_loss /= 4.0
            v_loss /= 4.0

        print(f"Partie [{episode:3d}/{args.num_games}] | Coups: {steps:3d} | Résultat: {result:12s} | Score: {reward:+.1f} | Buffer: {len(buffer):5d} | Loss P: {p_loss:.4f} V: {v_loss:.4f}")

        # 3. Sauvegarde régulière des poids
        if episode % 10 == 0 or episode == args.num_games:
            agent.save(args.save_path)
            print(f"  --> Checkpoint sauvegardé sous '{args.save_path}'")

    elapsed = time.time() - start_time
    print(f"\n=== ENTRAÎNEMENT TERMINÉ EN {elapsed:.2f} s ===")

if __name__ == "__main__":
    main()
