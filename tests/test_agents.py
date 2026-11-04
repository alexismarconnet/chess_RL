import sys
import os
import chess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import GreedyAgent, EpsilonGreedyAgent, UCBAgent

def test_agents():
    print("=== Test des 3 Agents (Greedy, Epsilon-Greedy, UCB) ===")
    board = chess.Board()

    agents = [
        ("Greedy", GreedyAgent()),
        ("Epsilon-Greedy", EpsilonGreedyAgent(epsilon=0.2)),
        ("UCB", UCBAgent(c_param=1.41))
    ]

    for name, agent in agents:
        move = agent.select_action(board)
        print(f"Agent [{name}] a sélectionné le coup : {move}")
        assert move in board.legal_moves, f"L'agent {name} a joué un coup illégal : {move}"

    print("Tous les tests d'agents ont réussi avec succès !")

if __name__ == "__main__":
    test_agents()
