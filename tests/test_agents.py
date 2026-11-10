import sys
import os
import chess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent import GreedyAgent, EpsilonGreedyAgent, UCBAgent, MCTSAgent

def test_agents():
    print("=== Test des Agents (Greedy, Epsilon-Greedy, UCB, MCTS) ===")
    board = chess.Board()

    agents = [
        ("Greedy", GreedyAgent()),
        ("Epsilon-Greedy", EpsilonGreedyAgent(epsilon=0.2)),
        ("UCB", UCBAgent(c_param=1.41)),
        ("MCTS (temp=1.0)", MCTSAgent(num_simulations=5, temperature=1.0)),
        ("MCTS (temp=0.0)", MCTSAgent(num_simulations=5, temperature=0.0))
    ]

    for name, agent in agents:
        move = agent.select_action(board)
        print(f"Agent [{name}] a sélectionné le coup : {move}")
        assert move in board.legal_moves, f"L'agent {name} a joué un coup illégal : {move}"

    print("Tous les tests d'agents ont réussi avec succès !")

if __name__ == "__main__":
    test_agents()

