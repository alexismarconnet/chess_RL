import random
import sys
import os

# Ajout du chemin du projet si nécessaire pour les imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import ChessEnvironment

def test_environment():
    print("=== Test de l'environnement d'échecs (via package environment) ===")
    env = ChessEnvironment()
    state = env.reset()
    
    print("Position initiale :")
    env.render()

    done = False
    step_count = 0

    while not done and step_count < 10:
        legal_moves = env.get_legal_moves()
        random_move = random.choice(legal_moves)
        print(f"Coup joué : {random_move}")
        
        state, reward, done, info = env.step(random_move)
        env.render()
        step_count += 1

    print(f"Test terminé après {step_count} coups. Partie terminée ? {done}")

if __name__ == "__main__":
    test_environment()


