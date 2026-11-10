import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import chess
from environment import ChessEnvironment
from environment.board_encoder import move_to_action
from agent import BaseChessAgent
from training.replay_buffer import ReplayBuffer

class Trainer:
    """
    Gestionnaire d'entraînement effectuant la rétropropagation (descente de gradient)
    et la génération de parties autonomes (self-play) AlphaZero.
    """
    def __init__(self, agent: BaseChessAgent, lr: float = 1e-3, gamma: float = 0.99):
        self.agent = agent
        self.gamma = gamma
        self.optimizer = optim.AdamW(self.agent.model.parameters(), lr=lr, weight_decay=1e-4)

    def train_step(self, buffer: ReplayBuffer, batch_size: int = 64):
        """
        Effectue une mise à jour des poids du réseau par descente de gradient (AlphaZero style) :
        - Policy Loss : Cross-Entropy entre les probabilités MCTS pi et les logits du réseau.
        - Value Loss : MSE entre la valeur prédite et le résultat final de la partie z.
        """
        if len(buffer) < batch_size:
            return 0.0, 0.0

        self.agent.model.train()
        states, target_policies, target_values = buffer.sample(batch_size, self.agent.device)

        # 1. Prédictions actuelles du réseau
        policy_logits, values = self.agent.model(states)

        # 2. Calcul des pertes
        value_loss = F.mse_loss(values, target_values)
        log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -(target_policies * log_probs).sum(dim=-1).mean()

        total_loss = policy_loss + value_loss

        # 3. Rétropropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.agent.model.eval()

        return policy_loss.item(), value_loss.item()

    def play_self_game(self, env: ChessEnvironment, buffer: ReplayBuffer, max_steps: int = 150):
        """
        Simule une partie d'auto-apprentissage (self-play) AlphaZero.
        Commence par des coups d'ouverture aléatoires (AlphaZero style) pour diversifier les positions,
        puis enregistre chaque état et la distribution des probabilités MCTS pi.
        """
        state = env.reset()
        done = False

        # Phase d'ouverture aléatoire (2 à 4 coups légaux aléatoires)
        num_random_moves = np.random.randint(2, 5)
        for _ in range(num_random_moves):
            if env.board.is_game_over():
                break
            legal_moves = env.get_legal_moves()
            if not legal_moves:
                break
            rand_move = np.random.choice(legal_moves)
            state, _, done, _ = env.step(rand_move)
            if done:
                break

        steps = 0
        total_reward = 0.0
        game_history = []

        while not done and steps < max_steps:
            temperature = 1.0 if steps < 30 else 0.0
            if hasattr(self.agent, 'get_action_and_policy'):
                move, policy = self.agent.get_action_and_policy(env.board, temperature=temperature, add_noise=True)
            else:
                move = self.agent.select_action(env.board)
                policy = np.zeros(4288, dtype=np.float32)
                if move:
                    policy[move_to_action(move)] = 1.0

            if move is None:
                break

            turn = env.board.turn
            game_history.append((state, policy, turn))

            next_state, reward, done, info = env.step(move)
            state = next_state
            total_reward += reward
            steps += 1

        result = info.get("result", "max_steps") if done else "max_steps"
        winner = None
        if result == "white_win":
            winner = chess.WHITE
        elif result == "black_win":
            winner = chess.BLACK

        # Attribuer la valeur z du point de vue de chaque joueur à chaque tour
        for st, pol, turn in game_history:
            if winner is None:
                z = 0.0
            elif winner == turn:
                z = 1.0
            else:
                z = -1.0
            buffer.push(st, pol, z)

        return steps, total_reward, result
