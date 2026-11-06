import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from environment import ChessEnvironment
from environment.board_encoder import move_to_action
from agent import BaseChessAgent
from training.replay_buffer import ReplayBuffer

class Trainer:
    """
    Gestionnaire d'entraînement effectuant la rétropropagation (descente de gradient)
    et la génération de parties autonomes (self-play).
    """
    def __init__(self, agent: BaseChessAgent, lr: float = 1e-3, gamma: float = 0.99):
        self.agent = agent
        self.gamma = gamma
        self.optimizer = optim.AdamW(self.agent.model.parameters(), lr=lr, weight_decay=1e-4)

    def train_step(self, buffer: ReplayBuffer, batch_size: int = 64):
        """
        Effectue une mise à jour des poids du réseau par descente de gradient
        sur un mini-lot d'expériences tiré du ReplayBuffer.
        """
        if len(buffer) < batch_size:
            return 0.0, 0.0

        self.agent.model.train()
        states, actions, rewards, next_states, dones = buffer.sample(batch_size, self.agent.device)

        # 1. Prédictions actuelles du réseau
        policy_logits, values = self.agent.model(states)

        # 2. Valeurs cibles (Temporal Difference / Q-value)
        with torch.no_grad():
            _, next_values = self.agent.model(next_states)
            target_values = rewards + (1.0 - dones) * self.gamma * next_values

        # 3. Calcul des pertes (Value Loss + Policy Loss)
        value_loss = F.mse_loss(values, target_values)
        policy_loss = F.cross_entropy(policy_logits, actions)

        total_loss = policy_loss + value_loss

        # 4. Rétropropagation (Backpropagation)
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.agent.model.eval()

        return policy_loss.item(), value_loss.item()

    def play_self_game(self, env: ChessEnvironment, buffer: ReplayBuffer, max_steps: int = 150):
        """
        Simule une partie d'auto-apprentissage (self-play) où l'agent joue contre lui-même.
        Chaque transition est ajoutée au ReplayBuffer.
        """
        state = env.reset()
        done = False
        steps = 0
        total_reward = 0.0

        while not done and steps < max_steps:
            # L'agent choisit un coup sur la position actuelle
            move = self.agent.select_action(env.board)
            if move is None:
                break

            action_idx = move_to_action(move)

            # L'environnement exécute le coup
            next_state, reward, done, info = env.step(move)

            # Enregistrement de la transition dans le ReplayBuffer
            buffer.push(state, action_idx, reward, next_state, float(done))

            state = next_state
            total_reward += reward
            steps += 1

        result = info.get("result", "max_steps") if done else "max_steps"
        return steps, total_reward, result
