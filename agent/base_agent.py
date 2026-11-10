import os
import torch
import numpy as np
import chess
from models import ChessNet
from environment.board_encoder import board_to_tensor, get_legal_action_mask, action_to_move

class BaseChessAgent:
    """
    Classe de base pour les agents d'échecs RL.
    Gère le réseau de neurones, l'encodage et le masquage des coups illégaux.
    """
    def __init__(self, model: ChessNet = None, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model if model is not None else ChessNet().to(self.device)
        try:
            self.model = torch.compile(self.model)
        except Exception:
            pass
        self.model.eval()

    def _get_legal_logits(self, board: chess.Board):
        """
        Passe l'état dans le réseau et applique le masque des coups légaux.
        """
        state_t = torch.tensor(board_to_tensor(board), dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            policy_logits, value = self.model(state_t)
            policy_logits = policy_logits.squeeze(0).cpu().numpy()
            value = value.squeeze(0).item()

        legal_mask = get_legal_action_mask(board)
        legal_indices = np.where(legal_mask > 0)[0]

        if len(legal_indices) == 0:
            return None, value, []

        masked_logits = np.full_like(policy_logits, -1e9)
        masked_logits[legal_indices] = policy_logits[legal_indices]

        return masked_logits, value, legal_indices

    def select_action(self, board: chess.Board) -> chess.Move:
        raise NotImplementedError("La méthode select_action doit être implémentée par les sous-classes.")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> bool:
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            return True
        return False


class GreedyAgent(BaseChessAgent):
    def select_action(self, board: chess.Board) -> chess.Move:
        masked_logits, _, legal_indices = self._get_legal_logits(board)
        if len(legal_indices) == 0:
            return None

        best_action = int(np.argmax(masked_logits))
        return action_to_move(best_action, board)


class EpsilonGreedyAgent(BaseChessAgent):
    def __init__(self, epsilon: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def select_action(self, board: chess.Board) -> chess.Move:
        masked_logits, _, legal_indices = self._get_legal_logits(board)
        if len(legal_indices) == 0:
            return None

        if np.random.rand() < self.epsilon:
            chosen_action = int(np.random.choice(legal_indices))
        else:
            chosen_action = int(np.argmax(masked_logits))

        return action_to_move(chosen_action, board)


class UCBAgent(BaseChessAgent):
    def __init__(self, c_param: float = 1.41, **kwargs):
        super().__init__(**kwargs)
        self.c_param = c_param
        self.visit_counts = {}
        self.state_visits = {}

    def select_action(self, board: chess.Board) -> chess.Move:
        masked_logits, _, legal_indices = self._get_legal_logits(board)
        if len(legal_indices) == 0:
            return None

        fen = board.fen()
        if fen not in self.visit_counts:
            self.visit_counts[fen] = {act: 0 for act in legal_indices}
            self.state_visits[fen] = 0

        total_state_visits = self.state_visits[fen]

        ucb_scores = {}
        for act in legal_indices:
            q_value = masked_logits[act]
            n_a = self.visit_counts[fen][act]
            exploration_bonus = self.c_param * np.sqrt(np.log(1 + total_state_visits) / (1 + n_a))
            ucb_scores[act] = q_value + exploration_bonus

        best_action = max(ucb_scores, key=ucb_scores.get)

        self.visit_counts[fen][best_action] += 1
        self.state_visits[fen] += 1

        return action_to_move(best_action, board)
