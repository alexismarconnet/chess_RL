import math
import numpy as np
import torch
import chess
from agent.base_agent import BaseChessAgent
from environment.board_encoder import board_to_tensor, get_legal_action_mask

class MCTSNode:
    """
    Nœud de l'arbre de recherche MCTS.
    """
    def __init__(self, board: chess.Board, parent=None, move: chess.Move = None, prior: float = 0.0):
        self.board = board
        self.parent = parent
        self.move = move
        self.is_expanded = False
        self.children = {}
        
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


class MCTSAgent(BaseChessAgent):
    """
    Agent MCTS guidé par le Réseau de Neurones (inspiré d'AlphaZero).
    Anticipe les futurs coups par simulations dans un arbre de recherche avant de décider.
    """
    def __init__(self, num_simulations: int = 50, c_puct: float = 1.41, **kwargs):
        super().__init__(**kwargs)
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def select_action(self, board: chess.Board) -> chess.Move:
        root = MCTSNode(board.copy())

        # 1. Expansion initiale de la racine
        self._expand_node(root)

        # 2. Exécuter les simulations MCTS
        for _ in range(self.num_simulations):
            node = root
            search_board = board.copy()

            # A. SÉLECTION (descendre dans l'arbre via PUCT)
            while node.is_expanded and len(node.children) > 0:
                node, move = self._select_child(node)
                search_board.push(move)

            # B. ÉVALUATION & EXPANSION
            value = 0.0
            if search_board.is_game_over():
                outcome = search_board.outcome()
                if outcome is not None:
                    if outcome.winner == search_board.turn:
                        value = -1.0 # Du point de vue du joueur précédent
                    elif outcome.winner is not None:
                        value = 1.0
                    else:
                        value = 0.0
            else:
                value = self._expand_node(node, search_board)

            # C. RÉTROPROPAGATION
            self._backpropagate(node, value)

        # 3. Choix du coup final basé sur le plus grand nombre de visites
        if not root.children:
            legal_moves = list(board.legal_moves)
            return legal_moves[0] if legal_moves else None

        best_move = max(root.children, key=lambda m: root.children[m].visit_count)
        return best_move

    def _expand_node(self, node: MCTSNode, board: chess.Board = None) -> float:
        if board is None:
            board = node.board

        # Obtenir les prédictions du réseau de neurones
        state_t = torch.tensor(board_to_tensor(board), dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            policy_logits, value = self.model(state_t)
            policy_logits = policy_logits.squeeze(0).cpu().numpy()
            value = value.squeeze(0).item()

        legal_mask = get_legal_action_mask(board)
        legal_indices = np.where(legal_mask > 0)[0]

        if len(legal_indices) == 0:
            return value

        # Softmax sur les logits légaux pour obtenir les priors P
        masked_logits = np.full_like(policy_logits, -1e9)
        masked_logits[legal_indices] = policy_logits[legal_indices]
        
        # Stabilité numérique softmax
        exp_logits = np.exp(masked_logits[legal_indices] - np.max(masked_logits[legal_indices]))
        probs = exp_logits / np.sum(exp_logits)

        # Créer les enfants
        for idx, prob in zip(legal_indices, probs):
            # Retrouver le coup correspondant à l'indice
            move = None
            for m in board.legal_moves:
                from environment.board_encoder import move_to_action
                if move_to_action(m) == idx:
                    move = m
                    break
            if move is not None and move not in node.children:
                next_board = board.copy()
                next_board.push(move)
                node.children[move] = MCTSNode(next_board, parent=node, move=move, prior=float(prob))

        node.is_expanded = True
        return value

    def _select_child(self, node: MCTSNode):
        best_score = -float('inf')
        best_move = None
        best_child = None

        # Somme des visites des enfants pour PUCT
        sum_visits = sum(child.visit_count for child in node.children.values())
        sqrt_sum_visits = math.sqrt(sum_visits + 1)

        for move, child in node.children.items():
            # Formule PUCT (AlphaZero)
            u_score = self.c_puct * child.prior * (sqrt_sum_visits / (1 + child.visit_count))
            score = child.q_value + u_score

            if score > best_score:
                best_score = score
                best_move = move
                best_child = child

        return best_child, best_move

    def _backpropagate(self, node: MCTSNode, value: float):
        current = node
        val = value
        while current is not None:
            current.visit_count += 1
            current.value_sum += val
            val = -val # Inverser la valeur pour l'adversaire à chaque niveau
            current = current.parent
