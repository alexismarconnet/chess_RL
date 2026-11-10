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
    def __init__(self, num_simulations: int = 50, c_puct: float = 1.41, temperature: float = 1.0, 
                 dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25, **kwargs):
        super().__init__(**kwargs)
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.temperature = temperature
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon

    def get_action_and_policy(self, board: chess.Board, temperature: float = None, add_noise: bool = False):
        if temperature is None:
            temperature = self.temperature
        root = MCTSNode(board.copy())
        self.eval_cache = {} # Cache de transposition pour la recherche MCTS

        # 1. Expansion initiale de la racine (avec ou sans bruit de Dirichlet)
        self._expand_node(root, is_root=add_noise)

        # 2. Exécuter les simulations MCTS par batch (ex: batch_size = 8)
        batch_size = 8
        sims_done = 0

        while sims_done < self.num_simulations:
            current_batch_size = min(batch_size, self.num_simulations - sims_done)
            sims_done += current_batch_size

            batch_paths = []
            for _ in range(current_batch_size):
                node = root
                search_board = board.copy()

                # A. SÉLECTION (descendre dans l'arbre via PUCT)
                while node.is_expanded and len(node.children) > 0:
                    node, move = self._select_child(node)
                    search_board.push(move)

                # Vérifier si game over
                is_over = False
                goval = 0.0
                if search_board.is_game_over():
                    is_over = True
                    outcome = search_board.outcome()
                    if outcome is not None:
                        if outcome.winner == search_board.turn:
                            goval = -1.0 # Du point de vue du joueur précédent
                        elif outcome.winner is not None:
                            goval = 1.0
                        else:
                            goval = 0.0

                batch_paths.append((node, search_board, is_over, goval))

            # B. ÉVALUATION BATCHÉE (Inférence réseau en un seul appel pour le batch)
            to_eval_boards = []
            to_eval_fens = []

            for node, sboard, is_over, goval in batch_paths:
                if is_over:
                    continue
                fen = sboard.fen()
                if fen not in self.eval_cache and fen not in to_eval_fens:
                    to_eval_boards.append(sboard)
                    to_eval_fens.append(fen)

            if len(to_eval_boards) > 0:
                tensors = [torch.tensor(board_to_tensor(b), dtype=torch.float32, device=self.device) for b in to_eval_boards]
                states_t = torch.stack(tensors)
                with torch.no_grad():
                    policy_logits_batch, values_batch = self.model(states_t)
                    policy_logits_batch = policy_logits_batch.cpu().numpy()
                    values_batch = values_batch.squeeze(-1).cpu().numpy()

                for fen, plog, val in zip(to_eval_fens, policy_logits_batch, values_batch):
                    self.eval_cache[fen] = (plog, float(val))

            # C. EXPANSION & RÉTROPROPAGATION
            for node, sboard, is_over, goval in batch_paths:
                if is_over:
                    value = goval
                else:
                    fen = sboard.fen()
                    plog, val = self.eval_cache[fen]
                    if not node.is_expanded:
                        value = self._expand_node(node, sboard, precomputed_logits=plog, precomputed_value=val)
                    else:
                        value = val
                self._backpropagate(node, value)

        # 3. Choix du coup final et construction du vecteur de politique MCTS pi
        if not root.children:
            legal_moves = list(board.legal_moves)
            move = legal_moves[0] if legal_moves else None
            policy = np.zeros(4288, dtype=np.float32)
            if move:
                from environment.board_encoder import move_to_action
                policy[move_to_action(move)] = 1.0
            return move, policy

        moves = list(root.children.keys())
        visits = np.array([child.visit_count for child in root.children.values()], dtype=np.float32)

        policy = np.zeros(4288, dtype=np.float32)
        for move, child in root.children.items():
            from environment.board_encoder import move_to_action
            policy[move_to_action(move)] = child.visit_count

        s_vis = np.sum(visits)
        if s_vis > 0:
            policy /= s_vis

        if temperature == 0.0:
            best_idx = np.argmax(visits)
            chosen_move = moves[best_idx]
        else:
            scaled_visits = visits ** (1.0 / temperature)
            s = np.sum(scaled_visits)
            probs = scaled_visits / s if s > 0 else np.ones_like(scaled_visits) / len(scaled_visits)
            chosen_idx = np.random.choice(len(moves), p=probs)
            chosen_move = moves[chosen_idx]

        return chosen_move, policy

    def select_action(self, board: chess.Board, temperature: float = None) -> chess.Move:
        move, _ = self.get_action_and_policy(board, temperature=temperature)
        return move

    def _expand_node(self, node: MCTSNode, board: chess.Board = None, is_root: bool = False, precomputed_logits=None, precomputed_value=None) -> float:
        if board is None:
            board = node.board

        fen = board.fen()
        if precomputed_logits is not None and precomputed_value is not None:
            policy_logits, value = precomputed_logits, precomputed_value
        elif hasattr(self, 'eval_cache') and fen in self.eval_cache:
            policy_logits, value = self.eval_cache[fen]
        else:
            # Obtenir les prédictions du réseau de neurones
            state_t = torch.tensor(board_to_tensor(board), dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                policy_logits, value = self.model(state_t)
                policy_logits = policy_logits.squeeze(0).cpu().numpy()
                value = value.squeeze(0).item()
            if hasattr(self, 'eval_cache'):
                self.eval_cache[fen] = (policy_logits, value)

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

        # Ajout du bruit de Dirichlet si demandé (uniquement à la racine pendant le self-play)
        if is_root and len(legal_indices) > 0:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(legal_indices))
            probs = (1.0 - self.dirichlet_epsilon) * probs + self.dirichlet_epsilon * noise

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
