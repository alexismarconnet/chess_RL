import chess
import numpy as np
from .board_encoder import board_to_tensor, action_to_move, get_legal_action_mask

MATERIAL_REWARDS = {
    chess.PAWN: 0.02,
    chess.KNIGHT: 0.06,
    chess.BISHOP: 0.06,
    chess.ROOK: 0.1,
    chess.QUEEN: 0.18,
    chess.KING: 0.0
}

class ChessEnvironment:
    """
    Environnement d'échecs standardisé pour l'Apprentissage par Renforcement.
    Il renvoie des tenseurs (observations) à l'agent et gère l'espace d'actions.
    """
    def __init__(self):
        self.board = chess.Board()

    def reset(self) -> np.ndarray:
        """
        Remet l'échiquier à zéro et retourne l'état sous forme de tenseur (14, 8, 8).
        """
        self.board.reset()
        return board_to_tensor(self.board)

    def get_state(self) -> np.ndarray:
        """
        Retourne l'état actuel sous forme de tenseur (14, 8, 8).
        """
        return board_to_tensor(self.board)

    def get_legal_mask(self) -> np.ndarray:
        """
        Retourne le masque des coups légaux pour l'état actuel.
        """
        return get_legal_action_mask(self.board)

    def get_legal_moves(self):
        """
        Retourne la liste des objets chess.Move légaux.
        """
        return list(self.board.legal_moves)

    def step(self, action_or_move):
        """
        Exécute une action (indice entier ou objet chess.Move), met à jour l'échiquier et retourne :
        (next_state_tensor, reward, done, info)
        """
        if isinstance(action_or_move, chess.Move):
            move = action_or_move
        else:
            move = action_to_move(action_or_move, self.board)

        if move is None or move not in self.board.legal_moves:
            raise ValueError(f"Action / coup illégal tenté : {action_or_move}")

        # Détecter si une pièce est capturée avant de pousser le coup
        captured_piece = self.board.piece_at(move.to_square)
        is_capture = self.board.is_capture(move)

        reward = 0.0
        if is_capture:
            if self.board.is_en_passant(move):
                reward += MATERIAL_REWARDS[chess.PAWN]
            elif captured_piece is not None:
                reward += MATERIAL_REWARDS.get(captured_piece.piece_type, 0.0)

        self.board.push(move)

        done = self.board.is_game_over()
        info = {}

        if done:
            outcome = self.board.outcome()
            if outcome is not None:
                if outcome.winner == chess.WHITE:
                    reward += 1.0
                    info["result"] = "white_win"
                elif outcome.winner == chess.BLACK:
                    reward -= 1.0  # Si l'agent gagne en tant que noir, la perspective globale dépend du tour mais +1/-1 standard
                    info["result"] = "black_win"
                else:
                    reward -= 0.05  # Légère pénalité pour match nul
                    info["result"] = "draw"
        else:
            # Si le jeu continue mais qu'on a atteint la limite max_steps dans une boucle externe
            pass

        return self.get_state(), reward, done, info

    def render(self):
        """
        Affiche l'échiquier dans le terminal.
        """
        print("\n" + str(self.board) + "\n")



