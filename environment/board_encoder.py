import chess
import numpy as np

# Mappage des types de pièces pour les 12 premiers canaux
PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Convertit un objet chess.Board en un tenseur 3D de dimensions (14, 8, 8) de type float32.
    - Canaux 0 à 5   : Pièces blanches [Pion, Cavalier, Fou, Tour, Dame, Roi]
    - Canaux 6 à 11  : Pièces noires [Pion, Cavalier, Fou, Tour, Dame, Roi]
    - Canal 12       : Tour de jeu (1.0 si c'est aux Blancs, 0.0 si c'est aux Noirs)
    - Canal 13       : Droits de roque normalisés
    """
    tensor = np.zeros((14, 8, 8), dtype=np.float32)

    # Remplissage des canaux de pièces (0 à 11)
    for sq, piece in board.piece_map().items():
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        piece_idx = PIECE_TYPES.index(piece.piece_type)
        channel = piece_idx if piece.color == chess.WHITE else 6 + piece_idx
        tensor[channel, rank, file] = 1.0

    # Canal du tour (12)
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    # Canal des droits de roque (13)
    castling_val = 0.0
    if board.has_kingside_castling_rights(chess.WHITE):
        castling_val += 0.25
    if board.has_queenside_castling_rights(chess.WHITE):
        castling_val += 0.25
    if board.has_kingside_castling_rights(chess.BLACK):
        castling_val += 0.25
    if board.has_queenside_castling_rights(chess.BLACK):
        castling_val += 0.25
    tensor[13, :, :] = castling_val

    return tensor

def move_to_action(move: chess.Move) -> int:
    """
    Encode un objet chess.Move en un entier unique (action).
    Espace d'actions de base : 64 * 64 = 4096 (cases départ * cases arrivée)
    plus les promotions sous-optimales.
    """
    from_sq = move.from_square
    to_sq = move.to_square

    if move.promotion is None or move.promotion == chess.QUEEN:
        return from_sq * 64 + to_sq
    elif move.promotion == chess.KNIGHT:
        return 4096 + from_sq * 3 + 0
    elif move.promotion == chess.BISHOP:
        return 4096 + from_sq * 3 + 1
    elif move.promotion == chess.ROOK:
        return 4096 + from_sq * 3 + 2

    return from_sq * 64 + to_sq

def action_to_move(action: int, board: chess.Board) -> chess.Move:
    """
    Décode un entier action en un chess.Move valide pour la position actuelle.
    """
    legal_moves = list(board.legal_moves)
    
    for m in legal_moves:
        if move_to_action(m) == action:
            return m

    # Fallback si correspondance exacte non trouvée
    if action < 4096:
        from_sq = action // 64
        to_sq = action % 64
        for m in legal_moves:
            if m.from_square == from_sq and m.to_square == to_sq:
                return m
    else:
        rem = action - 4096
        from_sq = rem // 3
        promo_type = [chess.KNIGHT, chess.BISHOP, chess.ROOK][rem % 3]
        for m in legal_moves:
            if m.from_square == from_sq and m.promotion == promo_type:
                return m

    return legal_moves[0] if legal_moves else None

def get_legal_action_mask(board: chess.Board) -> np.ndarray:
    """
    Renvoie un masque booléen/binaire de taille 4288 où 1.0 indique un coup légal
    et 0.0 un coup illégal.
    """
    mask = np.zeros(4288, dtype=np.float32)
    for m in board.legal_moves:
        act = move_to_action(m)
        mask[act] = 1.0
    return mask
