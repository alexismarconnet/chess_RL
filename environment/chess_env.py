import chess

class ChessEnvironment:
    """
    Environnement d'échecs pour l'Apprentissage par Renforcement.
    Encapsule la bibliothèque python-chess pour fournir une interface simple :
    reset(), get_legal_moves(), step(move), render().
    """
    def __init__(self):
        # Initialise un nouvel échiquier officiel d'échecs
        self.board = chess.Board()

    def reset(self):
        """
        Remet l'échiquier dans sa position initiale de début de partie.
        Retourne l'état initial (l'objet board).
        """
        self.board.reset()
        return self.board

    def get_legal_moves(self):
        """
        Retourne la liste de tous les coups légaux (valides selon les règles des échecs)
        sous forme d'objets chess.Move.
        """
        return list(self.board.legal_moves)

    def step(self, move: chess.Move):
        """
        Exécute un coup sur l'échiquier, puis calcule la récompense et 
        vérifie si la partie est terminée.
        
        Args:
            move (chess.Move): Le coup à jouer (ex: chess.Move.from_uci("e2e4"))
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # 1. Vérification de sécurité
        if move not in self.board.legal_moves:
            raise ValueError(f"Coup illégal tenté : {move}")

        # 2. Jouer le coup sur l'échiquier
        self.board.push(move)

        # 3. Vérifier si la partie est terminée et attribuer la récompense
        done = self.board.is_game_over()
        reward = 0.0
        info = {}

        if done:
            outcome = self.board.outcome()
            if outcome is not None:
                if outcome.winner == chess.WHITE:
                    reward = 1.0  # Victoire des blancs
                    info["result"] = "white_win"
                elif outcome.winner == chess.BLACK:
                    reward = -1.0 # Victoire des noirs (défaite des blancs)
                    info["result"] = "black_win"
                else:
                    reward = 0.0  # Match nul (pat, répétition, etc.)
                    info["result"] = "draw"
        else:
            reward = 0.0  # Pas de récompense intermédiaire pour l'instant

        return self.board, reward, done, info

    def render(self):
        """
        Affiche l'échiquier dans le terminal.
        """
        print("\n" + str(self.board) + "\n")

