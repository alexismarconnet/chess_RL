# GEMINI.md - Guide pour les Agents / Développeurs sur chess_RL

## 🎯 Vue d'ensemble du Projet
Projet d'Apprentissage par Renforcement (RL) pour les échecs (inspiré d'AlphaZero) basé sur `python-chess`, `PyTorch` et `numpy`.

## 📂 Architecture du Codebase
- `agent/` : Logique de décision (Greedy, Epsilon-Greedy, UCB, MCTS).
- `environment/` : Règles du jeu (`chess_env.py`) et encodage tenseur/action (`board_encoder.py`).
- `models/` : Réseau de neurones ResNet (`ChessNet` dans `models/network.py`).
- `training/` : Boucle d'entraînement, pertes Policy/Value et Self-Play (`trainer.py`, `replay_buffer.py`).
- `tools/` : Interface graphique Tkinter (`play_gui.py`).
- `tests/` : Tests unitaires (`test_env.py`, `test_agents.py`).

## ⚙️ Conventions et Règles de Développement
1. **Gestion des Tenseurs et Appareils (Device)** : Toujours transmettre et utiliser `self.device` (`cuda` ou `cpu`) lors de la création de tenseurs dans les agents et réseaux.
2. **Masquage des coups illégaux** : Toujours appliquer `get_legal_action_mask` lors de l'évaluation par le réseau de neurones pour éviter les actions illégales.
3. **Espace d'Actions** : L'espace de sortie du réseau est de taille fixe (4288 actions) géré par `board_encoder.py`. Ne pas modifier les fonctions d'encodage/décodage sans valider la cohérence.
4. **Température MCTS** : `MCTSAgent` gère la température (`temperature`) pour l'échantillonnage des coups (température=0.0 pour un choix déterministe, température>0.0 pour un échantillonnage stochastique basé sur les visites $N(s, a)^{1/T}$).
5. **Typage et Docstrings** : Maintenir le typage Python (`chess.Board`, `chess.Move`, `float`, etc.) et documenter les méthodes clés.

## 🚀 Commandes Utiles
- **Entraîner l'agent (MCTS)** :
  ```bash
  python train.py --agent_type mcts --num_games 50 --num_simulations 30
  ```
- **Jouer contre l'agent (GUI)** :
  ```bash
  python play.py --mode gui
  ```
- **Lancer les tests** :
  ```bash
  python tests/test_agents.py
  python tests/test_env.py
  ```
