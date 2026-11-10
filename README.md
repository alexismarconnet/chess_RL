# Chess Reinforcement Learning (AlphaZero-Inspired)

Un projet complet d'Apprentissage par Renforcement (Reinforcement Learning) pour entraîner de zéro un agent d'échecs inspiré de l'architecture **AlphaZero** (Réseau de Neurones Résiduel + Recherche MCTS guidée par Policy et Value).

## 🚀 Fonctionnalités Clés

- **Agent MCTS AlphaZero** : Arbre de recherche Monte Carlo guidé par les prédictions d'un réseau de neurones (`Policy` et `Value`) via la formule PUCT.
- **Évaluation MCTS par Batch (Batched MCTS)** : Inférence GPU groupée pour accélérer considérablement la recherche MCTS (accélération x3 à x5).
- **Cache de Transposition FEN** : Évite les évaluations redondantes de positions identiques dans l'arbre de recherche.
- **Ouvertures Aléatoires (Random Openings)** : Chaque partie de self-play commence par 2 à 4 coups légaux aléatoires pour diversifier les positions et casser les boucles de matchs nuls déterministes.
- **Apprentissage par Auto-Apprentissage (Self-Play)** : L'agent joue contre lui-même, stocke ses transitions dans un `ReplayBuffer` et met à jour ses poids via Policy Loss (Cross-Entropy) et Value Loss (MSE).
- **Interface Graphique (GUI)** : Permet de jouer contre l'agent entraîné via Tkinter.

---

## 📁 Structure du Projet

```text
chess_RL/
├── agent/
│   ├── base_agent.py      # Classes de base et agents heuristiques (Greedy, UCB, etc.)
│   └── mcts_agent.py      # Implémentation MCTS guidée par le NN (avec cache et batch)
├── environment/
│   ├── chess_env.py       # Wrapper de l'échiquier (règles et gestion des fins de partie)
│   └── board_encoder.py   # Encodage tenseur 3D (14, 8, 8) et gestion des actions (4288)
├── models/
│   └── network.py         # Réseau de neurones ResNet (ChessNet) avec têtes Policy et Value
├── training/
│   ├── trainer.py         # Boucle de Self-Play et étapes d'optimisation par gradient
│   └── replay_buffer.py   # Tampon de relecture des expériences
├── tests/
│   ├── test_agents.py     # Tests unitaires des agents
│   └── test_env.py        # Tests de l'environnement d'échecs
├── saved_models/          # Poids sauvegardés du modèle (.pt)
├── train.py               # Script principal d'entraînement RL
├── play.py                # Script pour jouer contre l'agent (mode GUI)
└── requirements.txt       # Dépendances du projet
```

---

## 🛠️ Installation

1. Cloner le dépôt et se placer dans le dossier :
   ```bash
   cd chess_RL
   ```

2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```

---

## 🏋️‍♂️ Entraînement de l'Agent

Pour lancer l'entraînement de l'agent MCTS en mode self-play :

```bash
python train.py --num_games 2000 --lr 1e-3 --num_simulations 20 --agent_type mcts
```

### Options disponibles :
- `--num_games` : Nombre de parties d'auto-apprentissage (défaut : 20).
- `--num_simulations` : Nombre de simulations MCTS par coup (défaut : 30, recommandé : 10 à 25 pour un entraînement rapide).
- `--lr` : Taux d'apprentissage (défaut : `1e-3`).
- `--agent_type` : Type d'agent (`mcts`, `greedy`, `epsilon`, `ucb`).
- `--resume` : Reprendre l'entraînement à partir du dernier checkpoint sauvegardé.

---

## 🎮 Jouer contre l'Agent

Une fois le modèle entraîné (ou pour tester l'agent actuel), lancez l'interface graphique :

```bash
python play.py --mode gui
```

---

## 🧪 Tests Unitaires

Pour vérifier le bon fonctionnement des agents et de l'environnement :

```bash
python tests/test_agents.py
python tests/test_env.py
```
