# Agent RL Échecs

Projet d'apprentissage par renforcement (Reinforcement Learning) pour apprendre à jouer aux échecs de zéro.

## Structure du Projet
- `environment/` : Logique de l'échiquier et règles (wrapper python-chess).
- `agent/` : Politique de l'agent et choix des coups.
- `models/` : Réseaux de neurones (si utilisation de Deep RL).
- `training/` : Boucles d'entraînement et self-play.
- `saved_models/` : Poids sauvegardés du modèle.
- `train.py` : Script principal pour entraîner l'agent.
- `play.py` : Script pour jouer contre l'agent.
