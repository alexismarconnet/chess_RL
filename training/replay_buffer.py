import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    """
    Mémoire tampon (Experience Replay) pour stocker et échantillonner
    des transitions AlphaZero (état, politique MCTS pi, valeur cible z).
    """
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, policy, value):
        """
        Ajoute une transition (état, politique MCTS, valeur de fin de partie z) dans le buffer.
        """
        self.buffer.append((state, policy, value))

    def sample(self, batch_size: int, device: torch.device):
        """
        Tire un mini-lot aléatoire de transitions et les convertit en tenseurs PyTorch.
        """
        batch = random.sample(self.buffer, batch_size)
        states, policies, values = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=device),
            torch.tensor(np.array(policies), dtype=torch.float32, device=device),
            torch.tensor(np.array(values), dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)
