import random
from collections import deque
import numpy as np
import torch

class ReplayBuffer:
    """
    Mémoire tampon (Experience Replay) pour stocker et échantillonner
    des transitions (état, action, récompense, état_suivant, done).
    """
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Ajoute une transition dans le buffer.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int, device: torch.device):
        """
        Tire un mini-lot aléatoire de transitions et les convertit en tenseurs PyTorch.
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32, device=device),
            torch.tensor(np.array(actions), dtype=torch.long, device=device),
            torch.tensor(np.array(rewards), dtype=torch.float32, device=device).unsqueeze(1),
            torch.tensor(np.array(next_states), dtype=torch.float32, device=device),
            torch.tensor(np.array(dones), dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)
