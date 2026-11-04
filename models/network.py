import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Bloc résiduel (ResNet) standard pour stabiliser et approfondir l'apprentissage.
    """
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)


class ChessNet(nn.Module):
    """
    Réseau de neurones Policy-Value flexible et modulaire pour le jeu d'échecs.
    - Entrée : Tenseur de forme (B, in_channels, 8, 8)
    - Sortie 1 (Policy) : Logits sur l'espace d'actions (B, num_actions)
    - Sortie 2 (Value) : Estimation de la position entre -1.0 et +1.0 (B, 1)
    """
    def __init__(self, in_channels: int = 14, num_actions: int = 4288, 
                 num_blocks: int = 4, hidden_dim: int = 128):
        super(ChessNet, self).__init__()

        # 1. Convolution initiale
        self.conv_in = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.bn_in = nn.BatchNorm2d(hidden_dim)

        # 2. Tour résiduel (blocs cachés modulables)
        self.res_blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(num_blocks)])

        # 3. Tête Policy (choix des coups)
        self.policy_conv = nn.Conv2d(hidden_dim, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 8 * 8, num_actions)

        # 4. Tête Value (évaluation de la position)
        self.value_conv = nn.Conv2d(hidden_dim, 8, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(8)
        self.value_fc1 = nn.Linear(8 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor):
        # Passage dans la tour convolutive
        out = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            out = block(out)

        # Tête Policy
        p = F.relu(self.policy_bn(self.policy_conv(out)))
        p = p.view(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # Tête Value
        v = F.relu(self.value_bn(self.value_conv(out)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        return policy_logits, value

