import torch
import torch.nn as nn
import torch.nn.functional as F

class RewardModel(nn.Module):
    """Neural reward model for RLHF preferences."""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        # state: (batch, state_dim), action: (batch,)
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        onehot = F.one_hot(action.long(), num_classes=self.action_dim).float()
        x = torch.cat([state, onehot], dim=-1)
        return self.net(x).squeeze(-1)
