import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1):
        super(LSTMModel, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, sequence_length, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        # out: (batch_size, sequence_length, hidden_dim)
        # We take the output from the last time step
        out = self.linear(out[:, -1, :])
        return out

class ActorLSTMModel(LSTMModel):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1):
        super().__init__(input_dim, hidden_dim, output_dim, num_layers)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.softmax(super().forward(x))