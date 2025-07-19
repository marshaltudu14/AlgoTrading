import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 1):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_dim)
        
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)

        # We need to ensure the input to LSTM is 3D: (batch_size, sequence_length, input_size)
        # If input is 2D (batch_size, input_dim), we need to unsqueeze sequence_length dimension
        if x.dim() == 2:
            x = x.unsqueeze(1) # Add sequence_length dimension of 1

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: (batch_size, sequence_length, hidden_size)

        # Decode the hidden state of the last time step
        out = self.linear(out[:, -1, :]) # Take the output from the last time step
        return out
