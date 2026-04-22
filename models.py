import torch
import torch.nn as nn

# ================================
# MLP BASELINE MODEL
# ================================
class MLP(nn.Module):
    def __init__(self, input_size, horizon):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, horizon)
        )

    def forward(self, x):
        # WHY: MLP cannot understand sequence order, so we flatten input
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ================================
# CUSTOM GRU MODEL (FROM SCRATCH)
# ================================
class CustomGRU(nn.Module):
    def __init__(self, input_size, hidden_size, horizon):
        super().__init__()
        self.hidden_size = hidden_size

        self.z = nn.Linear(input_size + hidden_size, hidden_size)  # update gate
        self.r = nn.Linear(input_size + hidden_size, hidden_size)  # reset gate
        self.h_hat = nn.Linear(input_size + hidden_size, hidden_size)

        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # WHY: hidden state stores memory of past sequence
        h = torch.zeros(x.size(0), self.hidden_size)

        for t in range(x.size(1)):
            combined = torch.cat((x[:, t], h), dim=1)

            z = torch.sigmoid(self.z(combined))  # update gate
            r = torch.sigmoid(self.r(combined))  # reset gate

            combined_r = torch.cat((x[:, t], r * h), dim=1)
            h_tilde = torch.tanh(self.h_hat(combined_r))

            # WHY: selectively update memory
            h = (1 - z) * h + z * h_tilde

        return self.fc(h)
