import torch, torch.nn as nn

class StatusMLP(nn.Module):
    def __init__(self, input_dim=3):
        super(StatusMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


import torch.nn as nn

class ProfitMLP(nn.Module):
    def __init__(self, input_dim):
        super(ProfitMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2) 
        )

    def forward(self, x):
        return self.net(x)

class StatusMLP(nn.Module):
    def __init__(self, input_dim):
        super(StatusMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2) 
        )

    def forward(self, x):
        return self.model(x)


