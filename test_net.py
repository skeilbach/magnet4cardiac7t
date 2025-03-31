import torch
import torch.nn as nn
import torch.optim as optim
import einops

class FunctionApproximator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1)  # 6 fields (E,B) × (real, imag)
        )

    def forward(self, x):
        output = self.net(x)
        return output

def loss_fn(pred, input, target_func: callable = lambda x, y, z: 10):
    return torch.mean((pred - target_func(input[:, 0], input[:, 1], input[:, 2]))**2)   

target_func = lambda x, y, z: 2*x + 3*y + 4*z


# Training Data: Randomly sample points in space
