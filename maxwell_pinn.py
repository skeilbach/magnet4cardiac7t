import torch
import torch.nn as nn
import torch.optim as optim
import einops

class HarmonicMaxwellPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 12)  # 6 fields (E,B) × (real, imag)
        )

    def forward(self, x):
        x.requires_grad_(True)
        output = self.net(x)  # Shape: (N, 12)
        output = einops.rearrange(output, "n (xyz reim eb) -> n xyz reim eb", xyz=3, reim=2, eb=2)  # Shape: (N, x/y/z, real/imag, E/B)
        return output
    
class PartialDerivatives(nn.Module):
    def forward(self, field, x):
        # field: (N, 3, 2, 2) = (N, x/y/z, real/imag, E/B)
        # x: (N, 3)
        # output: (N, 3, 3, 2, 2) = (N, ∂x/∂y/∂z, x/y/z, real/imag, E/B)
        x.requires_grad_(True)
        field = einops.rearrange(field, "n xyz reim eb -> n (xyz reim eb)")

        output = torch.empty((x.shape[0], x.shape[1], field.shape[1]), device=x.device)
        for i in range(output.shape[-1]):
            grads = torch.autograd.grad(field[:, i], x, torch.ones_like(field[:, i]), create_graph=True)[0]
            output[:, :, i] = grads

        output = einops.rearrange(output, "n partialxyz (xyz reim eb) -> n partialxyz xyz reim eb", xyz=3, reim=2, eb=2)

        return output
    
class Divergence(nn.Module):
    def forward(self, partial_derivatives):
        # partial_derivatives: (N, 3, 3, 2, 2) = (N, ∂x/∂y/∂z, x/y/z, real/imag, E/B)
        # output: (N, 2, 2) = (N, x/y/z, real/imag, E/B) -> divergence of E and B fields
        divergence = einops.einsum(partial_derivatives, "n i i reim eb -> n reim eb")
        return divergence
    
class Curl(nn.Module):
    def forward(self, partial_derivatives):
        # partial_derivatives: (N, 3, 3, 2, 2) = (N, ∂x/∂y/∂z, x/y/z, real/imag, E/B)
        # output: (N, 3, 2, 2) = (N, x/y/z, real/imag, E/B) -> curl of E and B fields
        output = torch.empty((partial_derivatives.shape[0], 3, 4), device=partial_derivatives.device)
        partial_derivatives = einops.rearrange(partial_derivatives, "n partialxyz xyz reim eb -> n partialxyz xyz (reim eb)")

        output[:, 0] = partial_derivatives[:, 1,2] - partial_derivatives[:, 2,1]
        output[:, 1] = partial_derivatives[:, 2,0] - partial_derivatives[:, 0,2]
        output[:, 2] = partial_derivatives[:, 0,1] - partial_derivatives[:, 1,0]
        output = einops.rearrange(output, "n xyz (reim eb) -> n xyz reim eb", reim=2, eb=2)
        return output
    
class MaxwellLoss(nn.Module):
    def __init__(self, omega, mu0, epsilon0):
        super().__init__()
        self.omega = omega
        self.mu0 = mu0
        self.epsilon0 = epsilon0
        self.pd = PartialDerivatives()
        self.curl = Curl()
        self.div = Divergence()
        self.target_func = lambda x,y,z: torch.sin(x) * torch.sin(y) * torch.sin(z)+10
        self.target_func = lambda x,y,z: 10

    def forward(self, model, x, J_real, J_imag):
        x.requires_grad_(True)
        field = model(x)
        field = einops.rearrange(field, "n xyz reim eb -> n (xyz reim eb)")
        target = self.target_func(x[:, 0], x[:, 1], x[:, 2])

        approx_loss = torch.mean((field[:, 0] - target) ** 2)
        return approx_loss

        partial_derivatives = self.pd(field, x)

        divergence = self.div(partial_derivatives)

        curls = self.curl(partial_derivatives)

        # Shapes
        # field: (N, 3, 2, 2) = (N, x/y/z, real/imag, E/B)
        # partial_derivatives: (N, 3, 3, 2, 2) = (N, ∂x/∂y/∂z, x/y/z, real/imag, E/B)
        # divergence: (N, 2, 2) = (N, x/y/z, real/imag, E/B)
        # curls: (N, 3, 2, 2) = (N, x/y/z, real/imag, E/B)

        # fields 
        E_real = field[:, :, 0, 0]
        E_imag = field[:, :, 1, 0]
        B_real = field[:, :, 0, 1]
        B_imag = field[:, :, 1, 1]

        # Compute curls
        curl_E_real = curls[:, :, 0, 0]
        curl_E_imag = curls[:, :, 1, 0]
        curl_B_real = curls[:, :, 0, 1]
        curl_B_imag = curls[:, :, 1, 1]

        # Faraday’s Law: ∇ × E = -jωB
        faraday_real = torch.mean((curl_E_real + self.omega * B_imag) ** 2)
        faraday_imag = torch.mean((curl_E_imag - self.omega * B_real) ** 2)

        # Ampère’s Law: ∇ × B = μ0J + jωμ0ε0E
        ampere_real = torch.mean((curl_B_real - self.mu0 * J_real - self.omega * self.mu0 * self.epsilon0 * E_imag) ** 2)
        ampere_imag = torch.mean((curl_B_imag - self.mu0 * J_imag + self.omega * self.mu0 * self.epsilon0 * E_real) ** 2)

        # Gauss’s Laws: ∇ · E = ρ/ε0, ∇ · B = 0
        div_E_real = divergence[:, 0, 0]
        div_E_imag = divergence[:, 0, 1]
        div_B_real = divergence[:, 1, 0]
        div_B_imag = divergence[:, 1, 1]

        gauss_E_loss = torch.mean((div_E_real) ** 2) + torch.mean((div_E_imag) ** 2)
        gauss_B_loss = torch.mean((div_B_real) ** 2) + torch.mean((div_B_imag) ** 2)

        return faraday_real + faraday_imag + ampere_real + ampere_imag + gauss_E_loss + gauss_B_loss


# Training Data: Randomly sample points in space
x_train = torch.rand((1000, 3), device="cuda") * 2 - 1  # Random points in [-1,1]^3

# Initialize the model and optimizer
model = HarmonicMaxwellPINN().to("cuda")
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Assume zero free charge (ρ=0) and zero current J for now
J_real = torch.zeros((1000, 3), device="cuda")
J_imag = torch.zeros((1000, 3), device="cuda")

# Constants
omega = 1.0
mu0 = 4 * torch.pi * 1e-7
epsilon0 = 8.85e-12

# Define loss function (Maxwell's Equations Residuals)
maxwell_loss = MaxwellLoss(omega, mu0, epsilon0)

# Training loop
for i in range(1000):
    optimizer.zero_grad()
    loss = maxwell_loss(model, x_train, J_real, J_imag)
    loss.backward()
    optimizer.step()
    print(f"Epoch {i}, Loss: {loss.item()}")

# Visualize a slice of the fields at z=0
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)
Z = np.zeros_like(X)

# Field at z=0
field = model(torch.tensor(np.stack([X, Y, Z], axis=-1).reshape(-1, 3), device="cuda", dtype=torch.float32)).cpu().detach().numpy()
E_real = field[:, :, 0, 0].reshape(3, 100, 100)
E_imag = field[:, :, 1, 0].reshape(3, 100, 100)
B_real = field[:, :, 0, 1].reshape(3, 100, 100)
B_imag = field[:, :, 1, 1].reshape(3, 100, 100)

fig, ax = plt.subplots(2, 2, figsize=(10, 10))
im00 = ax[0, 0].imshow(E_real[0], cmap="viridis")
ax[0, 0].set_title("E_real at z=0")
divider = make_axes_locatable(ax[0, 0])
cax = divider.append_axes("right", size="5%", pad=0.05)
plt.colorbar(im00, cax=cax)

ax[0, 1].imshow(E_imag[0], cmap="viridis")
ax[0, 1].set_title("E_imag at z=0")
ax[1, 0].imshow(B_real[0], cmap="viridis")
ax[1, 0].set_title("B_real at z=0")
ax[1, 1].imshow(B_imag[0], cmap="viridis")
ax[1, 1].set_title("B_imag at z=0")
plt.show()