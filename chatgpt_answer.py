import torch
import torch.nn as nn
import torch.optim as optim
import einops
from maxwell_pinn import HarmonicMaxwellPINN, PartialDerivatives, Divergence, Curl

        

# Define loss function (Maxwell's Equations Residuals)
def maxwell_loss(model, x, J_real, J_imag, omega, mu0, epsilon0):
    x.requires_grad_(True)
    #E_real, E_imag, B_real, B_imag = model(x)
    #E_real = 2*x
    E_real = model(x)

    # Compute spatial gradients
    grad_E_real = torch.autograd.grad(E_real, x, torch.ones_like(E_real), create_graph=True)[0]
    print(grad_E_real)
    grad_E_imag = torch.autograd.grad(E_imag, x, torch.ones_like(E_imag), create_graph=True)[0]
    grad_B_real = torch.autograd.grad(B_real, x, torch.ones_like(B_real), create_graph=True)[0]
    grad_B_imag = torch.autograd.grad(B_imag, x, torch.ones_like(B_imag), create_graph=True)[0]


    # Compute curls
    curl_E_real = torch.cross(grad_E_real, torch.tensor([1.0, 1.0, 1.0], device=x.device), dim=1)
    curl_E_imag = torch.cross(grad_E_imag, torch.tensor([1.0, 1.0, 1.0], device=x.device), dim=1)
    curl_B_real = torch.cross(grad_B_real, torch.tensor([1.0, 1.0, 1.0], device=x.device), dim=1)
    curl_B_imag = torch.cross(grad_B_imag, torch.tensor([1.0, 1.0, 1.0], device=x.device), dim=1)

    # Faraday’s Law: ∇ × E = -jωB
    faraday_real = torch.mean((curl_E_real + omega * B_imag) ** 2)
    faraday_imag = torch.mean((curl_E_imag - omega * B_real) ** 2)

    # Ampère’s Law: ∇ × B = μ0J + jωμ0ε0E
    ampere_real = torch.mean((curl_B_real - mu0 * J_real - omega * mu0 * epsilon0 * E_imag) ** 2)
    ampere_imag = torch.mean((curl_B_imag - mu0 * J_imag + omega * mu0 * epsilon0 * E_real) ** 2)

    # Gauss’s Laws: ∇ · E = ρ/ε0, ∇ · B = 0
    div_E = torch.sum(grad_E_real, dim=1)
    div_B = torch.sum(grad_B_real, dim=1)

    gauss_E_loss = torch.mean((div_E) ** 2)
    gauss_B_loss = torch.mean((div_B) ** 2)

    return faraday_real + faraday_imag + ampere_real + ampere_imag + gauss_E_loss + gauss_B_loss


# Training Data: Randomly sample points in space
x_train = torch.rand((1000, 3), device="cuda") * 2 - 1  # Random points in [-1,1]^3

# Initialize the model and optimizer
model = HarmonicMaxwellPINN().to("cuda")
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Assume zero free charge (ρ=0) and zero current J for now
J_real = torch.zeros((1000, 3), device="cuda")
J_imag = torch.zeros((1000, 3), device="cuda")

# Constants
omega = 1.0
mu0 = 4 * torch.pi * 1e-7
epsilon0 = 8.85e-12

# Train the PINN
for epoch in range(10000):
    optimizer.zero_grad()
    loss = maxwell_loss(model, x_train, J_real, J_imag, omega, mu0, epsilon0)
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")