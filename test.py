import torch

# Initial raw (unconstrained) parameters
raw_x1 = torch.randn(8, requires_grad=True)
raw_x2 = torch.randn(8, requires_grad=True)

# Optimizer
optimizer = torch.optim.Adam([raw_x1, raw_x2], lr=0.01)

# Scheduler: decay LR by 0.9 every 20 steps
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)

# Dummy simulation function (replace this)
def simulate_cost(x1, x2):
    # Just a mock cost for testing
    return -((x1 - torch.pi).pow(2).sum() + (x2 - 0.5).pow(2).sum())

# Optimization loop
for i in range(100):
    optimizer.zero_grad()

    # Apply bounds using sigmoid
    x1 = 2 * torch.pi * torch.sigmoid(raw_x1)  # ∈ [0, 2π]
    x2 = torch.sigmoid(raw_x2)                 # ∈ [0, 1]

    # Evaluate cost
    cost = simulate_cost(x1, x2)

    # Loss is negative cost (because we want to maximize)
    loss = -cost
    loss.backward()
    optimizer.step()
    scheduler.step()  # Update learning rate

    # Print current LR and cost
    if i % 10 == 0:
        current_lr = scheduler.get_last_lr()[0]
        print(f"Iter {i}, Cost: {cost.item():.4f}, LR: {current_lr:.6f}")
