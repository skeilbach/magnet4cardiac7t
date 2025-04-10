from ..data.simulation import Simulation, SimulationDataTorch, CoilConfigTorch
from ..costs.base import BaseCost
from .base import BaseOptimizer

from typing import Callable
import numpy as np
import torch
from tqdm import trange
import einops
import h5py
from tqdm import trange

class OurOptimizer(BaseOptimizer):
    """
    DummyOptimizer is a dummy optimizer that randomly samples coil configurations and returns the best one.
    """
    def __init__(self,
                 cost_function: BaseCost,
                 max_iter: int = 100) -> None:
        super().__init__(cost_function)
        self.max_iter = max_iter

    def optimize(self, simulation: Simulation):
      
        raw_x1 = torch.rand(8, dtype=torch.float64, requires_grad=True) #phase
        raw_x2 = torch.rand(8, dtype=torch.float64, requires_grad=True) #amplitude
        # Optimizer
        optimizer = torch.optim.Adam([raw_x1, raw_x2], lr=0.1)
        # Scheduler: decay LR by 0.9 every 20 steps
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.9)
        # Optimization loop
        for i in range(self.max_iter):
            optimizer.zero_grad()

            # Apply bounds using sigmoid
            x1 = 2 * torch.pi * torch.sigmoid(raw_x1)  # ∈ [0, 2π]
            x2 = torch.sigmoid(raw_x2)                 # ∈ [0, 1]

            coil_config = CoilConfigTorch(phase=x1, amplitude=x2)
            simulation_data = simulation(coil_config)
            
            cost = self.cost_function(simulation_data)

            # Loss is negative cost (because we want to maximize)
            loss = -cost
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate

            # Print current LR and cost
            if i % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Iter {i}, Cost: {cost.item():.4f}, LR: {current_lr:.6f}")
        return coil_config