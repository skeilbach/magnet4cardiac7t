from ..data.simulation import Simulation, SimulationDataTorch, CoilConfigTorch, CoilConfig
from ..costs.base import BaseCost
from .base import BaseOptimizer

from typing import Callable
import numpy as np
import torch
from tqdm import trange
import time 

class OurOptimizer(BaseOptimizer):
    """
    DummyOptimizer is a dummy optimizer that randomly samples coil configurations and returns the best one.
    """
    def __init__(self,
                 cost_function: BaseCost,
                 max_iter: int = 200) -> None:
        super().__init__(cost_function)
        self.max_iter = max_iter

    def conv_coilconfig(self, coilconfig_torch):
        x1 = coilconfig_torch.phase.detach().numpy()
        x2 = coilconfig_torch.amplitude.detach().numpy()
        return CoilConfig(phase=x1, amplitude=x2)

    def optimize(self, simulation: Simulation):
      
        raw_x1 = torch.rand(8, dtype=torch.float64, requires_grad=True) #phase
        raw_x2 = torch.rand(8, dtype=torch.float64, requires_grad=True) #amplitude
        # Optimizer
        optimizer = torch.optim.Adam([raw_x1, raw_x2], lr=0.1)
        # Scheduler: decay LR by 0.9 every 20 steps
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)
        # Optimization loop
        #pbar = trange(self.max_iter)
        start_time = time.time()
        timeout = 275#
        i = 0
        while time.time() - start_time < timeout:
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
            i += 1

            # Print current LR and cost
            if i % 10 == 0:
                current_lr = scheduler.get_last_lr()[0]
                print(f"Time: {time.time() - start_time:.1f}s, Iter {i}, Cost: {cost.item():.4f}, LR: {current_lr:.6f}")
        return coil_config