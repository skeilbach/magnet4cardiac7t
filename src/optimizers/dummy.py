from ..data import Simulation, SimulationData, CoilConfig
from .base import BaseOptimizer

from typing import Callable
import numpy as np

from tqdm import trange


class DummyOptimizer(BaseOptimizer):
    """
    DummyOptimizer is a dummy optimizer that randomly samples coil configurations and returns the best one.
    """
    def __init__(self,
                 cost_function: Callable[[SimulationData], float],
                 direction: str = "min",
                 num_samples: int = 100) -> None:
        super().__init__(cost_function, direction)
        self.num_samples = num_samples
        
    def _sample_coil_config(self) -> CoilConfig:
        phase = np.random.uniform(low=0, high=2*np.pi, size=(8,))
        amplitude = np.random.uniform(low=0, high=10, size=(8,))
        return CoilConfig(phase=phase, amplitude=amplitude)
        
    def optimize(self, simulation: Simulation):
        best_coil_config = None
        best_cost = -np.inf if self.direction == "max" else np.inf
        
        pbar = trange(self.num_samples)
        for i in pbar:
            coil_config = self._sample_coil_config()
            simulation_data = simulation(coil_config)
            
            cost = self.cost_function(simulation_data)
            if (self.direction == "min" and cost < best_cost) or (self.direction == "max" and cost > best_cost):
                best_cost = cost
                best_coil_config = coil_config
                pbar.set_postfix_str(f"Best cost {best_cost:.2f}")
        
        return best_coil_config