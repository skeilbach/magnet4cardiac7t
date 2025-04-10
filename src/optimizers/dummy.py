from ..data.simulation_torch import Simulation, SimulationDataTorch, CoilConfigTorch
from ..costs.base import BaseCost
from .base import BaseOptimizer
import time
from typing import Callable
import numpy as np
from tqdm import trange


class DummyOptimizer(BaseOptimizer):
    """
    DummyOptimizer is a dummy optimizer that randomly samples coil configurations and returns the best one.
    """
    def __init__(self,
                 cost_function: BaseCost,
                 max_iter: int = 100) -> None:
        super().__init__(cost_function)
        self.max_iter = max_iter
        
    def _sample_coil_config(self) -> CoilConfigTorch:
        phase = np.random.uniform(low=0, high=2*np.pi, size=(8,))
        amplitude = np.random.uniform(low=0, high=1, size=(8,))
        return CoilConfigTorch(phase=phase, amplitude=amplitude)
        
    def optimize(self, simulation: Simulation):
        best_coil_config = None
        best_cost = -np.inf if self.direction == "maximize" else np.inf

        start_time = time.time()
        pbar = trange(self.max_iter)

        for i in pbar:
            if time.time() - start_time > timeout:
                print("⏱️ Timeout reached — stopping optimization.")
                break

            coil_config = self._sample_coil_config()
            simulation_data = simulation(coil_config)
            cost = self.cost_function(simulation_data)

            if (self.direction == "minimize" and cost < best_cost) or \
                    (self.direction == "maximize" and cost > best_cost):
                best_cost = cost
                best_coil_config = coil_config
                pbar.set_postfix_str(f"Best cost {best_cost:.2f}")

        return best_coil_config