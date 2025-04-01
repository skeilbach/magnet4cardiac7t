from ..data import Simulation, SimulationData, CoilConfig
from abc import ABC, abstractmethod

from typing import Callable

class BaseOptimizer(ABC):
    def __init__(self,
                 cost_function: Callable[[SimulationData], float],
                 direction: str = "min") -> None:
        self.cost_function = cost_function
        self.direction = direction
        assert self.direction in ["min", "max"], f"Invalid direction: {self.direction}"

    @abstractmethod
    def optimize(self, simulation: Simulation) -> CoilConfig:
        raise NotImplementedError