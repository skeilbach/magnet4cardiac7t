from ..data.simulation import Simulation, SimulationDataTorch, CoilConfigTorch
from ..costs.base import BaseCost
from abc import ABC, abstractmethod

from typing import Callable

class BaseOptimizer(ABC):
    def __init__(self,
                 cost_function: BaseCost) -> None:
        self.cost_function = cost_function
        self.direction = cost_function.direction
        assert self.direction in ["minimize", "maximize"], f"Invalid direction: {self.direction}"

    @abstractmethod
    def optimize(self, simulation: Simulation) -> CoilConfigTorch:
        raise NotImplementedError