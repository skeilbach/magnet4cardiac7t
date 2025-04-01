from ..data import SimulationData
from abc import ABC, abstractmethod

class BaseCost(ABC):
    def __init__(self) -> None:
        self.direction = "min"
        assert self.direction in ["min", "max"], f"Invalid direction: {self.direction}"
        
    def __call__(self, simulation_data: SimulationData) -> float:
        return self.calculate_cost(simulation_data)

    @abstractmethod
    def calculate_cost(self, simulation_data: SimulationData) -> float:
        raise NotImplementedError