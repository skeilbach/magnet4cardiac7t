from ..data.simulation_torch import SimulationDataTorch
from ..data.simulation import SimulationData
from abc import ABC, abstractmethod

class BaseCost(ABC):
    def __init__(self) -> None:
        self.direction = "minimize"
        assert self.direction in ["minimize", "maximize"], f"Invalid direction: {self.direction}"
        
    def __call__(self, simulation_data: SimulationDataTorch | SimulationData) -> float:
        return self.calculate_cost(simulation_data)

    @abstractmethod
    def calculate_cost(self, simulation_data: SimulationDataTorch | SimulationData) -> float:
        raise NotImplementedError