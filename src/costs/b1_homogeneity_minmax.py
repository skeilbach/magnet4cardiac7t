from .base import BaseCost
from ..data.simulation import SimulationData
from ..data.utils import B1Calculator

import numpy as np


class B1HomogeneityMinMaxCost(BaseCost):
    def __init__(self) -> None:
        super().__init__()
        self.direction = "maximize"
        self.b1_calculator = B1Calculator()

    def calculate_cost(self, simulation_data: SimulationData) -> float:
        b1_field = self.b1_calculator(simulation_data)
        #subject = simulation_data.subject
        
        b1_field_abs = np.abs(b1_field)
        #b1_field_subject_voxels = b1_field_abs[subject]
        return np.mean(b1_field_abs)/(np.max(b1_field_abs) - np.min(b1_field_abs))