from .base import BaseCost
from ..data.simulation import SimulationData

import numpy as np


class B1HomogeneityCost(BaseCost):
    def __init__(self) -> None:
        super().__init__()
        self.direction = "minimize"
        
    def _calculate_b1_field(self, field: np.ndarray) -> np.ndarray:
        b_field = field[1]

        # b1_plus = b_x + i*b_y
        b_field_complex = b_field[0] + 1j*b_field[1]
        b1_plus = 0.5*(b_field_complex[0] + 1j*b_field_complex[1])
        return b1_plus

    def calculate_cost(self, simulation_data: SimulationData) -> float:
        b1_field = self._calculate_b1_field(simulation_data.field)
        subject = simulation_data.subject
        
        b1_field_abs = np.abs(b1_field)
        b1_field_subject_voxels = b1_field_abs[subject]
        return np.mean(b1_field_subject_voxels)/np.std(b1_field_subject_voxels)