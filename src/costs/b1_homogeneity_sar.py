from .base import BaseCost
from ..data.simulation import SimulationData
from ..data.utils import B1Calculator, SARCalculator

import numpy as np


class B1HomogeneitySARCost(BaseCost):
    def __init__(self,
                 weight: float = 100) -> None:
        super().__init__()
        self.direction = "maximize"
        self.b1_calculator = B1Calculator()
        self.sar_calculator = SARCalculator()
        self.weight = weight

    def calculate_cost(self, simulation_data: SimulationData) -> float:
        b1_field = self.b1_calculator(simulation_data)
        sar = self.sar_calculator(simulation_data)
        
        subject = simulation_data.subject
        
        b1_field_abs = np.abs(b1_field)
        b1_field_subject_voxels = b1_field_abs[subject]
        
        sar_subject_voxels = sar[subject]
        
        one_over_cov = np.mean(b1_field_subject_voxels)/np.std(b1_field_subject_voxels)
        peak_sar = np.max(sar_subject_voxels)
        peak_sar_sqrt = np.sqrt(peak_sar)
        min_b1 = np.min(b1_field_subject_voxels)
        print("one_over_cov", one_over_cov)
        print("peak_sar", peak_sar)
        print("peak_sar_sqrt", peak_sar_sqrt)
        print("min_b1", min_b1)
        return one_over_cov + self.weight * min_b1 / peak_sar_sqrt