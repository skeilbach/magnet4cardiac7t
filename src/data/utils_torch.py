import numpy as np
from .dataclasses_torch import SimulationData

class B1Calculator:
    """
    Class to calculate B1 field from simulation data.
    """

    def __call__(self, simulation_data: SimulationData) -> np.ndarray:
        return self.calculate_b1_field(simulation_data)

    def calculate_b1_field(self, simulation_data: SimulationData) -> np.ndarray:
        b_field = simulation_data.field[1]

        # b1_plus = b_x + i*b_y
        b_field_complex = b_field[0] + 1j*b_field[1]
        b1_plus = 0.5*(b_field_complex[0] + 1j*b_field_complex[1])
        return b1_plus

import torch

class B1CalculatorTorch:
    """
    Class to calculate B1 field from simulation data using PyTorch.
    """

    def __call__(self, simulation_data: SimulationData) -> torch.Tensor:
        return self.calculate_b1_field(simulation_data)

    def calculate_b1_field(self, simulation_data: SimulationData) -> torch.Tensor:
        # Convert the field data to PyTorch tensors
        b_field = simulation_data.field[1]

        # b1_plus = b_x + i*b_y
        b_field_complex = b_field[0] + 1j * b_field[1]
        b1_plus = 0.5 * (b_field_complex[0] + 1j * b_field_complex[1])
        return b1_plus   

class SARCalculator:
    """
    Class to calculate SAR from simulation data.
    """

    def __call__(self, simulation_data: SimulationData) -> np.ndarray:
        return self.calculate_sar(simulation_data)

    def calculate_sar(self, simulation_data: SimulationData) -> np.ndarray:
        e_field = simulation_data.field[0]
        abs_efield_sq = np.sum(e_field**2, axis=(0,1))

        # get the conductivity and density tensors
        conductivity = simulation_data.properties[0]
        density = simulation_data.properties[2]

        pointwise_sar = conductivity*abs_efield_sq/density
        return pointwise_sar
    
class SARCalculatorTorch:
    """
    Class to calculate SAR from simulation data.
    """

    def __call__(self, simulation_data: SimulationData) -> torch.Tensor:
        return self.calculate_sar(simulation_data)

    def calculate_sar(self, simulation_data: SimulationData) -> torch.Tensor:
        e_field = simulation_data.field[0]
        abs_efield_sq = torch.sum(e_field**2, dim=(0,1))

        # get the conductivity and density tensors
        conductivity = simulation_data.properties[0]
        density = simulation_data.properties[2]

        pointwise_sar = conductivity*abs_efield_sq/density
        return pointwise_sar