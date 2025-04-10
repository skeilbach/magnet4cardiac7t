from dataclasses import dataclass, field
import numpy.typing as npt
import numpy as np
from typing import Optional

@dataclass
class CoilConfig:
    """
    Stores the coil configuration data i.e. the phase and amplitude of each coil.
    """
    phase: npt.NDArray[np.float64] = field(default_factory=lambda: np.zeros((8,), dtype=np.float64))
    amplitude: npt.NDArray[np.float64] = field(default_factory=lambda: np.ones((8,), dtype=np.float64))

    def _post_init_(self):
        self.phase = np.array(self.phase)
        self.amplitude = np.array(self.amplitude)
        
        assert self.phase.shape == self.amplitude.shape, "Phase and amplitude must have the same shape."
        assert self.phase.shape == (8,), "Phase and amplitude must have shape (8,)."

@dataclass
class SimulationData:
    """
    Stores the simulation data for a specific coil configuration.
    """
    simulation_name: str
    properties: npt.NDArray[np.float64]
    field: npt.NDArray[np.float64]
    subject: npt.NDArray[np.bool_]  # The mask of the ROI (subject) in the entire simulated domain
    coil_config: CoilConfig
    
@dataclass
class SimulationRawData:
    """
    Stores the raw simulation data. Each coil contribution is stored separately along an additional dimension.
    """
    simulation_name: str
    properties: npt.NDArray[np.float64]
    field: npt.NDArray[np.float64]
    subject: npt.NDArray[np.bool_]
    coil: npt.NDArray[np.float64]


import torch

@dataclass
class CoilConfigTorch:
    """
    Stores the coil configuration data i.e. the phase and amplitude of each coil using PyTorch tensors.
    """
    phase: torch.Tensor = field(default_factory=lambda: torch.zeros((8,), dtype=torch.float64))
    amplitude: torch.Tensor = field(default_factory=lambda: torch.ones((8,), dtype=torch.float64))
    
    # def __post_init__(self):
    #     self.phase = torch.tensor(self.phase, dtype=torch.float64, requires_grad=True)
    #     self.amplitude = torch.tensor(self.amplitude, dtype=torch.float64, requires_grad=True)
        
    #     assert self.phase.shape == self.amplitude.shape, "Phase and amplitude must have the same shape."
    #     assert self.phase.shape == (8,), "Phase and amplitude must have shape (8,)."

@dataclass
class SimulationDataTorch:
    """
    Stores the simulation data for a specific coil configuration using PyTorch tensors.
    """
    simulation_name: str
    properties: torch.Tensor  # Tensor for properties
    field: torch.Tensor       # Tensor for field data
    subject: torch.Tensor     # Tensor for subject data
    coil_config: CoilConfigTorch  # Coil configuration using PyTorch tensors
    
    # def __post_init__(self):
    #     self.properties = torch.tensor(self.properties, dtype=torch.float64, requires_grad=True)
    #     self.field = torch.tensor(self.field, dtype=torch.float64, requires_grad=True)
    #     self.subject = torch.tensor(self.subject, dtype=torch.bool)

@dataclass
class SimulationRawDataTorch:
    """
    Stores the raw simulation data. Each coil contribution is stored separately along an additional dimension using PyTorch tensors.
    """
    simulation_name: str
    properties: torch.Tensor  # Tensor for properties
    field: torch.Tensor       # Tensor for field data
    subject: torch.Tensor     # Tensor for subject data
    coil: torch.Tensor        # Tensor for coil data
    
    # def __post_init__(self):
    #     self.properties = torch.tensor(self.properties, dtype=torch.float64, requires_grad=True)
    #     self.field = torch.tensor(self.field, dtype=torch.float64, requires_grad=True)
    #     self.subject = torch.tensor(self.subject, dtype=torch.bool)
    #     self.coil = torch.tensor(self.coil, dtype=torch.float64, requires_grad=True)
