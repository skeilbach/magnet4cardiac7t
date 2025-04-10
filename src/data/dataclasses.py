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
    coil: Optional[npt.NDArray[np.float64]]