from dataclasses import dataclass
import numpy.typing as npt
import numpy as np

@dataclass
class Dataitem:
    simulation_name: str

@dataclass
class CoilConfig:
    phase: npt.NDArray[np.float32]
    amplitude: npt.NDArray[np.float32]

class Simulation:
    def __init__(self, 
                 path: str):
        self.path = path

    def phase_shift(self, coil_config: CoilConfig) -> Dataitem:
        return coil_config.phase
        