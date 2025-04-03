
import numpy.typing as npt
import numpy as np
import h5py
import os
import einops

from typing import Tuple
from .dataclasses import SimulationRawData, SimulationData, CoilConfig



class Simulation:
    def __init__(self, 
                 path: str,
                 coil_path: str = "data/antenna/antenna.h5"):
        self.path = path
        self.coil_path = coil_path
        
        self.simulation_raw_data = self._load_raw_simulation_data()
        
    def _load_raw_simulation_data(self) -> SimulationRawData:
        # Load raw simulation data from path
        
        def read_field() -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
            with h5py.File(self.path) as f:
                re_efield, im_efield = f["efield"]["re"][:], f["efield"]["im"][:]
                re_hfield, im_hfield = f["hfield"]["re"][:], f["hfield"]["im"][:]
                field = np.stack([np.stack([re_efield, im_efield], axis=0), np.stack([re_hfield, im_hfield], axis=0)], axis=0)
            return field

        def read_physical_properties() -> npt.NDArray[np.float32]:
            with h5py.File(self.path) as f:
                physical_properties = f["input"][:]
            return physical_properties
        
        def read_subject_mask() -> npt.NDArray[np.bool_]:
            with h5py.File(self.path) as f:
                subject = f["subject"][:]
            subject = np.max(subject, axis=-1)
            return subject
        
        def read_coil_mask() -> npt.NDArray[np.float32]:
            with h5py.File(self.coil_path) as f:
                coil = f["masks"][:]
            return coil
        
        def read_simulation_name() -> str:
            return os.path.basename(self.path)[:-3]

        simulation_raw_data = SimulationRawData(
            simulation_name=read_simulation_name(),
            properties=read_physical_properties(),
            field=read_field(),
            subject=read_subject_mask(),
            coil=read_coil_mask()
        )
        
        return simulation_raw_data
    
    def _shift_field(self, 
                     field: npt.NDArray[np.float32], 
                     phase: npt.NDArray[np.float32], 
                     amplitude: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        """
        Shift the field calculating field_shifted = field * amplitude (e ^ (phase * 1j)) and summing over all coils.
        """
        re_phase = np.cos(phase) * amplitude
        im_phase = np.sin(phase) * amplitude
        coeffs_real = np.stack((re_phase, -im_phase), axis=0)
        coeffs_im = np.stack((im_phase, re_phase), axis=0)
        coeffs = np.stack((coeffs_real, coeffs_im), axis=0)
        coeffs = einops.repeat(coeffs, 'reimout reim coils -> hf reimout reim coils', hf=2)
        field_shift = einops.einsum(field, coeffs, 'hf reim fieldxyz ... coils, hf reimout reim coils -> hf reimout fieldxyz ...')
        return field_shift

    def phase_shift(self, coil_config: CoilConfig) -> SimulationData:
        
        field_shifted = self._shift_field(self.simulation_raw_data.field, coil_config.phase, coil_config.amplitude)
        
        simulation_data = SimulationData(
            simulation_name=self.simulation_raw_data.simulation_name,
            properties=self.simulation_raw_data.properties,
            field=field_shifted,
            subject=self.simulation_raw_data.subject,
            coil_config=coil_config
        )
        return simulation_data
    
    def __call__(self, coil_config: CoilConfig) -> SimulationData:
        return self.phase_shift(coil_config)