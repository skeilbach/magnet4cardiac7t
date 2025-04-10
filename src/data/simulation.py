
import numpy.typing as npt
import numpy as np
import h5py
import os
import einops
import torch
from typing import Tuple
from .dataclasses import SimulationRawDataTorch, SimulationDataTorch, CoilConfigTorch



class Simulation:
    def __init__(self, 
                 path: str,
                 coil_path: str = None):
        self.path = path
        self.coil_path = coil_path
        
        self.simulation_raw_data = self._load_raw_simulation_data()
        
    def _load_raw_simulation_data(self) -> SimulationRawDataTorch:
        # Load raw simulation data from path
        
        def read_field() -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
            with h5py.File(self.path) as f:
                re_efield, im_efield = f["efield"]["re"][:], f["efield"]["im"][:]
                re_hfield, im_hfield = f["hfield"]["re"][:], f["hfield"]["im"][:]
                field = np.stack([np.stack([re_efield, im_efield], axis=0), np.stack([re_hfield, im_hfield], axis=0)], axis=0)
            return  torch.tensor(field, dtype=torch.float64, requires_grad=True)

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
            if self.coil_path:
                with h5py.File(self.coil_path) as f:
                    coil = f["masks"][:]
            else:
                coil = None
            return coil
        
        def read_simulation_name() -> str:
            return os.path.basename(self.path)[:-3]

        simulation_raw_data = SimulationRawDataTorch(
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
                     amplitude: npt.NDArray[np.float32],
                     subject: npt.NDArray[np.bool_]) -> npt.NDArray[np.float32]:
        """
        Shift the field calculating field_shifted = field * amplitude (e ^ (phase * 1j)) and summing over all coils in the ROI defined by subject.
        """
        re_phase = np.cos(phase) * amplitude  # shape (8,) -> for 8 dipoles
        im_phase = np.sin(phase) * amplitude
        coeffs_real = np.stack((re_phase, -im_phase), axis=0)  # shape (2, 8)
        coeffs_im = np.stack((im_phase, re_phase), axis=0)
        coeffs = np.stack((coeffs_real, coeffs_im), axis=0)  # shape (2, 2, 8)
        coeffs = einops.repeat(coeffs, 'reimout reim coils -> hf reimout reim coils', hf=2)

        # Determine which indices to keep along each axis
        # For axis 0: Keep index 'i' if *any* value in the mask slice mask[i, :, :] is True.
        # We check for 'any' across axes 1 and 2.
        keep_axis0 = subject.any(axis=(1, 2))  # Result is 1D array of shape (data.shape[0],)

        # For axis 1: Keep index 'j' if *any* value in the mask slice mask[:, j, :] is True.
        # We check for 'any' across axes 0 and 2.
        keep_axis1 = subject.any(axis=(0, 2))  # Result is 1D array of shape (data.shape[1],)

        # For axis 2: Keep index 'k' if *any* value in the mask slice mask[:, :, k] is True.
        # We check for 'any' across axes 0 and 1.
        keep_axis2 = subject.any(axis=(0, 1))  # Result is 1D array of shape (data.shape[2],)

        # Use np.ix_ for advanced indexing (use prior knowledge of the shape of the field data which has shape (2, 2, 3, 121, 111, 126, 8))
        indexers = np.ix_(
            np.full(2, True),  # Dimension 0 (shape 2) - keep all
            np.full(2, True),  # Dimension 1 (shape 2) - keep all
            np.full(3, True),  # Dimension 2 (shape 3) - keep all
            keep_axis0,  # Dimension 3 (shape 121) - filter using mask's 1st dim logic
            keep_axis1,  # Dimension 4 (shape 111) - filter using mask's 2nd dim logic
            keep_axis2,  # Dimension 5 (shape 126) - filter using mask's 3rd dim logic
            np.full(8, True)  # Dimension 6 (shape 8) - keep all
        )

        # Apply the indexing to the high-dimensional data array
        field_sliced = field[indexers]

        # Apply the indexing to the data array
        field_shift = einops.einsum(field_sliced, coeffs, 'hf reim fieldxyz ... coils, hf reimout reim coils -> hf reimout fieldxyz ...')

        return field_shift
    
    def _shift_fieldTorch(self, 
                    field: torch.Tensor, 
                    phase: torch.Tensor, 
                    amplitude: torch.Tensor) -> torch.Tensor:
        """
        Shift the field calculating field_shifted = field * amplitude (e ^ (phase * 1j)) and summing over all coils.
        """
        re_phase = torch.cos(phase) * amplitude
        im_phase = torch.sin(phase) * amplitude
        coeffs_real = torch.stack((re_phase, -im_phase), dim=0)
        coeffs_im = torch.stack((im_phase, re_phase), dim=0)
        coeffs = torch.stack((coeffs_real, coeffs_im), dim=0)
        coeffs = einops.repeat(coeffs, 'reimout reim coils -> hf reimout reim coils', hf=2)
        field_shift = einops.einsum(field, coeffs, 'hf reim fieldxyz ... coils, hf reimout reim coils -> hf reimout fieldxyz ...')

        return field_shift
    
    def phase_shift(self, coil_config: CoilConfigTorch) -> SimulationDataTorch:
        
        field_shifted = self._shift_fieldTorch(self.simulation_raw_data.field, coil_config.phase, coil_config.amplitude)
        
        simulation_data = SimulationDataTorch(
            simulation_name=self.simulation_raw_data.simulation_name,
            properties=self.simulation_raw_data.properties,
            field=field_shifted,
            subject=self.simulation_raw_data.subject,
            coil_config=coil_config
        )
        return simulation_data
    
    def __call__(self, coil_config: CoilConfigTorch) -> SimulationDataTorch:
        return self.phase_shift(coil_config)