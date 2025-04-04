# Hackathon 
### Spring School on *Physics Informed Machine Learning for Medical Sciences*
Welcome to the Hackathon accompanying the 2025 Spring School on *Physics Informed Machine Learning for Medical Sciences*.

This repo will serve you as a starting point for solving the task described below.

## Getting Started
To setup the environment, clone the repository, create a new environment and run the following commands:
```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Task description
You are given a 3D object sitting inside an array of 8 dipoles, positioned radially around the object as seen in Figure 1.
Your task is to optimize the phase and amplitude of the coils (coil configuration) with respect to a specified cost function (e.g. $B_1$ homogeneity, minimal peak SAR etc.).

The domain is a simulated MRI environment with electric $E$ and magnetic $H$ field already calculated at each point using the CST studio suite and exported at a voxel size of 4mm. The contribution of each dipole $i$ is calculated and stored separately $E_i, H_i$, and the total fields are the sum of the contributions of each dipole.

When the phase and amplitude for a specific dipole $i$ is changed to $\varphi$ and $A_i$ respectively, the resulting contribution of this dipole can then be calculated as 
$$
\hat{E_i} = A_i e^{i\varphi} E_i\\
\hat{H_i} = A_i e^{i\varphi} H_i 
$$

## Ideas where to go from here
- Implement parallel processing and evaluation
- Optimize the calculation of phase-shifted fields
- Optimize the cost function calculation
- Run on GPU