from .data import CoilConfigTorch, Simulation
from .costs.base import BaseCost

from typing import Dict, Any

def evaluate_coil_config(coil_config: CoilConfigTorch, 
                         simulation: Simulation,
                         cost_function: BaseCost) -> Dict[str, Any]:
    """
    Evaluates the coil configuration using the cost function.

    Args:
        coil_config: Coil configuration to evaluate.
        simulation: Simulation object.
        cost_function: Cost function object.

    Returns:
        A dictionary containing the best coil configuration, cost, and cost improvement.
    """
    default_coil_config = CoilConfigTorch()

    simulation_data = simulation(coil_config)
    simulation_data_default = simulation(default_coil_config)

    # Calculate cost for both configurations
    default_coil_config_cost = cost_function(simulation_data_default)
    best_coil_config_cost = cost_function(simulation_data)

    # Calculate cost improvement
    cost_improvement_absolute = (default_coil_config_cost - best_coil_config_cost)
    cost_improvement_relative = (best_coil_config_cost - default_coil_config_cost) / default_coil_config_cost

    # Create a dictionary to store the results
    result = {
        "best_coil_phase": coil_config.phase.detach().cpu().tolist(),
        "best_coil_amplitude": coil_config.amplitude.detach().cpu().tolist(),
        "best_coil_config_cost": best_coil_config_cost.detach().cpu().item(),
        "default_coil_config_cost": default_coil_config_cost.detach().cpu().item(),
        "cost_improvement_absolute": cost_improvement_absolute.detach().cpu().item(),
        "cost_improvement_relative": cost_improvement_relative.detach().cpu().item(),
        "cost_function_name": cost_function.__class__.__name__,
        "cost_function_direction": cost_function.direction,
        "simulation_data": simulation_data.simulation_name,
    }
    return result