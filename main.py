from src.costs.base import BaseCost
from src.optimizers import OurOptimizer
from src.data import Simulation, CoilConfig
from src.costs.b1_homogeneity_torch import B1HomogeneityCost
import time

import numpy as np

def run(simulation: Simulation, 
        cost_function: BaseCost,
        timeout: int = 10) -> CoilConfig:
    """
        Main function to run the optimization, returns the best coil configuration

        Args:
            simulation: Simulation object
            cost_function: Cost function object
            timeout: Time (in seconds) after which the evaluation script will be terminated
    """
    optimizer = OurOptimizer(cost_function=cost_function)
    best_coil_config = optimizer.optimize(simulation)#, timeout=timeout)
    return best_coil_config

if __name__ == "__main__":
    # Example usage
    simulation = Simulation(path="data/simulations/children_1_tubes_6_id_23713.h5")
    cost_function = B1HomogeneityCost()

    best_coil_config = run(simulation, cost_function)

    print("Best coil configuration:")
    print(best_coil_config)
