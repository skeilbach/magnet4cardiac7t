from src.costs.base import BaseCost
from src.optimizers import DummyOptimizer
from src.data import Simulation, CoilConfig
from src.costs.b1_homogeneity import B1HomogeneityCost
from src.optimizers import DifferentialEvolutionOptimizer
import time

import numpy as np

def run(simulation: Simulation, 
        cost_function: BaseCost,
        timeout: int = 300) -> CoilConfig:
    """
        Main function to run the optimization, returns the best coil configuration

        Args:
            simulation: Simulation object
            cost_function: Cost function object
            timeout: Time (in seconds) after which the evaluation script will be terminated
    """
    #optimizer = DummyOptimizer(cost_function=cost_function)
    #best_coil_config = optimizer.optimize(simulation, timeout=timeout)
    initial_solutions = [
                            #[np.random.uniform(low=0, high=2*np.pi) for _ in range(8)] + [np.random.uniform(low=0, high=1) for _ in range(8)]
    ]
    # Create the optimizer, passing the guesses
    optimizer = DifferentialEvolutionOptimizer(
         cost_function=cost_function,
         simulation=simulation,
         population_size=50,
         max_generations=100,
         initial_guess=initial_solutions, # Pass the list here
         minimize=False,
         seed=42
    )
    best_coil_config, best_score, log = optimizer.optimize(timeout=timeout)

    return best_coil_config

if __name__ == "__main__":
    # Example usage
    simulation = Simulation(path="data/simulations/children_1_tubes_6_id_23713.h5", sparse_sampling=True)
    cost_function = B1HomogeneityCost()

    best_coil_config = run(simulation, cost_function)

    print("\nBest coil configuration:")
    print(best_coil_config)
