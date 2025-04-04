from main import run

from src.costs import B1HomogeneityCost
from src.data import Simulation
from src.utils import evaluate_coil_config

import numpy as np
import json

if __name__ == "__main__":
    # Load simulation data
    simulation = Simulation("data/simulations/children_2_tubes_7_id_3012.h5")
    
    # Define cost function
    cost_function = B1HomogeneityCost()
    
    # Run optimization
    best_coil_config = run(simulation=simulation, cost_function=cost_function)
    
    # Evaluate best coil configuration
    result = evaluate_coil_config(best_coil_config, simulation, cost_function)

    # Save results to JSON file
    with open("results.json", "w") as f:
        json.dump(result, f, indent=4)
