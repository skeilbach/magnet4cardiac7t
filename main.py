from src.costs import B1HomogeneityCost
from src.optimizers import DummyOptimizer
from src.data import Simulation, CoilConfig

import numpy as np


        
if __name__ == "__main__":
    # Load simulation data
    simulation = Simulation("data/simulations/children_2_tubes_7_id_3012.h5")
    
    # Define optimizer
    optimizer = DummyOptimizer(cost_function=B1HomogeneityCost(), direction="max")
    
    # Optimize
    best_coil_config = optimizer.optimize(simulation)
    
    # Shift field
    simulation_data = simulation(best_coil_config)
    
    # Calculate cost
    cost_function = B1HomogeneityCost()
    b1_field = cost_function._calculate_b1_field(simulation_data.field)
    b1_field = np.abs(b1_field)*simulation_data.subject
    
    # plot field
    import matplotlib.pyplot as plt
    efield_abs = np.linalg.norm(simulation_data.field[1], axis=(0, 1))
    plt.imshow(b1_field[:, :, 50])
    plt.colorbar()
    plt.show()