# Imports remain the same...
from ..data.simulation import Simulation, SimulationData, CoilConfig
from ..costs.base import BaseCost
from .base import BaseOptimizer
import time
from typing import Callable, Tuple, List, Optional # Added type hints
import numpy as np
import random
from tqdm import tqdm

# --- DEAP Imports ---
from deap import base, creator, tools, algorithms

class DifferentialEvolutionOptimizer(BaseOptimizer):
    """
    Encapsulates the Differential Evolution algorithm using DEAP,
    allowing for initial population seeding.
    """
    def __init__(self,
                 cost_function: BaseCost,
                 simulation: Simulation,
                 num_dimensions: int = 16,
                 bounds: list[tuple[float, float]] = None,
                 population_size: int = 50,
                 max_generations: int = 100,
                 F_mutation: float = 0.7,
                 CR_crossover: float = 0.9,
                 minimize: bool = False,
                 initial_guess: Optional[List[List[float]]] = None, # <-- Add initial_guess parameter
                 phase_indices = None,
                 seed=None):
        """
        Initializes the DE optimizer.

        Args:
            cost_function: The cost function to optimize.
            simulation: The simulation object used for evaluation.
            num_dimensions: The total number of parameters (e.g., phases + amplitudes).
            bounds: List of (min, max) tuples for each dimension. Defaults to phases(0,2pi), amps(0,1).
            population_size: The number of individuals in the population.
            max_generations: The maximum number of generations to run.
            F_mutation: The differential weight (mutation factor).
            CR_crossover: The crossover probability.
            minimize: If True, minimize the objective function; otherwise, maximize.
            initial_guess: Optional list of initial individuals (each as a list/tuple of parameters)
                           to seed the population.
            phase_indices: Optional range or list of indices for phase parameters (for wrapping). Default: range(num_dimensions/2).
            seed: Optional random seed for reproducibility.
        """
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        super().__init__(cost_function)
        self.simulation = simulation
        self.num_dimensions = num_dimensions

        if bounds is None:
            # Default bounds assuming even split: first half phases, second half amplitudes
            if num_dimensions % 2 != 0:
                raise ValueError("num_dimensions must be even if default bounds are used")
            phase_dim = num_dimensions // 2
            amp_dim = num_dimensions // 2
            self.bounds = [(0, 2*np.pi)] * phase_dim + [(0, 1)] * amp_dim
        else:
             if len(bounds) != num_dimensions:
                 raise ValueError(f"Length of bounds ({len(bounds)}) must match num_dimensions ({num_dimensions})")
             self.bounds = bounds

        self.population_size = population_size
        self.max_generations = max_generations
        self.F = F_mutation
        self.CR = CR_crossover
        self.minimize = minimize

        # --- Store and validate initial_guess ---
        self.initial_guess = []
        if initial_guess:
             print(f"Received {len(initial_guess)} initial guess(es).")
             for i, guess in enumerate(initial_guess):
                 if len(guess) != self.num_dimensions:
                     raise ValueError(f"Initial guess {i} has length {len(guess)}, but expected {self.num_dimensions} dimensions.")
                 # Store the valid guess
                 self.initial_guess.append(list(guess)) # Ensure it's a mutable list internally
             # Warn if more guesses provided than population size
             if len(self.initial_guess) > self.population_size:
                 print(f"Warning: Provided {len(self.initial_guess)} initial guesses, but population size is {self.population_size}. Only the first {self.population_size} will be used.")
                 self.initial_guess = self.initial_guess[:self.population_size]


        if phase_indices is None and num_dimensions % 2 == 0:
            self.phase_indices = range(num_dimensions // 2)
        elif phase_indices is not None:
            self.phase_indices = phase_indices
        else:
            self.phase_indices = []

        # --- DEAP Creator Setup (Unique per instance) ---
        creator_suffix = f"_{id(self)}"
        fitness_name = f"Fitness{creator_suffix}"
        individual_name = f"Individual{creator_suffix}"

        if hasattr(creator, fitness_name): delattr(creator, fitness_name)
        if hasattr(creator, individual_name): delattr(creator, individual_name)

        creator.create(fitness_name, base.Fitness, weights=(-1.0,) if minimize else (1.0,))
        creator.create(individual_name, list, fitness=getattr(creator, fitness_name))
        self.Individual = getattr(creator, individual_name) # Store reference

        # --- DEAP Toolbox Setup ---
        self.toolbox = base.Toolbox()
        # _initIndividual is now only used for the *random* part of the population
        self.toolbox.register("attribute_generator", self._generateAttribute)
        self.toolbox.register("individual_random", tools.initRepeat, self.Individual, self.toolbox.attribute_generator, n=self.num_dimensions)
        self.toolbox.register("population_random", tools.initRepeat, list, self.toolbox.individual_random)
        self.toolbox.register("evaluate", self._objective_func)
        self.toolbox.register("clone", lambda ind: self.Individual(ind[:]))

        # --- Statistics Setup ---
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        min_max_key = "min" if minimize else "max"
        self.stats.register(min_max_key, np.min if minimize else np.max)

    # Need a way to generate single attributes respecting bounds for initRepeat
    def _generateAttribute(self, index):
        low, up = self.bounds[index]
        val = random.uniform(low, up)
        # Apply phase wrapping during generation if applicable
        if self.phase_indices is not None and index in self.phase_indices:
            if low == 0.0 and up > low:
                val = val % up
        return val

    # Modify _initIndividual to use the new attribute generator
    # This is now only used if we need *random* individuals
    def _initRandomIndividual(self, icls):
        """Initializes a single random individual respecting bounds & wrapping."""
        ind = [self._generateAttribute(i) for i in range(self.num_dimensions)]
        return icls(ind) # Wrap in the Individual class


    # (Keep _checkBounds and _objective_func as they were in the previous good answer)
    def _checkBounds(self, individual):
        """Helper method to enforce bounds and wrap phases."""
        for i in range(self.num_dimensions):
            low, up = self.bounds[i]
            # Clip first
            individual[i] = np.clip(individual[i], low, up)
            # Optional: Phase wrapping
            if self.phase_indices is not None and i in self.phase_indices:
                if low == 0.0 and up > low:
                    individual[i] = individual[i] % up
        return individual

    def _objective_func(self, individual: List[float]) -> Tuple[float]:
        """Evaluates an individual."""
        if self.num_dimensions % 2 != 0:
            raise ValueError("Cannot automatically split odd num_dimensions")
        phase_dim = self.num_dimensions // 2
        phases = np.array(individual[:phase_dim])
        amplitudes = np.array(individual[phase_dim:])
        coil_config = CoilConfig(phase=phases, amplitude=amplitudes)
        simulation_data = self.simulation(coil_config)
        fitness_value = self.cost_function(simulation_data)
        if not isinstance(fitness_value, (int, float)) or np.isnan(fitness_value) or np.isinf(fitness_value):
            # print(f"Warning: Objective func returned invalid value: {fitness_value}. Assigning worst.")
            worst_fitness = float('-inf') if not self.minimize else float('inf')
            return (worst_fitness,)
        return (fitness_value,)


    def optimize(self, timeout: Optional[int] = None) -> Tuple[Optional[List[float]], float, tools.Logbook]:
        """
        Runs the Differential Evolution optimization process.
        Uses initial_guess (if provided) to seed the population.
        """
        print(f"Starting Differential Evolution {'Minimization' if self.minimize else 'Maximization'}...")
        # ... (rest of initial prints) ...

        start_time = time.time()
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (self.stats.fields if self.stats else [])
        hof = tools.HallOfFame(1)

        # --- Initialize Population with Seeding ---
        pop = []
        try:
            # 1. Add individuals from initial_guess
            if self.initial_guess:
                print(f"Seeding population with {len(self.initial_guess)} provided individual(s).")
                for guess in self.initial_guess:
                    # Create DEAP individual, check bounds/wrap phases
                    ind = self._checkBounds(self.Individual(guess))
                    pop.append(ind)

            # 2. Fill the rest of the population randomly
            num_random_needed = self.population_size - len(pop)
            if num_random_needed < 0: num_random_needed = 0 # Should not happen due to check in init, but safe

            if num_random_needed > 0:
                 print(f"Generating {num_random_needed} random individuals to reach population size {self.population_size}.")
                 # Use the registered random individual generator
                 # Need to regenerate toolbox function if using _initRandomIndividual directly
                 # Re-register toolbox.individual to use the correct method if needed
                 self.toolbox.register("individual", self._initRandomIndividual, self.Individual)
                 for _ in range(num_random_needed):
                     pop.append(self.toolbox.individual()) # Call the registered random generator

            # Shuffle the initial population? Optional, but can be good practice.
            random.shuffle(pop)

            if len(pop) != self.population_size:
                 # This case should ideally be handled by the logic above/in init
                 print(f"Warning: Final initial population size is {len(pop)}, expected {self.population_size}.")


        except Exception as e:
            print(f"\nCRITICAL ERROR during population initialization: {e}")
            return None, (np.inf if self.minimize else -np.inf), logbook

        # --- Evaluate Initial Population ---
        print("Evaluating initial population...")
        # ... (rest of the evaluation, timeout check, HOF update for Gen 0 remains the same) ...
        eval_start_time = time.time()
        invalid_count = 0
        try:
            fitnesses = list(map(self.toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
                if not ind.fitness.valid:
                    invalid_count += 1
            if invalid_count > 0:
                 print(f"Warning: {invalid_count}/{len(pop)} individuals had invalid fitness in initial population.")
        except Exception as e:
            print(f"\nCRITICAL ERROR during initial population evaluation: {e}")
            valid_pop = [ind for ind in pop if hasattr(ind, 'fitness') and ind.fitness.valid]
            if valid_pop: hof.update(valid_pop)
            best_ind_data = list(hof[0]) if hof else None
            best_fit_val = hof[0].fitness.values[0] if hof else (np.inf if self.minimize else -np.inf)
            return best_ind_data, best_fit_val, logbook

        eval_end_time = time.time()
        print(f"Initial evaluation took {eval_end_time - eval_start_time:.2f} seconds.")

        # Timeout check after initial eval
        if timeout is not None and (time.time() - start_time) > timeout:
            print(f"\n⏱️ Timeout ({timeout}s) reached during initial evaluation — stopping optimization.")
            valid_pop = [ind for ind in pop if ind.fitness.valid]
            if valid_pop: hof.update(valid_pop)
            best_ind_data = list(hof[0]) if hof else None
            best_fit_val = hof[0].fitness.values[0] if hof else (np.inf if self.minimize else -np.inf)
            return best_ind_data, best_fit_val, logbook

        # Update HOF and Logbook for Gen 0
        valid_pop = [ind for ind in pop if ind.fitness.valid]
        if not valid_pop:
            print("\nCRITICAL ERROR: No valid individuals in the initial population after evaluation.")
            return None, (np.inf if self.minimize else -np.inf), logbook

        hof.update(valid_pop)
        try:
            record = self.stats.compile(valid_pop) if self.stats else {}
            logbook.record(gen=0, nevals=len(pop), **record)
            if logbook.stream: print("Gen 0:", logbook.stream)
        except Exception as e:
            print(f"Warning: Could not compile or record stats for Gen 0: {e}")


        # --- Begin Evolution Loop ---
        print("\nStarting Evolution Generations...")
        # ... (The evolution loop itself: mutation, crossover, selection remains the same) ...
        gen = 0
        with tqdm(total=self.max_generations, desc="Generations", unit="gen", initial=1) as pbar:
            for gen in range(1, self.max_generations + 1):
                # ... (Timeout Check) ...
                elapsed_time = time.time() - start_time
                if timeout is not None and elapsed_time > timeout:
                    print(f"\n⏱️ Timeout ({timeout}s) reached at generation {gen-1} (elapsed: {elapsed_time:.2f}s) — stopping optimization.")
                    break

                # ... (Standard DE Generation Logic: new_pop loop, mutation, crossover, selection) ...
                new_pop = []
                gen_nevals = 0
                invalid_trial_count = 0
                eval_errors = 0

                for i in range(len(pop)): # Use len(pop) in case it changed unexpectedly
                    target = pop[i]
                    # Mutation (ensure enough individuals)
                    idxs = list(range(len(pop)))
                    idxs.remove(i)
                    if len(idxs) < 3:
                         choices = idxs * (3 // len(idxs) + 1) if idxs else []
                         if len(choices) < 3: choices.extend([i] * (3 - len(choices)))
                         a_idx, b_idx, c_idx = random.choices(choices, k=3)
                    else:
                         a_idx, b_idx, c_idx = random.sample(idxs, 3)
                    a, b, c = pop[a_idx], pop[b_idx], pop[c_idx]

                    mutant = self.toolbox.clone(a)
                    for j in range(self.num_dimensions):
                        diff = b[j] - c[j]
                        if isinstance(diff, (int,float)) and np.isfinite(diff):
                            mutant[j] += self.F * diff
                    mutant = self._checkBounds(mutant)

                    # Crossover
                    trial = self.toolbox.clone(target)
                    rand_j = random.randrange(self.num_dimensions)
                    for j in range(self.num_dimensions):
                        if random.random() < self.CR or j == rand_j:
                            trial[j] = mutant[j]

                    # Selection
                    try:
                        trial.fitness.values = self.toolbox.evaluate(trial)
                        gen_nevals += 1
                        if not trial.fitness.valid:
                            invalid_trial_count += 1
                            new_pop.append(target)
                            continue
                    except Exception as e:
                        eval_errors += 1
                        new_pop.append(target)
                        continue

                    if trial.fitness.dominates(target.fitness):
                        new_pop.append(trial)
                    else:
                        new_pop.append(target)

                # Update postfix for progress bar
                postfix_str = ""
                if hof: postfix_str += f"Best: {hof[0].fitness.values[0]:.4e}"
                if invalid_trial_count > 0: postfix_str += f" | InvTrials: {invalid_trial_count}"
                if eval_errors > 0: postfix_str += f" | EvalErrs: {eval_errors}"
                pbar.set_postfix_str(postfix_str.strip(" |"), refresh=False)


                pop[:] = new_pop # Replace old population

                # Update HoF, Logbook, Progress Bar
                valid_pop = [ind for ind in pop if ind.fitness.valid]
                if not valid_pop:
                    print(f"\nWarning: No valid individuals left in population at generation {gen}. Stopping.")
                    break
                hof.update(valid_pop)
                try:
                    record = self.stats.compile(valid_pop) if self.stats else {}
                    logbook.record(gen=gen, nevals=gen_nevals, **record)
                except Exception as e:
                    print(f"Warning: Could not compile/record stats for Gen {gen}: {e}")

                pbar.update(1)


        # --- Process and Return Final Results ---
        # ... (This part remains the same - extracting from hof) ...
        print("\n--- Final Results ---")
        final_gen = gen

        if hof:
            best_individual_obj = hof[0]
            best_individual_list = list(best_individual_obj)
            best_fitness = best_individual_obj.fitness.values[0]
            print(f"Optimization finished after {final_gen} generations.")
            print(f"Best Fitness Achieved ({'minimum' if self.minimize else 'maximum'}): {best_fitness:.6e}")
            print(f"Best Individual (Parameters): {best_individual_list}")
            # ... (Optional printing of phases/amplitudes) ...
            if self.num_dimensions % 2 == 0:
                 phase_dim = self.num_dimensions // 2
                 final_phases = best_individual_list[:phase_dim]
                 final_amplitudes = best_individual_list[phase_dim:]
                 print(f"Corresponding Phases: {np.array(final_phases)}")
                 print(f"Corresponding Amplitudes: {np.array(final_amplitudes)}")
            return CoilConfig(phase = np.array(final_phases), amplitude=np.array(final_amplitudes)), best_fitness, logbook
        else:
            print("Optimization did not yield a valid result (Hall of Fame is empty).")
            return None, (np.inf if self.minimize else -np.inf), logbook





