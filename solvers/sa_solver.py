# solvers/sa_solver.py

import time
from typing import Any, Dict
import dimod
import os
import json

from .base_solver import BaseSolver
from ..qubo_formulator import build_qubo
from ..parser import parse_input_file, build_conflict_matrix

class SimulatedAnnealingSolver(BaseSolver):
    """
    A solver class that uses the dimod Simulated Annealing Sampler to solve the TFISP.

    This class extends the BaseSolver and provides an implementation of the
    `solve` method to execute a simulated annealing algorithm on a given BQM.
    It leverages the dimod library, which is part of the D-Wave Ocean SDK.
    """

    def solve(self, bqm_problem: dimod.BinaryQuadraticModel) -> Dict[str, Any]:
        """
        Solves the QUBO problem using dimod's Simulated Annealing.

        This method initializes a `SimulatedAnnealingSampler` and samples from the
        provided BQM. It looks for parameters like 'num_reads' in the config file
        to control the solver's behavior. The best solution from the sampling
        is returned as the final result.

        Args:
            bqm_problem: A problem instance, typically a dimod.BinaryQuadraticModel
                         from the qubo_formulator.py file.

        Returns:
            A dictionary containing the solution, cost, and execution time.
            Expected format: {'cost': float, 'time': float, 'assignment': dict}
        """
        start_time = time.time()
        
        
        # 1. Initialize the sampler and get parameters from config
         
        sampler = dimod.SimulatedAnnealingSampler()
        
        sa_params = self.config.get('sa_params', {})
        num_reads = sa_params.get('num_reads', 100) # Use a default if not in config
        
        print(f"Solving with Simulated Annealing, num_reads={num_reads}...")


        # 2. Sample the BQM
        
        
        try:
            # The sampler returns a SampleSet containing the solutions found
            sampleset = sampler.sample(bqm_problem, num_reads=num_reads)
        except Exception as e:
            print(f"[ERROR] dimod.SimulatedAnnealingSampler failed: {e}")
            return {
                'cost': float('inf'),
                'time': time.time() - start_time,
                'assignment': {}
            }

        end_time = time.time()
        
        # 3. Extract and format the solution
    
        # The sampleset contains multiple samples, each with an energy value
        # The best solution is the one with the lowest energy in the sampleset
        best_solution = sampleset.first
        
        cost = best_solution.energy
        # Convert the dimod.Sample object to a standard dictionary
        assignment = dict(best_solution.sample)
        
        return {
            'cost': cost,
            'time': end_time - start_time,
            'assignment': assignment
        }

if __name__ == '__main__':
    # This block demonstrates how to use the SimulatedAnnealingSolver class with sample data
    from ..parser import Job

    print("Simulated Annealing Solver Demo: ")
    
    # Sample data (should match the format of the parser output)
    num_jobs = 5
    num_resources = 3
    resource_costs = [100.0, 200.0, 150.0]
    jobs = [
        Job(0, 10, 20),
        Job(1, 15, 25),
        Job(2, 30, 40),
        Job(3, 35, 45),
        Job(4, 5, 15)
    ]
    disqualified_pairs = [(0, 2), (1, 0), (1, 1), (2, 0), (3, 1)]
    
    # Manually create a conflict matrix for demonstration
    conflict_matrix = build_conflict_matrix(jobs)
    
    # Simple configuration dictionary for the QUBO formulator and SA solver
    config = {
        'qubo_params': {
            'penalty_P1': 10000.0,
            'penalty_P2': 10000.0,
            'penalty_P3': 10000.0,
            'penalty_P4': 10000.0,
        },
        'sa_params': {
            'num_reads': 1000
        }
    }
    
    # Create a dummy config file for the solver's constructor
    config_file_path = "config.json"
    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Build the BQM using the qubo_formulator
    bqm_problem = build_qubo(num_jobs, num_resources, resource_costs, jobs, disqualified_pairs, conflict_matrix, config)

    # Instantiate the Simulated Annealing solver
    sa_solver = SimulatedAnnealingSolver(config_file_path)

    # Solve the problem
    solution = sa_solver.solve(bqm_problem)

    print("\n--- Solution Results ---")
    print(f"Objective Cost: {solution['cost']}")
    print(f"Time Taken: {solution['time']:.4f} seconds")
    print("Variable Assignments:")
    for var, val in solution['assignment'].items():
        if val == 1:
            print(f"  {var} = {val}")

    # Clean up the dummy config file
    os.remove(config_file_path)