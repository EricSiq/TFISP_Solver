# solvers/dwave_hybrid_solver.py

import time
from typing import Any, Dict
import os
import json

from dimod import BinaryQuadraticModel
from hybrid.flow import LoopUntilNoImprovement, RacingBranches
from hybrid.decomposers import IdentityDecomposer
from hybrid.samplers import SimulatedAnnealingSampler, QPUSubproblemAutoEmbeddingSampler
from hybrid.composers import SplittingComposer
from hybrid import State

from .base_solver import BaseSolver
from ..qubo_formulator import build_qubo
from ..parser import parse_input_file, build_conflict_matrix

class DwaveHybridSolver(BaseSolver):
    """
    A solver class that uses the D-Wave hybrid workflow to solve the TFISP.

    This class extends the BaseSolver and provides an implementation of the
    `solve` method to execute a hybrid quantum-classical algorithm on a given BQM.
    It leverages the D-Wave `hybrid` library, which allows for building complex
    workflows that combine classical samplers (like simulated annealing) with
    quantum processing units (QPU) or cloud-based QPU-like solvers.
    """

    def solve(self, bqm_problem: BinaryQuadraticModel) -> Dict[str, Any]:
        """
        Solves the QUBO problem using a D-Wave hybrid workflow.

        This method defines a hybrid workflow to solve the provided BQM. The workflow
        starts with a simulated annealing sampler, then iteratively tries a QPU
        sampler to improve the solution. This process continues until a better
        solution is no longer found.

        Args:
            bqm_problem: A problem instance, typically a dimod.BinaryQuadraticModel
                         from the qubo_formulator.py file.

        Returns:
            A dictionary containing the best solution found, its cost, and
            the total execution time.
            Expected format: {'cost': float, 'time': float, 'assignment': dict}
        """
        start_time = time.time()
        
        hybrid_params = self.config.get('dwave_hybrid_params', {})
        # num_reads can be passed to the QPU sampler
        num_reads = hybrid_params.get('num_reads', 100)
        
        print(f"Solving with D-Wave Hybrid Solver, num_reads={num_reads}...")

        try:
            # 1. Define the hybrid workflow
            # A common pattern is to start with a simulated annealing solution
            # and then use a QPU to refine it.
            # This workflow will loop until no improvement is found.
            workflow = LoopUntilNoImprovement(
                RacingBranches(
                    # Branch 1: Use a Simulated Annealing sampler
                    SimulatedAnnealingSampler(num_reads=num_reads),

                    # Branch 2: Decompose the problem and solve subproblems on a QPU
                    # NOTE: This requires access to a D-Wave QPU or Hybrid Solver.
                    # This workflow is simplified and uses the IdentityDecomposer
                    # to keep the example self-contained, but it can be replaced
                    # with a more advanced decomposition strategy.
                    IdentityDecomposer()
                    | QPUSubproblemAutoEmbeddingSampler()
                    | SplittingComposer()
                )
            )

            # 2. Run the workflow on the BQM
            state = workflow.run(State.from_problem(bqm_problem))

        except Exception as e:
            print(f"[ERROR] D-Wave Hybrid Solver failed: {e}")
            return {
                'cost': float('inf'),
                'time': time.time() - start_time,
                'assignment': {}
            }
        
        end_time = time.time()
        
        # 3. Extract and format the solution
        # The hybrid workflow's final state contains the best sample found
        best_solution = state.result.first
        
        cost = best_solution.energy
        # Convert the dimod.Sample object to a standard dictionary
        assignment = dict(best_solution.sample)
        
        return {
            'cost': cost,
            'time': end_time - start_time,
            'assignment': assignment
        }

if __name__ == '__main__':
    # This block demonstrates how to use the DwaveHybridSolver class with sample data
    from ..parser import Job
    
    print("--- D-Wave Hybrid Solver Demo ---")
    
    # Sample data
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
    conflict_matrix = build_conflict_matrix(jobs)
    
    config = {
        'qubo_params': {
            'penalty_P1': 10000.0, 'penalty_P2': 10000.0, 'penalty_P3': 10000.0, 'penalty_P4': 10000.0,
        },
        'dwave_hybrid_params': {
            'num_reads': 1000
        }
    }
    
    config_file_path = "config.json"
    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=4)

    bqm_problem = build_qubo(num_jobs, num_resources, resource_costs, jobs, disqualified_pairs, conflict_matrix, config)
    dwave_hybrid_solver = DwaveHybridSolver(config_file_path)
    solution = dwave_hybrid_solver.solve(bqm_problem)

    print("\n--- Solution Results ---")
    print(f"Objective Cost: {solution['cost']}")
    print(f"Time Taken: {solution['time']:.4f} seconds")
    print("Variable Assignments:")
    for var, val in solution['assignment'].items():
        if val == 1:
            print(f"  {var} = {val}")

    os.remove(config_file_path)