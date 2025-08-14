# solvers/sa_solver.py

import time
from typing import Any, Dict
import dimod
import os
import json

from .base_solver import BaseSolver

class SimulatedAnnealingSolver(BaseSolver):
    """
    A solver class that uses the dimod Simulated Annealing Sampler to solve the TFISP.
    """

    def solve(self, bqm_problem: dimod.BinaryQuadraticModel) -> Dict[str, Any]:
        """
        Solves the QUBO problem using dimod's Simulated Annealing.
        """
        start_time = time.time()
        
        sampler = dimod.SimulatedAnnealingSampler()
        
        sa_params = self.config.get('sa_params', {})
        num_reads = sa_params.get('num_reads', 100)
        
        print(f"Solving with Simulated Annealing, num_reads={num_reads}...")

        try:
            sampleset = sampler.sample(bqm_problem, num_reads=num_reads)
        except Exception as e:
            print(f"[ERROR] dimod.SimulatedAnnealingSampler failed: {e}")
            return {
                'cost': float('inf'),
                'time': time.time() - start_time,
                'assignment': {}
            }

        end_time = time.time()
        
        best_solution = sampleset.first
        
        cost = best_solution.energy
        assignment = dict(best_solution.sample)
        
        return {
            'cost': cost,
            'time': end_time - start_time,
            'assignment': assignment
        }

if __name__ == '__main__':
    from ..qubo_formulator import build_qubo
    from ..parser import build_conflict_matrix
    from ..parser import Job
    
    print("--- Simulated Annealing Solver Demo ---")
    
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
        'sa_params': { 'num_reads': 1000 }
    }
    
    config_file_path = "config.json"
    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=4)

    bqm_problem = build_qubo(num_jobs, num_resources, resource_costs, jobs, disqualified_pairs, conflict_matrix, config)
    sa_solver = SimulatedAnnealingSolver(config_file_path)
    solution = sa_solver.solve(bqm_problem)

    print("\nSolution Results :")
    print(f"Objective Cost: {solution['cost']}")
    print(f"Time Taken: {solution['time']:.4f} seconds")
    print("Variable Assignments:")
    for var, val in solution['assignment'].items():
        if val == 1:
            print(f"  {var} = {val}")

    os.remove(config_file_path)