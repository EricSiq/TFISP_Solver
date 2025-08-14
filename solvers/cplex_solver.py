# solvers/cplex_solver.py

import time
from typing import Any, Dict
import cplex
from dimod import BinaryQuadraticModel

from .base_solver import BaseSolver

class CplexSolver(BaseSolver):
    """
    A solver class that uses the IBM CPLEX Optimizer to solve the TFISP.
    """

    def solve(self, bqm_problem: BinaryQuadraticModel) -> Dict[str, Any]:
        """
        Solves the QUBO problem using the CPLEX Python API.
        """
        start_time = time.time()
        
        cplex_problem = cplex.Cplex()
        cplex_problem.objective.set_sense(cplex_problem.objective.sense.minimize)

        var_names = list(bqm_problem.linear.keys())
        variable_types = [cplex_problem.variables.type.binary] * len(var_names)
        
        cplex_problem.variables.add(names=var_names, types=variable_types)

        linear_terms = [(var, bias) for var, bias in bqm_problem.linear.items()]
        cplex_problem.objective.set_linear(linear_terms)
        
        quadratic_terms = [(str(u), str(v), bias) for (u, v), bias in bqm_problem.quadratic.items()]
        cplex_problem.objective.set_quadratic_coefficients(quadratic_terms)

        cplex_params = self.config.get('cplex_params', {})
        for param, value in cplex_params.items():
            try:
                cplex_problem.parameters.set_by_name(param, value)
            except cplex.exceptions.CplexError as e:
                print(f"[WARN] Failed to set CPLEX parameter '{param}': {e}")
        
        try:
            cplex_problem.solve()
        except cplex.exceptions.CplexSolverError as e:
            print(f"[ERROR] CPLEX Solver Error: {e}")
            return {
                'cost': float('inf'),
                'time': time.time() - start_time,
                'assignment': {}
            }

        end_time = time.time()
        
        solution = cplex_problem.solution
        
        if solution.get_status() == solution.status.optimal:
            objective_value = solution.get_objective_value() + bqm_problem.offset
            variable_values = solution.get_values()
            
            assignment = {var_names[i]: int(round(val)) for i, val in enumerate(variable_values)}
            
            return {
                'cost': objective_value,
                'time': end_time - start_time,
                'assignment': assignment
            }
        else:
            print(f"[WARN] CPLEX did not find an optimal solution. Status: {solution.get_status_string()}")
            return {
                'cost': float('inf'),
                'time': end_time - start_time,
                'assignment': {}
            }

if __name__ == '__main__':
    # This block demonstrates how to use the CplexSolver class with sample data
    # Note: Requires a config file named 'config.json' and assumes the
    # tfisp_solver package structure is in the PYTHONPATH.
    from ..qubo_formulator import build_qubo
    from ..parser import build_conflict_matrix
    from ..parser import Job
    import os
    import json

    print("CPLEX Solver Demo :\n")
    
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
        'cplex_params': { 'timelimit': 60.0 }
    }
    
    config_file_path = "config.json"
    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=4)

    bqm_problem = build_qubo(num_jobs, num_resources, resource_costs, jobs, disqualified_pairs, conflict_matrix, config)
    cplex_solver = CplexSolver(config_file_path)
    solution = cplex_solver.solve(bqm_problem)

    print("\n Solution Results: ")
    print(f"Objective Cost: {solution['cost']}")
    print(f"Time Taken: {solution['time']:.4f} seconds")
    print("Variable Assignments:")
    for var, val in solution['assignment'].items():
        if val == 1:
            print(f"  {var} = {val}")

    os.remove(config_file_path)