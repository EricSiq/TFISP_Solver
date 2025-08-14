# solvers/gurobi_solver.py

import time
from typing import Any, Dict
import gurobipy as gp
from gurobipy import GRB
from dimod import BinaryQuadraticModel

from .base_solver import BaseSolver

class GurobiSolver(BaseSolver):
    """
    A solver class that uses the Gurobi Optimizer to solve the TFISP.
    """

    def solve(self, bqm_problem: BinaryQuadraticModel) -> Dict[str, Any]:
        """
        Solves the QUBO problem using the Gurobi Python API.
        """
        start_time = time.time()
        
        try:
            model = gp.Model("TFISP_QUBO_Solver")
            model.modelSense = GRB.MINIMIZE
            model.setParam('OutputFlag', 0)

            gurobi_vars = {
                var_name: model.addVar(vtype=GRB.BINARY, name=str(var_name))
                for var_name in bqm_problem.variables
            }
            
            linear_objective = gp.quicksum(gurobi_vars[v] * b for v, b in bqm_problem.linear.items())
            
            quadratic_objective = gp.QuadExpr()
            for (u, v), b in bqm_problem.quadratic.items():
                quadratic_objective += gurobi_vars[u] * gurobi_vars[v] * b

            model.setObjective(linear_objective + quadratic_objective + bqm_problem.offset)

            gurobi_params = self.config.get('gurobi_params', {})
            for param, value in gurobi_params.items():
                try:
                    model.setParam(param, value)
                except gp.GurobiError as e:
                    print(f"[WARN] Failed to set Gurobi parameter '{param}': {e}")
            
            model.optimize()

        except gp.GurobiError as e:
            print(f"[ERROR] Gurobi Solver Error: {e.message}")
            return {
                'cost': float('inf'),
                'time': time.time() - start_time,
                'assignment': {}
            }
        
        end_time = time.time()
        
        if model.status == GRB.OPTIMAL:
            objective_value = model.objVal
            assignment = {name: int(round(var.x)) for name, var in gurobi_vars.items()}
            
            return {
                'cost': objective_value,
                'time': end_time - start_time,
                'assignment': assignment
            }
        else:
            print(f"[WARN] Gurobi did not find an optimal solution. Status: {model.status}")
            return {
                'cost': float('inf'),
                'time': end_time - start_time,
                'assignment': {}
            }

if __name__ == '__main__':
    from ..qubo_formulator import build_qubo
    from ..parser import build_conflict_matrix
    from ..parser import Job
    import os
    import json

    print("Gurobi Solver Demo:")
    
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
        'gurobi_params': { 'TimeLimit': 60.0 }
    }
    
    config_file_path = "config.json"
    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=4)

    bqm_problem = build_qubo(num_jobs, num_resources, resource_costs, jobs, disqualified_pairs, conflict_matrix, config)
    gurobi_solver = GurobiSolver(config_file_path)
    solution = gurobi_solver.solve(bqm_problem)

    print("\n--- Solution Results ---")
    print(f"Objective Cost: {solution['cost']}")
    print(f"Time Taken: {solution['time']:.4f} seconds")
    print("Variable Assignments:")
    for var, val in solution['assignment'].items():
        if val == 1:
            print(f"  {var} = {val}")

    os.remove(config_file_path)