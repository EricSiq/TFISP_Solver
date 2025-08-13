# gurobi_solver.py

import time
from typing import Any, Dict
import gurobipy as gp
from gurobipy import GRB

from .base_solver import BaseSolver
from ..qubo_formulator import build_qubo
from ..parser import parse_input_file, build_conflict_matrix

class GurobiSolver(BaseSolver):
    """
    A solver class that uses the Gurobi Optimizer to solve the TFISP.

    This class extends the BaseSolver and provides an implementation of the
    `solve` method to translate a QUBO problem into a Gurobi model, solve it,
    and return the solution.
    """

    def solve(self, bqm_problem: Any) -> Dict[str, Any]:
        """
        Solves the QUBO problem using the Gurobi Python API.

        This method translates the Binary Quadratic Model (BQM) into a Gurobi
        quadratic program and solves it to find the optimal assignment. The
        BQM is passed from the `qubo_formulator.py`, and this method
        converts its linear and quadratic terms into a Gurobi objective function.

        Args:
            bqm_problem: A problem instance, typically a dimod.BinaryQuadraticModel
                         from the qubo_formulator.py file.

        Returns:
            A dictionary containing the solution, cost, and execution time.
            Expected format: {'cost': float, 'time': float, 'assignment': dict}
        """
        start_time = time.time()
        
        
        # 1. Initialize Gurobi model and variables
         
        try:
            model = gp.Model("TFISP_QUBO_Solver")
            # Set the model to minimize the objective
            model.modelSense = GRB.MINIMIZE
            # Suppress Gurobi output to keep things clean
            model.setParam('OutputFlag', 0)

            # Create a dictionary to map variable names to Gurobi variable objects
            gurobi_vars = {}
            for var_name in bqm_problem.variables:
                gurobi_vars[var_name] = model.addVar(vtype=GRB.BINARY, name=str(var_name))
            
            
            # 2. Set the objective function
            # The BQM's linear and quadratic terms are used to set the objective function.
            # Gurobi's model expects a linear expression for the objective function.
            # The objective function in Gurobi is defined by linear and quadratic terms.
            # Convert the BQM's linear terms to Gurobi's linear expression
            linear_objective = gp.quicksum(gurobi_vars[v] * b for v, b in bqm_problem.linear.items())
            
            # Convert the BQM's quadratic terms to Gurobi's quadratic expression
            quadratic_objective = gp.QuadExpr()
            for (u, v), b in bqm_problem.quadratic.items():
                quadratic_objective += gurobi_vars[u] * gurobi_vars[v] * b

            # Set the total objective function
            model.setObjective(linear_objective + quadratic_objective + bqm_problem.offset)


            # 3. Solve the model
            model.update()  # Ensure the model is updated with the new variables and objective
            # Apply solver parameters from the config file if available
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
        
        
        # 4. Extract and format the solution
        if model.status == GRB.OPTIMAL:
            objective_value = model.objVal
            # Extract the variable assignments
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
    # This block demonstrates how to use the GurobiSolver class with sample data
    from ..qubo_formulator import build_qubo
    from ..parser import build_conflict_matrix
    from ..parser import Job
    import os
    import json

    print("--- Gurobi Solver Demo ---")
    
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
    
    # Simple configuration dictionary for the QUBO formulator and Gurobi solver
    config = {
        'qubo_params': {
            'penalty_P1': 10000.0,
            'penalty_P2': 10000.0,
            'penalty_P3': 10000.0,
            'penalty_P4': 10000.0,
        },
        'gurobi_params': {
            'TimeLimit': 60.0
        }
    }
    
    # Create a dummy config file for the solver's constructor
    config_file_path = "config.json"
    with open(config_file_path, 'w') as f:
        json.dump(config, f, indent=4)

    # Build the BQM using the qubo_formulator
    bqm_problem = build_qubo(num_jobs, num_resources, resource_costs, jobs, disqualified_pairs, conflict_matrix, config)

    # Instantiate the Gurobi solver
    gurobi_solver = GurobiSolver(config_file_path)

    # Solve the problem
    solution = gurobi_solver.solve(bqm_problem)

    print("\n Solution Results:")
    print(f"Objective Cost: {solution['cost']}")
    print(f"Time Taken: {solution['time']:.4f} seconds")
    print("Variable Assignments:")
    for var, val in solution['assignment'].items():
        if val == 1:
            print(f"  {var} = {val}")

    # Clean up the dummy config file
    os.remove(config_file_path)