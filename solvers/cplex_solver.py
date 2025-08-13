# solvers/cplex_solver.py

import time
from typing import Any, Dict
import cplex

from .base_solver import BaseSolver
from ..qubo_formulator import build_qubo
from ..parser import parse_input_file, build_conflict_matrix

class CplexSolver(BaseSolver):
    """
    A solver class that uses the IBM CPLEX Optimizer to solve the TFISP.

    This class extends the BaseSolver and provides an implementation of the
    `solve` method to translate a QUBO problem into a CPLEX model, solve it,
    and return the solution.
    """

    def solve(self, bqm_problem: cplex.Cplex) -> Dict[str, Any]:
        """
        Solves the QUBO problem using the CPLEX Python API.

        This method translates the Binary Quadratic Model (BQM) into a CPLEX
        quadratic program and solves it to find the optimal assignment. The
        CPLEX documentation provides details on how to build models by adding
        variables and setting the objective function.

        Args:
            bqm_problem: A problem instance, typically a dimod.BinaryQuadraticModel
                         from the qubo_formulator.py file.

        Returns:
            A dictionary containing the solution, cost, and execution time.
            Expected format: {'cost': float, 'time': float, 'assignment': dict}
        """
        start_time = time.time()
        
        # ---------------------------------------------------------------------
        # 1. Initialize CPLEX model and variables
        # ---------------------------------------------------------------------
        cplex_problem = cplex.Cplex()
        cplex_problem.objective.set_sense(cplex_problem.objective.sense.minimize)

        # Get the variable names from the BQM
        var_names = list(bqm_problem.linear.keys())
        
        # Set all variables to be binary
        variable_types = [cplex_problem.variables.type.binary] * len(var_names)
        
        # Add the variables to the CPLEX model
        cplex_problem.variables.add(names=var_names, types=variable_types)

        # ---------------------------------------------------------------------
        # 2. Set the objective function
        # ---------------------------------------------------------------------
        # The objective function in CPLEX is defined by linear and quadratic terms
        
        # Add linear terms from the BQM
        linear_terms = []
        for var, bias in bqm_problem.linear.items():
            linear_terms.append((var, bias))
        cplex_problem.objective.set_linear(linear_terms)
        
        # Add quadratic terms from the BQM
        quadratic_terms = []
        for (u, v), bias in bqm_problem.quadratic.items():
            quadratic_terms.append((str(u), str(v), bias))
        cplex_problem.objective.set_quadratic_coefficients(quadratic_terms)

        # ---------------------------------------------------------------------
        # 3. Solve the model
        # ---------------------------------------------------------------------
        # Apply solver parameters from the config file if available
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
        
        # ---------------------------------------------------------------------
        # 4. Extract and format the solution
        # ---------------------------------------------------------------------
        solution = cplex_problem.solution
        
        if solution.get_status() == solution.status.optimal:
            objective_value = solution.get_objective_value()
            variable_values = solution.get_values()
            
            # Map the variable values to the original variable names
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
    from ..qubo_formulator import build_qubo
    from ..parser import build_conflict_matrix

    print("--- CPLEX Solver Demo ---")
    
    # Sample data (should match the format of the parser output)
    num_jobs = 5
    num_resources = 3
    resource_costs = [100.0, 200.0, 150.0]
    jobs = [('job_0', 10, 20), ('job_1', 15, 25), ('job_2', 30, 40), ('job_3', 35, 45), ('job_4', 5, 15)]
    disqualified_pairs = [(0, 2), (1, 0), (1, 1), (2, 0), (3, 1)]
    
    # Manually create a conflict matrix for demonstration
    conflict_matrix = build_conflict_matrix([cplex.Cplex().linear_constraints.add(names=['job_0', 'job_1', 'job_2', 'job_3', 'job_4'])])
    
    # Simple configuration dictionary for the QUBO formulator
    config = {
        'qubo_params': {
            'penalty_P1': 10000.0,
            'penalty_P2': 10000.0,
            'penalty_P3': 10000.0,
            'penalty_P4': 10000.0,
        },
        'cplex_params': {
            'timelimit': 60.0
        }
    }

    # Build the BQM using the qubo_formulator
    bqm_problem = build_qubo(num_jobs, num_resources, resource_costs, jobs, disqualified_pairs, conflict_matrix, config)

    # Instantiate the CPLEX solver
    cplex_solver = CplexSolver('config.json')  # Assuming config.json exists or will be created

    # Solve the problem
    solution = cplex_solver.solve(bqm_problem)

    print("\n--- Solution Results ---")
    print(f"Objective Cost: {solution['cost']}")
    print(f"Time Taken: {solution['time']:.4f} seconds")
    print("Variable Assignments:")
    for var, val in solution['assignment'].items():
        if val == 1:
            print(f"  {var} = {val}")