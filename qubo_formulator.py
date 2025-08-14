# qubo_formulator.py

import dimod
from typing import List, Dict, Tuple

def build_qubo(
    num_jobs: int,
    num_resources: int,
    resource_costs: List[float],
    jobs: List,
    disqualified_pairs: List[Tuple[int, int]],
    conflict_matrix: List[List[int]],
    config: Dict
) -> dimod.BinaryQuadraticModel:
    
    # Constraint Term - H_link_resource_usage) 
    # Links resource usage (y_k) to job assignments (x_ik)
    # Correct QUBO formulation for this constraint term is:
    # H_link_resource_usage = P4 * sum_k [ M_k*y_k + sum_i(x_ik) - (M_k+1)*y_k*sum_i(x_ik) ]

    for k in range(num_resources):
        y_k_var = f'y_{k}'
        
        # Add linear terms
        bqm.add_variable(y_k_var, P4 * M_k)
        for i in range(num_jobs):
            x_ik_var = f'x_{i}_{k}'
            bqm.add_variable(x_ik_var, P4)
        
        # Add quadratic terms
        for i in range(num_jobs):
            x_ik_var = f'x_{i}_{k}'
            bqm.add_interaction(x_ik_var, y_k_var, -P4 * (M_k + 1))
    """
    Formulates TFISP into a Binary Quadratic Model (BQM), which is a general
    representation of a QUBO problem, using dimod.

    Arguments used:
        num_jobs: The number of jobs (N).
        num_resources: The number of resources (M).
        resource_costs: A list of costs for each resource (C_k).
        jobs: A list of job data (Job namedtuples).
        disqualified_pairs: A list of (job_id, resource_id) pairs that are invalid.
        conflict_matrix: A pre-computed matrix where conflict_matrix[i][j] = 1 if jobs i and j overlap.
        config: A dictionary containing penalty coefficients and other parameters.

    Returns:
        A dimod.BinaryQuadraticModel object representing the QUBO formulation.
    """
    
    # 1. Initialize the BQM
    
    bqm = dimod.BinaryQuadraticModel(dimod.BINARY)
    
    
    # 2. Get penalty coefficients from config
    
    qubo_params = config.get('qubo_params', {})
    P1 = qubo_params.get('penalty_P1', 1000.0)
    P2 = qubo_params.get('penalty_P2', 1000.0)
    P3 = qubo_params.get('penalty_P3', 1000.0)
    P4 = qubo_params.get('penalty_P4', 1000.0)
    
    # M_k is a sufficiently large positive constant, N+1 is a good choice
    M_k = num_jobs + 1
    
    print(f"Using QUBO penalty coefficients: P1={P1}, P2={P2}, P3={P3}, P4={P4}, M_k={M_k}")

    # 3. Add QUBO terms
    # Objective Term (H_objective) 
    # Minimizes the total resource activation cost.
    # H_objective = sum(C_k * y_k)
    for k in range(num_resources):
        y_k_var = f'y_{k}'
        bqm.add_variable(y_k_var, resource_costs[k])
        
    # Constraint Term 1 (H_job_assigned)
    # Ensures each job is assigned to exactly one resource.
    # H_job_assigned = P1 * sum(1 - sum(x_ik))^2
    for i in range(num_jobs):
        job_vars = [f'x_{i}_{k}' for k in range(num_resources)]
        # This is a one-hot constraint, dimod has a helper for it.
        # Alternatively, you can add it manually:
        # bqm.add_linear_equality_constraint(
        #     terms=[(var, 1) for var in job_vars],
        #     constant=-1,
        #     lagrange_multiplier=P1
        # )
        # Manual expansion of the squared term for clarity:
        bqm.add_variable(None, P1) 
        for k_outer in range(num_resources):
            x_ik_outer = f'x_{i}_{k_outer}'
            bqm.add_variable(x_ik_outer, -2 * P1)
            for k_inner in range(k_outer, num_resources):
                x_ik_inner = f'x_{i}_{k_inner}'
                if k_outer == k_inner:
                    bqm.add_variable(x_ik_inner, P1)
                else:
                    bqm.add_interaction(x_ik_outer, x_ik_inner, 2 * P1)

    # Constraint Term 2 (H_overlap)
    # Penalizes overlapping jobs on the same resource.
    # H_overlap = P2 * sum(O_ij * x_ik * x_jk)
    for k in range(num_resources):
        for i in range(num_jobs):
            for j in range(i + 1, num_jobs):
                if conflict_matrix[i][j] == 1:
                    x_ik_var = f'x_{i}_{k}'
                    x_jk_var = f'x_{j}_{k}'
                    bqm.add_interaction(x_ik_var, x_jk_var, P2)

    # Constraint Term 3 (H_disqualified)
    # Penalizes assigning a job to a disqualified resource.
    # H_disqualified = P3 * sum(D_ik * x_ik)
    for job_id, resource_id in disqualified_pairs:
        x_ik_var = f'x_{job_id}_{resource_id}'
        bqm.add_variable(x_ik_var, P3)

    # Constraint Term 4 (H_link_resource_usage)
    # Links resource usage (y_k) to job assignments (x_ik).
    # H_link_resource_usage = P4 * sum[ (sum(x_ik)) * (1-y_k) + M_k * y_k * (1-sum(x_ik)) ]
    for k in range(num_resources):
        y_k_var = f'y_{k}'
        
        # Term 1: (sum(x_ik)) * (1-y_k)
        # Expansion: sum(x_ik) - y_k*sum(x_ik)
        bqm.add_variable(y_k_var, -P4 * M_k) # from term 2 below
        for i in range(num_jobs):
            x_ik_var = f'x_{i}_{k}'
            bqm.add_variable(x_ik_var, P4 + P4 * M_k) # combines with terms from below
            bqm.add_interaction(x_ik_var, y_k_var, -P4 - P4 * M_k)
        
        # We need to add the following linear and quadratic terms from the expansion of the provided formula
        # P4 * (sum_i x_ik - sum_i x_ik*y_k + M_k*y_k - M_k*y_k*sum_i x_ik)
        
        # sum_i x_ik*y_k
        for i in range(num_jobs):
            x_ik_var = f'x_{i}_{k}'
            bqm.add_interaction(x_ik_var, y_k_var, -P4)

        # M_k * y_k
        bqm.add_variable(y_k_var, P4 * M_k)

        # M_k * y_k * sum_i x_ik
        for i in range(num_jobs):
            x_ik_var = f'x_{i}_{k}'
            bqm.add_interaction(x_ik_var, y_k_var, -P4 * M_k)

    return bqm

if __name__ == '__main__':
    # This example demonstrates how to use the function with sample data
    # that matches the format of tfisp_input.txt.
    print(" Example QUBO Formulation")
    
    # Sample data
    num_jobs = 5
    num_resources = 3
    resource_costs = [100.0, 200.0, 150.0]
    jobs = [
        ('job_0', 10, 20),
        ('job_1', 15, 25),
        ('job_2', 30, 40),
        ('job_3', 35, 45),
        ('job_4', 5, 15)
    ]
    disqualified_pairs = [(0, 2), (1, 0), (1, 1), (2, 0), (3, 1)]
    
    # A simple, manually-built conflict matrix based on the jobs data
    conflict_matrix = [
        [0, 1, 0, 0, 1],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [1, 0, 0, 0, 0],
    ]
    
    config = {
        'qubo_params': {
            'penalty_P1': 1000.0,
            'penalty_P2': 1000.0,
            'penalty_P3': 1000.0,
            'penalty_P4': 1000.0,
        }
    }

    # Build the QUBO
    bqm_model = build_qubo(num_jobs, num_resources, resource_costs, jobs, disqualified_pairs, conflict_matrix, config)

    print("\n BQM Summary")
    print(f"Number of variables: {bqm_model.num_variables}")
    print(f"Type of variables: {bqm_model.vartype}")
    
    # Print some terms for verification
    print("\n Linear Terms (first 10) ")
    for i, (var, bias) in enumerate(bqm_model.linear.items()):
        print(f"  {var}: {bias}")
        if i >= 9:
            break
            
    print("\n Quadratic Terms (first 10) ")
    for i, (interaction, bias) in enumerate(bqm_model.quadratic.items()):
        print(f"  {interaction}: {bias}")
        if i >= 9:
            break