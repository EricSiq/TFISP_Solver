# parser.py

import os
from collections import namedtuple
from typing import List, Dict, Tuple

# Define a named tuple for better code readability
Job = namedtuple("Job", ["id", "start_time", "end_time"])

def parse_input_file(file_path: str) -> Tuple[int, int, List[float], List[Job], List[Tuple[int, int]]]:
    """
    Parses a TFISP problem instance from a human-readable text file.

    Args:
        file_path: The path to the input text file.

    Returns:
        A tuple containing:
        - The number of jobs.
        - The number of resources.
        - A list of resource costs.
        - A list of Job namedtuples.
        - A list of disqualified (job_id, resource_id) pairs.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    num_jobs = 0
    num_resources = 0
    resource_costs = []
    jobs = []
    disqualified_pairs = []

    with open(file_path, 'r') as f:
        # State machine for parsing
        parsing_jobs = False
        parsing_disqualified = False
        
        for line in f:
            line = line.strip()
            # Ignore comments and empty lines
            if not line or line.startswith('#'):
                continue

            if line.startswith("NUM_JOBS="):
                num_jobs = int(line.split('=')[1])
                parsing_jobs = False
                parsing_disqualified = False
            elif line.startswith("NUM_RESOURCES="):
                num_resources = int(line.split('=')[1])
                parsing_jobs = False
                parsing_disqualified = False
            elif line.startswith("RESOURCE_COSTS="):
                costs_str = line.split('=')[1]
                resource_costs = [float(c.strip()) for c in costs_str.split(',')]
                parsing_jobs = False
                parsing_disqualified = False
            elif line.startswith("JOBS:"):
                parsing_jobs = True
                parsing_disqualified = False
            elif line.startswith("DISQUALIFIED_RESOURCES:"):
                parsing_disqualified = True
                parsing_jobs = False
            elif parsing_jobs:
                parts = [p.strip() for p in line.split(',')]
                job_id, start_time, end_time = int(parts[0]), float(parts[1]), float(parts[2])
                jobs.append(Job(job_id, start_time, end_time))
            elif parsing_disqualified:
                parts = [p.strip() for p in line.split(',')]
                job_id, resource_id = int(parts[0]), int(parts[1])
                disqualified_pairs.append((job_id, resource_id))

    return num_jobs, num_resources, resource_costs, jobs, disqualified_pairs

def build_conflict_matrix(jobs: List[Job]) -> List[List[int]]:
    """
    Constructs an overlap matrix O_ij: 1 when job i and job j intervals overlap, 0 otherwise.

    Args:
        jobs: A list of Job namedtuples.

    Returns:
        A 2D list representing the conflict matrix.
    """
    n = len(jobs)
    O = [[0] * n for _ in range(n)]
    for i in range(n):
        si, ei = jobs[i].start_time, jobs[i].end_time
        for j in range(i + 1, n):
            sj, ej = jobs[j].start_time, jobs[j].end_time
            if (ei > sj) and (ej > si):
                O[i][j] = O[j][i] = 1
    return O

if __name__ == '__main__':
    # Example usage with the provided tfisp_input.txt content
    # This block assumes the file is in the same directory for demonstration
    file_path = "tfisp_input.txt"
    if os.path.exists(file_path):
        num_jobs, num_resources, resource_costs, jobs, disqualified_pairs = parse_input_file(file_path)
        print("--- Parsed Data ---")
        print(f"NUM_JOBS: {num_jobs}")
        print(f"NUM_RESOURCES: {num_resources}")
        print(f"RESOURCE_COSTS: {resource_costs}")
        print(f"JOBS: {jobs}")
        print(f"DISQUALIFIED_RESOURCES: {disqualified_pairs}")

        conflict_matrix = build_conflict_matrix(jobs)
        print("\n--- Conflict Matrix ---")
        for row in conflict_matrix:
            print(row)
    else:
        print(f"Test file '{file_path}' not found. Cannot demonstrate.")