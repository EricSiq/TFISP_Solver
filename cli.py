
# cli.py - Command Line Interface for TFISP Solver
import argparse
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
import inspect
import sys
import traceback

#from tfisp_solver import parser as input_parser
#from tfisp_solver.qubo_formulator import build_qubo
#from tfisp_solver.solvers import solver_registry

 
# Helper utilities (robust)
 
def file_exists_or_exit(path, friendly_name="file"):
    if not path:
        print(f"[ERROR] No {friendly_name} path provided.")
        sys.exit(1)
    if not os.path.isfile(path):
        print(f"[ERROR] {friendly_name} not found: {path}")
        sys.exit(1)

def safe_load_json(path):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[WARN] Config file '{path}' not found. Using empty/default config.")
        return {}
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in config file '{path}': {e}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error loading config '{path}': {e}")
        sys.exit(1)

def job_get_times(job):
    """Support job as namedtuple/object or tuple/list (job_id, start, end)."""
    try:
        # namedtuple or object with attributes
        start = getattr(job, "start_time", None)
        end = getattr(job, "end_time", None)
        if start is not None and end is not None:
            return float(start), float(end)
    except Exception:
        pass

    # fallback: sequence
    try:
        return float(job[1]), float(job[2])
    except Exception:
        raise ValueError(f"Cannot extract start/end from job: {job}")

def build_conflict_matrix_safe(jobs):
    """Construct Oij: 1 when intervals overlap, 0 otherwise. Works with job tuple or object."""
    n = len(jobs)
    O = [[0] * n for _ in range(n)]
    for i in range(n):
        si, ei = job_get_times(jobs[i])
        for j in range(i + 1, n):
            sj, ej = job_get_times(jobs[j])
            # overlap if i.start < j.end and j.start < i.end (strict overlap)
            if (ei > sj) and (ej > si):
                O[i][j] = O[j][i] = 1
    return O

def qubo_to_iterable(qubo):
    """
    Accept different QUBO/BQM representations and return an iterable of ((v1, v2), coeff).
    Handles:
      - dict-like ( {(v1,v2):coeff, ...} )
      - dimod.BinaryQuadraticModel (converted with to_qubo or .quadratic)
      - dimod.SampleSet (not expected here, but we try to extract bqm)
    """
    # If it's a dict of pairs -> coeff
    if isinstance(qubo, dict):
        # maybe nested dict (qubo dict of dicts), convert to pair keys
        # try flat pairs first
        try:
            for k, v in qubo.items():
                if isinstance(k, tuple) and len(k) == 2:
                    yield k, v
                else:
                    # nested dict attempt
                    break
            else:
                return
        except Exception:
            pass

        # nested dict case: map to ((i,j), coeff)
        for u, row in qubo.items():
            if isinstance(row, dict):
                for v, coeff in row.items():
                    # stable ordering
                    key = tuple(sorted((u, v)))
                    yield key, coeff
        return

    # If it's a dimod BinaryQuadraticModel
    try:
        import dimod
        if isinstance(qubo, dimod.BinaryQuadraticModel):
            # we can convert to (quad, lin, offset)
            # but to present sample terms, iterate quadratic + linear
            for (u, v), coeff in qubo.quadratic.items():
                yield (u, v), coeff
            for v, coeff in qubo.linear.items():
                yield (v, v), coeff
            return
    except Exception:
        pass

    # Fallback: try .to_qubo()
    try:
        t = getattr(qubo, "to_qubo", None)
        if callable(t):
            Q, offset = t() if t.__code__.co_argcount == 0 else t()
            # Q is dict-of-dicts
            for u, row in Q.items():
                for v, coeff in row.items():
                    key = tuple(sorted((u, v)))
                    yield key, coeff
            return
    except Exception:
        pass

    # If nothing worked, raise
    raise TypeError("Unsupported QUBO/BQM type for printing sample terms.")


# Main

def main(input_file, solvers_list, output_dir=None, config_file="config.json"):
    # Basic validation of input/config files
    if input_file is None:
        print("[ERROR] --input is required.")
        return # Use return instead of sys.exit(1) in notebook context
    if not os.path.isfile(input_file):
        print(f"[ERROR] Input file does not exist: {input_file}")
        return # Use return instead of sys.exit(1) in notebook context

    config = safe_load_json(config_file)
    # Print parsed arguments
    print("Input file:", input_file)
    solvers = [s.strip().lower() for s in solvers_list.split(",") if s.strip()]
    if not solvers:
        print("[ERROR] No solvers provided in --solvers.")
        return # Use return instead of sys.exit(1) in notebook context
    print("Solvers:", solvers)
    print("Output directory:", output_dir or "Auto-generated")
    print("Config file:", config_file)

    print("\nLoading problem instance...")
    try:
        # Replace with actual import once package is available
        # from tfisp_solver import parser as input_parser
        # num_jobs, num_resources, resource_costs, jobs, disqualified_pairs = input_parser.parse_input_file(input_file)
        # Placeholder for now:
        num_jobs = 0
        num_resources = 0
        resource_costs = {}
        jobs = []
        disqualified_pairs = []
        print("[WARN] Using placeholder data for parsing due to missing tfisp_solver.parser")

    except Exception as e:
        print(f"[ERROR] Failed to parse input file '{input_file}': {e}")
        traceback.print_exc()
        return # Use return instead of sys.exit(1) in notebook context

    # extra sanity checks
    try:
        if num_jobs != len(jobs):
            print(f"[WARN] Declared NUM_JOBS={num_jobs} but parsed {len(jobs)} job entries. Using parsed count.")
            num_jobs = len(jobs)
    except Exception:
        pass

    print("Parsed Data Summary:")
    print("  Number of Jobs:", num_jobs)
    print("  Number of Resources:", num_resources)
    print("  Resource Costs:", resource_costs)
    print("  Jobs:", jobs)
    print("  Disqualified Pairs:", disqualified_pairs)

    # Build conflict matrix (use parser-provided function if available, else local safe builder)
    conflict_matrix = None
    try:
        # if hasattr(input_parser, "build_conflict_matrix"):
        #     conflict_matrix = input_parser.build_conflict_matrix(jobs)
        # else:
        conflict_matrix = build_conflict_matrix_safe(jobs)
    except Exception as e:
        print(f"[WARN] Could not build conflict matrix using parser: {e}. Falling back to safe builder.")
        conflict_matrix = build_conflict_matrix_safe(jobs)

    print("\nFormulating QUBO...")
    # Build qubo in a robust way depending on build_qubo signature
    try:
        # Replace with actual import once package is available
        # from tfisp_solver.qubo_formulator import build_qubo
        # sig = inspect.signature(build_qubo)
        # kwargs = {
        #     "num_jobs": num_jobs,
        #     "num_resources": num_resources,
        #     "resource_costs": resource_costs,
        #     "jobs": jobs,
        #     "disqualified_pairs": disqualified_pairs
        # }
        # # pass conflict_matrix/config if function supports them
        # if "conflict_matrix" in sig.parameters:
        #     kwargs["conflict_matrix"] = conflict_matrix
        # if "config" in sig.parameters:
        #     kwargs["config"] = config

        # qubo = build_qubo(**kwargs)
        # Placeholder for now:
        qubo = {}
        print("[WARN] Using placeholder QUBO formulation due to missing tfisp_solver.qubo_formulator")

    except TypeError as e:
        # fallback: try basic call (older/newer API mismatch)
        print(f"[WARN] build_qubo call mismatch: {e}. Attempting fallback call without optional args.")
        try:
            # qubo = build_qubo(num_jobs, num_resources, resource_costs, jobs, disqualified_pairs)
            qubo = {} # Placeholder
        except Exception as e2:
            print(f"[ERROR] Failed to build QUBO: {e2}")
            traceback.print_exc()
            return # Use return instead of sys.exit(1) in notebook context
    except Exception as e:
        print(f"[ERROR] Unexpected error building QUBO: {e}")
        traceback.print_exc()
        return # Use return instead of sys.exit(1) in notebook context

    # Try to get a size/length for QUBO for friendly info
    try:
        # If it's a dimod.BQM, len() may return something meaningful; else try to convert
        q_sample_count = None
        try:
            import dimod
            if isinstance(qubo, dimod.BinaryQuadraticModel):
                q_sample_count = len(qubo.linear) + len(qubo.quadratic)
        except Exception:
            pass
        if q_sample_count is None:
            try:
                q_sample_count = len(qubo)
            except Exception:
                q_sample_count = "unknown"
        print(f"QUBO built with approx {q_sample_count} terms.")
    except Exception:
        print("QUBO built (size unknown).")

    # Optionally, print a few QUBO entries (robustly)
    print("\nSample QUBO terms:")
    try:
        for i, (key, coeff) in enumerate(qubo_to_iterable(qubo)):
            var1, var2 = key
            print(f"{var1} * {var2}: {coeff}")
            if i >= 9:
                break
    except Exception as e:
        print(f"[WARN] Could not iterate QUBO terms for printing: {e}")

    # Prepare output directories (respect user-provided output dir)
    results_dir = output_dir if output_dir else "results"
    plots_dir = os.path.join(results_dir, "plots")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Collect results for plotting
    all_results = {}

    print("\nSolving QUBO with selected solvers...")

    for solver_name in solvers:
        # Replace with actual import once package is available
        # SolverClass = solver_registry.get(solver_name)
        # Placeholder for now:
        SolverClass = None
        print(f"[WARN] Using placeholder solver due to missing tfisp_solver.solvers")

        if not SolverClass:
            print(f"[!] Solver '{solver_name}' not recognized or available. Skipping.")
            continue

        try:
            # instantiate solver - allow both config_path or config dict usage
            try:
                solver = SolverClass(config_path=config_file)
            except TypeError:
                # fallback: some solver classes accept a dict
                solver = SolverClass(config)

            result = solver.solve(qubo)

            # Validate result fields
            if not isinstance(result, dict):
                print(f"[WARN] Solver '{solver_name}' returned unexpected result type ({type(result)}). Attempting to interpret.")
                # try to interpret results from a dimod.SampleSet (if returned)
                if hasattr(result, "first"):
                    # sample set fallback (very heuristic)
                    sample = result.first.sample
                    cost = result.first.energy if hasattr(result.first, "energy") else None
                    duration = getattr(result, "info", {}).get("timings", {}).get("total_time", 0)
                    assignment = sample
                else:
                    print(f"[ERROR] Cannot interpret solver result for '{solver_name}'. Skipping.")
                    continue
            else:
                cost = result.get("cost", None)
                duration = result.get("time", None)
                assignment = result.get("assignment", None)

                # Best-effort fill-ins
                if cost is None and "raw" in result:
                    cost = result["raw"].get("cost")
                if duration is None:
                    duration = result.get("duration") or result.get("runtime") or 0.0
                if assignment is None:
                    assignment = result.get("solution") or result.get("sample")

            # Final validation
            if assignment is None:
                print(f"[WARN] Solver '{solver_name}' returned no assignment. Storing partial results.")
                assignment = {}

            # Ensure numeric values exist for cost/time
            try:
                cost = float(cost) if cost is not None else float("inf")
            except Exception:
                cost = float("inf")
            try:
                duration = float(duration) if duration is not None else 0.0
            except Exception:
                duration = 0.0

            print(f"\nSolver: {solver_name}")
            print(f"  Cost: {cost}")
            print(f"  Time: {duration:.3f}s")
            print(f"  Assignments: {assignment if isinstance(assignment, dict) else 'see result object'}")

            # Save results to file with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_path = os.path.join(results_dir, f"{solver_name}_{timestamp}.txt")
            try:
                with open(result_path, "w") as f:
                    f.write(f"Solver: {solver_name}\n")
                    f.write(f"Cost: {cost}\n")
                    f.write(f"Time: {duration:.3f}s\n")
                    f.write(f"Assignments:\n")
                    if isinstance(assignment, dict):
                        for var, val in assignment.items():
                            f.write(f"{var}: {val}\n")
                    else:
                        f.write(repr(assignment) + "\n")
                print(f"Saved result to: {result_path}")
            except Exception as e:
                print(f"[WARN] Failed to save result file for solver '{solver_name}': {e}")

            all_results[solver_name] = {
                "cost": cost,
                "time": duration,
                "num_assignments": len(assignment) if isinstance(assignment, dict) else 1
            }

        except Exception as e:
            print(f"[!] Error while solving with '{solver_name}': {e}")
            traceback.print_exc()
            continue

    # Final Report: Combined Execution Time + Normalized Cost
    if all_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        solver_names = list(all_results.keys())
        times = [all_results[s]["time"] for s in solver_names]
        raw_costs = [all_results[s]["cost"] for s in solver_names]

        # Normalize costs; avoid division by zero
        min_cost = min(raw_costs) if raw_costs else 0
        normalized_costs = []
        for cost in raw_costs:
            try:
                normalized_costs.append(cost / min_cost if min_cost not in (0, float("inf")) else 0.0)
            except Exception:
                normalized_costs.append(0.0)

        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Execution Time Plot
        axes[0].bar(solver_names, times, color='skyblue')
        axes[0].set_title("Solver Execution Time")
        axes[0].set_ylabel("Time (s)")
        axes[0].tick_params(axis='x', rotation=45)

        # Add numeric labels inside bars (Execution Time)
        for idx, time in enumerate(times):
            axes[0].text(
                idx,
                time / 2 if time else 0,
                f"{time:.3f}s",
                ha='center',
                va='center',
                color='black',
                fontsize=9
            )

        # Normalized Cost Plot
        axes[1].bar(solver_names, normalized_costs, color='salmon')
        axes[1].set_title("Solver Solution Quality")
        axes[1].set_ylabel("Normalized Cost (Best = 1.0)")
        axes[1].tick_params(axis='x', rotation=45)

        # Add numeric labels inside bars (Normalized Cost)
        for idx, cost in enumerate(normalized_costs):
            axes[1].text(
                idx,
                cost / 2 if cost else 0,
                f"{cost:.2f}",
                ha='center',
                va='center',
                color='black',
                fontsize=9
            )

        plt.tight_layout()
        report_path = os.path.join(plots_dir, f"performance_report_{timestamp}.png")
        try:
            plt.savefig(report_path)
            print(f"Performance report saved: {report_path}")
        except Exception as e:
            print(f"[WARN] Could not save performance report: {e}")

# Example usage within Colab:
# Define your input file and solvers here
input_file = "/path/to/your/input.txt" # <--- CHANGE THIS
solvers_list = "simulated_annealing,dwave_ocean" # <--- CHANGE THIS

# Run the main function
main(input_file, solvers_list)