# solvers/base_solver.py
import abc
import os
import json
import sys
from typing import Any, Dict

class BaseSolver(abc.ABC):
    """
    An abstract base class for all TFISP solvers.

    This class defines the standard interface that every solver must implement
    (the 'solve' method). It also provides a shared utility for loading solver
    configuration from a JSON file, as required by the system architecture
    """
    def __init__(self, config_path: str):
        """
        Initializes the solver with configuration data.

        Args:
            config_path: The path to the JSON configuration file.
        """
        self.config = self._safe_load_json(config_path)

    @staticmethod
    def _safe_load_json(path: str) -> Dict[str, Any]:
        """
        Safely loads and parses a JSON file, mirroring the robust logic
        from cli.py's helper function.
        """
        if not os.path.isfile(path):
            print(f"[WARN] Config file '{path}' not found. Using empty config.")
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Invalid JSON in config file '{path}': {e}")
            sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Unexpected error loading config '{path}': {e}")
            sys.exit(1)

    @abc.abstractmethod
    def solve(self, bqm_problem: Any) -> Dict[str, Any]:
        """
        Solves the QUBO problem instance.

        This method must be implemented by all concrete solver classes.

        Args:
            bqm_problem: A problem instance, typically a dimod.BinaryQuadraticModel.

        Returns:
            A dictionary containing the solution, cost, and execution time.
            Expected format: {'cost': float, 'time': float, 'assignment': dict}
        """
        pass