# tfisp_solver/__init__.py

from . import parser
from . import qubo_formulator

# Import the solvers

from .solvers.gurobi_solver import GurobiSolver
from .solvers.cplex_solver import CplexSolver
from .solvers.sa_solver import SimulatedAnnealingSolver
from .solvers.dwave_hybrid_solver import DwaveHybridSolver

# Create a solver registry for easy access by cli.py
solver_registry = {
    'gurobi': GurobiSolver,
    'cplex': CplexSolver,
    'simulated_annealing': SimulatedAnnealingSolver,
    'dwave_hybrid': DwaveHybridSolver,
}

__all__ = [
    'parser',
    'qubo_formulator',
    'solver_registry',
]