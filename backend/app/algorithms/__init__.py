"""
Routing optimization algorithms.

This module provides multiple algorithms for solving the order routing
optimization problem, which is modeled as a variant of the Traveling
Salesman Problem (TSP).

Algorithms are categorized as:

Exact Algorithms (guarantee optimal solution):
- Brute Force: O(n!) - for n <= 10
- Held-Karp: O(n² · 2ⁿ) - for n <= 20

Heuristic Algorithms (near-optimal solutions):
- Nearest Neighbor: O(n²) - fast greedy baseline
- 2-Opt: O(n² · k) - local search improvement
- Simulated Annealing: O(iterations · n) - escapes local optima
- Genetic Algorithm: O(generations · population · n) - evolutionary
"""

from typing import Type

from app.algorithms.base import BaseRoutingAlgorithm, RoutingSolution
from app.algorithms.brute_force import BruteForceAlgorithm
from app.algorithms.genetic import GeneticAlgorithm
from app.algorithms.held_karp import HeldKarpAlgorithm
from app.algorithms.nearest_neighbor import NearestNeighborAlgorithm
from app.algorithms.simulated_annealing import SimulatedAnnealingAlgorithm
from app.algorithms.two_opt import TwoOptAlgorithm
from app.models.order import AlgorithmType


# Registry of available algorithms
ALGORITHM_REGISTRY: dict[AlgorithmType, Type[BaseRoutingAlgorithm]] = {
    AlgorithmType.BRUTE_FORCE: BruteForceAlgorithm,
    AlgorithmType.HELD_KARP: HeldKarpAlgorithm,
    AlgorithmType.NEAREST_NEIGHBOR: NearestNeighborAlgorithm,
    AlgorithmType.TWO_OPT: TwoOptAlgorithm,
    AlgorithmType.SIMULATED_ANNEALING: SimulatedAnnealingAlgorithm,
    AlgorithmType.GENETIC: GeneticAlgorithm,
}


def get_algorithm(algorithm_type: AlgorithmType, **kwargs) -> BaseRoutingAlgorithm:
    """
    Factory function to get an algorithm instance.
    
    Args:
        algorithm_type: Type of algorithm to instantiate
        **kwargs: Algorithm-specific parameters
        
    Returns:
        Configured algorithm instance
        
    Raises:
        ValueError: If algorithm type is not recognized
    """
    if algorithm_type not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm type: {algorithm_type}")
    
    algorithm_class = ALGORITHM_REGISTRY[algorithm_type]
    return algorithm_class(**kwargs)


def get_algorithm_info() -> list[dict]:
    """
    Get information about all available algorithms.
    
    Returns:
        List of algorithm info dictionaries
    """
    info = []
    for algo_type, algo_class in ALGORITHM_REGISTRY.items():
        instance = algo_class()
        info.append({
            "type": algo_type.value,
            "name": instance.name,
            "time_complexity": instance.time_complexity,
            "space_complexity": instance.space_complexity,
            "is_exact": instance.is_exact,
        })
    return info


__all__ = [
    "BaseRoutingAlgorithm",
    "RoutingSolution",
    "BruteForceAlgorithm",
    "HeldKarpAlgorithm",
    "NearestNeighborAlgorithm",
    "TwoOptAlgorithm",
    "SimulatedAnnealingAlgorithm",
    "GeneticAlgorithm",
    "ALGORITHM_REGISTRY",
    "get_algorithm",
    "get_algorithm_info",
]
