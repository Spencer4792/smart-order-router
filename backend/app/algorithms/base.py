"""
Abstract base class for routing optimization algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class RoutingSolution:
    """
    Solution from a routing optimization algorithm.
    
    Attributes:
        route: Ordered list of venue indices representing execution sequence
        allocations: Fraction of order allocated to each venue (same order as route)
        total_cost: Total cost in basis points
        iterations: Number of iterations/evaluations performed
        nodes_explored: Number of nodes explored (for exact algorithms)
        metadata: Algorithm-specific metadata
    """
    
    route: list[int]
    allocations: list[float]
    total_cost: float
    iterations: Optional[int] = None
    nodes_explored: Optional[int] = None
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        # Normalize allocations to sum to 1
        total = sum(self.allocations)
        if total > 0:
            self.allocations = [a / total for a in self.allocations]


class BaseRoutingAlgorithm(ABC):
    """
    Abstract base class for order routing optimization algorithms.
    
    All routing algorithms must implement:
    1. optimize() - Find optimal routing given a cost matrix
    2. name - Human-readable algorithm name
    3. time_complexity - Big-O time complexity
    4. space_complexity - Big-O space complexity
    
    The cost matrix is NxN where:
    - N is the number of venues
    - cost_matrix[i][j] is the cost of routing to venue j after venue i
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable algorithm name."""
        pass
    
    @property
    @abstractmethod
    def time_complexity(self) -> str:
        """Big-O time complexity."""
        pass
    
    @property
    @abstractmethod
    def space_complexity(self) -> str:
        """Big-O space complexity."""
        pass
    
    @property
    def is_exact(self) -> bool:
        """Whether this algorithm guarantees optimal solution."""
        return False
    
    @abstractmethod
    def optimize(
        self,
        cost_matrix: np.ndarray,
        **kwargs,
    ) -> RoutingSolution:
        """
        Find optimal or near-optimal routing.
        
        Args:
            cost_matrix: NxN matrix of transition costs between venues
            **kwargs: Algorithm-specific parameters
            
        Returns:
            RoutingSolution with optimal route and allocations
        """
        pass
    
    def _calculate_route_cost(
        self,
        route: list[int],
        cost_matrix: np.ndarray,
    ) -> float:
        """
        Calculate total cost of a route.
        
        Args:
            route: List of venue indices in execution order
            cost_matrix: NxN cost matrix
            
        Returns:
            Total cost in basis points
        """
        if len(route) <= 1:
            return 0.0
        
        total_cost = 0.0
        for i in range(len(route) - 1):
            total_cost += cost_matrix[route[i]][route[i + 1]]
        
        return total_cost
    
    def _default_allocations(self, n_venues: int) -> list[float]:
        """
        Generate default equal allocations.
        
        For more sophisticated allocation, see the AllocationOptimizer.
        """
        return [1.0 / n_venues] * n_venues
