"""
Brute Force algorithm for order routing optimization.

Time Complexity: O(n!)
Space Complexity: O(n)

This algorithm exhaustively searches all possible permutations of venues
to find the globally optimal routing. Only practical for small numbers
of venues (n <= 10).
"""

from itertools import permutations
from typing import Optional

import numpy as np

from app.algorithms.base import BaseRoutingAlgorithm, RoutingSolution


class BruteForceAlgorithm(BaseRoutingAlgorithm):
    """
    Exhaustive search over all permutations.
    
    Guarantees optimal solution but has factorial time complexity.
    Only suitable for small venue sets (n <= 10).
    """
    
    def __init__(self, max_venues: int = 10):
        """
        Initialize brute force algorithm.
        
        Args:
            max_venues: Maximum number of venues to process (safety limit)
        """
        self.max_venues = max_venues
    
    @property
    def name(self) -> str:
        return "Brute Force"
    
    @property
    def time_complexity(self) -> str:
        return "O(n!)"
    
    @property
    def space_complexity(self) -> str:
        return "O(n)"
    
    @property
    def is_exact(self) -> bool:
        return True
    
    def optimize(
        self,
        cost_matrix: np.ndarray,
        **kwargs,
    ) -> RoutingSolution:
        """
        Find optimal routing by exhaustive search.
        
        Args:
            cost_matrix: NxN cost matrix
            
        Returns:
            RoutingSolution with globally optimal route
            
        Raises:
            ValueError: If n > max_venues
        """
        n = len(cost_matrix)
        
        if n > self.max_venues:
            raise ValueError(
                f"Brute force is impractical for {n} venues. "
                f"Maximum is {self.max_venues}. Use a heuristic algorithm instead."
            )
        
        if n == 0:
            return RoutingSolution(
                route=[],
                allocations=[],
                total_cost=0.0,
                iterations=0,
                nodes_explored=0,
            )
        
        if n == 1:
            return RoutingSolution(
                route=[0],
                allocations=[1.0],
                total_cost=0.0,
                iterations=1,
                nodes_explored=1,
            )
        
        best_route: Optional[list[int]] = None
        best_cost = float("inf")
        nodes_explored = 0
        
        # Generate all permutations
        for perm in permutations(range(n)):
            route = list(perm)
            cost = self._calculate_route_cost(route, cost_matrix)
            nodes_explored += 1
            
            if cost < best_cost:
                best_cost = cost
                best_route = route
        
        # Equal allocation across all venues in the route
        allocations = self._default_allocations(n)
        
        return RoutingSolution(
            route=best_route or list(range(n)),
            allocations=allocations,
            total_cost=best_cost,
            iterations=nodes_explored,
            nodes_explored=nodes_explored,
            metadata={"algorithm": "brute_force", "optimal": True},
        )
