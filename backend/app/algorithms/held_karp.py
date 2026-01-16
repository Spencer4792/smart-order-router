"""
Held-Karp algorithm for order routing optimization.

Time Complexity: O(n² · 2ⁿ)
Space Complexity: O(n · 2ⁿ)

This algorithm uses dynamic programming with bitmask state compression
to find the optimal routing. It's significantly faster than brute force
but still exponential, practical for n <= 20.

Reference:
    Held, M., & Karp, R. M. (1962). A dynamic programming approach 
    to sequencing problems. Journal of SIAM.
"""

from typing import Optional

import numpy as np

from app.algorithms.base import BaseRoutingAlgorithm, RoutingSolution


class HeldKarpAlgorithm(BaseRoutingAlgorithm):
    """
    Dynamic programming with bitmask state compression.
    
    The Held-Karp algorithm reduces the complexity from O(n!) to O(n²·2ⁿ)
    by using memoization over subsets of venues.
    
    State: dp[S][i] = minimum cost to visit all venues in set S, ending at venue i
    
    Recurrence:
        dp[S][i] = min(dp[S\\{i}][j] + cost[j][i]) for all j in S\\{i}
    
    The bitmask S represents which venues have been visited.
    """
    
    def __init__(self, max_venues: int = 20):
        """
        Initialize Held-Karp algorithm.
        
        Args:
            max_venues: Maximum venues to process (memory limit)
        """
        self.max_venues = max_venues
    
    @property
    def name(self) -> str:
        return "Held-Karp (Dynamic Programming)"
    
    @property
    def time_complexity(self) -> str:
        return "O(n² · 2ⁿ)"
    
    @property
    def space_complexity(self) -> str:
        return "O(n · 2ⁿ)"
    
    @property
    def is_exact(self) -> bool:
        return True
    
    def optimize(
        self,
        cost_matrix: np.ndarray,
        start_venue: int = 0,
        **kwargs,
    ) -> RoutingSolution:
        """
        Find optimal routing using dynamic programming.
        
        Args:
            cost_matrix: NxN cost matrix
            start_venue: Index of starting venue (default: 0)
            
        Returns:
            RoutingSolution with globally optimal route
            
        Raises:
            ValueError: If n > max_venues (memory constraint)
        """
        n = len(cost_matrix)
        
        if n > self.max_venues:
            raise ValueError(
                f"Held-Karp requires O(n·2ⁿ) memory. "
                f"For {n} venues, this exceeds the limit of {self.max_venues}. "
                f"Use a heuristic algorithm instead."
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
        
        # dp[mask][i] = (min_cost, prev_venue)
        # mask is a bitmask where bit j is set if venue j has been visited
        INF = float("inf")
        dp = [[INF] * n for _ in range(1 << n)]
        parent = [[-1] * n for _ in range(1 << n)]
        
        nodes_explored = 0
        
        # Base case: start at start_venue
        dp[1 << start_venue][start_venue] = 0
        nodes_explored += 1
        
        # Fill DP table
        for mask in range(1 << n):
            for last in range(n):
                if not (mask & (1 << last)):
                    continue
                if dp[mask][last] == INF:
                    continue
                
                # Try extending to each unvisited venue
                for next_venue in range(n):
                    if mask & (1 << next_venue):
                        continue  # Already visited
                    
                    new_mask = mask | (1 << next_venue)
                    new_cost = dp[mask][last] + cost_matrix[last][next_venue]
                    nodes_explored += 1
                    
                    if new_cost < dp[new_mask][next_venue]:
                        dp[new_mask][next_venue] = new_cost
                        parent[new_mask][next_venue] = last
        
        # Find the best ending venue (all venues visited)
        full_mask = (1 << n) - 1
        best_cost = INF
        best_end = -1
        
        for i in range(n):
            if dp[full_mask][i] < best_cost:
                best_cost = dp[full_mask][i]
                best_end = i
        
        # Reconstruct the path
        route = self._reconstruct_path(parent, full_mask, best_end, n)
        
        # Equal allocation
        allocations = self._default_allocations(n)
        
        return RoutingSolution(
            route=route,
            allocations=allocations,
            total_cost=best_cost,
            iterations=nodes_explored,
            nodes_explored=nodes_explored,
            metadata={
                "algorithm": "held_karp",
                "optimal": True,
                "dp_states": 1 << n,
            },
        )
    
    def _reconstruct_path(
        self,
        parent: list[list[int]],
        mask: int,
        last: int,
        n: int,
    ) -> list[int]:
        """Reconstruct the optimal path from parent pointers."""
        path = []
        current = last
        current_mask = mask
        
        while current != -1:
            path.append(current)
            prev = parent[current_mask][current]
            current_mask ^= (1 << current)
            current = prev
        
        path.reverse()
        return path
