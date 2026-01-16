"""
Nearest Neighbor algorithm for order routing optimization.

Time Complexity: O(n²)
Space Complexity: O(n)

A greedy heuristic that builds a route by always choosing the
lowest-cost next venue. Fast but may produce suboptimal solutions.
"""

import numpy as np

from app.algorithms.base import BaseRoutingAlgorithm, RoutingSolution


class NearestNeighborAlgorithm(BaseRoutingAlgorithm):
    """
    Greedy nearest neighbor heuristic.
    
    At each step, selects the unvisited venue with lowest transition cost
    from the current venue. Simple and fast, but can get stuck in local optima.
    
    Typical performance: Within 25% of optimal on average.
    """
    
    @property
    def name(self) -> str:
        return "Nearest Neighbor"
    
    @property
    def time_complexity(self) -> str:
        return "O(n²)"
    
    @property
    def space_complexity(self) -> str:
        return "O(n)"
    
    def optimize(
        self,
        cost_matrix: np.ndarray,
        start_venue: int = 0,
        **kwargs,
    ) -> RoutingSolution:
        """
        Find a route using greedy nearest neighbor selection.
        
        Args:
            cost_matrix: NxN cost matrix
            start_venue: Index of starting venue
            
        Returns:
            RoutingSolution with greedy route
        """
        n = len(cost_matrix)
        
        if n == 0:
            return RoutingSolution(
                route=[],
                allocations=[],
                total_cost=0.0,
                iterations=0,
            )
        
        if n == 1:
            return RoutingSolution(
                route=[0],
                allocations=[1.0],
                total_cost=0.0,
                iterations=1,
            )
        
        visited = [False] * n
        route = [start_venue]
        visited[start_venue] = True
        current = start_venue
        total_cost = 0.0
        iterations = 0
        
        # Greedily select nearest unvisited venue
        for _ in range(n - 1):
            best_next = -1
            best_cost = float("inf")
            
            for next_venue in range(n):
                iterations += 1
                if not visited[next_venue]:
                    cost = cost_matrix[current][next_venue]
                    if cost < best_cost:
                        best_cost = cost
                        best_next = next_venue
            
            if best_next != -1:
                route.append(best_next)
                visited[best_next] = True
                total_cost += best_cost
                current = best_next
        
        allocations = self._default_allocations(n)
        
        return RoutingSolution(
            route=route,
            allocations=allocations,
            total_cost=total_cost,
            iterations=iterations,
            metadata={"algorithm": "nearest_neighbor", "start_venue": start_venue},
        )
    
    def optimize_multi_start(
        self,
        cost_matrix: np.ndarray,
        **kwargs,
    ) -> RoutingSolution:
        """
        Run nearest neighbor from each possible start and return best.
        
        This improves solution quality at the cost of O(n³) time.
        
        Args:
            cost_matrix: NxN cost matrix
            
        Returns:
            Best RoutingSolution across all starting points
        """
        n = len(cost_matrix)
        
        if n <= 1:
            return self.optimize(cost_matrix)
        
        best_solution = None
        total_iterations = 0
        
        for start in range(n):
            solution = self.optimize(cost_matrix, start_venue=start)
            total_iterations += solution.iterations or 0
            
            if best_solution is None or solution.total_cost < best_solution.total_cost:
                best_solution = solution
        
        if best_solution:
            best_solution.iterations = total_iterations
            best_solution.metadata = best_solution.metadata or {}
            best_solution.metadata["multi_start"] = True
            best_solution.metadata["starts_tried"] = n
        
        return best_solution or self.optimize(cost_matrix)
