"""
2-Opt local search algorithm for order routing optimization.

Time Complexity: O(n²) per iteration, O(n² · k) total where k = iterations
Space Complexity: O(n)

2-Opt is a local search algorithm that iteratively improves a route by
reversing segments. It's commonly used to refine solutions from other
heuristics like Nearest Neighbor.
"""

from typing import Optional

import numpy as np

from app.algorithms.base import BaseRoutingAlgorithm, RoutingSolution
from app.algorithms.nearest_neighbor import NearestNeighborAlgorithm
from app.config import get_settings


class TwoOptAlgorithm(BaseRoutingAlgorithm):
    """
    2-Opt local search improvement algorithm.
    
    Starting from an initial route, 2-Opt repeatedly:
    1. Selects two edges (i, i+1) and (j, j+1) in the route
    2. Removes them and reconnects by reversing the segment between
    3. Keeps the change if it improves the total cost
    
    The algorithm terminates when no improving 2-opt move exists
    (local optimum) or iteration limit is reached.
    
    2-Opt Move Visualization:
        Before: ... - A - B - ... - C - D - ...
        After:  ... - A - C - ... - B - D - ...
        (segment B...C is reversed)
    """
    
    def __init__(
        self,
        max_iterations: Optional[int] = None,
        no_improve_limit: Optional[int] = None,
        initial_route: Optional[list[int]] = None,
    ):
        """
        Initialize 2-Opt algorithm.
        
        Args:
            max_iterations: Maximum iterations (default from config)
            no_improve_limit: Stop after this many iterations without improvement
            initial_route: Starting route (if None, uses nearest neighbor)
        """
        settings = get_settings()
        self.max_iterations = max_iterations or settings.TWO_OPT_MAX_ITERATIONS
        self.no_improve_limit = no_improve_limit or settings.TWO_OPT_NO_IMPROVE_LIMIT
        self.initial_route = initial_route
    
    @property
    def name(self) -> str:
        return "2-Opt Local Search"
    
    @property
    def time_complexity(self) -> str:
        return "O(n² · k)"
    
    @property
    def space_complexity(self) -> str:
        return "O(n)"
    
    def optimize(
        self,
        cost_matrix: np.ndarray,
        initial_route: Optional[list[int]] = None,
        **kwargs,
    ) -> RoutingSolution:
        """
        Improve a route using 2-opt moves.
        
        Args:
            cost_matrix: NxN cost matrix
            initial_route: Starting route (uses nearest neighbor if not provided)
            
        Returns:
            RoutingSolution with locally optimal route
        """
        n = len(cost_matrix)
        
        if n <= 2:
            return RoutingSolution(
                route=list(range(n)),
                allocations=self._default_allocations(n) if n > 0 else [],
                total_cost=0.0 if n <= 1 else cost_matrix[0][1],
                iterations=0,
            )
        
        # Get initial route
        if initial_route is not None:
            route = list(initial_route)
        elif self.initial_route is not None:
            route = list(self.initial_route)
        else:
            # Use nearest neighbor to generate initial solution
            nn = NearestNeighborAlgorithm()
            nn_solution = nn.optimize_multi_start(cost_matrix)
            route = nn_solution.route
        
        current_cost = self._calculate_route_cost(route, cost_matrix)
        iterations = 0
        no_improve_count = 0
        improvement_history = [current_cost]
        
        improved = True
        while improved and iterations < self.max_iterations:
            improved = False
            
            for i in range(n - 1):
                for j in range(i + 2, n):
                    iterations += 1
                    
                    # Calculate cost change for 2-opt move
                    delta = self._calculate_2opt_delta(route, cost_matrix, i, j)
                    
                    if delta < -1e-10:  # Improvement found (with floating point tolerance)
                        # Apply 2-opt move (reverse segment)
                        route[i + 1:j + 1] = reversed(route[i + 1:j + 1])
                        current_cost += delta
                        improvement_history.append(current_cost)
                        improved = True
                        no_improve_count = 0
                        break
                
                if improved:
                    break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count >= self.no_improve_limit:
                    break
        
        allocations = self._default_allocations(n)
        
        return RoutingSolution(
            route=route,
            allocations=allocations,
            total_cost=current_cost,
            iterations=iterations,
            metadata={
                "algorithm": "two_opt",
                "converged": not improved,
                "improvements": len(improvement_history) - 1,
                "improvement_history": improvement_history[-10:],  # Last 10 improvements
            },
        )
    
    def _calculate_2opt_delta(
        self,
        route: list[int],
        cost_matrix: np.ndarray,
        i: int,
        j: int,
    ) -> float:
        """
        Calculate the cost change from a 2-opt move.
        
        The 2-opt move removes edges (route[i], route[i+1]) and 
        (route[j], route[j+1]) and adds edges (route[i], route[j])
        and (route[i+1], route[j+1]).
        
        For a path (not cycle), we only consider internal edges.
        
        Args:
            route: Current route
            cost_matrix: NxN cost matrix
            i: First edge starts at route[i]
            j: Second edge starts at route[j]
            
        Returns:
            Cost change (negative = improvement)
        """
        n = len(route)
        
        # Current edges
        # Edge 1: route[i] -> route[i+1]
        # Edge 2: route[j] -> route[j+1] (if j+1 exists)
        
        old_cost = cost_matrix[route[i]][route[i + 1]]
        if j + 1 < n:
            old_cost += cost_matrix[route[j]][route[j + 1]]
        
        # New edges after reversal
        # Edge 1: route[i] -> route[j]
        # Edge 2: route[i+1] -> route[j+1] (if j+1 exists)
        
        new_cost = cost_matrix[route[i]][route[j]]
        if j + 1 < n:
            new_cost += cost_matrix[route[i + 1]][route[j + 1]]
        
        return new_cost - old_cost
