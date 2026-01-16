"""
Simulated Annealing algorithm for order routing optimization.

Time Complexity: O(iterations · n)
Space Complexity: O(n)

Simulated Annealing is a probabilistic metaheuristic that can escape
local optima by occasionally accepting worse solutions. The probability
of accepting worse solutions decreases over time (cooling schedule).
"""

import math
import random
from typing import Optional

import numpy as np

from app.algorithms.base import BaseRoutingAlgorithm, RoutingSolution
from app.algorithms.nearest_neighbor import NearestNeighborAlgorithm
from app.config import get_settings


class SimulatedAnnealingAlgorithm(BaseRoutingAlgorithm):
    """
    Simulated Annealing metaheuristic.
    
    Inspired by the physical process of annealing in metallurgy, this
    algorithm explores the solution space by:
    
    1. Starting with an initial solution and high "temperature"
    2. Making random perturbations (swaps, reversals)
    3. Always accepting improvements
    4. Accepting worse solutions with probability exp(-delta/T)
    5. Gradually reducing temperature (cooling)
    
    The cooling schedule determines how quickly T decreases:
        T(k+1) = T(k) * cooling_rate
    
    As T approaches 0, the algorithm behaves like pure local search.
    """
    
    def __init__(
        self,
        initial_temp: Optional[float] = None,
        cooling_rate: Optional[float] = None,
        min_temp: Optional[float] = None,
        max_iterations: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize Simulated Annealing.
        
        Args:
            initial_temp: Starting temperature (default from config)
            cooling_rate: Temperature multiplier each iteration (0 < rate < 1)
            min_temp: Minimum temperature before stopping
            max_iterations: Maximum iterations
            seed: Random seed for reproducibility
        """
        settings = get_settings()
        self.initial_temp = initial_temp or settings.SA_INITIAL_TEMP
        self.cooling_rate = cooling_rate or settings.SA_COOLING_RATE
        self.min_temp = min_temp or settings.SA_MIN_TEMP
        self.max_iterations = max_iterations or settings.SA_ITERATIONS
        
        if seed is not None:
            random.seed(seed)
    
    @property
    def name(self) -> str:
        return "Simulated Annealing"
    
    @property
    def time_complexity(self) -> str:
        return "O(iterations · n)"
    
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
        Find near-optimal routing using simulated annealing.
        
        Args:
            cost_matrix: NxN cost matrix
            initial_route: Starting route (uses nearest neighbor if not provided)
            
        Returns:
            RoutingSolution with optimized route
        """
        n = len(cost_matrix)
        
        if n <= 2:
            return RoutingSolution(
                route=list(range(n)),
                allocations=self._default_allocations(n) if n > 0 else [],
                total_cost=0.0 if n <= 1 else cost_matrix[0][1],
                iterations=0,
            )
        
        # Initialize
        if initial_route is not None:
            current_route = list(initial_route)
        else:
            nn = NearestNeighborAlgorithm()
            nn_solution = nn.optimize_multi_start(cost_matrix)
            current_route = nn_solution.route
        
        current_cost = self._calculate_route_cost(current_route, cost_matrix)
        
        best_route = list(current_route)
        best_cost = current_cost
        
        temperature = self.initial_temp
        iterations = 0
        accepted_moves = 0
        rejected_moves = 0
        improvements = 0
        
        cost_history = [current_cost]
        temp_history = [temperature]
        
        while temperature > self.min_temp and iterations < self.max_iterations:
            # Generate neighbor solution
            new_route, move_type = self._generate_neighbor(current_route)
            new_cost = self._calculate_route_cost(new_route, cost_matrix)
            
            delta = new_cost - current_cost
            
            # Accept or reject
            if delta < 0:
                # Improvement - always accept
                current_route = new_route
                current_cost = new_cost
                accepted_moves += 1
                improvements += 1
                
                # Update best if this is globally best
                if current_cost < best_cost:
                    best_route = list(current_route)
                    best_cost = current_cost
            else:
                # Worse solution - accept with probability exp(-delta/T)
                acceptance_prob = math.exp(-delta / temperature)
                if random.random() < acceptance_prob:
                    current_route = new_route
                    current_cost = new_cost
                    accepted_moves += 1
                else:
                    rejected_moves += 1
            
            # Cool down
            temperature *= self.cooling_rate
            iterations += 1
            
            # Record history (sample every 100 iterations)
            if iterations % 100 == 0:
                cost_history.append(current_cost)
                temp_history.append(temperature)
        
        allocations = self._default_allocations(n)
        
        return RoutingSolution(
            route=best_route,
            allocations=allocations,
            total_cost=best_cost,
            iterations=iterations,
            metadata={
                "algorithm": "simulated_annealing",
                "final_temperature": temperature,
                "accepted_moves": accepted_moves,
                "rejected_moves": rejected_moves,
                "improvements": improvements,
                "acceptance_rate": accepted_moves / (accepted_moves + rejected_moves) if (accepted_moves + rejected_moves) > 0 else 0,
                "cost_history": cost_history[-20:],  # Last 20 samples
                "initial_temp": self.initial_temp,
                "cooling_rate": self.cooling_rate,
            },
        )
    
    def _generate_neighbor(self, route: list[int]) -> tuple[list[int], str]:
        """
        Generate a neighboring solution by random perturbation.
        
        Randomly selects one of three move types:
        1. Swap: Exchange two random cities
        2. Insert: Remove a city and insert elsewhere
        3. Reverse: Reverse a random segment (2-opt move)
        
        Args:
            route: Current route
            
        Returns:
            Tuple of (new_route, move_type)
        """
        n = len(route)
        new_route = list(route)
        
        move_type = random.choice(["swap", "insert", "reverse"])
        
        if move_type == "swap":
            # Swap two random positions
            i, j = random.sample(range(n), 2)
            new_route[i], new_route[j] = new_route[j], new_route[i]
        
        elif move_type == "insert":
            # Remove from position i and insert at position j
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                city = new_route.pop(i)
                new_route.insert(j, city)
        
        else:  # reverse
            # Reverse a random segment (2-opt style)
            i, j = sorted(random.sample(range(n), 2))
            new_route[i:j + 1] = reversed(new_route[i:j + 1])
        
        return new_route, move_type
