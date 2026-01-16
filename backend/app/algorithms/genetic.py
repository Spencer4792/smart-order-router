"""
Genetic Algorithm for order routing optimization.

Time Complexity: O(generations · population_size · n)
Space Complexity: O(population_size · n)

A population-based evolutionary algorithm that maintains a diverse
set of solutions and evolves them through selection, crossover, and mutation.
"""

import random
from typing import Optional

import numpy as np

from app.algorithms.base import BaseRoutingAlgorithm, RoutingSolution
from app.algorithms.nearest_neighbor import NearestNeighborAlgorithm
from app.config import get_settings


class GeneticAlgorithm(BaseRoutingAlgorithm):
    """
    Genetic Algorithm for combinatorial optimization.
    
    Evolutionary approach that maintains a population of solutions:
    
    1. Initialize population with random/heuristic solutions
    2. Evaluate fitness (inverse of cost) for each individual
    3. Select parents based on fitness (tournament selection)
    4. Create offspring through crossover (OX - Order Crossover)
    5. Apply mutation (swap, insert, reverse)
    6. Replace population with offspring (elitism preserved)
    7. Repeat for specified generations
    
    Key operators:
    - Selection: Tournament selection with elitism
    - Crossover: Order Crossover (OX) preserves relative order
    - Mutation: Swap, insert, or reverse operations
    """
    
    def __init__(
        self,
        population_size: Optional[int] = None,
        generations: Optional[int] = None,
        mutation_rate: Optional[float] = None,
        crossover_rate: Optional[float] = None,
        elite_size: Optional[int] = None,
        tournament_size: int = 5,
        seed: Optional[int] = None,
    ):
        """
        Initialize Genetic Algorithm.
        
        Args:
            population_size: Number of individuals in population
            generations: Number of generations to evolve
            mutation_rate: Probability of mutation per offspring
            crossover_rate: Probability of crossover vs cloning
            elite_size: Number of best individuals preserved unchanged
            tournament_size: Number of individuals in tournament selection
            seed: Random seed for reproducibility
        """
        settings = get_settings()
        self.population_size = population_size or settings.GA_POPULATION_SIZE
        self.generations = generations or settings.GA_GENERATIONS
        self.mutation_rate = mutation_rate or settings.GA_MUTATION_RATE
        self.crossover_rate = crossover_rate or settings.GA_CROSSOVER_RATE
        self.elite_size = elite_size or settings.GA_ELITE_SIZE
        self.tournament_size = tournament_size
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    @property
    def name(self) -> str:
        return "Genetic Algorithm"
    
    @property
    def time_complexity(self) -> str:
        return "O(generations · population · n)"
    
    @property
    def space_complexity(self) -> str:
        return "O(population · n)"
    
    def optimize(
        self,
        cost_matrix: np.ndarray,
        **kwargs,
    ) -> RoutingSolution:
        """
        Find near-optimal routing using genetic algorithm.
        
        Args:
            cost_matrix: NxN cost matrix
            
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
                metadata={"algorithm": "genetic", "generations": 0},
            )
        
        # Initialize population
        population = self._initialize_population(n, cost_matrix)
        
        # Track best solution
        best_individual = min(population, key=lambda x: x[1])
        best_route = best_individual[0]
        best_cost = best_individual[1]
        
        generations_completed = 0
        fitness_history = [best_cost]
        diversity_history = []
        
        for gen in range(self.generations):
            # Sort population by fitness (cost, lower is better)
            population.sort(key=lambda x: x[1])
            
            # Track diversity
            unique_routes = len(set(tuple(ind[0]) for ind in population))
            diversity_history.append(unique_routes / self.population_size)
            
            # Update best
            if population[0][1] < best_cost:
                best_route = list(population[0][0])
                best_cost = population[0][1]
            
            # Create new generation
            new_population = []
            
            # Elitism: preserve best individuals
            for i in range(self.elite_size):
                new_population.append((list(population[i][0]), population[i][1]))
            
            # Fill rest of population with offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)
                
                # Crossover
                if random.random() < self.crossover_rate:
                    child = self._order_crossover(parent1[0], parent2[0])
                else:
                    child = list(parent1[0]) if parent1[1] < parent2[1] else list(parent2[0])
                
                # Mutation
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                child_cost = self._calculate_route_cost(child, cost_matrix)
                new_population.append((child, child_cost))
            
            population = new_population
            generations_completed = gen + 1
            
            # Record history (every 10 generations)
            if gen % 10 == 0:
                fitness_history.append(best_cost)
        
        allocations = self._default_allocations(n)
        
        return RoutingSolution(
            route=best_route,
            allocations=allocations,
            total_cost=best_cost,
            iterations=generations_completed * self.population_size,
            metadata={
                "algorithm": "genetic",
                "generations": generations_completed,
                "population_size": self.population_size,
                "mutation_rate": self.mutation_rate,
                "crossover_rate": self.crossover_rate,
                "elite_size": self.elite_size,
                "final_diversity": diversity_history[-1] if diversity_history else 0,
                "fitness_history": fitness_history[-20:],
            },
        )
    
    def _initialize_population(
        self,
        n: int,
        cost_matrix: np.ndarray,
    ) -> list[tuple[list[int], float]]:
        """
        Initialize population with mix of heuristic and random solutions.
        
        Args:
            n: Number of venues
            cost_matrix: NxN cost matrix
            
        Returns:
            List of (route, cost) tuples
        """
        population = []
        
        # Add nearest neighbor solutions from different starts
        nn = NearestNeighborAlgorithm()
        for start in range(min(n, self.population_size // 4)):
            solution = nn.optimize(cost_matrix, start_venue=start)
            population.append((solution.route, solution.total_cost))
        
        # Fill rest with random permutations
        while len(population) < self.population_size:
            route = list(range(n))
            random.shuffle(route)
            cost = self._calculate_route_cost(route, cost_matrix)
            population.append((route, cost))
        
        return population
    
    def _tournament_select(
        self,
        population: list[tuple[list[int], float]],
    ) -> tuple[list[int], float]:
        """
        Select individual using tournament selection.
        
        Args:
            population: Current population
            
        Returns:
            Selected individual (route, cost)
        """
        tournament = random.sample(population, min(self.tournament_size, len(population)))
        return min(tournament, key=lambda x: x[1])
    
    def _order_crossover(
        self,
        parent1: list[int],
        parent2: list[int],
    ) -> list[int]:
        """
        Order Crossover (OX) operator.
        
        1. Select a random segment from parent1
        2. Copy segment to child at same position
        3. Fill remaining positions with cities from parent2 in order
        
        This preserves relative ordering of cities.
        
        Args:
            parent1: First parent route
            parent2: Second parent route
            
        Returns:
            Child route
        """
        n = len(parent1)
        
        # Select crossover points
        start, end = sorted(random.sample(range(n), 2))
        
        # Initialize child with None
        child = [None] * n
        
        # Copy segment from parent1
        child[start:end + 1] = parent1[start:end + 1]
        
        # Fill remaining from parent2
        segment_set = set(parent1[start:end + 1])
        remaining = [city for city in parent2 if city not in segment_set]
        
        idx = 0
        for i in range(n):
            if child[i] is None:
                child[i] = remaining[idx]
                idx += 1
        
        return child
    
    def _mutate(self, route: list[int]) -> list[int]:
        """
        Apply mutation to a route.
        
        Randomly selects one of:
        - Swap: Exchange two cities
        - Insert: Move a city to a different position
        - Reverse: Reverse a segment
        
        Args:
            route: Route to mutate
            
        Returns:
            Mutated route
        """
        n = len(route)
        mutated = list(route)
        
        mutation_type = random.choice(["swap", "insert", "reverse"])
        
        if mutation_type == "swap":
            i, j = random.sample(range(n), 2)
            mutated[i], mutated[j] = mutated[j], mutated[i]
        
        elif mutation_type == "insert":
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                city = mutated.pop(i)
                mutated.insert(j, city)
        
        else:  # reverse
            i, j = sorted(random.sample(range(n), 2))
            mutated[i:j + 1] = reversed(mutated[i:j + 1])
        
        return mutated
