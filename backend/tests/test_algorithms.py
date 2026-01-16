"""
Unit tests for routing optimization algorithms.
"""

import numpy as np
import pytest

from app.algorithms import (
    BruteForceAlgorithm,
    GeneticAlgorithm,
    HeldKarpAlgorithm,
    NearestNeighborAlgorithm,
    SimulatedAnnealingAlgorithm,
    TwoOptAlgorithm,
    get_algorithm,
)
from app.models.order import AlgorithmType


class TestCostMatrixGeneration:
    """Test cost matrix utilities."""
    
    @staticmethod
    def create_symmetric_matrix(n: int) -> np.ndarray:
        """Create a symmetric cost matrix for testing."""
        np.random.seed(42)
        matrix = np.random.rand(n, n) * 10
        matrix = (matrix + matrix.T) / 2
        np.fill_diagonal(matrix, 0)
        return matrix
    
    @staticmethod
    def create_known_optimal_matrix() -> tuple[np.ndarray, list[int], float]:
        """
        Create a small matrix with known optimal solution.
        
        Returns:
            Tuple of (matrix, optimal_route, optimal_cost)
        """
        # Simple 4-node problem where optimal is 0->1->2->3
        matrix = np.array([
            [0, 1, 10, 10],
            [1, 0, 1, 10],
            [10, 1, 0, 1],
            [10, 10, 1, 0],
        ], dtype=float)
        # Optimal route: 0->1->2->3 with cost 1+1+1 = 3
        return matrix, [0, 1, 2, 3], 3.0


class TestBruteForce:
    """Test brute force algorithm."""
    
    def test_empty_matrix(self):
        algo = BruteForceAlgorithm()
        result = algo.optimize(np.array([]))
        assert result.route == []
        assert result.total_cost == 0.0
    
    def test_single_venue(self):
        algo = BruteForceAlgorithm()
        result = algo.optimize(np.array([[0]]))
        assert result.route == [0]
        assert result.total_cost == 0.0
    
    def test_two_venues(self):
        algo = BruteForceAlgorithm()
        matrix = np.array([[0, 5], [5, 0]])
        result = algo.optimize(matrix)
        assert len(result.route) == 2
        assert result.total_cost == 5.0
    
    def test_known_optimal(self):
        matrix, expected_route, expected_cost = TestCostMatrixGeneration.create_known_optimal_matrix()
        algo = BruteForceAlgorithm()
        result = algo.optimize(matrix)
        assert result.total_cost == pytest.approx(expected_cost, abs=0.01)
    
    def test_max_venues_limit(self):
        algo = BruteForceAlgorithm(max_venues=5)
        matrix = TestCostMatrixGeneration.create_symmetric_matrix(10)
        with pytest.raises(ValueError, match="impractical"):
            algo.optimize(matrix)


class TestHeldKarp:
    """Test Held-Karp dynamic programming algorithm."""
    
    def test_empty_matrix(self):
        algo = HeldKarpAlgorithm()
        result = algo.optimize(np.array([]))
        assert result.route == []
        assert result.total_cost == 0.0
    
    def test_single_venue(self):
        algo = HeldKarpAlgorithm()
        result = algo.optimize(np.array([[0]]))
        assert result.route == [0]
    
    def test_known_optimal(self):
        matrix, expected_route, expected_cost = TestCostMatrixGeneration.create_known_optimal_matrix()
        algo = HeldKarpAlgorithm()
        result = algo.optimize(matrix)
        assert result.total_cost == pytest.approx(expected_cost, abs=0.01)
    
    def test_matches_brute_force(self):
        """Held-Karp should produce same result as brute force."""
        matrix = TestCostMatrixGeneration.create_symmetric_matrix(8)
        
        bf = BruteForceAlgorithm()
        hk = HeldKarpAlgorithm()
        
        bf_result = bf.optimize(matrix)
        hk_result = hk.optimize(matrix)
        
        assert bf_result.total_cost == pytest.approx(hk_result.total_cost, abs=0.01)


class TestNearestNeighbor:
    """Test nearest neighbor greedy algorithm."""
    
    def test_empty_matrix(self):
        algo = NearestNeighborAlgorithm()
        result = algo.optimize(np.array([]))
        assert result.route == []
    
    def test_single_venue(self):
        algo = NearestNeighborAlgorithm()
        result = algo.optimize(np.array([[0]]))
        assert result.route == [0]
    
    def test_greedy_selection(self):
        # Matrix where nearest neighbor should pick 0->1->2
        matrix = np.array([
            [0, 1, 100],
            [1, 0, 1],
            [100, 1, 0],
        ], dtype=float)
        
        algo = NearestNeighborAlgorithm()
        result = algo.optimize(matrix, start_venue=0)
        
        assert result.route[0] == 0
        assert result.route[1] == 1  # Nearest to 0
        assert result.route[2] == 2  # Nearest to 1
    
    def test_multi_start_improves(self):
        matrix = TestCostMatrixGeneration.create_symmetric_matrix(10)
        algo = NearestNeighborAlgorithm()
        
        single_start = algo.optimize(matrix, start_venue=0)
        multi_start = algo.optimize_multi_start(matrix)
        
        # Multi-start should be at least as good
        assert multi_start.total_cost <= single_start.total_cost


class TestTwoOpt:
    """Test 2-opt local search algorithm."""
    
    def test_improves_solution(self):
        matrix = TestCostMatrixGeneration.create_symmetric_matrix(10)
        
        nn = NearestNeighborAlgorithm()
        two_opt = TwoOptAlgorithm()
        
        nn_result = nn.optimize(matrix)
        two_opt_result = two_opt.optimize(matrix, initial_route=nn_result.route)
        
        # 2-opt should improve or maintain the solution
        assert two_opt_result.total_cost <= nn_result.total_cost + 0.01
    
    def test_preserves_valid_route(self):
        matrix = TestCostMatrixGeneration.create_symmetric_matrix(8)
        algo = TwoOptAlgorithm()
        result = algo.optimize(matrix)
        
        # Check route is a valid permutation
        assert sorted(result.route) == list(range(8))


class TestSimulatedAnnealing:
    """Test simulated annealing algorithm."""
    
    def test_produces_valid_route(self):
        matrix = TestCostMatrixGeneration.create_symmetric_matrix(10)
        algo = SimulatedAnnealingAlgorithm(max_iterations=1000, seed=42)
        result = algo.optimize(matrix)
        
        # Check route is valid permutation
        assert sorted(result.route) == list(range(10))
    
    def test_reproducibility_with_seed(self):
        matrix = TestCostMatrixGeneration.create_symmetric_matrix(10)
        
        algo1 = SimulatedAnnealingAlgorithm(max_iterations=500, seed=42)
        algo2 = SimulatedAnnealingAlgorithm(max_iterations=500, seed=42)
        
        result1 = algo1.optimize(matrix)
        result2 = algo2.optimize(matrix)
        
        assert result1.total_cost == result2.total_cost
    
    def test_quality_on_known_problem(self):
        matrix, _, optimal_cost = TestCostMatrixGeneration.create_known_optimal_matrix()
        algo = SimulatedAnnealingAlgorithm(max_iterations=5000, seed=42)
        result = algo.optimize(matrix)
        
        # SA should get within 50% of optimal on this small problem
        assert result.total_cost <= optimal_cost * 1.5


class TestGeneticAlgorithm:
    """Test genetic algorithm."""
    
    def test_produces_valid_route(self):
        matrix = TestCostMatrixGeneration.create_symmetric_matrix(10)
        algo = GeneticAlgorithm(population_size=50, generations=100, seed=42)
        result = algo.optimize(matrix)
        
        assert sorted(result.route) == list(range(10))
    
    def test_reproducibility_with_seed(self):
        matrix = TestCostMatrixGeneration.create_symmetric_matrix(8)
        
        algo1 = GeneticAlgorithm(population_size=30, generations=50, seed=42)
        algo2 = GeneticAlgorithm(population_size=30, generations=50, seed=42)
        
        result1 = algo1.optimize(matrix)
        result2 = algo2.optimize(matrix)
        
        assert result1.total_cost == result2.total_cost


class TestAlgorithmFactory:
    """Test algorithm factory function."""
    
    def test_get_all_algorithms(self):
        for algo_type in AlgorithmType:
            algo = get_algorithm(algo_type)
            assert algo is not None
            assert hasattr(algo, 'optimize')


class TestAlgorithmComparison:
    """Compare algorithm quality across different problem sizes."""
    
    @pytest.mark.parametrize("n", [5, 8])
    def test_heuristics_vs_exact(self, n):
        """Heuristics should be within reasonable range of optimal."""
        matrix = TestCostMatrixGeneration.create_symmetric_matrix(n)
        
        # Get optimal solution
        exact = HeldKarpAlgorithm()
        optimal = exact.optimize(matrix)
        
        # Test each heuristic
        heuristics = [
            NearestNeighborAlgorithm(),
            TwoOptAlgorithm(),
            SimulatedAnnealingAlgorithm(max_iterations=2000, seed=42),
            GeneticAlgorithm(population_size=50, generations=100, seed=42),
        ]
        
        for heuristic in heuristics:
            result = heuristic.optimize(matrix)
            # Heuristics should be within 50% of optimal
            assert result.total_cost <= optimal.total_cost * 1.5, \
                f"{heuristic.name} exceeded 50% optimality gap"
