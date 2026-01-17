"""
Order routing service.

Orchestrates the routing optimization process:
1. Fetch market data
2. Build cost matrix
3. Run optimization algorithm
4. Apply smart allocation (optional)
5. Generate execution schedule (for VWAP/TWAP/IS)
6. Return routing decision with cost breakdown
"""

import time
from datetime import datetime
from typing import Optional

import numpy as np

from app.algorithms import get_algorithm
from app.algorithms.allocation_optimizer import AllocationOptimizer, AllocationResult
from app.algorithms.execution_strategies import ExecutionStrategyEngine, ExecutionStrategy
from app.config import VenueConfig
from app.models.cost import CostMatrixBuilder, CostModel
from app.models.order import (
    AlgorithmMetrics,
    AlgorithmType,
    BenchmarkRequest,
    BenchmarkResult,
    CostBreakdown,
    ExecutionScheduleInfo,
    ExecutionSlice,
    ExecutionStrategyType,
    MarketData,
    OrderRequest,
    OrderSide,
    OrderUrgency,
    RoutingResult,
    Venue,
    VenueAllocation,
    VenueType,
)
from app.services.market_data import MarketDataService


class OrderRoutingService:
    """
    Service for optimizing order routing across venues.
    
    This service:
    1. Accepts order parameters (symbol, quantity, side, urgency)
    2. Fetches current market data
    3. Constructs the cost model and matrix
    4. Runs the selected optimization algorithm
    5. Applies smart allocation optimization
    6. Generates execution schedule (VWAP/TWAP/IS)
    7. Returns detailed routing recommendations
    """
    
    def __init__(
        self,
        market_data_service: Optional[MarketDataService] = None,
        cost_model: Optional[CostModel] = None,
    ):
        self.market_data = market_data_service or MarketDataService()
        self.cost_model = cost_model or CostModel()
        self.cost_matrix_builder = CostMatrixBuilder(self.cost_model)
        self.allocation_optimizer = AllocationOptimizer()
        self.execution_engine = ExecutionStrategyEngine()
    
    def _get_venues(self, include_dark_pools: bool = False) -> list[Venue]:
        """
        Get list of available trading venues.
        
        Args:
            include_dark_pools: Whether to include dark pool venues
            
        Returns:
            List of Venue objects
        """
        venue_configs = VenueConfig.get_all_venues() if include_dark_pools else VenueConfig.get_lit_venues()
        
        venues = []
        for venue_id, config in venue_configs.items():
            venues.append(Venue(
                id=venue_id,
                name=config["name"],
                type=VenueType(config["type"]),
                maker_fee_bps=config["maker_fee_bps"],
                taker_fee_bps=config["taker_fee_bps"],
                latency_ms=config["latency_ms"],
                min_order_size=config["min_order_size"],
                supports_dark=config.get("supports_dark", False),
                avg_fill_rate=config.get("avg_fill_rate", 1.0),
            ))
        
        return venues
    
    async def route_order(self, request: OrderRequest) -> RoutingResult:
        """
        Calculate optimal order routing.
        
        Args:
            request: Order routing request
            
        Returns:
            RoutingResult with optimal routing and cost analysis
        """
        start_time = time.perf_counter()
        
        # Get market data
        market_data = await self.market_data.get_quote(request.symbol)
        adv = await self.market_data.get_adv(request.symbol)
        
        # Get venues
        venues = self._get_venues(include_dark_pools=request.include_dark_pools)
        
        # Limit venues if requested
        if request.max_venues and len(venues) > request.max_venues:
            # Sort by taker fee and take the cheapest
            venues = sorted(venues, key=lambda v: v.taker_fee_bps)[:request.max_venues]
        
        # Build cost matrix
        cost_matrix = self.cost_matrix_builder.build_cost_matrix(
            venues=venues,
            total_quantity=request.quantity,
            market_data=market_data,
            side=request.side,
            urgency=request.urgency,
            adv=adv,
        )
        
        # Run routing sequence optimization
        algorithm = get_algorithm(request.algorithm)
        solution = algorithm.optimize(cost_matrix)
        
        # Apply smart allocation or equal allocation
        if request.smart_allocation:
            allocation_results = self.allocation_optimizer.optimize_allocation(
                venues=venues,
                total_quantity=request.quantity,
                market_data=market_data,
                adv=adv,
                urgency=request.urgency,
            )
            allocation_method = "smart"
        else:
            # Equal allocation (original behavior)
            allocation_results = [
                AllocationResult(
                    venue_id=venues[i].id,
                    allocation=1.0 / len(venues),
                    quantity=request.quantity // len(venues),
                    reasoning="Equal allocation"
                )
                for i in range(len(venues))
            ]
            # Fix rounding
            total_qty = sum(a.quantity for a in allocation_results)
            if allocation_results and total_qty != request.quantity:
                allocation_results[0].quantity += (request.quantity - total_qty)
            allocation_method = "equal"
        
        # Create venue lookup for allocation results
        allocation_by_venue = {a.venue_id: a for a in allocation_results}
        
        execution_time_ms = (time.perf_counter() - start_time) * 1000
        
        # Build routing result using optimized sequence and smart allocations
        routing = self._build_routing_allocations_smart(
            solution=solution,
            venues=venues,
            allocation_by_venue=allocation_by_venue,
            total_quantity=request.quantity,
            market_data=market_data,
            side=request.side,
            urgency=request.urgency,
            adv=adv,
        )
        
        # Calculate total cost breakdown
        cost = self._calculate_total_cost(routing, request.quantity, market_data.mid_price)
        
        # Generate execution schedule if not instant
        execution_schedule = None
        if request.execution_strategy != ExecutionStrategyType.INSTANT:
            strategy_map = {
                ExecutionStrategyType.VWAP: ExecutionStrategy.VWAP,
                ExecutionStrategyType.TWAP: ExecutionStrategy.TWAP,
                ExecutionStrategyType.IMPLEMENTATION_SHORTFALL: ExecutionStrategy.IMPLEMENTATION_SHORTFALL,
                ExecutionStrategyType.AGGRESSIVE: ExecutionStrategy.AGGRESSIVE,
                ExecutionStrategyType.PASSIVE: ExecutionStrategy.PASSIVE,
            }
            
            schedule = self.execution_engine.generate_schedule(
                strategy=strategy_map[request.execution_strategy],
                total_quantity=request.quantity,
                adv=adv,
                duration_minutes=request.duration_minutes or 120,
            )
            
            execution_schedule = ExecutionScheduleInfo(
                strategy=request.execution_strategy,
                duration_minutes=schedule.duration_minutes,
                num_slices=len(schedule.slices),
                slices=[
                    ExecutionSlice(
                        slice_id=s.slice_id,
                        start_time=s.start_time,
                        end_time=s.end_time,
                        target_quantity=s.target_quantity,
                        target_percentage=s.target_percentage,
                        cumulative_percentage=s.cumulative_percentage,
                        volume_participation=s.volume_participation,
                        urgency_factor=s.urgency_factor,
                    )
                    for s in schedule.slices
                ],
                expected_participation_rate=schedule.expected_participation_rate,
                risk_score=schedule.risk_score,
            )
        
        # Calculate baseline (single best venue) for comparison
        baseline_cost = self._calculate_baseline_cost(
            venues=venues,
            quantity=request.quantity,
            market_data=market_data,
            side=request.side,
            urgency=request.urgency,
            adv=adv,
        )
        
        savings = baseline_cost.total_bps - cost.total_bps if baseline_cost else None
        
        return RoutingResult(
            symbol=request.symbol,
            side=request.side,
            total_quantity=request.quantity,
            urgency=request.urgency,
            market_data=market_data,
            routing=routing,
            cost=cost,
            algorithm_metrics=AlgorithmMetrics(
                algorithm=request.algorithm,
                execution_time_ms=execution_time_ms,
                iterations=solution.iterations,
                nodes_explored=solution.nodes_explored,
                final_temperature=solution.metadata.get("final_temperature") if solution.metadata else None,
                generations=solution.metadata.get("generations") if solution.metadata else None,
            ),
            execution_schedule=execution_schedule,
            allocation_method=allocation_method,
            baseline_cost=baseline_cost,
            savings_vs_baseline_bps=savings,
        )
    
    def _build_routing_allocations(
        self,
        solution,
        venues: list[Venue],
        total_quantity: int,
        market_data: MarketData,
        side: OrderSide,
        urgency: OrderUrgency,
        adv: int,
    ) -> list[VenueAllocation]:
        """Build detailed venue allocations from optimization solution."""
        allocations = []
        
        for seq_idx, venue_idx in enumerate(solution.route):
            venue = venues[venue_idx]
            allocation_fraction = solution.allocations[seq_idx] if seq_idx < len(solution.allocations) else 1.0 / len(solution.route)
            quantity = int(total_quantity * allocation_fraction)
            
            # Calculate costs for this allocation
            costs = self.cost_model.calculate_venue_cost(
                venue=venue,
                quantity=quantity,
                market_data=market_data,
                side=side,
                urgency=urgency,
                adv=adv,
                sequence_position=seq_idx,
            )
            
            allocations.append(VenueAllocation(
                venue_id=venue.id,
                venue_name=venue.name,
                venue_type=venue.type,
                allocation=allocation_fraction,
                quantity=quantity,
                estimated_fee_usd=costs.fee_usd,
                estimated_spread_cost_usd=costs.spread_usd,
                estimated_impact_cost_usd=costs.impact_usd,
                estimated_total_cost_usd=costs.total_usd,
                execution_sequence=seq_idx + 1,
                allocation_reasoning="Equal allocation",
            ))
        
        # Ensure quantities sum to total (handle rounding)
        total_allocated = sum(a.quantity for a in allocations)
        if total_allocated != total_quantity and allocations:
            allocations[0].quantity += (total_quantity - total_allocated)
        
        return allocations
    
    def _build_routing_allocations_smart(
        self,
        solution,
        venues: list[Venue],
        allocation_by_venue: dict[str, AllocationResult],
        total_quantity: int,
        market_data: MarketData,
        side: OrderSide,
        urgency: OrderUrgency,
        adv: int,
    ) -> list[VenueAllocation]:
        """Build detailed venue allocations using smart allocation results."""
        allocations = []
        
        # Use solution route for sequence, but smart allocation for quantities
        for seq_idx, venue_idx in enumerate(solution.route):
            venue = venues[venue_idx]
            
            # Get smart allocation for this venue
            smart_alloc = allocation_by_venue.get(venue.id)
            if smart_alloc:
                allocation_fraction = smart_alloc.allocation
                quantity = smart_alloc.quantity
                reasoning = smart_alloc.reasoning
            else:
                # Fallback to equal
                allocation_fraction = 1.0 / len(solution.route)
                quantity = int(total_quantity * allocation_fraction)
                reasoning = "Equal allocation (fallback)"
            
            # Calculate costs for this allocation
            costs = self.cost_model.calculate_venue_cost(
                venue=venue,
                quantity=quantity,
                market_data=market_data,
                side=side,
                urgency=urgency,
                adv=adv,
                sequence_position=seq_idx,
            )
            
            allocations.append(VenueAllocation(
                venue_id=venue.id,
                venue_name=venue.name,
                venue_type=venue.type,
                allocation=allocation_fraction,
                quantity=quantity,
                estimated_fee_usd=costs.fee_usd,
                estimated_spread_cost_usd=costs.spread_usd,
                estimated_impact_cost_usd=costs.impact_usd,
                estimated_total_cost_usd=costs.total_usd,
                execution_sequence=seq_idx + 1,
                allocation_reasoning=reasoning,
            ))
        
        # Sort by allocation (highest first) for better display
        allocations.sort(key=lambda x: x.allocation, reverse=True)
        
        # Re-assign sequence numbers after sorting
        for i, alloc in enumerate(allocations):
            alloc.execution_sequence = i + 1
        
        # Ensure quantities sum to total (handle rounding)
        total_allocated = sum(a.quantity for a in allocations)
        if total_allocated != total_quantity and allocations:
            allocations[0].quantity += (total_quantity - total_allocated)
        
        return allocations
    
    def _calculate_total_cost(
        self,
        routing: list[VenueAllocation],
        total_quantity: int,
        price: float,
    ) -> CostBreakdown:
        """Calculate total cost breakdown from routing allocations."""
        total_fees = sum(a.estimated_fee_usd for a in routing)
        total_spread = sum(a.estimated_spread_cost_usd for a in routing)
        total_impact = sum(a.estimated_impact_cost_usd for a in routing)
        total_usd = sum(a.estimated_total_cost_usd for a in routing)
        
        notional = total_quantity * price
        
        return CostBreakdown(
            total_bps=(total_usd / notional) * 10000 if notional > 0 else 0,
            total_usd=total_usd,
            fees_usd=total_fees,
            fees_bps=(total_fees / notional) * 10000 if notional > 0 else 0,
            spread_cost_usd=total_spread,
            spread_cost_bps=(total_spread / notional) * 10000 if notional > 0 else 0,
            impact_cost_usd=total_impact,
            impact_cost_bps=(total_impact / notional) * 10000 if notional > 0 else 0,
            latency_cost_usd=total_usd - total_fees - total_spread - total_impact,
            latency_cost_bps=((total_usd - total_fees - total_spread - total_impact) / notional) * 10000 if notional > 0 else 0,
        )
    
    def _calculate_baseline_cost(
        self,
        venues: list[Venue],
        quantity: int,
        market_data: MarketData,
        side: OrderSide,
        urgency: OrderUrgency,
        adv: int,
    ) -> CostBreakdown:
        """Calculate cost if routed to single best venue (for comparison)."""
        best_cost = float("inf")
        best_venue = None
        
        for venue in venues:
            costs = self.cost_model.calculate_venue_cost(
                venue=venue,
                quantity=quantity,
                market_data=market_data,
                side=side,
                urgency=urgency,
                adv=adv,
                sequence_position=0,
            )
            if costs.total_bps < best_cost:
                best_cost = costs.total_bps
                best_venue = venue
        
        if best_venue is None:
            return CostBreakdown(
                total_bps=0, total_usd=0, fees_usd=0, fees_bps=0,
                spread_cost_usd=0, spread_cost_bps=0,
                impact_cost_usd=0, impact_cost_bps=0,
                latency_cost_usd=0, latency_cost_bps=0,
            )
        
        costs = self.cost_model.calculate_venue_cost(
            venue=best_venue,
            quantity=quantity,
            market_data=market_data,
            side=side,
            urgency=urgency,
            adv=adv,
            sequence_position=0,
        )
        
        notional = quantity * market_data.mid_price
        
        return CostBreakdown(
            total_bps=costs.total_bps,
            total_usd=costs.total_usd,
            fees_usd=costs.fee_usd,
            fees_bps=costs.fee_bps,
            spread_cost_usd=costs.spread_usd,
            spread_cost_bps=costs.spread_bps,
            impact_cost_usd=costs.impact_usd,
            impact_cost_bps=costs.impact_bps,
            latency_cost_usd=costs.latency_usd,
            latency_cost_bps=costs.latency_bps,
        )
    
    async def benchmark_algorithms(self, request: BenchmarkRequest) -> BenchmarkResult:
        """
        Run multiple algorithms and compare results.
        
        Args:
            request: Benchmark request parameters
            
        Returns:
            BenchmarkResult with comparison data
        """
        # Get market data once
        market_data = await self.market_data.get_quote(request.symbol)
        adv = await self.market_data.get_adv(request.symbol)
        venues = self._get_venues(include_dark_pools=False)
        
        algorithms = list(request.algorithms)
        if request.include_exact:
            if len(venues) <= 10:
                algorithms.append(AlgorithmType.BRUTE_FORCE)
            if len(venues) <= 20:
                algorithms.append(AlgorithmType.HELD_KARP)
        
        results = []
        for algo_type in algorithms:
            order_request = OrderRequest(
                symbol=request.symbol,
                quantity=request.quantity,
                side=request.side,
                urgency=request.urgency,
                algorithm=algo_type,
            )
            result = await self.route_order(order_request)
            results.append(result)
        
        # Find best and worst
        sorted_results = sorted(results, key=lambda r: r.cost.total_bps)
        best = sorted_results[0].algorithm_metrics.algorithm
        worst = sorted_results[-1].algorithm_metrics.algorithm
        cost_range = sorted_results[-1].cost.total_bps - sorted_results[0].cost.total_bps
        
        return BenchmarkResult(
            symbol=request.symbol,
            quantity=request.quantity,
            num_venues=len(venues),
            results=results,
            best_algorithm=best,
            worst_algorithm=worst,
            cost_range_bps=cost_range,
        )


# Singleton instance
_routing_service: Optional[OrderRoutingService] = None


def get_routing_service() -> OrderRoutingService:
    """Get the singleton routing service."""
    global _routing_service
    if _routing_service is None:
        _routing_service = OrderRoutingService()
    return _routing_service
