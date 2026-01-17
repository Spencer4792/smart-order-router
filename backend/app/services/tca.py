"""
Transaction Cost Analysis (TCA) Module.

This module provides post-trade analysis to evaluate execution quality.
TCA is the industry standard for measuring how well an order was executed.

Key Metrics:
- Implementation Shortfall: Total cost vs decision price
- VWAP Slippage: Performance vs volume-weighted average price
- Arrival Price Slippage: Drift from when execution started
- Market Impact: Price movement caused by our order
- Timing Cost: Cost from waiting to execute

Reference:
    Kissell, R. (2013). The Science of Algorithmic Trading and Portfolio Management.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import random
import math

from app.models.order import (
    ExecutionScheduleInfo,
    MarketData,
    OrderSide,
    RoutingResult,
    VenueAllocation,
)


class BenchmarkType(str, Enum):
    """Benchmark types for TCA comparison."""
    ARRIVAL_PRICE = "arrival_price"  # Price when order was received
    VWAP = "vwap"  # Volume-weighted average price
    TWAP = "twap"  # Time-weighted average price
    CLOSE = "close"  # Closing price
    OPEN = "open"  # Opening price
    PREVIOUS_CLOSE = "previous_close"


@dataclass
class ExecutionFill:
    """A single execution fill."""
    fill_id: int
    timestamp: datetime
    venue_id: str
    quantity: int
    price: float
    side: OrderSide
    fee_usd: float
    
    @property
    def notional(self) -> float:
        return self.quantity * self.price


@dataclass
class SimulatedExecution:
    """Result of simulating an order execution."""
    order_id: str
    symbol: str
    side: OrderSide
    total_quantity: int
    
    # Timing
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    
    # Fills
    fills: list[ExecutionFill]
    
    # Prices
    arrival_price: float
    vwap_benchmark: float
    average_fill_price: float
    final_price: float
    
    # Completion
    fill_rate: float  # What % of order was filled
    num_fills: int


@dataclass
class CostAttribution:
    """Attribution of execution costs to different factors."""
    spread_cost_bps: float  # Cost from crossing bid-ask spread
    impact_cost_bps: float  # Cost from our order moving the market
    timing_cost_bps: float  # Cost from price drift while waiting
    fee_cost_bps: float  # Exchange/venue fees
    opportunity_cost_bps: float  # Cost from unfilled portion
    
    @property
    def total_bps(self) -> float:
        return (
            self.spread_cost_bps +
            self.impact_cost_bps +
            self.timing_cost_bps +
            self.fee_cost_bps +
            self.opportunity_cost_bps
        )


@dataclass
class TCAReport:
    """Comprehensive Transaction Cost Analysis report."""
    
    # Order Info
    order_id: str
    symbol: str
    side: OrderSide
    total_quantity: int
    filled_quantity: int
    
    # Execution Summary
    execution: SimulatedExecution
    
    # Benchmark Prices
    arrival_price: float
    vwap_benchmark: float
    twap_benchmark: float
    
    # Performance vs Benchmarks (negative = outperformed, positive = underperformed)
    arrival_slippage_bps: float
    vwap_slippage_bps: float
    twap_slippage_bps: float
    implementation_shortfall_bps: float
    
    # Cost Attribution
    cost_attribution: CostAttribution
    
    # Quality Metrics
    fill_rate: float
    participation_rate: float  # Our volume as % of market volume
    average_fill_price: float
    price_improvement_bps: float  # vs worst case (crossing full spread)
    
    # Risk Metrics
    execution_risk_score: float  # 0-1, how volatile was execution
    timing_risk_realized: float  # How much price moved during execution
    
    # Venue Analysis
    venue_performance: dict[str, dict]  # Per-venue metrics
    
    # Timestamps
    analysis_timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def summary(self) -> dict:
        """Get a summary of key metrics."""
        return {
            "implementation_shortfall_bps": self.implementation_shortfall_bps,
            "vwap_slippage_bps": self.vwap_slippage_bps,
            "arrival_slippage_bps": self.arrival_slippage_bps,
            "total_cost_bps": self.cost_attribution.total_bps,
            "fill_rate": self.fill_rate,
            "avg_fill_price": self.average_fill_price,
        }


class ExecutionSimulator:
    """
    Simulates order execution for TCA analysis.
    
    Since we don't have real execution data, this simulator creates
    realistic execution scenarios based on:
    - Market microstructure models
    - Volume patterns
    - Price impact models
    - Random market movements
    """
    
    def __init__(
        self,
        volatility: float = 0.02,
        spread_bps: float = 2.0,
        impact_coefficient: float = 0.1,
        seed: Optional[int] = None,
    ):
        self.volatility = volatility
        self.spread_bps = spread_bps
        self.impact_coefficient = impact_coefficient
        if seed is not None:
            random.seed(seed)
    
    def simulate_execution(
        self,
        routing_result: RoutingResult,
        execution_schedule: Optional[ExecutionScheduleInfo] = None,
    ) -> SimulatedExecution:
        """
        Simulate execution of a routed order.
        
        Args:
            routing_result: The routing plan to execute
            execution_schedule: Optional time schedule (for VWAP/TWAP)
            
        Returns:
            SimulatedExecution with fill details
        """
        symbol = routing_result.symbol
        side = routing_result.side
        total_quantity = routing_result.total_quantity
        arrival_price = routing_result.market_data.mid_price
        
        start_time = datetime.utcnow()
        fills = []
        fill_id = 0
        
        # Determine execution duration
        if execution_schedule and execution_schedule.slices:
            duration_minutes = execution_schedule.duration_minutes
        else:
            # Instant execution - simulate as 1-minute execution
            duration_minutes = 1
        
        # Simulate price path during execution
        price_path = self._simulate_price_path(
            start_price=arrival_price,
            duration_minutes=duration_minutes,
            volatility=self.volatility,
        )
        
        # Track cumulative impact
        cumulative_impact = 0.0
        cumulative_quantity = 0
        
        # Generate fills for each venue allocation
        for alloc in routing_result.routing:
            venue_quantity = alloc.quantity
            if venue_quantity <= 0:
                continue
            
            # Determine timing of fills for this venue
            if execution_schedule and execution_schedule.slices:
                # Spread fills across schedule
                venue_fills = self._generate_scheduled_fills(
                    venue_id=alloc.venue_id,
                    quantity=venue_quantity,
                    schedule=execution_schedule,
                    price_path=price_path,
                    start_time=start_time,
                    side=side,
                    cumulative_impact=cumulative_impact,
                    fee_per_share=alloc.estimated_fee_usd / venue_quantity if venue_quantity > 0 else 0,
                    fill_id_start=fill_id,
                )
            else:
                # Instant execution - all at once
                venue_fills = self._generate_instant_fills(
                    venue_id=alloc.venue_id,
                    quantity=venue_quantity,
                    base_price=arrival_price,
                    start_time=start_time,
                    side=side,
                    cumulative_impact=cumulative_impact,
                    fee_per_share=alloc.estimated_fee_usd / venue_quantity if venue_quantity > 0 else 0,
                    fill_id_start=fill_id,
                )
            
            fills.extend(venue_fills)
            fill_id += len(venue_fills)
            
            # Update cumulative impact
            cumulative_quantity += venue_quantity
            cumulative_impact = self._calculate_cumulative_impact(
                cumulative_quantity, total_quantity, arrival_price
            )
        
        # Calculate execution metrics
        end_time = start_time + timedelta(minutes=duration_minutes)
        filled_quantity = sum(f.quantity for f in fills)
        
        if filled_quantity > 0:
            average_fill_price = sum(f.notional for f in fills) / filled_quantity
        else:
            average_fill_price = arrival_price
        
        # VWAP benchmark (simulated market VWAP)
        vwap_benchmark = self._calculate_simulated_vwap(price_path)
        
        return SimulatedExecution(
            order_id=str(routing_result.order_id),
            symbol=symbol,
            side=side,
            total_quantity=total_quantity,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration_minutes * 60,
            fills=fills,
            arrival_price=arrival_price,
            vwap_benchmark=vwap_benchmark,
            average_fill_price=average_fill_price,
            final_price=price_path[-1] if price_path else arrival_price,
            fill_rate=filled_quantity / total_quantity if total_quantity > 0 else 0,
            num_fills=len(fills),
        )
    
    def _simulate_price_path(
        self,
        start_price: float,
        duration_minutes: int,
        volatility: float,
    ) -> list[float]:
        """Simulate a price path using geometric Brownian motion."""
        # One price point per minute
        num_points = max(duration_minutes, 1)
        prices = [start_price]
        
        # Scale volatility to per-minute
        minute_vol = volatility / math.sqrt(390)  # 390 minutes in trading day
        
        for _ in range(num_points - 1):
            # Random return
            drift = 0  # No drift assumption
            shock = random.gauss(0, minute_vol)
            return_pct = drift + shock
            
            new_price = prices[-1] * (1 + return_pct)
            prices.append(new_price)
        
        return prices
    
    def _calculate_cumulative_impact(
        self,
        cumulative_quantity: int,
        total_quantity: int,
        base_price: float,
    ) -> float:
        """Calculate cumulative market impact in price terms."""
        # Square root impact model
        participation = cumulative_quantity / total_quantity if total_quantity > 0 else 0
        impact_bps = self.impact_coefficient * math.sqrt(participation) * 100  # Scale to bps
        return impact_bps * base_price / 10000
    
    def _generate_scheduled_fills(
        self,
        venue_id: str,
        quantity: int,
        schedule: ExecutionScheduleInfo,
        price_path: list[float],
        start_time: datetime,
        side: OrderSide,
        cumulative_impact: float,
        fee_per_share: float,
        fill_id_start: int,
    ) -> list[ExecutionFill]:
        """Generate fills spread across execution schedule."""
        fills = []
        remaining = quantity
        fill_id = fill_id_start
        
        for i, slice_info in enumerate(schedule.slices):
            if remaining <= 0:
                break
            
            # Quantity for this slice (proportional to schedule)
            slice_qty = min(
                int(quantity * slice_info.target_percentage),
                remaining
            )
            
            if slice_qty <= 0:
                continue
            
            # Price at this time point
            price_idx = min(i * (len(price_path) // len(schedule.slices)), len(price_path) - 1)
            base_price = price_path[price_idx]
            
            # Add spread cost and impact
            spread_cost = base_price * (self.spread_bps / 2 / 10000)
            impact_cost = cumulative_impact
            
            if side == OrderSide.BUY:
                fill_price = base_price + spread_cost + impact_cost
            else:
                fill_price = base_price - spread_cost - impact_cost
            
            # Add some randomness
            fill_price *= (1 + random.gauss(0, 0.0001))
            
            fill_time = start_time + timedelta(
                minutes=i * (schedule.duration_minutes / len(schedule.slices))
            )
            
            fills.append(ExecutionFill(
                fill_id=fill_id,
                timestamp=fill_time,
                venue_id=venue_id,
                quantity=slice_qty,
                price=round(fill_price, 4),
                side=side,
                fee_usd=slice_qty * fee_per_share,
            ))
            
            fill_id += 1
            remaining -= slice_qty
        
        # Handle any remaining quantity in last fill
        if remaining > 0 and fills:
            fills[-1].quantity += remaining
            fills[-1].fee_usd += remaining * fee_per_share
        
        return fills
    
    def _generate_instant_fills(
        self,
        venue_id: str,
        quantity: int,
        base_price: float,
        start_time: datetime,
        side: OrderSide,
        cumulative_impact: float,
        fee_per_share: float,
        fill_id_start: int,
    ) -> list[ExecutionFill]:
        """Generate fills for instant execution."""
        # Split into a few fills to simulate partial fills
        num_fills = min(3, max(1, quantity // 1000))
        fills = []
        remaining = quantity
        
        for i in range(num_fills):
            if remaining <= 0:
                break
            
            fill_qty = remaining if i == num_fills - 1 else remaining // (num_fills - i)
            
            # Add spread and impact
            spread_cost = base_price * (self.spread_bps / 2 / 10000)
            
            if side == OrderSide.BUY:
                fill_price = base_price + spread_cost + cumulative_impact
            else:
                fill_price = base_price - spread_cost - cumulative_impact
            
            # Small random variation
            fill_price *= (1 + random.gauss(0, 0.0002))
            
            fills.append(ExecutionFill(
                fill_id=fill_id_start + i,
                timestamp=start_time + timedelta(seconds=i * 2),
                venue_id=venue_id,
                quantity=fill_qty,
                price=round(fill_price, 4),
                side=side,
                fee_usd=fill_qty * fee_per_share,
            ))
            
            remaining -= fill_qty
        
        return fills
    
    def _calculate_simulated_vwap(self, price_path: list[float]) -> float:
        """Calculate simulated market VWAP from price path."""
        if not price_path:
            return 0
        
        # Simulate volume profile (U-shaped)
        n = len(price_path)
        volumes = []
        for i in range(n):
            # Higher volume at start and end
            t = i / max(n - 1, 1)
            vol = 1 + 0.5 * (4 * (t - 0.5) ** 2)  # U-shape
            volumes.append(vol)
        
        # Calculate VWAP
        total_volume = sum(volumes)
        vwap = sum(p * v for p, v in zip(price_path, volumes)) / total_volume
        return vwap


class TCAAnalyzer:
    """
    Analyzes execution quality and generates TCA reports.
    """
    
    def __init__(self, simulator: Optional[ExecutionSimulator] = None):
        self.simulator = simulator or ExecutionSimulator()
    
    def analyze(
        self,
        routing_result: RoutingResult,
        execution: Optional[SimulatedExecution] = None,
        market_volume: Optional[int] = None,
    ) -> TCAReport:
        """
        Perform Transaction Cost Analysis on an execution.
        
        Args:
            routing_result: The routing plan
            execution: Actual execution (simulated if not provided)
            market_volume: Total market volume during execution period
            
        Returns:
            TCAReport with comprehensive analysis
        """
        # Simulate execution if not provided
        if execution is None:
            execution = self.simulator.simulate_execution(
                routing_result,
                routing_result.execution_schedule,
            )
        
        # Extract key prices
        arrival_price = execution.arrival_price
        avg_fill_price = execution.average_fill_price
        vwap_benchmark = execution.vwap_benchmark
        twap_benchmark = sum(
            f.price for f in execution.fills
        ) / len(execution.fills) if execution.fills else arrival_price
        
        side = execution.side
        
        # Calculate slippage (positive = underperformed)
        if side == OrderSide.BUY:
            arrival_slippage_bps = (avg_fill_price - arrival_price) / arrival_price * 10000
            vwap_slippage_bps = (avg_fill_price - vwap_benchmark) / vwap_benchmark * 10000
            twap_slippage_bps = (avg_fill_price - twap_benchmark) / twap_benchmark * 10000
        else:
            arrival_slippage_bps = (arrival_price - avg_fill_price) / arrival_price * 10000
            vwap_slippage_bps = (vwap_benchmark - avg_fill_price) / vwap_benchmark * 10000
            twap_slippage_bps = (twap_benchmark - avg_fill_price) / twap_benchmark * 10000
        
        # Implementation shortfall
        implementation_shortfall_bps = arrival_slippage_bps
        
        # Cost attribution
        cost_attribution = self._attribute_costs(
            execution=execution,
            routing_result=routing_result,
            arrival_price=arrival_price,
        )
        
        # Calculate participation rate
        if market_volume and market_volume > 0:
            participation_rate = execution.total_quantity / market_volume
        else:
            # Estimate based on ADV assumption
            estimated_volume = 10_000_000 * (execution.duration_seconds / 23400)  # 23400 sec in trading day
            participation_rate = execution.total_quantity / max(estimated_volume, 1)
        
        # Price improvement vs worst case (full spread crossing)
        worst_case_cost_bps = routing_result.market_data.spread_bps
        actual_cost_bps = abs(arrival_slippage_bps)
        price_improvement_bps = worst_case_cost_bps - actual_cost_bps
        
        # Execution risk score
        price_volatility = self._calculate_execution_volatility(execution)
        execution_risk_score = min(1.0, price_volatility / 0.01)  # Normalize to 0-1
        
        # Timing risk realized
        timing_risk = abs(execution.final_price - arrival_price) / arrival_price * 10000
        
        # Per-venue analysis
        venue_performance = self._analyze_venue_performance(execution, arrival_price)
        
        return TCAReport(
            order_id=execution.order_id,
            symbol=execution.symbol,
            side=side,
            total_quantity=execution.total_quantity,
            filled_quantity=sum(f.quantity for f in execution.fills),
            execution=execution,
            arrival_price=arrival_price,
            vwap_benchmark=vwap_benchmark,
            twap_benchmark=twap_benchmark,
            arrival_slippage_bps=arrival_slippage_bps,
            vwap_slippage_bps=vwap_slippage_bps,
            twap_slippage_bps=twap_slippage_bps,
            implementation_shortfall_bps=implementation_shortfall_bps,
            cost_attribution=cost_attribution,
            fill_rate=execution.fill_rate,
            participation_rate=participation_rate,
            average_fill_price=avg_fill_price,
            price_improvement_bps=price_improvement_bps,
            execution_risk_score=execution_risk_score,
            timing_risk_realized=timing_risk,
            venue_performance=venue_performance,
        )
    
    def _attribute_costs(
        self,
        execution: SimulatedExecution,
        routing_result: RoutingResult,
        arrival_price: float,
    ) -> CostAttribution:
        """Attribute total costs to different factors."""
        total_quantity = execution.total_quantity
        filled_quantity = sum(f.quantity for f in execution.fills)
        avg_fill_price = execution.average_fill_price
        
        # Spread cost (estimated from routing)
        spread_cost_bps = routing_result.cost.spread_cost_bps
        
        # Impact cost
        impact_cost_bps = routing_result.cost.impact_cost_bps
        
        # Timing cost (price drift during execution)
        price_drift = execution.final_price - arrival_price
        if execution.side == OrderSide.BUY:
            timing_cost_bps = (price_drift / arrival_price) * 10000
        else:
            timing_cost_bps = (-price_drift / arrival_price) * 10000
        timing_cost_bps = max(0, timing_cost_bps)  # Only count adverse drift
        
        # Fee cost
        total_fees = sum(f.fee_usd for f in execution.fills)
        notional = filled_quantity * avg_fill_price
        fee_cost_bps = (total_fees / notional * 10000) if notional > 0 else 0
        
        # Opportunity cost (unfilled portion)
        unfilled = total_quantity - filled_quantity
        if unfilled > 0 and execution.side == OrderSide.BUY:
            # Price went up, missed opportunity
            opportunity_cost_bps = (execution.final_price - arrival_price) / arrival_price * 10000 * (unfilled / total_quantity)
        elif unfilled > 0:
            opportunity_cost_bps = (arrival_price - execution.final_price) / arrival_price * 10000 * (unfilled / total_quantity)
        else:
            opportunity_cost_bps = 0
        opportunity_cost_bps = max(0, opportunity_cost_bps)
        
        return CostAttribution(
            spread_cost_bps=spread_cost_bps,
            impact_cost_bps=impact_cost_bps,
            timing_cost_bps=timing_cost_bps,
            fee_cost_bps=fee_cost_bps,
            opportunity_cost_bps=opportunity_cost_bps,
        )
    
    def _calculate_execution_volatility(self, execution: SimulatedExecution) -> float:
        """Calculate price volatility during execution."""
        if len(execution.fills) < 2:
            return 0
        
        prices = [f.price for f in execution.fills]
        returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
        
        if not returns:
            return 0
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        return math.sqrt(variance)
    
    def _analyze_venue_performance(
        self,
        execution: SimulatedExecution,
        arrival_price: float,
    ) -> dict[str, dict]:
        """Analyze performance by venue."""
        venue_fills: dict[str, list[ExecutionFill]] = {}
        
        for fill in execution.fills:
            if fill.venue_id not in venue_fills:
                venue_fills[fill.venue_id] = []
            venue_fills[fill.venue_id].append(fill)
        
        venue_performance = {}
        for venue_id, fills in venue_fills.items():
            quantity = sum(f.quantity for f in fills)
            notional = sum(f.notional for f in fills)
            avg_price = notional / quantity if quantity > 0 else 0
            fees = sum(f.fee_usd for f in fills)
            
            if execution.side == OrderSide.BUY:
                slippage_bps = (avg_price - arrival_price) / arrival_price * 10000
            else:
                slippage_bps = (arrival_price - avg_price) / arrival_price * 10000
            
            venue_performance[venue_id] = {
                "quantity": quantity,
                "num_fills": len(fills),
                "avg_price": round(avg_price, 4),
                "slippage_bps": round(slippage_bps, 2),
                "fees_usd": round(fees, 2),
                "notional": round(notional, 2),
            }
        
        return venue_performance


# Convenience function
def run_tca(routing_result: RoutingResult, seed: Optional[int] = None) -> TCAReport:
    """Run TCA analysis on a routing result."""
    simulator = ExecutionSimulator(seed=seed)
    analyzer = TCAAnalyzer(simulator)
    return analyzer.analyze(routing_result)
