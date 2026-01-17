"""
Data models for orders, venues, and routing results.
"""

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator


class OrderSide(str, Enum):
    """Order side (buy or sell)."""
    BUY = "buy"
    SELL = "sell"


class OrderUrgency(str, Enum):
    """
    Order urgency level affects the cost model weights.
    
    - LOW: Minimize fees, accept longer execution time
    - MEDIUM: Balance between cost and speed
    - HIGH: Prioritize speed, accept higher market impact
    """
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ExecutionStrategyType(str, Enum):
    """Available execution strategies."""
    VWAP = "vwap"
    TWAP = "twap"
    IMPLEMENTATION_SHORTFALL = "is"
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    INSTANT = "instant"  # Execute immediately (original behavior)


class AlgorithmType(str, Enum):
    """Available routing optimization algorithms."""
    BRUTE_FORCE = "brute_force"
    HELD_KARP = "held_karp"
    NEAREST_NEIGHBOR = "nearest_neighbor"
    TWO_OPT = "two_opt"
    SIMULATED_ANNEALING = "simulated_annealing"
    GENETIC = "genetic"


class VenueType(str, Enum):
    """Type of trading venue."""
    EXCHANGE = "exchange"
    DARK_POOL = "dark_pool"
    ATS = "ats"  # Alternative Trading System


class Venue(BaseModel):
    """Trading venue representation."""
    
    id: str
    name: str
    type: VenueType
    maker_fee_bps: float = Field(description="Maker fee in basis points (negative = rebate)")
    taker_fee_bps: float = Field(description="Taker fee in basis points")
    latency_ms: float = Field(description="Estimated latency in milliseconds")
    min_order_size: int = Field(default=1, description="Minimum order size in shares")
    supports_dark: bool = Field(default=False, description="Whether venue supports dark orders")
    avg_fill_rate: Optional[float] = Field(
        default=1.0, 
        description="Average fill rate (1.0 for lit venues, <1.0 for dark pools)"
    )


class MarketData(BaseModel):
    """Real-time market data for a symbol."""
    
    symbol: str
    bid: float
    ask: float
    bid_size: int
    ask_size: int
    last_price: float
    volume: int
    vwap: Optional[float] = None
    timestamp: datetime
    
    @property
    def spread(self) -> float:
        """Calculate bid-ask spread."""
        return self.ask - self.bid
    
    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points."""
        mid = (self.bid + self.ask) / 2
        return (self.spread / mid) * 10000 if mid > 0 else 0
    
    @property
    def mid_price(self) -> float:
        """Calculate mid price."""
        return (self.bid + self.ask) / 2


class OrderRequest(BaseModel):
    """Request to route an order."""
    
    symbol: str = Field(description="Stock symbol (e.g., 'AAPL')")
    quantity: int = Field(gt=0, description="Number of shares to trade")
    side: OrderSide = Field(description="Buy or sell")
    urgency: OrderUrgency = Field(default=OrderUrgency.MEDIUM)
    algorithm: AlgorithmType = Field(default=AlgorithmType.SIMULATED_ANNEALING)
    include_dark_pools: bool = Field(default=False, description="Include dark pools in routing")
    max_venues: Optional[int] = Field(
        default=None, 
        ge=2, 
        le=15,
        description="Maximum number of venues to consider"
    )
    
    # New execution strategy options
    execution_strategy: ExecutionStrategyType = Field(
        default=ExecutionStrategyType.INSTANT,
        description="Execution strategy (VWAP, TWAP, IS, or instant)"
    )
    duration_minutes: Optional[int] = Field(
        default=120,
        ge=15,
        le=390,
        description="Execution duration in minutes (for VWAP/TWAP/IS)"
    )
    smart_allocation: bool = Field(
        default=True,
        description="Use smart allocation optimization (vs equal allocation)"
    )
    
    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        """Ensure symbol is uppercase."""
        return v.upper().strip()


class VenueAllocation(BaseModel):
    """Allocation to a specific venue."""
    
    venue_id: str
    venue_name: str
    venue_type: VenueType
    allocation: float = Field(ge=0, le=1, description="Fraction of order (0-1)")
    quantity: int = Field(ge=0, description="Number of shares")
    estimated_fee_usd: float
    estimated_spread_cost_usd: float
    estimated_impact_cost_usd: float
    estimated_total_cost_usd: float
    execution_sequence: int = Field(description="Order in execution sequence (1-indexed)")
    allocation_reasoning: Optional[str] = Field(default=None, description="Why this allocation was chosen")


class ExecutionSlice(BaseModel):
    """A single time slice in the execution schedule."""
    slice_id: int
    start_time: datetime
    end_time: datetime
    target_quantity: int
    target_percentage: float
    cumulative_percentage: float
    volume_participation: float
    urgency_factor: float


class ExecutionScheduleInfo(BaseModel):
    """Execution schedule information."""
    strategy: ExecutionStrategyType
    duration_minutes: int
    num_slices: int
    slices: list[ExecutionSlice]
    expected_participation_rate: float
    risk_score: float


class CostBreakdown(BaseModel):
    """Detailed cost breakdown for a routing decision."""
    
    total_bps: float = Field(description="Total cost in basis points")
    total_usd: float = Field(description="Total cost in USD")
    fees_usd: float = Field(description="Exchange/venue fees")
    fees_bps: float
    spread_cost_usd: float = Field(description="Bid-ask spread crossing cost")
    spread_cost_bps: float
    impact_cost_usd: float = Field(description="Estimated market impact")
    impact_cost_bps: float
    latency_cost_usd: float = Field(description="Cost from execution delay")
    latency_cost_bps: float


class AlgorithmMetrics(BaseModel):
    """Performance metrics for the routing algorithm."""
    
    algorithm: AlgorithmType
    execution_time_ms: float
    iterations: Optional[int] = None
    nodes_explored: Optional[int] = None
    final_temperature: Optional[float] = None  # For SA
    generations: Optional[int] = None  # For GA
    improvement_history: Optional[list[float]] = None


class RoutingResult(BaseModel):
    """Complete routing optimization result."""
    
    order_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Request details
    symbol: str
    side: OrderSide
    total_quantity: int
    urgency: OrderUrgency
    
    # Market context
    market_data: MarketData
    
    # Results
    routing: list[VenueAllocation]
    cost: CostBreakdown
    algorithm_metrics: AlgorithmMetrics
    
    # Execution schedule (for VWAP/TWAP/IS)
    execution_schedule: Optional[ExecutionScheduleInfo] = None
    
    # Allocation method used
    allocation_method: str = Field(default="equal", description="'smart' or 'equal'")
    
    # Comparison (optional)
    baseline_cost: Optional[CostBreakdown] = Field(
        default=None,
        description="Cost if routed to single best venue (for comparison)"
    )
    savings_vs_baseline_bps: Optional[float] = None


class BenchmarkRequest(BaseModel):
    """Request to benchmark multiple algorithms."""
    
    symbol: str
    quantity: int = Field(gt=0)
    side: OrderSide = Field(default=OrderSide.BUY)
    urgency: OrderUrgency = Field(default=OrderUrgency.MEDIUM)
    algorithms: list[AlgorithmType] = Field(
        default=[
            AlgorithmType.NEAREST_NEIGHBOR,
            AlgorithmType.TWO_OPT,
            AlgorithmType.SIMULATED_ANNEALING,
            AlgorithmType.GENETIC,
        ]
    )
    include_exact: bool = Field(
        default=False,
        description="Include exact algorithms (slow for many venues)"
    )
    
    @field_validator("symbol")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        return v.upper().strip()


class BenchmarkResult(BaseModel):
    """Result of algorithm benchmarking."""
    
    symbol: str
    quantity: int
    num_venues: int
    results: list[RoutingResult]
    best_algorithm: AlgorithmType
    worst_algorithm: AlgorithmType
    cost_range_bps: float = Field(description="Difference between best and worst in bps")
