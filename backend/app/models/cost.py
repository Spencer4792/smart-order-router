"""
Execution cost model for Smart Order Router.

This module implements a realistic cost model incorporating:
- Exchange fees (maker/taker)
- Bid-ask spread crossing costs
- Market impact (square-root model)
- Latency costs (price drift)
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.config import VenueConfig, get_settings
from app.models.order import MarketData, OrderSide, OrderUrgency, Venue


@dataclass
class CostComponents:
    """Individual cost components for a venue allocation."""
    
    fee_bps: float
    fee_usd: float
    spread_bps: float
    spread_usd: float
    impact_bps: float
    impact_usd: float
    latency_bps: float
    latency_usd: float
    
    @property
    def total_bps(self) -> float:
        return self.fee_bps + self.spread_bps + self.impact_bps + self.latency_bps
    
    @property
    def total_usd(self) -> float:
        return self.fee_usd + self.spread_usd + self.impact_usd + self.latency_usd


class CostModel:
    """
    Execution cost model for order routing optimization.
    
    The model calculates total execution cost as:
        Total Cost = Fees + Spread + Market Impact + Latency Cost
    
    Market Impact Model (Square-Root):
        Impact = σ * η * sqrt(Q / ADV)
        
        Where:
        - σ = daily volatility
        - η = impact coefficient (calibrated parameter)
        - Q = order quantity
        - ADV = average daily volume
    
    Latency Cost:
        Cost = latency_ms * price_drift_per_ms * quantity
    """
    
    def __init__(
        self,
        volatility: Optional[float] = None,
        impact_coefficient: Optional[float] = None,
        latency_cost_bps: Optional[float] = None,
    ):
        settings = get_settings()
        self.volatility = volatility or settings.DEFAULT_VOLATILITY
        self.impact_coefficient = impact_coefficient or settings.IMPACT_COEFFICIENT
        self.latency_cost_bps = latency_cost_bps or settings.LATENCY_COST_BPS
        self.spread_multiplier = settings.SPREAD_MULTIPLIER
    
    def calculate_fee_cost(
        self,
        venue: Venue,
        quantity: int,
        price: float,
        is_taker: bool = True,
    ) -> tuple[float, float]:
        """
        Calculate exchange fee cost.
        
        Args:
            venue: Trading venue
            quantity: Number of shares
            price: Execution price
            is_taker: Whether this is a taker order (True) or maker (False)
            
        Returns:
            Tuple of (cost_bps, cost_usd)
        """
        fee_bps = venue.taker_fee_bps if is_taker else venue.maker_fee_bps
        notional = quantity * price
        fee_usd = (fee_bps / 10000) * notional
        return fee_bps, fee_usd
    
    def calculate_spread_cost(
        self,
        market_data: MarketData,
        quantity: int,
        side: OrderSide,
    ) -> tuple[float, float]:
        """
        Calculate bid-ask spread crossing cost.
        
        For a buy order, we pay half the spread (from mid to ask).
        For a sell order, we pay half the spread (from mid to bid).
        
        Args:
            market_data: Current market data
            quantity: Number of shares
            side: Order side (buy/sell)
            
        Returns:
            Tuple of (cost_bps, cost_usd)
        """
        half_spread = market_data.spread / 2
        spread_bps = market_data.spread_bps * self.spread_multiplier
        spread_usd = half_spread * quantity * self.spread_multiplier
        return spread_bps, spread_usd
    
    def calculate_market_impact(
        self,
        quantity: int,
        price: float,
        adv: int,
        urgency: OrderUrgency = OrderUrgency.MEDIUM,
    ) -> tuple[float, float]:
        """
        Calculate market impact using the square-root model.
        
        Impact = σ * η * sqrt(Q / ADV) * urgency_multiplier
        
        Args:
            quantity: Number of shares in this child order
            price: Current price
            adv: Average daily volume
            urgency: Order urgency (affects impact scaling)
            
        Returns:
            Tuple of (impact_bps, impact_usd)
        """
        if adv <= 0:
            adv = 1000000  # Default assumption if ADV unknown
        
        # Urgency multiplier: higher urgency = faster execution = more impact
        urgency_multipliers = {
            OrderUrgency.LOW: 0.5,
            OrderUrgency.MEDIUM: 1.0,
            OrderUrgency.HIGH: 2.0,
        }
        urgency_mult = urgency_multipliers[urgency]
        
        # Square-root impact model
        participation_rate = quantity / adv
        impact_fraction = self.volatility * self.impact_coefficient * math.sqrt(participation_rate)
        impact_fraction *= urgency_mult
        
        impact_bps = impact_fraction * 10000
        impact_usd = impact_fraction * price * quantity
        
        return impact_bps, impact_usd
    
    def calculate_latency_cost(
        self,
        venue: Venue,
        quantity: int,
        price: float,
        sequence_position: int = 0,
    ) -> tuple[float, float]:
        """
        Calculate cost from execution latency (price drift).
        
        Later executions in a sequence face more price drift as
        information leaks into the market.
        
        Args:
            venue: Trading venue
            quantity: Number of shares
            price: Current price
            sequence_position: Position in execution sequence (0-indexed)
            
        Returns:
            Tuple of (cost_bps, cost_usd)
        """
        # Base latency cost from venue
        base_latency_ms = venue.latency_ms
        
        # Additional latency from sequence position (information leakage)
        # Each position adds ~0.5ms effective latency
        effective_latency_ms = base_latency_ms + (sequence_position * 0.5)
        
        latency_bps = effective_latency_ms * self.latency_cost_bps
        notional = quantity * price
        latency_usd = (latency_bps / 10000) * notional
        
        return latency_bps, latency_usd
    
    def calculate_venue_cost(
        self,
        venue: Venue,
        quantity: int,
        market_data: MarketData,
        side: OrderSide,
        urgency: OrderUrgency,
        adv: int,
        sequence_position: int = 0,
        is_taker: bool = True,
    ) -> CostComponents:
        """
        Calculate total cost for executing at a specific venue.
        
        Args:
            venue: Trading venue
            quantity: Number of shares to execute
            market_data: Current market data
            side: Order side
            urgency: Order urgency
            adv: Average daily volume
            sequence_position: Position in execution sequence
            is_taker: Whether this is a taker order
            
        Returns:
            CostComponents with all cost breakdowns
        """
        price = market_data.mid_price
        
        fee_bps, fee_usd = self.calculate_fee_cost(venue, quantity, price, is_taker)
        spread_bps, spread_usd = self.calculate_spread_cost(market_data, quantity, side)
        impact_bps, impact_usd = self.calculate_market_impact(quantity, price, adv, urgency)
        latency_bps, latency_usd = self.calculate_latency_cost(
            venue, quantity, price, sequence_position
        )
        
        # Adjust for dark pool fill probability
        if venue.avg_fill_rate and venue.avg_fill_rate < 1.0:
            # If fill is uncertain, we may need to route elsewhere
            # This increases effective cost
            fill_adjustment = 1.0 / venue.avg_fill_rate
            impact_bps *= fill_adjustment
            impact_usd *= fill_adjustment
        
        return CostComponents(
            fee_bps=fee_bps,
            fee_usd=fee_usd,
            spread_bps=spread_bps,
            spread_usd=spread_usd,
            impact_bps=impact_bps,
            impact_usd=impact_usd,
            latency_bps=latency_bps,
            latency_usd=latency_usd,
        )


class CostMatrixBuilder:
    """
    Builds cost matrices for TSP-style optimization.
    
    The cost matrix represents transition costs between venues,
    capturing the incremental cost of routing to venue j after venue i.
    """
    
    def __init__(self, cost_model: Optional[CostModel] = None):
        self.cost_model = cost_model or CostModel()
    
    def build_cost_matrix(
        self,
        venues: list[Venue],
        total_quantity: int,
        market_data: MarketData,
        side: OrderSide,
        urgency: OrderUrgency,
        adv: int,
    ) -> np.ndarray:
        """
        Build a cost matrix for routing optimization.
        
        The matrix entry [i][j] represents the cost of routing to venue j
        when venue i was the previous venue in the sequence.
        
        For TSP formulation:
        - Diagonal entries are 0 (no self-loops)
        - Entry [i][j] captures venue j's base cost plus transition effects
        
        Args:
            venues: List of available venues
            total_quantity: Total order quantity
            market_data: Current market data
            side: Order side
            urgency: Order urgency
            adv: Average daily volume
            
        Returns:
            NxN cost matrix where N = len(venues)
        """
        n = len(venues)
        
        # For simplicity, assume equal allocation initially
        # The actual allocation will be optimized separately
        quantity_per_venue = total_quantity // n
        
        cost_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    cost_matrix[i][j] = 0
                    continue
                
                # Cost of executing at venue j, given we previously executed at i
                # Sequence position is approximated by j (later in sequence = higher position)
                costs = self.cost_model.calculate_venue_cost(
                    venue=venues[j],
                    quantity=quantity_per_venue,
                    market_data=market_data,
                    side=side,
                    urgency=urgency,
                    adv=adv,
                    sequence_position=j,  # Approximate
                )
                
                cost_matrix[i][j] = costs.total_bps
        
        return cost_matrix
    
    def build_allocation_cost_function(
        self,
        venues: list[Venue],
        market_data: MarketData,
        side: OrderSide,
        urgency: OrderUrgency,
        adv: int,
    ):
        """
        Build a cost function for allocation optimization.
        
        Returns a function that takes (venue_index, quantity, sequence_position)
        and returns the cost in basis points.
        """
        def cost_function(
            venue_idx: int,
            quantity: int,
            sequence_position: int,
        ) -> float:
            if quantity <= 0:
                return 0.0
            
            costs = self.cost_model.calculate_venue_cost(
                venue=venues[venue_idx],
                quantity=quantity,
                market_data=market_data,
                side=side,
                urgency=urgency,
                adv=adv,
                sequence_position=sequence_position,
            )
            return costs.total_bps
        
        return cost_function
