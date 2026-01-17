"""
Allocation Optimizer for Smart Order Router.

This module determines optimal allocation percentages across venues,
moving beyond equal allocation to consider:
- Venue liquidity and fill rates
- Fee structures (maker/taker incentives)
- Market impact at each venue
- Order book depth when available
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from app.models.order import MarketData, OrderSide, OrderUrgency, Venue


@dataclass
class VenueLiquidity:
    """Liquidity estimate for a venue."""
    venue_id: str
    available_liquidity: int  # Estimated shares available
    liquidity_score: float  # 0-1 normalized score
    avg_daily_participation: float  # What % of ADV this venue typically sees


@dataclass 
class AllocationResult:
    """Result of allocation optimization."""
    venue_id: str
    allocation: float  # 0-1 fraction
    quantity: int
    reasoning: str


class AllocationOptimizer:
    """
    Optimizes order allocation across venues based on multiple factors.
    
    The optimizer balances:
    1. Fee minimization - prefer venues with lower fees
    2. Liquidity matching - allocate proportional to available liquidity
    3. Impact minimization - don't overwhelm small venues
    4. Fill probability - favor venues with higher fill rates
    
    Optimization approach:
    - Score each venue on multiple dimensions
    - Apply constraints (min/max allocation, minimum order sizes)
    - Use quadratic optimization to minimize total expected cost
    """
    
    def __init__(
        self,
        fee_weight: float = 0.3,
        liquidity_weight: float = 0.35,
        impact_weight: float = 0.25,
        fill_rate_weight: float = 0.1,
    ):
        """
        Initialize optimizer with factor weights.
        
        Args:
            fee_weight: Importance of fee minimization (0-1)
            liquidity_weight: Importance of liquidity matching (0-1)
            impact_weight: Importance of impact minimization (0-1)
            fill_rate_weight: Importance of fill probability (0-1)
        """
        # Normalize weights
        total = fee_weight + liquidity_weight + impact_weight + fill_rate_weight
        self.fee_weight = fee_weight / total
        self.liquidity_weight = liquidity_weight / total
        self.impact_weight = impact_weight / total
        self.fill_rate_weight = fill_rate_weight / total
    
    def estimate_venue_liquidity(
        self,
        venue: Venue,
        market_data: MarketData,
        adv: int,
    ) -> VenueLiquidity:
        """
        Estimate liquidity available at a venue.
        
        In practice, this would use real order book data.
        Here we estimate based on venue characteristics and market data.
        """
        # Estimate venue's share of market volume
        # These are approximate market share percentages
        venue_market_shares = {
            "NYSE": 0.22,
            "NASDAQ": 0.18,
            "ARCA": 0.08,
            "IEX": 0.03,
            "CBOE_BZX": 0.12,
            "CBOE_EDGX": 0.10,
            "MEMX": 0.05,
            "DARK_POOL_1": 0.08,
            "DARK_POOL_2": 0.06,
        }
        
        market_share = venue_market_shares.get(venue.id, 0.05)
        
        # Estimate available liquidity based on displayed size and market share
        if venue.type.value == "exchange":
            # Lit venues - estimate from bid/ask size and ADV
            displayed_liquidity = (market_data.bid_size + market_data.ask_size) / 2
            estimated_liquidity = int(displayed_liquidity * market_share * 10)  # Hidden liquidity multiplier
        else:
            # Dark pools - estimate from ADV and market share
            estimated_liquidity = int(adv * market_share * 0.1)  # Dark pools see ~10% of their share
        
        # Normalize to liquidity score
        liquidity_score = min(1.0, estimated_liquidity / (adv * 0.01))  # Score relative to 1% ADV
        
        return VenueLiquidity(
            venue_id=venue.id,
            available_liquidity=estimated_liquidity,
            liquidity_score=liquidity_score,
            avg_daily_participation=market_share,
        )
    
    def score_venue(
        self,
        venue: Venue,
        liquidity: VenueLiquidity,
        order_quantity: int,
        urgency: OrderUrgency,
    ) -> tuple[float, dict]:
        """
        Score a venue for allocation suitability.
        
        Returns:
            Tuple of (overall_score, component_scores)
            Higher score = better venue for allocation
        """
        # Fee score (lower fees = higher score)
        # Normalize fees to 0-1 range (assuming fees range from -0.3 to 0.5 bps)
        max_fee = 0.5
        min_fee = -0.3
        normalized_fee = (venue.taker_fee_bps - min_fee) / (max_fee - min_fee)
        fee_score = 1 - normalized_fee  # Invert so lower fee = higher score
        
        # Liquidity score (already 0-1)
        liquidity_score = liquidity.liquidity_score
        
        # Impact score (how much would our order impact this venue)
        # Lower participation rate = higher score
        if liquidity.available_liquidity > 0:
            participation_rate = order_quantity / liquidity.available_liquidity
            impact_score = max(0, 1 - participation_rate)  # 0 if we'd take all liquidity
        else:
            impact_score = 0
        
        # Fill rate score
        fill_score = venue.avg_fill_rate or 1.0
        
        # Urgency adjustment
        # High urgency: prioritize fill rate and liquidity
        # Low urgency: prioritize fees
        if urgency == OrderUrgency.HIGH:
            urgency_adjusted_weights = {
                'fee': self.fee_weight * 0.5,
                'liquidity': self.liquidity_weight * 1.3,
                'impact': self.impact_weight * 0.8,
                'fill': self.fill_rate_weight * 1.5,
            }
        elif urgency == OrderUrgency.LOW:
            urgency_adjusted_weights = {
                'fee': self.fee_weight * 1.5,
                'liquidity': self.liquidity_weight * 0.8,
                'impact': self.impact_weight * 1.2,
                'fill': self.fill_rate_weight * 0.7,
            }
        else:
            urgency_adjusted_weights = {
                'fee': self.fee_weight,
                'liquidity': self.liquidity_weight,
                'impact': self.impact_weight,
                'fill': self.fill_rate_weight,
            }
        
        # Normalize urgency weights
        total_weight = sum(urgency_adjusted_weights.values())
        for k in urgency_adjusted_weights:
            urgency_adjusted_weights[k] /= total_weight
        
        # Calculate weighted score
        overall_score = (
            urgency_adjusted_weights['fee'] * fee_score +
            urgency_adjusted_weights['liquidity'] * liquidity_score +
            urgency_adjusted_weights['impact'] * impact_score +
            urgency_adjusted_weights['fill'] * fill_score
        )
        
        component_scores = {
            'fee_score': fee_score,
            'liquidity_score': liquidity_score,
            'impact_score': impact_score,
            'fill_score': fill_score,
            'weights': urgency_adjusted_weights,
        }
        
        return overall_score, component_scores
    
    def optimize_allocation(
        self,
        venues: list[Venue],
        total_quantity: int,
        market_data: MarketData,
        adv: int,
        urgency: OrderUrgency,
        min_allocation: float = 0.05,  # Minimum 5% if included
        max_allocation: float = 0.40,  # Maximum 40% to any single venue
    ) -> list[AllocationResult]:
        """
        Determine optimal allocation across venues.
        
        Args:
            venues: Available trading venues
            total_quantity: Total order quantity
            market_data: Current market data
            adv: Average daily volume
            urgency: Order urgency level
            min_allocation: Minimum allocation if venue is used
            max_allocation: Maximum allocation to any single venue
            
        Returns:
            List of AllocationResult with optimized allocations
        """
        n = len(venues)
        
        if n == 0:
            return []
        
        if n == 1:
            return [AllocationResult(
                venue_id=venues[0].id,
                allocation=1.0,
                quantity=total_quantity,
                reasoning="Single venue available",
            )]
        
        # Score each venue
        venue_scores = []
        venue_liquidities = []
        
        for venue in venues:
            liquidity = self.estimate_venue_liquidity(venue, market_data, adv)
            score, components = self.score_venue(venue, liquidity, total_quantity // n, urgency)
            venue_scores.append((venue, score, components))
            venue_liquidities.append(liquidity)
        
        # Sort by score
        venue_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Allocate based on scores with constraints
        allocations = []
        remaining = 1.0
        
        # First pass: allocate proportional to scores
        total_score = sum(vs[1] for vs in venue_scores)
        
        if total_score == 0:
            # Equal allocation if all scores are 0
            raw_allocations = [1.0 / n] * n
        else:
            raw_allocations = [vs[1] / total_score for vs in venue_scores]
        
        # Apply constraints
        final_allocations = []
        for i, (venue, score, components) in enumerate(venue_scores):
            alloc = raw_allocations[i]
            
            # Apply min/max constraints
            if alloc > 0 and alloc < min_allocation:
                alloc = min_allocation
            if alloc > max_allocation:
                alloc = max_allocation
            
            # Check venue minimum order size
            if alloc * total_quantity < venue.min_order_size:
                alloc = 0  # Skip this venue
            
            final_allocations.append(alloc)
        
        # Normalize to sum to 1
        total_alloc = sum(final_allocations)
        if total_alloc > 0:
            final_allocations = [a / total_alloc for a in final_allocations]
        
        # Build results
        results = []
        for i, (venue, score, components) in enumerate(venue_scores):
            alloc = final_allocations[i]
            if alloc > 0:
                # Generate reasoning
                reasoning_parts = []
                if components['fee_score'] > 0.7:
                    reasoning_parts.append("low fees")
                if components['liquidity_score'] > 0.7:
                    reasoning_parts.append("high liquidity")
                if components['impact_score'] > 0.7:
                    reasoning_parts.append("low impact")
                if components['fill_score'] > 0.9:
                    reasoning_parts.append("high fill rate")
                
                reasoning = f"Score: {score:.2f}"
                if reasoning_parts:
                    reasoning += f" ({', '.join(reasoning_parts)})"
                
                results.append(AllocationResult(
                    venue_id=venue.id,
                    allocation=alloc,
                    quantity=int(alloc * total_quantity),
                    reasoning=reasoning,
                ))
        
        # Adjust quantities to sum to total (handle rounding)
        total_allocated = sum(r.quantity for r in results)
        if results and total_allocated != total_quantity:
            results[0].quantity += (total_quantity - total_allocated)
        
        return results


class LiquidityAwareOptimizer(AllocationOptimizer):
    """
    Extended optimizer that can incorporate real order book data.
    
    When order book data is available, this optimizer uses actual
    liquidity at each price level rather than estimates.
    """
    
    def optimize_with_order_book(
        self,
        venues: list[Venue],
        order_books: dict[str, dict],  # venue_id -> order book data
        total_quantity: int,
        side: OrderSide,
        urgency: OrderUrgency,
    ) -> list[AllocationResult]:
        """
        Optimize allocation using real order book data.
        
        Args:
            venues: Available venues
            order_books: Order book data keyed by venue ID
            total_quantity: Total order quantity
            side: Order side (buy/sell)
            urgency: Order urgency
            
        Returns:
            Optimized allocations based on real liquidity
        """
        # Calculate available liquidity at each venue
        venue_liquidity = {}
        
        for venue in venues:
            book = order_books.get(venue.id, {})
            
            if side == OrderSide.BUY:
                # Look at ask side for buys
                asks = book.get('asks', [])
                # Sum liquidity within 10 bps of best ask
                if asks:
                    best_ask = asks[0]['price']
                    threshold = best_ask * 1.001  # 10 bps
                    liquidity = sum(
                        level['size'] for level in asks
                        if level['price'] <= threshold
                    )
                else:
                    liquidity = 0
            else:
                # Look at bid side for sells
                bids = book.get('bids', [])
                if bids:
                    best_bid = bids[0]['price']
                    threshold = best_bid * 0.999
                    liquidity = sum(
                        level['size'] for level in bids
                        if level['price'] >= threshold
                    )
                else:
                    liquidity = 0
            
            venue_liquidity[venue.id] = liquidity
        
        # Allocate proportional to liquidity with fee adjustment
        total_liquidity = sum(venue_liquidity.values())
        
        if total_liquidity == 0:
            # Fall back to score-based allocation
            return super().optimize_allocation(
                venues=venues,
                total_quantity=total_quantity,
                market_data=MarketData(
                    symbol="UNKNOWN",
                    bid=100, ask=100.01,
                    bid_size=1000, ask_size=1000,
                    last_price=100, volume=1000000,
                    timestamp=None,
                ),
                adv=1000000,
                urgency=urgency,
            )
        
        results = []
        for venue in venues:
            liq = venue_liquidity[venue.id]
            
            # Base allocation on liquidity share
            base_alloc = liq / total_liquidity if total_liquidity > 0 else 0
            
            # Adjust for fees (bonus for low-fee venues)
            fee_adjustment = 1 - (venue.taker_fee_bps / 10)  # Scale factor
            adjusted_alloc = base_alloc * max(0.5, fee_adjustment)
            
            if adjusted_alloc > 0:
                results.append(AllocationResult(
                    venue_id=venue.id,
                    allocation=adjusted_alloc,
                    quantity=int(adjusted_alloc * total_quantity),
                    reasoning=f"Liquidity: {liq:,} shares available",
                ))
        
        # Normalize
        total_alloc = sum(r.allocation for r in results)
        if total_alloc > 0:
            for r in results:
                r.allocation /= total_alloc
                r.quantity = int(r.allocation * total_quantity)
        
        # Fix rounding
        total_qty = sum(r.quantity for r in results)
        if results and total_qty != total_quantity:
            results[0].quantity += (total_quantity - total_qty)
        
        return results
