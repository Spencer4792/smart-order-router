"""
Execution Strategies for Smart Order Router.

This module implements industry-standard execution algorithms:
- VWAP (Volume Weighted Average Price)
- TWAP (Time Weighted Average Price)
- Implementation Shortfall

These strategies determine HOW to execute over time, while the
routing optimizer determines WHERE to execute at each interval.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import math


class ExecutionStrategy(str, Enum):
    """Available execution strategies."""
    VWAP = "vwap"
    TWAP = "twap"
    IMPLEMENTATION_SHORTFALL = "is"
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"


@dataclass
class TimeSlice:
    """A single time slice in the execution schedule."""
    slice_id: int
    start_time: datetime
    end_time: datetime
    target_quantity: int
    target_percentage: float
    cumulative_percentage: float
    volume_participation: float  # Expected % of market volume
    urgency_factor: float  # 1.0 = normal, >1 = more aggressive


@dataclass
class ExecutionSchedule:
    """Complete execution schedule for an order."""
    strategy: ExecutionStrategy
    total_quantity: int
    start_time: datetime
    end_time: datetime
    duration_minutes: int
    slices: list[TimeSlice]
    expected_participation_rate: float
    risk_score: float  # 0-1, higher = more market impact risk
    
    def get_current_slice(self, current_time: datetime) -> Optional[TimeSlice]:
        """Get the active time slice for the current time."""
        for slice in self.slices:
            if slice.start_time <= current_time < slice.end_time:
                return slice
        return None
    
    def get_progress(self, current_time: datetime) -> float:
        """Get execution progress (0-1) at current time."""
        if current_time <= self.start_time:
            return 0.0
        if current_time >= self.end_time:
            return 1.0
        
        elapsed = (current_time - self.start_time).total_seconds()
        total = (self.end_time - self.start_time).total_seconds()
        return elapsed / total


@dataclass
class VolumeProfile:
    """Historical volume profile for VWAP calculation."""
    intervals: list[float]  # Volume percentage for each interval
    interval_minutes: int
    
    @classmethod
    def get_us_equity_profile(cls, interval_minutes: int = 30) -> 'VolumeProfile':
        """
        Get typical US equity intraday volume profile.
        
        US markets: 9:30 AM - 4:00 PM ET (6.5 hours = 390 minutes)
        Volume is typically U-shaped with high volume at open/close.
        """
        # 13 intervals of 30 minutes each
        # Percentages based on typical US equity volume patterns
        profile_30min = [
            0.12,  # 9:30-10:00 - High volume at open
            0.09,  # 10:00-10:30
            0.07,  # 10:30-11:00
            0.06,  # 11:00-11:30
            0.05,  # 11:30-12:00
            0.05,  # 12:00-12:30 - Lunch lull
            0.05,  # 12:30-1:00
            0.06,  # 1:00-1:30
            0.07,  # 1:30-2:00
            0.08,  # 2:00-2:30
            0.09,  # 2:30-3:00
            0.10,  # 3:00-3:30
            0.11,  # 3:30-4:00 - High volume at close
        ]
        
        if interval_minutes == 30:
            return cls(intervals=profile_30min, interval_minutes=30)
        elif interval_minutes == 15:
            # Interpolate to 15-minute intervals
            profile_15min = []
            for pct in profile_30min:
                profile_15min.extend([pct / 2, pct / 2])
            return cls(intervals=profile_15min, interval_minutes=15)
        elif interval_minutes == 60:
            # Aggregate to 60-minute intervals
            profile_60min = []
            for i in range(0, len(profile_30min) - 1, 2):
                profile_60min.append(profile_30min[i] + profile_30min[i + 1])
            if len(profile_30min) % 2 == 1:
                profile_60min.append(profile_30min[-1])
            return cls(intervals=profile_60min, interval_minutes=60)
        else:
            # Default to 30-min and let caller handle
            return cls(intervals=profile_30min, interval_minutes=30)


class ExecutionStrategyEngine:
    """
    Engine for generating execution schedules.
    
    Given an order and strategy, generates a time-sliced execution
    plan that can be fed to the routing optimizer.
    """
    
    def __init__(
        self,
        volume_profile: Optional[VolumeProfile] = None,
        max_participation_rate: float = 0.10,  # Don't exceed 10% of volume
    ):
        self.volume_profile = volume_profile or VolumeProfile.get_us_equity_profile()
        self.max_participation_rate = max_participation_rate
    
    def generate_vwap_schedule(
        self,
        total_quantity: int,
        adv: int,
        start_time: datetime,
        duration_minutes: int = 120,  # Default 2 hours
        interval_minutes: int = 15,
    ) -> ExecutionSchedule:
        """
        Generate a VWAP execution schedule.
        
        VWAP (Volume Weighted Average Price) execution aims to match
        the market's volume profile, executing more when volume is high
        and less when volume is low.
        
        Goal: Achieve execution price close to the day's VWAP
        
        Args:
            total_quantity: Total shares to execute
            adv: Average daily volume
            start_time: Execution start time
            duration_minutes: Total execution duration
            interval_minutes: Length of each time slice
            
        Returns:
            ExecutionSchedule with VWAP-weighted time slices
        """
        num_intervals = duration_minutes // interval_minutes
        
        # Get volume profile for these intervals
        profile = VolumeProfile.get_us_equity_profile(interval_minutes)
        
        # Map our execution window to market hours
        # Simplified: assume we start at market open
        relevant_profile = profile.intervals[:num_intervals]
        
        # Normalize to sum to 1
        total_profile = sum(relevant_profile)
        if total_profile > 0:
            normalized_profile = [p / total_profile for p in relevant_profile]
        else:
            normalized_profile = [1.0 / num_intervals] * num_intervals
        
        # Generate slices
        slices = []
        cumulative = 0.0
        current_time = start_time
        
        for i, pct in enumerate(normalized_profile):
            target_qty = int(total_quantity * pct)
            cumulative += pct
            
            # Calculate expected participation rate
            interval_volume = adv * profile.intervals[i] if i < len(profile.intervals) else adv / 13
            participation = target_qty / interval_volume if interval_volume > 0 else 0
            
            # Adjust if exceeding max participation
            urgency = 1.0
            if participation > self.max_participation_rate:
                urgency = self.max_participation_rate / participation
            
            slices.append(TimeSlice(
                slice_id=i + 1,
                start_time=current_time,
                end_time=current_time + timedelta(minutes=interval_minutes),
                target_quantity=target_qty,
                target_percentage=pct,
                cumulative_percentage=cumulative,
                volume_participation=min(participation, self.max_participation_rate),
                urgency_factor=urgency,
            ))
            
            current_time += timedelta(minutes=interval_minutes)
        
        # Adjust last slice to capture rounding
        total_scheduled = sum(s.target_quantity for s in slices)
        if slices and total_scheduled != total_quantity:
            slices[-1].target_quantity += (total_quantity - total_scheduled)
        
        # Calculate overall participation and risk
        avg_participation = total_quantity / (adv * (duration_minutes / 390))
        risk_score = min(1.0, avg_participation / 0.05)  # Risk increases above 5% participation
        
        return ExecutionSchedule(
            strategy=ExecutionStrategy.VWAP,
            total_quantity=total_quantity,
            start_time=start_time,
            end_time=slices[-1].end_time if slices else start_time,
            duration_minutes=duration_minutes,
            slices=slices,
            expected_participation_rate=avg_participation,
            risk_score=risk_score,
        )
    
    def generate_twap_schedule(
        self,
        total_quantity: int,
        adv: int,
        start_time: datetime,
        duration_minutes: int = 120,
        interval_minutes: int = 15,
    ) -> ExecutionSchedule:
        """
        Generate a TWAP execution schedule.
        
        TWAP (Time Weighted Average Price) execution spreads the order
        evenly across time, regardless of volume patterns.
        
        Simpler than VWAP but may result in higher impact during
        low-volume periods.
        
        Args:
            total_quantity: Total shares to execute
            adv: Average daily volume
            start_time: Execution start time
            duration_minutes: Total execution duration
            interval_minutes: Length of each time slice
            
        Returns:
            ExecutionSchedule with equal-weighted time slices
        """
        num_intervals = duration_minutes // interval_minutes
        quantity_per_interval = total_quantity // num_intervals
        
        slices = []
        current_time = start_time
        
        for i in range(num_intervals):
            # TWAP: equal allocation each interval
            pct = 1.0 / num_intervals
            target_qty = quantity_per_interval
            
            # Last slice gets remainder
            if i == num_intervals - 1:
                target_qty = total_quantity - (quantity_per_interval * (num_intervals - 1))
            
            # Estimate participation based on volume profile
            profile = VolumeProfile.get_us_equity_profile(interval_minutes)
            interval_volume = adv * profile.intervals[i] if i < len(profile.intervals) else adv / 13
            participation = target_qty / interval_volume if interval_volume > 0 else 0
            
            slices.append(TimeSlice(
                slice_id=i + 1,
                start_time=current_time,
                end_time=current_time + timedelta(minutes=interval_minutes),
                target_quantity=target_qty,
                target_percentage=pct,
                cumulative_percentage=(i + 1) / num_intervals,
                volume_participation=participation,
                urgency_factor=1.0,  # TWAP is steady
            ))
            
            current_time += timedelta(minutes=interval_minutes)
        
        avg_participation = total_quantity / (adv * (duration_minutes / 390))
        risk_score = min(1.0, avg_participation / 0.05)
        
        return ExecutionSchedule(
            strategy=ExecutionStrategy.TWAP,
            total_quantity=total_quantity,
            start_time=start_time,
            end_time=slices[-1].end_time if slices else start_time,
            duration_minutes=duration_minutes,
            slices=slices,
            expected_participation_rate=avg_participation,
            risk_score=risk_score,
        )
    
    def generate_is_schedule(
        self,
        total_quantity: int,
        adv: int,
        start_time: datetime,
        volatility: float = 0.02,
        urgency_factor: float = 1.0,
        duration_minutes: int = 120,
        interval_minutes: int = 15,
    ) -> ExecutionSchedule:
        """
        Generate an Implementation Shortfall minimization schedule.
        
        Implementation Shortfall (IS) strategy balances:
        - Market impact cost (favors slower execution)
        - Timing risk / volatility cost (favors faster execution)
        
        The optimal trade-off depends on volatility and urgency.
        
        Args:
            total_quantity: Total shares to execute
            adv: Average daily volume
            start_time: Execution start time
            volatility: Daily volatility (e.g., 0.02 = 2%)
            urgency_factor: Higher = more front-loaded (1.0 = balanced)
            duration_minutes: Total execution duration
            interval_minutes: Length of each time slice
            
        Returns:
            ExecutionSchedule with IS-optimized time slices
        """
        num_intervals = duration_minutes // interval_minutes
        
        # IS optimization: front-load based on urgency and volatility
        # Higher volatility = more front-loading to reduce timing risk
        # Higher urgency = more front-loading
        
        decay_rate = 0.1 * urgency_factor * (1 + volatility * 10)
        
        # Generate exponentially decaying weights
        weights = []
        for i in range(num_intervals):
            weight = math.exp(-decay_rate * i)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        slices = []
        current_time = start_time
        cumulative = 0.0
        
        for i, pct in enumerate(normalized_weights):
            target_qty = int(total_quantity * pct)
            cumulative += pct
            
            profile = VolumeProfile.get_us_equity_profile(interval_minutes)
            interval_volume = adv * profile.intervals[i] if i < len(profile.intervals) else adv / 13
            participation = target_qty / interval_volume if interval_volume > 0 else 0
            
            slices.append(TimeSlice(
                slice_id=i + 1,
                start_time=current_time,
                end_time=current_time + timedelta(minutes=interval_minutes),
                target_quantity=target_qty,
                target_percentage=pct,
                cumulative_percentage=cumulative,
                volume_participation=participation,
                urgency_factor=urgency_factor * (1 + decay_rate * (num_intervals - i - 1) / num_intervals),
            ))
            
            current_time += timedelta(minutes=interval_minutes)
        
        # Adjust last slice
        total_scheduled = sum(s.target_quantity for s in slices)
        if slices and total_scheduled != total_quantity:
            slices[-1].target_quantity += (total_quantity - total_scheduled)
        
        avg_participation = total_quantity / (adv * (duration_minutes / 390))
        # IS typically has higher risk due to front-loading
        risk_score = min(1.0, avg_participation / 0.04 * urgency_factor)
        
        return ExecutionSchedule(
            strategy=ExecutionStrategy.IMPLEMENTATION_SHORTFALL,
            total_quantity=total_quantity,
            start_time=start_time,
            end_time=slices[-1].end_time if slices else start_time,
            duration_minutes=duration_minutes,
            slices=slices,
            expected_participation_rate=avg_participation,
            risk_score=risk_score,
        )
    
    def generate_schedule(
        self,
        strategy: ExecutionStrategy,
        total_quantity: int,
        adv: int,
        start_time: Optional[datetime] = None,
        duration_minutes: int = 120,
        interval_minutes: int = 15,
        **kwargs,
    ) -> ExecutionSchedule:
        """
        Generate execution schedule for any strategy.
        
        Args:
            strategy: Execution strategy to use
            total_quantity: Total shares to execute
            adv: Average daily volume
            start_time: Execution start (defaults to now)
            duration_minutes: Total execution duration
            interval_minutes: Time slice length
            **kwargs: Strategy-specific parameters
            
        Returns:
            ExecutionSchedule for the specified strategy
        """
        if start_time is None:
            start_time = datetime.utcnow()
        
        if strategy == ExecutionStrategy.VWAP:
            return self.generate_vwap_schedule(
                total_quantity, adv, start_time, duration_minutes, interval_minutes
            )
        elif strategy == ExecutionStrategy.TWAP:
            return self.generate_twap_schedule(
                total_quantity, adv, start_time, duration_minutes, interval_minutes
            )
        elif strategy == ExecutionStrategy.IMPLEMENTATION_SHORTFALL:
            return self.generate_is_schedule(
                total_quantity, adv, start_time,
                volatility=kwargs.get('volatility', 0.02),
                urgency_factor=kwargs.get('urgency_factor', 1.0),
                duration_minutes=duration_minutes,
                interval_minutes=interval_minutes,
            )
        elif strategy == ExecutionStrategy.AGGRESSIVE:
            # Aggressive: 80% in first third, 20% in rest
            return self.generate_is_schedule(
                total_quantity, adv, start_time,
                urgency_factor=2.5,
                duration_minutes=duration_minutes,
                interval_minutes=interval_minutes,
            )
        elif strategy == ExecutionStrategy.PASSIVE:
            # Passive: back-loaded execution
            return self.generate_is_schedule(
                total_quantity, adv, start_time,
                urgency_factor=0.3,
                duration_minutes=duration_minutes,
                interval_minutes=interval_minutes,
            )
        else:
            # Default to TWAP
            return self.generate_twap_schedule(
                total_quantity, adv, start_time, duration_minutes, interval_minutes
            )
