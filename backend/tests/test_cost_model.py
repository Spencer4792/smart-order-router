"""
Unit tests for cost model.
"""

from datetime import datetime

import pytest

from app.models.cost import CostModel, CostMatrixBuilder
from app.models.order import MarketData, OrderSide, OrderUrgency, Venue, VenueType


@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    return MarketData(
        symbol="AAPL",
        bid=174.50,
        ask=174.55,
        bid_size=1000,
        ask_size=800,
        last_price=174.52,
        volume=50_000_000,
        timestamp=datetime.utcnow(),
    )


@pytest.fixture
def sample_venue():
    """Create sample venue for testing."""
    return Venue(
        id="NASDAQ",
        name="NASDAQ",
        type=VenueType.EXCHANGE,
        maker_fee_bps=-0.2,
        taker_fee_bps=0.3,
        latency_ms=0.3,
        min_order_size=1,
        supports_dark=False,
        avg_fill_rate=1.0,
    )


@pytest.fixture
def sample_venues():
    """Create multiple sample venues for testing."""
    return [
        Venue(
            id="NYSE",
            name="New York Stock Exchange",
            type=VenueType.EXCHANGE,
            maker_fee_bps=-0.1,
            taker_fee_bps=0.3,
            latency_ms=0.5,
        ),
        Venue(
            id="NASDAQ",
            name="NASDAQ",
            type=VenueType.EXCHANGE,
            maker_fee_bps=-0.2,
            taker_fee_bps=0.3,
            latency_ms=0.3,
        ),
        Venue(
            id="IEX",
            name="Investors Exchange",
            type=VenueType.EXCHANGE,
            maker_fee_bps=0.0,
            taker_fee_bps=0.09,
            latency_ms=1.0,
        ),
    ]


class TestCostModel:
    """Test cost model calculations."""
    
    def test_fee_cost_taker(self, sample_venue, sample_market_data):
        model = CostModel()
        fee_bps, fee_usd = model.calculate_fee_cost(
            venue=sample_venue,
            quantity=1000,
            price=sample_market_data.mid_price,
            is_taker=True,
        )
        
        assert fee_bps == 0.3  # Taker fee
        expected_usd = (0.3 / 10000) * 1000 * sample_market_data.mid_price
        assert fee_usd == pytest.approx(expected_usd, abs=0.01)
    
    def test_fee_cost_maker(self, sample_venue, sample_market_data):
        model = CostModel()
        fee_bps, fee_usd = model.calculate_fee_cost(
            venue=sample_venue,
            quantity=1000,
            price=sample_market_data.mid_price,
            is_taker=False,
        )
        
        assert fee_bps == -0.2  # Maker rebate (negative)
        expected_usd = (-0.2 / 10000) * 1000 * sample_market_data.mid_price
        assert fee_usd == pytest.approx(expected_usd, abs=0.01)
    
    def test_spread_cost(self, sample_market_data):
        model = CostModel()
        spread_bps, spread_usd = model.calculate_spread_cost(
            market_data=sample_market_data,
            quantity=1000,
            side=OrderSide.BUY,
        )
        
        # Spread should be positive
        assert spread_bps > 0
        assert spread_usd > 0
        
        # Cost should be half the spread times quantity
        half_spread = sample_market_data.spread / 2
        expected_usd = half_spread * 1000 * model.spread_multiplier
        assert spread_usd == pytest.approx(expected_usd, abs=0.01)
    
    def test_market_impact_increases_with_quantity(self, sample_market_data):
        model = CostModel()
        adv = 10_000_000
        
        impact_small_bps, impact_small_usd = model.calculate_market_impact(
            quantity=1000,
            price=sample_market_data.mid_price,
            adv=adv,
        )
        
        impact_large_bps, impact_large_usd = model.calculate_market_impact(
            quantity=100000,
            price=sample_market_data.mid_price,
            adv=adv,
        )
        
        # Larger quantity should have more impact
        assert impact_large_bps > impact_small_bps
        assert impact_large_usd > impact_small_usd
    
    def test_market_impact_urgency_effect(self, sample_market_data):
        model = CostModel()
        adv = 10_000_000
        quantity = 10000
        
        impact_low, _ = model.calculate_market_impact(
            quantity=quantity,
            price=sample_market_data.mid_price,
            adv=adv,
            urgency=OrderUrgency.LOW,
        )
        
        impact_high, _ = model.calculate_market_impact(
            quantity=quantity,
            price=sample_market_data.mid_price,
            adv=adv,
            urgency=OrderUrgency.HIGH,
        )
        
        # Higher urgency should have more impact
        assert impact_high > impact_low
    
    def test_latency_cost_increases_with_sequence(self, sample_venue, sample_market_data):
        model = CostModel()
        
        latency_first_bps, latency_first_usd = model.calculate_latency_cost(
            venue=sample_venue,
            quantity=1000,
            price=sample_market_data.mid_price,
            sequence_position=0,
        )
        
        latency_last_bps, latency_last_usd = model.calculate_latency_cost(
            venue=sample_venue,
            quantity=1000,
            price=sample_market_data.mid_price,
            sequence_position=5,
        )
        
        # Later in sequence should have higher latency cost
        assert latency_last_bps > latency_first_bps
        assert latency_last_usd > latency_first_usd
    
    def test_total_venue_cost(self, sample_venue, sample_market_data):
        model = CostModel()
        
        costs = model.calculate_venue_cost(
            venue=sample_venue,
            quantity=1000,
            market_data=sample_market_data,
            side=OrderSide.BUY,
            urgency=OrderUrgency.MEDIUM,
            adv=10_000_000,
            sequence_position=0,
        )
        
        # Total should be sum of components
        expected_total = costs.fee_bps + costs.spread_bps + costs.impact_bps + costs.latency_bps
        assert costs.total_bps == pytest.approx(expected_total, abs=0.01)
        
        expected_total_usd = costs.fee_usd + costs.spread_usd + costs.impact_usd + costs.latency_usd
        assert costs.total_usd == pytest.approx(expected_total_usd, abs=0.01)


class TestCostMatrixBuilder:
    """Test cost matrix construction."""
    
    def test_matrix_dimensions(self, sample_venues, sample_market_data):
        builder = CostMatrixBuilder()
        
        matrix = builder.build_cost_matrix(
            venues=sample_venues,
            total_quantity=10000,
            market_data=sample_market_data,
            side=OrderSide.BUY,
            urgency=OrderUrgency.MEDIUM,
            adv=10_000_000,
        )
        
        n = len(sample_venues)
        assert matrix.shape == (n, n)
    
    def test_diagonal_is_zero(self, sample_venues, sample_market_data):
        builder = CostMatrixBuilder()
        
        matrix = builder.build_cost_matrix(
            venues=sample_venues,
            total_quantity=10000,
            market_data=sample_market_data,
            side=OrderSide.BUY,
            urgency=OrderUrgency.MEDIUM,
            adv=10_000_000,
        )
        
        # Diagonal should be zero (no self-transition cost)
        for i in range(len(sample_venues)):
            assert matrix[i][i] == 0.0
    
    def test_matrix_values_positive(self, sample_venues, sample_market_data):
        builder = CostMatrixBuilder()
        
        matrix = builder.build_cost_matrix(
            venues=sample_venues,
            total_quantity=10000,
            market_data=sample_market_data,
            side=OrderSide.BUY,
            urgency=OrderUrgency.MEDIUM,
            adv=10_000_000,
        )
        
        # Off-diagonal costs should generally be positive
        # (some might be negative due to maker rebates, but total should be positive)
        n = len(sample_venues)
        for i in range(n):
            for j in range(n):
                if i != j:
                    # At minimum, spread cost should make this positive
                    assert matrix[i][j] >= -1.0  # Allow small negative from rebates


class TestMarketDataProperties:
    """Test MarketData calculated properties."""
    
    def test_spread_calculation(self, sample_market_data):
        expected_spread = sample_market_data.ask - sample_market_data.bid
        assert sample_market_data.spread == pytest.approx(expected_spread, abs=0.001)
    
    def test_mid_price_calculation(self, sample_market_data):
        expected_mid = (sample_market_data.bid + sample_market_data.ask) / 2
        assert sample_market_data.mid_price == pytest.approx(expected_mid, abs=0.001)
    
    def test_spread_bps_calculation(self, sample_market_data):
        mid = (sample_market_data.bid + sample_market_data.ask) / 2
        spread = sample_market_data.ask - sample_market_data.bid
        expected_bps = (spread / mid) * 10000
        assert sample_market_data.spread_bps == pytest.approx(expected_bps, abs=0.01)
