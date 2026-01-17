"""
API routes for Smart Order Router.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.algorithms import get_algorithm_info
from app.config import VenueConfig
from app.models.order import (
    AlgorithmType,
    BenchmarkRequest,
    BenchmarkResult,
    CostAttributionResponse,
    ExecutionFillResponse,
    MarketData,
    OrderRequest,
    RoutingResult,
    TCARequest,
    TCAResponse,
    Venue,
    VenuePerformance,
    VenueType,
)
from app.services.market_data import get_market_data_service
from app.services.router import get_routing_service
from app.services.tca import TCAAnalyzer, ExecutionSimulator

router = APIRouter(prefix="/api/v1", tags=["routing"])


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "smart-order-router"}


@router.get("/algorithms", response_model=list[dict])
async def list_algorithms():
    """
    List all available routing optimization algorithms.
    
    Returns information about each algorithm including:
    - Name and type
    - Time and space complexity
    - Whether it guarantees optimal solution
    """
    return get_algorithm_info()


@router.get("/venues", response_model=list[Venue])
async def list_venues(
    include_dark_pools: bool = Query(
        default=False,
        description="Include dark pool venues"
    ),
):
    """
    List all available trading venues.
    
    Returns venue configuration including fees, latency, and capabilities.
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


@router.get("/quote/{symbol}", response_model=MarketData)
async def get_quote(symbol: str):
    """
    Get real-time quote data for a symbol.
    
    Returns current bid, ask, last price, and volume.
    Uses Alpaca API if configured, otherwise returns simulated data.
    """
    try:
        market_data_service = get_market_data_service()
        return await market_data_service.get_quote(symbol.upper())
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch quote: {str(e)}")


@router.post("/route", response_model=RoutingResult)
async def route_order(request: OrderRequest):
    """
    Calculate optimal order routing.
    
    Given an order (symbol, quantity, side, urgency), this endpoint:
    1. Fetches current market data
    2. Evaluates all available venues
    3. Runs the selected optimization algorithm
    4. Returns optimal routing with detailed cost breakdown
    
    The response includes:
    - Recommended allocation to each venue
    - Execution sequence
    - Cost breakdown (fees, spread, impact, latency)
    - Comparison to single-venue baseline
    - Algorithm performance metrics
    """
    try:
        routing_service = get_routing_service()
        return await routing_service.route_order(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Routing failed: {str(e)}")


@router.post("/benchmark", response_model=BenchmarkResult)
async def benchmark_algorithms(request: BenchmarkRequest):
    """
    Benchmark multiple routing algorithms.
    
    Runs the specified algorithms on the same order and compares:
    - Total execution cost
    - Algorithm execution time
    - Solution quality
    
    Useful for understanding algorithm trade-offs and selecting
    the best algorithm for your use case.
    """
    try:
        routing_service = get_routing_service()
        return await routing_service.benchmark_algorithms(request)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@router.get("/adv/{symbol}")
async def get_average_daily_volume(
    symbol: str,
    days: int = Query(default=20, ge=1, le=90),
):
    """
    Get average daily volume for a symbol.
    
    ADV is used in the market impact model to estimate
    how much the order will move the price.
    """
    try:
        market_data_service = get_market_data_service()
        adv = await market_data_service.get_adv(symbol.upper(), days)
        return {"symbol": symbol.upper(), "adv": adv, "days": days}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate ADV: {str(e)}")


@router.post("/tca", response_model=TCAResponse)
async def run_tca_analysis(request: TCARequest):
    """
    Run Transaction Cost Analysis on a simulated execution.
    
    This endpoint:
    1. Creates an optimal routing plan
    2. Simulates execution with realistic market dynamics
    3. Analyzes execution quality vs benchmarks
    4. Attributes costs to different factors
    
    Key metrics returned:
    - Implementation Shortfall: Total cost vs arrival price
    - VWAP Slippage: Performance vs volume-weighted average
    - Cost Attribution: Breakdown of spread, impact, timing, fees
    - Venue Performance: How each venue contributed
    
    Use this to evaluate different strategies and parameters.
    """
    try:
        # First, get the routing plan
        routing_service = get_routing_service()
        routing_request = OrderRequest(
            symbol=request.symbol,
            quantity=request.quantity,
            side=request.side,
            urgency=request.urgency,
            algorithm=request.algorithm,
            execution_strategy=request.execution_strategy,
            duration_minutes=request.duration_minutes,
            smart_allocation=request.smart_allocation,
        )
        routing_result = await routing_service.route_order(routing_request)
        
        # Run TCA analysis
        simulator = ExecutionSimulator(seed=request.seed)
        analyzer = TCAAnalyzer(simulator)
        tca_report = analyzer.analyze(routing_result)
        
        # Build response
        return TCAResponse(
            order_id=tca_report.order_id,
            symbol=tca_report.symbol,
            side=tca_report.side,
            total_quantity=tca_report.total_quantity,
            filled_quantity=tca_report.filled_quantity,
            arrival_price=tca_report.arrival_price,
            average_fill_price=tca_report.average_fill_price,
            vwap_benchmark=tca_report.vwap_benchmark,
            twap_benchmark=tca_report.twap_benchmark,
            final_price=tca_report.execution.final_price,
            arrival_slippage_bps=round(tca_report.arrival_slippage_bps, 2),
            vwap_slippage_bps=round(tca_report.vwap_slippage_bps, 2),
            twap_slippage_bps=round(tca_report.twap_slippage_bps, 2),
            implementation_shortfall_bps=round(tca_report.implementation_shortfall_bps, 2),
            cost_attribution=CostAttributionResponse(
                spread_cost_bps=round(tca_report.cost_attribution.spread_cost_bps, 2),
                impact_cost_bps=round(tca_report.cost_attribution.impact_cost_bps, 2),
                timing_cost_bps=round(tca_report.cost_attribution.timing_cost_bps, 2),
                fee_cost_bps=round(tca_report.cost_attribution.fee_cost_bps, 2),
                opportunity_cost_bps=round(tca_report.cost_attribution.opportunity_cost_bps, 2),
                total_bps=round(tca_report.cost_attribution.total_bps, 2),
            ),
            fill_rate=round(tca_report.fill_rate, 4),
            participation_rate=round(tca_report.participation_rate, 4),
            price_improvement_bps=round(tca_report.price_improvement_bps, 2),
            execution_risk_score=round(tca_report.execution_risk_score, 2),
            timing_risk_realized_bps=round(tca_report.timing_risk_realized, 2),
            num_fills=tca_report.execution.num_fills,
            duration_seconds=tca_report.execution.duration_seconds,
            fills=[
                ExecutionFillResponse(
                    fill_id=f.fill_id,
                    timestamp=f.timestamp,
                    venue_id=f.venue_id,
                    quantity=f.quantity,
                    price=f.price,
                    fee_usd=round(f.fee_usd, 4),
                )
                for f in tca_report.execution.fills
            ],
            venue_performance=[
                VenuePerformance(
                    venue_id=venue_id,
                    quantity=perf["quantity"],
                    num_fills=perf["num_fills"],
                    avg_price=perf["avg_price"],
                    slippage_bps=perf["slippage_bps"],
                    fees_usd=perf["fees_usd"],
                    notional=perf["notional"],
                )
                for venue_id, perf in tca_report.venue_performance.items()
            ],
            routing_cost_estimate_bps=round(routing_result.cost.total_bps, 2),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TCA analysis failed: {str(e)}")
