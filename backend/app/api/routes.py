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
    MarketData,
    OrderRequest,
    RoutingResult,
    Venue,
    VenueType,
)
from app.services.market_data import get_market_data_service
from app.services.router import get_routing_service

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
