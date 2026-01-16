"""
Smart Order Router - FastAPI Application

A quantitative finance application that applies TSP optimization
algorithms to smart order routing.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import get_settings
from app.services.market_data import get_market_data_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    settings = get_settings()
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    if settings.alpaca_configured:
        print("Alpaca API configured - using real market data")
    else:
        print("Alpaca API not configured - using simulated market data")
    
    yield
    
    # Shutdown
    market_data_service = get_market_data_service()
    await market_data_service.close()
    print("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="""
## Smart Order Router (SOR) Optimizer

A quantitative finance application that applies Traveling Salesman Problem (TSP) 
optimization algorithms to smart order routing, minimizing execution costs when 
routing large trades across multiple trading venues.

### Features

- **Multiple Algorithms**: Compare exact (Brute Force, Held-Karp) and heuristic 
  (Nearest Neighbor, 2-Opt, Simulated Annealing, Genetic) algorithms
- **Real Market Data**: Integration with Alpaca Markets API for live quotes
- **Comprehensive Cost Model**: Fees, spread, market impact, and latency costs
- **Algorithm Benchmarking**: Compare algorithm performance side-by-side

### Cost Model

Total Cost = Fees + Spread + Market Impact + Latency Cost

Where market impact is estimated using the square-root model:
```
Impact = σ × η × √(Q / ADV)
```
        """,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routes
    app.include_router(router)
    
    return app


# Application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
