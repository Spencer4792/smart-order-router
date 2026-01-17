# Smart Order Router (SOR) Optimizer

A quantitative finance application that applies Traveling Salesman Problem (TSP) optimization algorithms to smart order routing, minimizing execution costs when routing large trades across multiple trading venues.

## Overview

When executing large orders, institutional traders face the challenge of splitting orders across multiple exchanges and dark pools to minimize total execution cost. This cost includes:

- **Explicit costs**: Exchange fees, bid-ask spread
- **Implicit costs**: Market impact, latency slippage, information leakage

This project models the order routing problem as a variant of the Traveling Salesman Problem, where:
- **Nodes** represent trading venues (NYSE, NASDAQ, IEX, CBOE, dark pools)
- **Edge weights** represent composite execution costs between venue routing decisions
- **Objective** is to find the optimal routing sequence and allocation that minimizes total execution cost

## Algorithms Implemented

### Exact Algorithms

| Algorithm | Time Complexity | Space Complexity | Description |
|-----------|----------------|------------------|-------------|
| Brute Force | O(n!) | O(n) | Exhaustive search, optimal for n ≤ 10 venues |
| Held-Karp (DP + Bitmask) | O(n²·2ⁿ) | O(n·2ⁿ) | Dynamic programming approach, optimal for n ≤ 20 |

### Heuristic Algorithms

| Algorithm | Time Complexity | Space Complexity | Description |
|-----------|----------------|------------------|-------------|
| Nearest Neighbor | O(n²) | O(n) | Greedy approach, fast baseline |
| 2-Opt Local Search | O(n²) per iteration | O(n) | Iterative improvement |
| Simulated Annealing | O(iterations · n) | O(n) | Metaheuristic, escapes local optima |
| Genetic Algorithm | O(generations · pop · n) | O(pop · n) | Evolutionary optimization |

## Cost Model

The execution cost model incorporates:

```
Total Cost = Σ (Fee_i + Spread_i + Impact_i + Latency_i) × Allocation_i
```

Where:
- **Fee_i**: Maker/taker fees at venue i
- **Spread_i**: Half the bid-ask spread (crossing cost)
- **Impact_i**: Market impact estimated via square-root model: σ × √(V/ADV)
- **Latency_i**: Price drift during execution based on venue latency

## Project Structure

```
smart-order-router/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI application
│   │   ├── config.py               # Configuration management
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── order.py            # Order and venue models
│   │   │   └── cost.py             # Cost model definitions
│   │   ├── algorithms/
│   │   │   ├── __init__.py
│   │   │   ├── base.py             # Abstract base class
│   │   │   ├── brute_force.py      # Exact: O(n!)
│   │   │   ├── held_karp.py        # Exact: O(n²·2ⁿ)
│   │   │   ├── nearest_neighbor.py # Heuristic: O(n²)
│   │   │   ├── two_opt.py          # Local search: O(n²)
│   │   │   ├── simulated_annealing.py
│   │   │   └── genetic.py          # Evolutionary
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── market_data.py      # Alpaca API integration
│   │   │   ├── cost_calculator.py  # Execution cost engine
│   │   │   └── router.py           # Order routing logic
│   │   └── api/
│   │       ├── __init__.py
│   │       └── routes.py           # API endpoints
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── test_algorithms.py
│   │   └── test_cost_model.py
│   └── requirements.txt
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── services/
│   │   ├── App.jsx
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── .env.example
├── .gitignore
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Alpaca Markets API key (free at https://alpaca.markets)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Set environment variables
export ALPACA_API_KEY="your_api_key"
export ALPACA_SECRET_KEY="your_secret_key"

# Run the server
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Docker Deployment

```bash
# Copy and configure environment
cp .env.example .env
# Edit .env with your Alpaca credentials

# Build and run
docker-compose up --build
```

## API Documentation

Once running, access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/route` | Calculate optimal order routing |
| GET | `/api/v1/venues` | List available trading venues |
| GET | `/api/v1/quote/{symbol}` | Get real-time quote data |
| POST | `/api/v1/benchmark` | Run algorithm comparison |
| GET | `/api/v1/algorithms` | List available algorithms |

### Example Request

```bash
curl -X POST "http://localhost:8000/api/v1/route" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "quantity": 10000,
    "side": "buy",
    "urgency": "medium",
    "algorithm": "simulated_annealing"
  }'
```

### Example Response

```json
{
  "order_id": "uuid",
  "symbol": "AAPL",
  "total_quantity": 10000,
  "estimated_cost": {
    "total_bps": 4.2,
    "fees_usd": 12.50,
    "spread_cost_usd": 18.30,
    "impact_cost_usd": 45.20,
    "total_usd": 76.00
  },
  "routing": [
    {"venue": "IEX", "allocation": 0.35, "quantity": 3500},
    {"venue": "NASDAQ", "allocation": 0.40, "quantity": 4000},
    {"venue": "NYSE", "allocation": 0.25, "quantity": 2500}
  ],
  "algorithm_used": "simulated_annealing",
  "execution_time_ms": 45,
  "iterations": 1000
}
```

## Algorithm Deep Dive

### Why TSP for Order Routing?

The order routing problem shares key characteristics with TSP:
1. **Combinatorial**: With n venues, there are n! possible routing sequences
2. **NP-Hard**: No known polynomial-time exact algorithm
3. **Metric properties**: Triangle inequality often holds (routing through intermediate venue typically costs more)

### Cost Matrix Construction

For each pair of venues (i, j), we compute a transition cost that captures:
- Fee differential when moving allocation between venues
- Spread arbitrage opportunities
- Latency cost of sequential execution
- Information leakage risk

### Held-Karp Algorithm

The Held-Karp algorithm uses dynamic programming with bitmask state compression:

```
dp[S][i] = minimum cost to visit all venues in set S, ending at venue i
dp[S][i] = min(dp[S\{i}][j] + cost[j][i]) for all j in S\{i}
```

This reduces complexity from O(n!) to O(n²·2ⁿ), making it tractable for up to ~20 venues.

### Simulated Annealing for Large Scale

For larger venue sets, simulated annealing provides near-optimal solutions:
1. Start with initial routing (e.g., from nearest neighbor)
2. Randomly perturb the solution (swap, insert, reverse)
3. Accept worse solutions with probability exp(-ΔE/T)
4. Gradually decrease temperature T

## Performance Benchmarks

Tested on M1 MacBook Pro with 10 trading venues:

| Algorithm | Avg Time (ms) | Avg Cost (bps) | Optimality Gap |
|-----------|---------------|----------------|----------------|
| Brute Force | 2,450 | 3.8 | 0% (optimal) |
| Held-Karp | 12 | 3.8 | 0% (optimal) |
| Nearest Neighbor | 0.3 | 4.9 | +29% |
| 2-Opt | 2.1 | 4.1 | +8% |
| Simulated Annealing | 45 | 3.9 | +2.6% |
| Genetic Algorithm | 120 | 4.0 | +5.3% |

## Configuration

Key configuration options in `backend/app/config.py`:

```python
class Settings:
    # Alpaca API
    ALPACA_API_KEY: str
    ALPACA_SECRET_KEY: str
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"
    
    # Cost Model Parameters
    DEFAULT_VOLATILITY: float = 0.02  # Daily volatility assumption
    IMPACT_COEFFICIENT: float = 0.1   # Market impact scaling
    LATENCY_COST_BPS: float = 0.5     # Cost per ms of latency
    
    # Algorithm Defaults
    SA_INITIAL_TEMP: float = 1000.0
    SA_COOLING_RATE: float = 0.995
    SA_ITERATIONS: int = 10000
    GA_POPULATION_SIZE: int = 100
    GA_GENERATIONS: int = 500
```

## Venue Data

Default venue configuration includes:

| Venue | Type | Maker Fee (bps) | Taker Fee (bps) | Latency (ms) |
|-------|------|-----------------|-----------------|--------------|
| NYSE | Exchange | -0.1 | 0.3 | 0.5 |
| NASDAQ | Exchange | -0.2 | 0.3 | 0.3 |
| IEX | Exchange | 0.0 | 0.09 | 1.0 |
| CBOE | Exchange | -0.3 | 0.3 | 0.4 |
| MEMX | Exchange | -0.2 | 0.25 | 0.2 |

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details.

## References

- Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. Journal of Risk.
- Held, M., & Karp, R. M. (1962). A dynamic programming approach to sequencing problems. Journal of SIAM.
- Kissell, R. (2013). The Science of Algorithmic Trading and Portfolio Management. Academic Press.

## Author

Spencer - BYU Computer Science 2025

---

*This project is for educational and personal use. It is not financial advice and should not be used for actual trading without proper risk management and regulatory compliance.*
