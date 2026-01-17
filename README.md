# Smart Order Router

A quantitative finance application that applies **Traveling Salesman Problem (TSP)** optimization algorithms to smart order routing, minimizing execution costs when routing large trades across multiple stock exchanges.

**[Live Demo](https://hales-sor.vercel.app)** | **[API Documentation](https://web-production-2a270.up.railway.app/docs)**

---

## The Problem

When institutional traders need to execute large orders (e.g., 10,000 shares of AAPL), sending the entire order to a single exchange is expensive due to:

- **Market Impact**: Large orders move prices against you
- **Fee Structures**: Different exchanges have different maker/taker fees
- **Information Leakage**: Early executions reveal your intent to the market
- **Latency**: Execution timing affects final cost

The solution is to split the order across multiple venues in an optimal sequence—but finding that optimal sequence is computationally hard.

## The TSP Connection

This problem maps elegantly to the **Traveling Salesman Problem**:

| TSP (Classic) | Order Routing |
|---------------|---------------|
| Cities to visit | Trading venues (NYSE, NASDAQ, IEX, etc.) |
| Distance between cities | Execution cost between venues |
| Find shortest route | Find lowest-cost execution path |
| Visit each city once | Allocate order across venues |

The "cost" between venues isn't physical distance—it's a composite of fees, spread, market impact, and latency that changes based on execution sequence.

## Features

### Routing Optimization
- **6 TSP Algorithms**: Compare exact solutions (Brute Force, Held-Karp) with heuristics (Nearest Neighbor, 2-Opt, Simulated Annealing, Genetic Algorithm)
- **Smart Allocation**: Liquidity-aware allocation that considers venue characteristics, not just equal splits
- **Real-time Market Data**: Integration with Alpaca Markets API for live quotes

### Execution Strategies
- **VWAP**: Volume Weighted Average Price - execute proportional to market volume
- **TWAP**: Time Weighted Average Price - spread evenly across time
- **Implementation Shortfall**: Front-loaded execution to minimize price drift risk
- **Aggressive/Passive**: Configurable execution urgency

### Transaction Cost Analysis (TCA)
- **Implementation Shortfall**: Total cost vs. decision price
- **Benchmark Comparison**: Performance vs. VWAP, TWAP, arrival price
- **Cost Attribution**: Break down costs into spread, impact, timing, and fees
- **Venue Performance**: Per-venue execution analysis

## Algorithms

| Algorithm | Time Complexity | Space Complexity | Type | Best For |
|-----------|----------------|------------------|------|----------|
| Brute Force | O(n!) | O(n) | Exact | n <= 10 venues |
| Held-Karp | O(n^2 * 2^n) | O(n * 2^n) | Exact | n <= 20 venues |
| Nearest Neighbor | O(n^2) | O(n) | Greedy | Quick baseline |
| 2-Opt | O(n^2 * k) | O(n) | Local Search | Improving solutions |
| Simulated Annealing | O(iterations * n) | O(n) | Metaheuristic | Balanced quality/speed |
| Genetic Algorithm | O(gen * pop * n) | O(pop * n) | Evolutionary | Large search spaces |

## Cost Model

Total execution cost is calculated as:

```
Total Cost = Fees + Spread + Market Impact + Latency Cost
```

Where market impact uses the square-root model:

```
Impact = volatility * eta * sqrt(quantity / ADV) * urgency_multiplier
```

## Tech Stack

**Backend**: Python 3.11, FastAPI, Pydantic, NumPy, Alpaca Markets API

**Frontend**: React 18, Vite, Tailwind CSS, Recharts

**Deployment**: Railway (Backend), Vercel (Frontend)

## Trading Venues

The router optimizes across these simulated venues:

| Venue | Type | Maker Fee | Taker Fee | Latency |
|-------|------|-----------|-----------|---------|
| NYSE | Exchange | -0.10 bps | 0.30 bps | 0.5 ms |
| NASDAQ | Exchange | -0.20 bps | 0.30 bps | 0.3 ms |
| IEX | Exchange | 0.00 bps | 0.09 bps | 1.0 ms |
| CBOE BZX | Exchange | -0.30 bps | 0.30 bps | 0.4 ms |
| CBOE EDGX | Exchange | -0.20 bps | 0.29 bps | 0.4 ms |
| MEMX | Exchange | -0.20 bps | 0.25 bps | 0.2 ms |
| NYSE Arca | Exchange | -0.15 bps | 0.30 bps | 0.4 ms |

Dark pools available with configurable fill rates.

## Local Development

### Prerequisites
- Python 3.11+
- Node.js 18+
- Alpaca API keys (optional - falls back to simulated data)

### Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

### Environment Variables

Create `backend/.env`:

```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/quote/{symbol}` | Get real-time quote |
| GET | `/api/v1/venues` | List trading venues |
| GET | `/api/v1/algorithms` | List available algorithms |
| POST | `/api/v1/route` | Calculate optimal routing |
| POST | `/api/v1/tca` | Run execution simulation with TCA |
| POST | `/api/v1/benchmark` | Compare algorithm performance |

## Example Request

```bash
curl -X POST https://web-production-2a270.up.railway.app/api/v1/route \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "quantity": 10000,
    "side": "buy",
    "urgency": "medium",
    "algorithm": "simulated_annealing",
    "smart_allocation": true
  }'
```

## Project Structure

```
smart-order-router/
├── backend/
│   ├── app/
│   │   ├── algorithms/      # TSP algorithm implementations
│   │   ├── api/             # FastAPI routes
│   │   ├── models/          # Pydantic models
│   │   ├── services/        # Business logic
│   │   ├── config.py        # Configuration
│   │   └── main.py          # Application entry
│   ├── tests/               # Unit tests
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main application
│   │   └── services/        # API client
│   └── package.json
└── README.md
```

## References

- Held, M., & Karp, R. M. (1962). A Dynamic Programming Approach to Sequencing Problems
- Almgren, R., & Chriss, N. (2001). Optimal Execution of Portfolio Transactions
- Kissell, R. (2013). The Science of Algorithmic Trading and Portfolio Management

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

Spencer Hales

---

Built as a demonstration of applying classical optimization algorithms to real-world quantitative finance problems.
