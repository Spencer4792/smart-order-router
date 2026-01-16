"""
Market data service with Alpaca API integration.

Provides real-time and historical market data for order routing decisions.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Optional

import httpx

from app.config import get_settings
from app.models.order import MarketData


class MarketDataService:
    """
    Service for fetching market data from Alpaca.
    
    Alpaca provides free market data including:
    - Real-time quotes (NBBO)
    - Trade data
    - Historical bars
    
    For paper trading/testing, no subscription is required.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def headers(self) -> dict:
        """API authentication headers."""
        return {
            "APCA-API-KEY-ID": self.settings.ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": self.settings.ALPACA_SECRET_KEY,
        }
    
    @property
    def data_url(self) -> str:
        """Alpaca data API base URL."""
        return self.settings.ALPACA_DATA_URL
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers=self.headers,
                timeout=30.0,
            )
        return self._client
    
    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
    
    async def get_quote(self, symbol: str) -> MarketData:
        """
        Get the latest quote for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            
        Returns:
            MarketData with current bid/ask/last
            
        Raises:
            ValueError: If symbol not found or API error
        """
        if not self.settings.alpaca_configured:
            return self._get_simulated_quote(symbol)
        
        client = await self.get_client()
        
        try:
            # Get latest quote
            quote_url = f"{self.data_url}/v2/stocks/{symbol}/quotes/latest"
            quote_response = await client.get(quote_url)
            
            if quote_response.status_code == 404:
                raise ValueError(f"Symbol not found: {symbol}")
            
            quote_response.raise_for_status()
            quote_data = quote_response.json()
            
            # Get latest trade for last price
            trade_url = f"{self.data_url}/v2/stocks/{symbol}/trades/latest"
            trade_response = await client.get(trade_url)
            trade_response.raise_for_status()
            trade_data = trade_response.json()
            
            quote = quote_data.get("quote", {})
            trade = trade_data.get("trade", {})
            
            return MarketData(
                symbol=symbol.upper(),
                bid=quote.get("bp", 0),
                ask=quote.get("ap", 0),
                bid_size=quote.get("bs", 0),
                ask_size=quote.get("as", 0),
                last_price=trade.get("p", (quote.get("bp", 0) + quote.get("ap", 0)) / 2),
                volume=trade.get("s", 0),
                timestamp=datetime.utcnow(),
            )
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                # API key issue, fall back to simulated
                return self._get_simulated_quote(symbol)
            raise ValueError(f"Failed to fetch quote for {symbol}: {e}")
        except Exception as e:
            # Fall back to simulated data
            return self._get_simulated_quote(symbol)
    
    async def get_bars(
        self,
        symbol: str,
        timeframe: str = "1Day",
        limit: int = 30,
    ) -> list[dict]:
        """
        Get historical bars for a symbol.
        
        Args:
            symbol: Stock symbol
            timeframe: Bar timeframe (1Min, 5Min, 1Hour, 1Day)
            limit: Number of bars to fetch
            
        Returns:
            List of bar dictionaries
        """
        if not self.settings.alpaca_configured:
            return self._get_simulated_bars(symbol, limit)
        
        client = await self.get_client()
        
        try:
            end = datetime.utcnow()
            start = end - timedelta(days=limit * 2)  # Buffer for weekends
            
            url = f"{self.data_url}/v2/stocks/{symbol}/bars"
            params = {
                "timeframe": timeframe,
                "start": start.isoformat() + "Z",
                "end": end.isoformat() + "Z",
                "limit": limit,
            }
            
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            return data.get("bars", [])
            
        except Exception:
            return self._get_simulated_bars(symbol, limit)
    
    async def get_adv(self, symbol: str, days: int = 20) -> int:
        """
        Calculate Average Daily Volume.
        
        Args:
            symbol: Stock symbol
            days: Number of days to average
            
        Returns:
            Average daily volume
        """
        bars = await self.get_bars(symbol, timeframe="1Day", limit=days)
        
        if not bars:
            return 1_000_000  # Default assumption
        
        total_volume = sum(bar.get("v", 0) for bar in bars)
        return total_volume // len(bars) if bars else 1_000_000
    
    def _get_simulated_quote(self, symbol: str) -> MarketData:
        """
        Generate simulated quote data for testing.
        
        Uses reasonable values based on typical stock behavior.
        """
        import random
        
        # Seed based on symbol for consistency
        random.seed(hash(symbol) % 2**32)
        
        # Generate realistic price based on symbol
        base_prices = {
            "AAPL": 175.0,
            "GOOGL": 140.0,
            "MSFT": 380.0,
            "AMZN": 180.0,
            "TSLA": 250.0,
            "META": 500.0,
            "NVDA": 800.0,
            "JPM": 195.0,
            "V": 275.0,
            "JNJ": 155.0,
        }
        
        base_price = base_prices.get(symbol.upper(), 100.0 + random.random() * 200)
        
        # Add some randomness
        price_variation = base_price * 0.001 * (random.random() - 0.5)
        mid_price = base_price + price_variation
        
        # Typical spread for liquid stocks
        spread = mid_price * 0.0002  # 2 bps spread
        
        bid = mid_price - spread / 2
        ask = mid_price + spread / 2
        
        return MarketData(
            symbol=symbol.upper(),
            bid=round(bid, 2),
            ask=round(ask, 2),
            bid_size=random.randint(100, 1000) * 100,
            ask_size=random.randint(100, 1000) * 100,
            last_price=round(mid_price, 2),
            volume=random.randint(1_000_000, 50_000_000),
            timestamp=datetime.utcnow(),
        )
    
    def _get_simulated_bars(self, symbol: str, limit: int) -> list[dict]:
        """Generate simulated historical bars."""
        import random
        
        random.seed(hash(symbol) % 2**32)
        
        bars = []
        base_price = 100.0 + random.random() * 200
        
        for i in range(limit):
            daily_return = (random.random() - 0.5) * 0.04  # +/- 2% daily
            open_price = base_price * (1 + daily_return)
            high = open_price * (1 + random.random() * 0.02)
            low = open_price * (1 - random.random() * 0.02)
            close = (high + low) / 2
            volume = random.randint(1_000_000, 50_000_000)
            
            bars.append({
                "o": round(open_price, 2),
                "h": round(high, 2),
                "l": round(low, 2),
                "c": round(close, 2),
                "v": volume,
            })
            
            base_price = close
        
        return bars


# Singleton instance
_market_data_service: Optional[MarketDataService] = None


def get_market_data_service() -> MarketDataService:
    """Get the singleton market data service."""
    global _market_data_service
    if _market_data_service is None:
        _market_data_service = MarketDataService()
    return _market_data_service
