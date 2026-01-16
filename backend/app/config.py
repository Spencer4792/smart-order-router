"""
Configuration management for Smart Order Router.
Uses Pydantic Settings for environment variable parsing and validation.
"""

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    APP_NAME: str = "Smart Order Router"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Alpaca API Configuration
    ALPACA_API_KEY: str = ""
    ALPACA_SECRET_KEY: str = ""
    ALPACA_BASE_URL: str = "https://paper-api.alpaca.markets"
    ALPACA_DATA_URL: str = "https://data.alpaca.markets"

    # Cost Model Parameters
    DEFAULT_VOLATILITY: float = 0.02  # Daily volatility (2%)
    IMPACT_COEFFICIENT: float = 0.1  # Market impact scaling factor
    LATENCY_COST_BPS: float = 0.5  # Cost per millisecond of latency in bps
    SPREAD_MULTIPLIER: float = 0.5  # Fraction of spread as cost (crossing)

    # Simulated Annealing Parameters
    SA_INITIAL_TEMP: float = 1000.0
    SA_COOLING_RATE: float = 0.995
    SA_MIN_TEMP: float = 0.01
    SA_ITERATIONS: int = 10000

    # Genetic Algorithm Parameters
    GA_POPULATION_SIZE: int = 100
    GA_GENERATIONS: int = 500
    GA_MUTATION_RATE: float = 0.1
    GA_CROSSOVER_RATE: float = 0.8
    GA_ELITE_SIZE: int = 5

    # 2-Opt Parameters
    TWO_OPT_MAX_ITERATIONS: int = 1000
    TWO_OPT_NO_IMPROVE_LIMIT: int = 100

    # API Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_WINDOW: int = 60  # seconds

    # CORS
    CORS_ORIGINS: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    @property
    def alpaca_configured(self) -> bool:
        """Check if Alpaca API credentials are configured."""
        return bool(self.ALPACA_API_KEY and self.ALPACA_SECRET_KEY)


class VenueConfig:
    """
    Trading venue configuration.
    
    Fees are in basis points (bps), where 1 bps = 0.01% = 0.0001.
    Latency is in milliseconds.
    """

    VENUES = {
        "NYSE": {
            "name": "New York Stock Exchange",
            "type": "exchange",
            "maker_fee_bps": -0.1,  # Rebate
            "taker_fee_bps": 0.30,
            "latency_ms": 0.5,
            "min_order_size": 1,
            "supports_dark": False,
        },
        "NASDAQ": {
            "name": "NASDAQ",
            "type": "exchange",
            "maker_fee_bps": -0.2,  # Rebate
            "taker_fee_bps": 0.30,
            "latency_ms": 0.3,
            "min_order_size": 1,
            "supports_dark": False,
        },
        "IEX": {
            "name": "Investors Exchange",
            "type": "exchange",
            "maker_fee_bps": 0.0,
            "taker_fee_bps": 0.09,
            "latency_ms": 1.0,  # Speed bump
            "min_order_size": 1,
            "supports_dark": False,
        },
        "CBOE_BZX": {
            "name": "Cboe BZX Exchange",
            "type": "exchange",
            "maker_fee_bps": -0.30,  # Higher rebate
            "taker_fee_bps": 0.30,
            "latency_ms": 0.4,
            "min_order_size": 1,
            "supports_dark": False,
        },
        "CBOE_EDGX": {
            "name": "Cboe EDGX Exchange",
            "type": "exchange",
            "maker_fee_bps": -0.20,
            "taker_fee_bps": 0.29,
            "latency_ms": 0.4,
            "min_order_size": 1,
            "supports_dark": False,
        },
        "MEMX": {
            "name": "Members Exchange",
            "type": "exchange",
            "maker_fee_bps": -0.20,
            "taker_fee_bps": 0.25,
            "latency_ms": 0.2,
            "min_order_size": 1,
            "supports_dark": False,
        },
        "ARCA": {
            "name": "NYSE Arca",
            "type": "exchange",
            "maker_fee_bps": -0.15,
            "taker_fee_bps": 0.30,
            "latency_ms": 0.4,
            "min_order_size": 1,
            "supports_dark": False,
        },
        "DARK_POOL_1": {
            "name": "Simulated Dark Pool A",
            "type": "dark_pool",
            "maker_fee_bps": 0.05,
            "taker_fee_bps": 0.10,
            "latency_ms": 2.0,
            "min_order_size": 100,
            "supports_dark": True,
            "avg_fill_rate": 0.3,  # 30% fill probability
        },
        "DARK_POOL_2": {
            "name": "Simulated Dark Pool B",
            "type": "dark_pool",
            "maker_fee_bps": 0.08,
            "taker_fee_bps": 0.12,
            "latency_ms": 1.5,
            "min_order_size": 50,
            "supports_dark": True,
            "avg_fill_rate": 0.4,  # 40% fill probability
        },
    }

    @classmethod
    def get_venue(cls, venue_id: str) -> Optional[dict]:
        """Get venue configuration by ID."""
        return cls.VENUES.get(venue_id)

    @classmethod
    def get_all_venues(cls) -> dict:
        """Get all venue configurations."""
        return cls.VENUES.copy()

    @classmethod
    def get_lit_venues(cls) -> dict:
        """Get only lit exchange venues (not dark pools)."""
        return {k: v for k, v in cls.VENUES.items() if v["type"] == "exchange"}

    @classmethod
    def get_dark_venues(cls) -> dict:
        """Get only dark pool venues."""
        return {k: v for k, v in cls.VENUES.items() if v["type"] == "dark_pool"}


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
