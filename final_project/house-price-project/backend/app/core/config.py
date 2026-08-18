"""Application configuration using pydantic-settings."""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    # CORS
    frontend_url: str = "http://localhost:5173"

    # Model paths (relative to backend/ directory)
    model_path: str = "models/house_price.pkl"
    locations_path: str = "locations.json"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        protected_namespaces=("settings_",),
    )


# Singleton instance
settings = Settings()
