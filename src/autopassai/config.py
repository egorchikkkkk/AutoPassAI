"""Application configuration and environment settings."""

from importlib.metadata import version

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    app_name: str = "AutoPassAI"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/autopassai"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def app_version(self) -> str:
        """Return the installed application version."""
        return version("autopassai")


settings = Settings()
