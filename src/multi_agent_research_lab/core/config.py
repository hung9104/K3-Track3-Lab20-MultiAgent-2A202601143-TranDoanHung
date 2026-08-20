"""Application configuration.

Keep config small and explicit. Do not read environment variables directly in agents.
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    app_env: str = Field(default="local", validation_alias="APP_ENV")
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o-mini", validation_alias="OPENAI_MODEL")

    langsmith_api_key: str | None = Field(default=None, validation_alias="LANGSMITH_API_KEY")
    langsmith_project: str = Field(
        default="multi-agent-research-lab", validation_alias="LANGSMITH_PROJECT"
    )

    langfuse_public_key: str | None = Field(default=None, validation_alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str | None = Field(default=None, validation_alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        default="https://cloud.langfuse.com", validation_alias="LANGFUSE_HOST"
    )

    tavily_api_key: str | None = Field(default=None, validation_alias="TAVILY_API_KEY")

    max_iterations: int = Field(default=6, ge=1, le=20, validation_alias="MAX_ITERATIONS")
    timeout_seconds: int = Field(default=60, ge=5, le=600, validation_alias="TIMEOUT_SECONDS")
    max_retries: int = Field(default=2, ge=0, le=10, validation_alias="MAX_RETRIES")
    retry_backoff_seconds: float = Field(
        default=1.0, ge=0, le=60, validation_alias="RETRY_BACKOFF_SECONDS"
    )

    @field_validator("app_env", "log_level", "openai_model", mode="before")
    @classmethod
    def reject_blank_required_strings(cls, value: object) -> object:
        """Reject whitespace-only values that would fail later and less clearly."""

        if isinstance(value, str) and not value.strip():
            raise ValueError("must not be blank")
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()
