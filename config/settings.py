"""Application configuration loaded from environment variables."""

from __future__ import annotations

from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REDACT_TOKENS = ("key", "secret", "webhook")


class EnvSettings(BaseSettings):
    """Base settings with a shared env-file configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class PhoenixSettings(EnvSettings):
    """Settings for Arize Phoenix."""

    host: str = Field(default="http://localhost:6006", validation_alias="PHOENIX_HOST")
    api_key: str | None = Field(default=None, validation_alias="PHOENIX_API_KEY")
    project_name: str = Field(
        default="prompt-guardian", validation_alias="PHOENIX_PROJECT_NAME"
    )


class GoogleCloudSettings(EnvSettings):
    """Settings for Google Cloud and Vertex AI."""

    project: str = Field(default="local-project", validation_alias="GOOGLE_CLOUD_PROJECT")
    region: str = Field(default="us-central1", validation_alias="GOOGLE_CLOUD_REGION")
    model: str = Field(default="gemini-2.0-flash", validation_alias="VERTEX_AI_MODEL")


class SlackSettings(EnvSettings):
    """Settings for Slack alerts."""

    webhook_url: str = Field(
        default="http://localhost:9999/webhook", validation_alias="SLACK_WEBHOOK_URL"
    )
    channel: str = Field(default="#prompt-alerts", validation_alias="SLACK_CHANNEL")


class AgentSettings(EnvSettings):
    """Settings that control agent behavior."""

    regression_threshold: float = Field(
        default=0.05, validation_alias="REGRESSION_THRESHOLD"
    )
    eval_judge_model: str = Field(
        default="gemini-2.0-flash", validation_alias="EVAL_JUDGE_MODEL"
    )
    poll_interval_seconds: int = Field(
        default=300, validation_alias="POLL_INTERVAL_SECONDS"
    )


class Settings(EnvSettings):
    """Top-level application settings."""

    phoenix: PhoenixSettings = Field(default_factory=PhoenixSettings)
    google_cloud: GoogleCloudSettings = Field(default_factory=GoogleCloudSettings)
    slack: SlackSettings = Field(default_factory=SlackSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)

    def to_redacted_dict(self) -> dict[str, Any]:
        """Return a dict of settings with sensitive values redacted.

        Returns:
            dict[str, Any]: Settings dictionary with secrets masked.
        """

        def redact(value: Any, key_path: str) -> Any:
            """Redact values based on key names.

            Args:
                value: The value to redact.
                key_path: The key name or path.

            Returns:
                Any: The redacted value or original value.
            """

            lowered = key_path.lower()
            if any(token in lowered for token in REDACT_TOKENS):
                return "***"
            if isinstance(value, dict):
                return {k: redact(v, k) for k, v in value.items()}
            return value

        raw = self.model_dump()
        return {key: redact(value, key) for key, value in raw.items()}


settings = Settings()
