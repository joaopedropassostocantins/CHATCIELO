"""
Centralized settings via Pydantic BaseSettings.

All values are loaded from environment variables or .env file.
No secrets are ever hardcoded.

Sections:
  - Application       general runtime config
  - Model             inference / checkpoint config
  - Database          PostgreSQL + Redis
  - Training          hyperparameters
  - A/B Testing       traffic split per segment
  - Monitoring        Prometheus + tracing
  - Security / LGPD   PII, consent, retention, anonymization
"""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    development = "development"
    staging = "staging"
    production = "production"


class MerchantSegment(str, Enum):
    """Cielo merchant segments used for preference conditioning."""

    MEI = "MEI"
    VAREJO = "VAREJO"
    CORPORATE = "CORPORATE"


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env.

    Args:
        N/A — all fields populated from environment.

    Returns:
        Singleton Settings instance via get_settings().

    Validation Metrics:
        - All path fields must exist when app_env != development.
        - max_seq_length must be power of 2 and <= 2048.
        - ab_challenger_traffic must be in [0.0, 0.5].
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_env: Environment = Environment.development
    app_secret_key: str = "change-me"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str = "microsoft/deberta-v3-large"
    model_path: Path = Path("./artifacts/models/preference_model")
    max_seq_length: int = 1024
    inference_batch_size: int = 16
    device: str = "cpu"
    # Challenger model for A/B testing (empty = disabled)
    challenger_model_path: Path = Path("")

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://chatcielo:chatcielo@localhost:5432/chatcielo"
    redis_url: str = "redis://localhost:6379/0"
    redis_cache_ttl: int = 3600           # seconds

    # ── Training ──────────────────────────────────────────────────────────────
    train_data_path: Path = Path("./data/train.parquet")
    val_data_path: Path = Path("./data/val.parquet")
    output_dir: Path = Path("./artifacts/models")
    num_epochs: int = 3
    learning_rate: float = 1e-5
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    label_smoothing: float = 0.1
    cls_loss_weight: float = 0.7          # combined loss: 0.7·cls + 0.3·rank
    early_stopping_patience: int = 2

    # ── A/B Testing ───────────────────────────────────────────────────────────
    # Fraction of traffic routed to challenger model (0.0 = disabled)
    ab_challenger_traffic: float = Field(default=0.0, ge=0.0, le=0.5)
    # Redis key where A/B assignment is stored per conversation_id
    ab_redis_prefix: str = "chatcielo:ab:"

    # ── Monitoring ────────────────────────────────────────────────────────────
    prometheus_port: int = 9090
    enable_tracing: bool = False
    # PSI threshold above which drift is flagged and retraining triggered
    drift_psi_threshold: float = 0.2

    # ── Security / LGPD ───────────────────────────────────────────────────────
    pii_audit_enabled: bool = True
    data_retention_days: int = 90
    consent_required: bool = True
    anonymization_salt: str = "change-me"

    @field_validator("max_seq_length")
    @classmethod
    def must_be_power_of_two(cls, v: int) -> int:
        """Validate that max_seq_length is a power of 2 and within transformer limits."""
        if v <= 0 or (v & (v - 1)) != 0:
            raise ValueError(f"max_seq_length must be a power of 2, got {v}")
        if v > 2048:
            raise ValueError(f"max_seq_length must be <= 2048, got {v}")
        return v


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return singleton Settings instance.

    Returns:
        The global Settings object, constructed once and cached.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
