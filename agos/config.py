"""Global configuration — loaded from environment variables."""

from pathlib import Path

from pydantic_settings import BaseSettings


class AgosSettings(BaseSettings):
    anthropic_api_key: str = ""
    default_model: str = "claude-sonnet-4-20250514"
    workspace_dir: Path = Path(".agos")
    db_path: Path = Path(".agos/agos.db")
    max_concurrent_agents: int = 50
    log_level: str = "INFO"
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8420

    # Evolution settings
    evolution_auto_merge: bool = False
    evolution_interval_hours: int = 168  # weekly
    evolution_days_lookback: int = 7
    evolution_max_papers: int = 20

    # Update settings
    auto_update_check: bool = True
    github_owner: str = "aliveclaw"
    github_repo: str = "agos"

    model_config = {"env_prefix": "AGOS_"}


settings = AgosSettings()
