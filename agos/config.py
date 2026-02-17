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

    # Node specialization (for multi-node fleet diversity)
    node_role: str = "general"  # knowledge|intent|orchestration|policy|general
    evolution_initial_delay: int = 0  # Seconds to wait before first evolution cycle (stagger fleet)

    # Update settings
    auto_update_check: bool = True
    github_owner: str = "aliveclaw"
    github_repo: str = "agos"
    github_token: str = ""  # GitHub PAT for community contributions (optional)
    auto_share_every: int = 0  # Auto-share via PR disabled by default (evolution writes code locally instead)

    # MCP (Model Context Protocol) settings
    mcp_auto_connect: bool = True  # Auto-connect to configured MCP servers on startup

    # A2A (Agent-to-Agent) protocol settings
    a2a_enabled: bool = True  # Expose AGOS as an A2A server
    a2a_remote_agents: str = ""  # Comma-separated URLs of remote A2A agents to auto-discover

    # Approval settings (dashboard human-in-the-loop)
    approval_mode: str = "auto"  # "auto", "confirm-dangerous", "confirm-all"
    approval_timeout_seconds: int = 300  # 5 minutes

    model_config = {"env_prefix": "AGOS_"}


settings = AgosSettings()
