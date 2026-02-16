"""AGenticOS live server — dashboard + real agent engine running together."""

from __future__ import annotations

import asyncio

import uvicorn

from agos.config import settings
from agos.events.bus import EventBus
from agos.policy.audit import AuditTrail
from agos.policy.engine import PolicyEngine
from agos.events.tracing import Tracer
from agos.knowledge.manager import TheLoom
from agos.evolution.state import EvolutionState
from agos.dashboard.app import dashboard_app, configure
from agos.demo import run_demo


async def main() -> None:
    event_bus = EventBus()
    policy_engine = PolicyEngine()
    tracer = Tracer()

    settings.workspace_dir.mkdir(parents=True, exist_ok=True)

    db_path = str(settings.workspace_dir / "agos.db")
    audit_trail = AuditTrail(db_path)
    await audit_trail.initialize()

    # Initialize TheLoom knowledge substrate for evolution
    loom_path = str(settings.workspace_dir / "knowledge.db")
    loom = TheLoom(loom_path)
    await loom.initialize()

    # Initialize evolution state persistence
    evolution_state = EvolutionState(settings.workspace_dir / "evolution_state.json")

    # Wire into dashboard (no runtime needed — agents run directly)
    configure(
        event_bus=event_bus,
        audit_trail=audit_trail,
        policy_engine=policy_engine,
        tracer=tracer,
        loom=loom,
        evolution_state=evolution_state,
    )

    # Start real agent engine + evolution as background task
    demo_task = asyncio.create_task(
        run_demo(None, event_bus, audit_trail, policy_engine, tracer,
                 loom=loom, evolution_state=evolution_state)
    )

    # Run uvicorn in the same event loop
    config = uvicorn.Config(
        dashboard_app,
        host="0.0.0.0",
        port=settings.dashboard_port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    await server.serve()

    demo_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
