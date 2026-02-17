"""AGenticOS live server — dashboard + real agent engine running together."""

from __future__ import annotations

import asyncio
import logging

import uvicorn

from agos.config import settings
from agos.events.bus import EventBus
from agos.policy.audit import AuditTrail
from agos.policy.engine import PolicyEngine
from agos.events.tracing import Tracer
from agos.knowledge.manager import TheLoom
from agos.evolution.state import EvolutionState
from agos.evolution.meta import MetaEvolver
from agos.processes.manager import ProcessManager
from agos.processes.workload import WorkloadDiscovery
from agos.processes.registry import AgentRegistry
from agos.os_agent import OSAgent
from agos.dashboard.app import dashboard_app, configure
from agos.demo import run_demo
from agos.mcp.client import MCPManager
from agos.mcp.config import load_mcp_configs
from agos.approval.gate import ApprovalGate, ApprovalMode

_logger = logging.getLogger(__name__)


async def _boot_os(
    agent_registry: AgentRegistry,
    event_bus: EventBus,
) -> None:
    """OS boot sequence: discover available agents (but don't start them).

    Like Windows discovering installed programs on boot —
    they show up in the Start Menu, but the user decides when to run them.
    """
    await event_bus.emit("os.boot", {"phase": "agent_discovery"}, source="kernel")

    # Discover bundled agents (shipped with the OS image)
    available = await agent_registry.discover_available()
    _logger.info("Discovered %d available agents", len(available))

    for agent in available:
        _logger.info(
            "  [%s] %s (%s) — %s",
            agent.runtime, agent.display_name, agent.name, agent.status.value,
        )

    await event_bus.emit("os.boot", {
        "phase": "complete",
        "agents_available": len(available),
        "agents": [{"name": a.name, "runtime": a.runtime} for a in available],
    }, source="kernel")


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

    # Initialize meta-evolver (ALMA-style all-component evolution)
    meta_evolver = MetaEvolver()

    # Initialize OS process management
    process_manager = ProcessManager(event_bus, audit_trail)
    workload_discovery = WorkloadDiscovery(event_bus, audit_trail)

    # Initialize agent registry (user-installed agents)
    agent_registry = AgentRegistry(
        event_bus=event_bus,
        audit_trail=audit_trail,
        process_manager=process_manager,
        workload_discovery=workload_discovery,
        state_path=settings.workspace_dir / "agent_registry.json",
    )

    # Initialize LLM if API key is available
    llm = None
    if settings.anthropic_api_key:
        from agos.llm.anthropic import AnthropicProvider
        llm = AnthropicProvider(
            api_key=settings.anthropic_api_key,
            model=settings.default_model,
        )

    # Initialize approval gate (human-in-the-loop for dashboard)
    approval_gate = ApprovalGate(
        mode=ApprovalMode(settings.approval_mode),
        event_bus=event_bus,
        timeout_seconds=settings.approval_timeout_seconds,
    )

    # Build sandbox config from default policy
    sandbox_config = None
    default_policy = policy_engine.get_policy("*")
    if default_policy.sandbox_level != "none":
        from agos.sandbox.executor import SandboxConfig, SandboxLevel
        sandbox_config = SandboxConfig(
            level=SandboxLevel(default_policy.sandbox_level),
            memory_limit_mb=default_policy.sandbox_memory_limit_mb,
            cpu_time_limit_s=default_policy.sandbox_cpu_time_limit_s,
            allowed_paths=default_policy.sandbox_allowed_paths,
        )

    # Initialize the OS agent (the brain) — with real Claude reasoning
    os_agent = OSAgent(
        event_bus=event_bus,
        audit_trail=audit_trail,
        agent_registry=agent_registry,
        process_manager=process_manager,
        policy_engine=policy_engine,
        llm=llm,
        approval_gate=approval_gate,
        sandbox_config=sandbox_config,
    )

    # Initialize MCP manager (external tool servers)
    # Register MCP tools on the inner registry so they also go through
    # the sandbox wrapper when executed.
    mcp_manager = MCPManager(
        registry=os_agent._inner_registry,
        event_bus=event_bus,
    )
    if settings.mcp_auto_connect:
        mcp_configs = await load_mcp_configs(settings.workspace_dir)
        for mc in mcp_configs:
            if mc.enabled:
                try:
                    await mcp_manager.add_server(mc)
                except Exception as e:
                    _logger.warning("Failed to connect MCP server '%s': %s", mc.name, e)

    # Initialize A2A server (Agent-to-Agent protocol)
    a2a_server = None
    if settings.a2a_enabled:
        from agos.a2a.server import A2AServer
        from agos.a2a.client import A2ADirectory
        a2a_server = A2AServer(
            os_agent=os_agent,
            agent_registry=agent_registry,
            event_bus=event_bus,
        )
        a2a_server.set_base_url(
            f"http://{settings.dashboard_host}:{settings.dashboard_port}"
        )

        # A2A discovery runs as a background task after uvicorn starts,
        # because in a fleet all nodes boot simultaneously and need time
        # for their HTTP servers to come up before they can discover each other.
        async def _discover_a2a_peers() -> None:
            await asyncio.sleep(30)  # wait for all nodes to start serving
            if not settings.a2a_remote_agents:
                return
            a2a_dir = A2ADirectory(
                state_path=settings.workspace_dir / "a2a_directory.json",
            )
            for url in settings.a2a_remote_agents.split(","):
                url = url.strip()
                if not url:
                    continue
                for attempt in range(3):
                    try:
                        await a2a_dir.register(url)
                        _logger.info("Registered remote A2A agent: %s", url)
                        break
                    except Exception as e:
                        if attempt < 2:
                            await asyncio.sleep(10)
                        else:
                            _logger.warning("Failed to discover A2A agent at '%s': %s", url, e)

    # Wire into dashboard
    configure(
        event_bus=event_bus,
        audit_trail=audit_trail,
        policy_engine=policy_engine,
        tracer=tracer,
        loom=loom,
        evolution_state=evolution_state,
        meta_evolver=meta_evolver,
        process_manager=process_manager,
        workload_discovery=workload_discovery,
        agent_registry=agent_registry,
        os_agent=os_agent,
        mcp_manager=mcp_manager,
        approval_gate=approval_gate,
        a2a_server=a2a_server,
    )

    # Boot: discover available agents (don't start them — user decides)
    boot_task = asyncio.create_task(
        _boot_os(agent_registry, event_bus)
    )

    # Start system-level agents + evolution engine
    demo_task = asyncio.create_task(
        run_demo(None, event_bus, audit_trail, policy_engine, tracer,
                 loom=loom, evolution_state=evolution_state,
                 meta_evolver=meta_evolver)
    )

    # Discover A2A peers (delayed to let all nodes boot first)
    if settings.a2a_enabled:
        asyncio.create_task(_discover_a2a_peers())

    # Run uvicorn in the same event loop
    config = uvicorn.Config(
        dashboard_app,
        host="0.0.0.0",
        port=settings.dashboard_port,
        log_level="warning",
    )
    server = uvicorn.Server(config)
    await server.serve()

    # Cleanup on shutdown
    await process_manager.shutdown()
    boot_task.cancel()
    demo_task.cancel()


if __name__ == "__main__":
    asyncio.run(main())
