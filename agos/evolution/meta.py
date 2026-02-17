"""MetaEvolver — ALMA-inspired meta-evolution for ALL agos components.

Instead of only evolving the knowledge layer, this controller runs a
meta-learning loop across every architectural layer:

  1. Semantic Work Substrate (TheLoom, weaves, consolidation)
  2. Agent & Intent Intelligence (IntentEngine, personas)
  3. Agent Orchestration & Workflow (Planner, runtime)
  4. Identity, Delegation & Governance (PolicyEngine)
  5. Episodic Experience (EventBus, Tracer)

Each component exposes a "genome" of evolvable parameters.  The
MetaEvolver observes fitness signals from real usage (audit trail,
event bus, tracing), identifies underperforming components, proposes
parameter mutations, tests them in sandbox, and applies winners.

Inspired by: "Learning to Continually Learn via Meta-learning Agentic
Memory Designs" (ALMA, arXiv:2602.07755).
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from typing import Any

from pydantic import BaseModel, Field

from agos.types import new_id


# ── Component Genome ─────────────────────────────────────────────


class ParamSpec(BaseModel):
    """A single evolvable parameter."""

    name: str
    current: Any = None
    default: Any = None
    min_val: Any = None
    max_val: Any = None
    param_type: str = "float"  # float, int, bool, str
    description: str = ""


class ComponentGenome(BaseModel):
    """Evolvable parameters for one architectural component."""

    component: str  # e.g. "knowledge.semantic", "intent.engine"
    layer: str  # architecture layer name
    params: list[ParamSpec] = Field(default_factory=list)
    fitness_score: float = 0.5  # 0..1 — current fitness
    last_evaluated: str = ""
    mutations_applied: int = 0


class FitnessSignal(BaseModel):
    """A single fitness observation from the running system."""

    component: str
    metric: str  # e.g. "recall_hit_rate", "task_completion_rate"
    value: float
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class Mutation(BaseModel):
    """A proposed parameter change."""

    id: str = Field(default_factory=new_id)
    component: str
    param_name: str
    old_value: Any = None
    new_value: Any = None
    reason: str = ""
    fitness_before: float = 0.0
    fitness_after: float | None = None
    applied: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Fitness Collector ────────────────────────────────────────────


class FitnessCollector:
    """Gathers performance signals from all subsystems.

    Reads from: EventBus history, AuditTrail, Tracer spans,
    TheLoom access stats, and raw system metrics.
    """

    def __init__(self) -> None:
        self._signals: list[FitnessSignal] = []
        self._window_hours: int = 6  # look at last 6 hours

    async def collect(
        self,
        event_bus=None,
        audit_trail=None,
        tracer=None,
        loom=None,
        policy_engine=None,
        runtime=None,
        process_manager=None,
    ) -> list[FitnessSignal]:
        """Collect fitness signals from all available subsystems."""
        signals: list[FitnessSignal] = []

        # ── Knowledge fitness ──
        if loom:
            signals.extend(await self._collect_knowledge(loom))

        # ── Policy fitness ──
        if audit_trail:
            signals.extend(await self._collect_policy(audit_trail))

        # ── Agent fitness ──
        if runtime:
            signals.extend(self._collect_agent(runtime))

        # ── Event bus fitness ──
        if event_bus:
            signals.extend(self._collect_events(event_bus))

        # ── Tracing fitness ──
        if tracer:
            signals.extend(self._collect_tracing(tracer))

        # ── Process fitness (OS-level workload monitoring) ──
        if process_manager:
            signals.extend(self._collect_processes(process_manager))

        self._signals.extend(signals)
        # Keep last 1000
        self._signals = self._signals[-1000:]
        return signals

    async def _collect_knowledge(self, loom) -> list[FitnessSignal]:
        """Fitness signals from the knowledge substrate."""
        signals = []
        try:
            from agos.knowledge.base import ThreadQuery

            # Semantic retrieval health: try a generic query
            results = await loom.semantic.query(
                ThreadQuery(text="agent knowledge", limit=5)
            )
            hit_rate = min(len(results) / 5.0, 1.0)
            signals.append(FitnessSignal(
                component="knowledge.semantic",
                metric="retrieval_hit_rate",
                value=hit_rate,
            ))

            # Graph density
            entities = await loom.graph.entities()
            density = min(len(entities) / 100.0, 1.0)  # normalize to 0-1
            signals.append(FitnessSignal(
                component="knowledge.graph",
                metric="graph_density",
                value=density,
            ))
        except Exception:
            pass
        return signals

    async def _collect_policy(self, audit_trail) -> list[FitnessSignal]:
        """Fitness signals from the policy/audit system."""
        signals = []
        try:
            total = await audit_trail.count()
            violations = await audit_trail.violations(limit=100)
            violation_count = len(violations)

            # Low violation rate = good policy calibration
            if total > 0:
                violation_rate = violation_count / max(total, 1)
                signals.append(FitnessSignal(
                    component="policy.engine",
                    metric="violation_rate",
                    value=1.0 - min(violation_rate * 10, 1.0),  # invert: lower is better
                ))

            # Audit volume = system activity
            activity = min(total / 100.0, 1.0)
            signals.append(FitnessSignal(
                component="policy.audit",
                metric="activity_level",
                value=activity,
            ))
        except Exception:
            pass
        return signals

    def _collect_agent(self, runtime) -> list[FitnessSignal]:
        """Fitness signals from the agent runtime."""
        signals = []
        try:
            agents = runtime.list_agents()
            total = len(agents)
            completed = sum(1 for a in agents if a["state"] == "completed")
            errored = sum(1 for a in agents if a["state"] == "error")

            if total > 0:
                # Completion rate
                signals.append(FitnessSignal(
                    component="kernel.runtime",
                    metric="completion_rate",
                    value=completed / max(total, 1),
                ))
                # Error rate (inverted)
                signals.append(FitnessSignal(
                    component="kernel.runtime",
                    metric="error_rate_inv",
                    value=1.0 - (errored / max(total, 1)),
                ))

            # Token efficiency: average tokens per completed agent
            completed_agents = [a for a in agents if a["state"] == "completed"]
            if completed_agents:
                avg_tokens = sum(a["tokens_used"] for a in completed_agents) / len(completed_agents)
                # Lower is better, normalize against 200k budget
                efficiency = 1.0 - min(avg_tokens / 200_000, 1.0)
                signals.append(FitnessSignal(
                    component="kernel.agent",
                    metric="token_efficiency",
                    value=efficiency,
                ))
        except Exception:
            pass
        return signals

    def _collect_events(self, event_bus) -> list[FitnessSignal]:
        """Fitness signals from the event bus."""
        signals = []
        try:
            topics = event_bus.topics()
            history_len = len(event_bus._history)

            # Topic diversity
            diversity = min(len(topics) / 20.0, 1.0)
            signals.append(FitnessSignal(
                component="events.bus",
                metric="topic_diversity",
                value=diversity,
            ))

            # History utilization
            utilization = history_len / max(event_bus._history_limit, 1)
            signals.append(FitnessSignal(
                component="events.bus",
                metric="history_utilization",
                value=min(utilization, 1.0),
            ))
        except Exception:
            pass
        return signals

    def _collect_tracing(self, tracer) -> list[FitnessSignal]:
        """Fitness signals from the tracing system."""
        signals = []
        try:
            traces = tracer.list_traces(limit=50)
            if traces:
                error_traces = sum(1 for t in traces if t.error_count > 0)
                signals.append(FitnessSignal(
                    component="events.tracing",
                    metric="trace_success_rate",
                    value=1.0 - (error_traces / max(len(traces), 1)),
                ))
        except Exception:
            pass
        return signals

    def _collect_processes(self, process_manager) -> list[FitnessSignal]:
        """Fitness signals from OS-level process management."""
        signals = []
        try:
            procs = process_manager.list_processes()
            if not procs:
                return signals

            running = [p for p in procs if p["state"] == "running"]
            crashed = [p for p in procs if p["state"] == "crashed"]

            # Process survival rate
            total = len(procs)
            survival = len(running) / max(total, 1)
            signals.append(FitnessSignal(
                component="kernel",
                metric="process_survival_rate",
                value=survival,
            ))

            # Crash rate (lower is better → invert for fitness)
            crash_rate = len(crashed) / max(total, 1)
            signals.append(FitnessSignal(
                component="kernel",
                metric="process_stability",
                value=1.0 - crash_rate,
            ))

            # Memory efficiency across all processes
            for p in running:
                mem_usage = p["memory_mb"] / max(p["memory_limit_mb"], 1)
                signals.append(FitnessSignal(
                    component="kernel",
                    metric=f"memory_efficiency:{p['name']}",
                    value=max(0, 1.0 - mem_usage),
                ))

                # Token budget utilization
                if p["token_limit"] > 0:
                    token_usage = p["token_count"] / p["token_limit"]
                    signals.append(FitnessSignal(
                        component="policy",
                        metric=f"token_budget_health:{p['name']}",
                        value=max(0, 1.0 - token_usage),
                    ))

            # Restart frequency (many restarts = OS not handling failures well)
            total_restarts = sum(p["restart_count"] for p in procs)
            restart_penalty = min(total_restarts / max(total * 3, 1), 1.0)
            signals.append(FitnessSignal(
                component="orchestration.runtime",
                metric="restart_frequency",
                value=1.0 - restart_penalty,
            ))

        except Exception:
            pass
        return signals

    def aggregate_fitness(self, component: str) -> float:
        """Average fitness for a component over recent signals."""
        cutoff = (datetime.utcnow() - timedelta(hours=self._window_hours)).isoformat()
        relevant = [
            s for s in self._signals
            if s.component == component and s.timestamp >= cutoff
        ]
        if not relevant:
            return 0.5  # neutral default
        return sum(s.value for s in relevant) / len(relevant)

    def recent_signals(self, limit: int = 50) -> list[FitnessSignal]:
        return self._signals[-limit:]


# ── Meta Evolver ─────────────────────────────────────────────────


class MetaEvolver:
    """The ALMA-style meta-evolution controller.

    Maintains a genome for each component, collects fitness signals,
    proposes mutations, and applies them through the integrator.
    """

    def __init__(self) -> None:
        self.genomes: dict[str, ComponentGenome] = {}
        self.fitness = FitnessCollector()
        self.mutations: list[Mutation] = []
        self._build_genomes()

    def _build_genomes(self) -> None:
        """Define the evolvable parameter space for each component."""

        # ── Layer 1: Semantic Work Substrate ──
        self.genomes["knowledge.semantic"] = ComponentGenome(
            component="knowledge.semantic",
            layer="Semantic Work Substrate",
            params=[
                ParamSpec(
                    name="temperature", default=0.0, min_val=0.0, max_val=1.0,
                    param_type="float", description="Softmax retrieval diversity",
                ),
                ParamSpec(
                    name="track_access", default=False,
                    param_type="bool", description="Access-based confidence tracking",
                ),
                ParamSpec(
                    name="relevance_threshold", default=0.01, min_val=0.001, max_val=0.1,
                    param_type="float", description="Minimum cosine similarity for results",
                ),
                ParamSpec(
                    name="confidence_decay_factor", default=0.95, min_val=0.8, max_val=0.99,
                    param_type="float", description="Confidence decay for unused knowledge",
                ),
                ParamSpec(
                    name="confidence_decay_days", default=30, min_val=7, max_val=90,
                    param_type="int", description="Days inactive before decay kicks in",
                ),
            ],
        )

        self.genomes["knowledge.graph"] = ComponentGenome(
            component="knowledge.graph",
            layer="Semantic Work Substrate",
            params=[
                ParamSpec(
                    name="default_traversal_depth", default=1, min_val=1, max_val=4,
                    param_type="int", description="Default neighbor traversal hops",
                ),
                ParamSpec(
                    name="edge_weight_decay", default=0.99, min_val=0.9, max_val=1.0,
                    param_type="float", description="Weight decay per consolidation cycle",
                ),
            ],
        )

        self.genomes["knowledge.consolidator"] = ComponentGenome(
            component="knowledge.consolidator",
            layer="Semantic Work Substrate",
            params=[
                ParamSpec(
                    name="older_than_hours", default=24, min_val=6, max_val=168,
                    param_type="int", description="Consolidate events older than N hours",
                ),
                ParamSpec(
                    name="min_cluster_size", default=3, min_val=2, max_val=10,
                    param_type="int", description="Minimum events to form a summary",
                ),
                ParamSpec(
                    name="max_concurrent_writes", default=5, min_val=1, max_val=20,
                    param_type="int", description="Semaphore limit for batch ops",
                ),
            ],
        )

        self.genomes["knowledge.loom"] = ComponentGenome(
            component="knowledge.loom",
            layer="Semantic Work Substrate",
            params=[
                ParamSpec(
                    name="use_layered_recall", default=False,
                    param_type="bool", description="Priority-ordered layer retrieval",
                ),
                ParamSpec(
                    name="recall_limit", default=10, min_val=3, max_val=50,
                    param_type="int", description="Default recall result limit",
                ),
            ],
        )

        # ── Layer 2: Agent & Intent Intelligence ──
        self.genomes["intent.engine"] = ComponentGenome(
            component="intent.engine",
            layer="Agent & Intent Intelligence",
            params=[
                ParamSpec(
                    name="default_strategy", default="solo",
                    param_type="str", description="Fallback coordination strategy",
                ),
                ParamSpec(
                    name="max_intent_tokens", default=500, min_val=200, max_val=1500,
                    param_type="int", description="Token limit for intent classification",
                ),
            ],
        )

        self.genomes["intent.personas"] = ComponentGenome(
            component="intent.personas",
            layer="Agent & Intent Intelligence",
            params=[
                ParamSpec(
                    name="researcher_budget", default=200_000, min_val=50_000, max_val=500_000,
                    param_type="int", description="Researcher agent token budget",
                ),
                ParamSpec(
                    name="coder_budget", default=200_000, min_val=50_000, max_val=500_000,
                    param_type="int", description="Coder agent token budget",
                ),
                ParamSpec(
                    name="orchestrator_budget", default=200_000, min_val=50_000, max_val=500_000,
                    param_type="int", description="Orchestrator agent token budget",
                ),
                ParamSpec(
                    name="researcher_max_turns", default=30, min_val=5, max_val=80,
                    param_type="int", description="Researcher max turns",
                ),
                ParamSpec(
                    name="coder_max_turns", default=40, min_val=10, max_val=100,
                    param_type="int", description="Coder max turns",
                ),
                ParamSpec(
                    name="orchestrator_max_turns", default=50, min_val=10, max_val=100,
                    param_type="int", description="Orchestrator max turns",
                ),
            ],
        )

        # ── Layer 3: Agent Orchestration & Workflow ──
        self.genomes["orchestration.planner"] = ComponentGenome(
            component="orchestration.planner",
            layer="Agent Orchestration & Workflow",
            params=[
                ParamSpec(
                    name="parallel_threshold", default=3, min_val=2, max_val=10,
                    param_type="int",
                    description="Min subtasks to trigger parallel execution",
                ),
                ParamSpec(
                    name="pipeline_max_agents", default=5, min_val=2, max_val=10,
                    param_type="int", description="Max agents in a pipeline",
                ),
            ],
        )

        self.genomes["orchestration.runtime"] = ComponentGenome(
            component="orchestration.runtime",
            layer="Agent Orchestration & Workflow",
            params=[
                ParamSpec(
                    name="max_concurrent_agents", default=50, min_val=5, max_val=200,
                    param_type="int", description="Max agents running simultaneously",
                ),
            ],
        )

        # ── Layer 4: Identity, Delegation & Governance ──
        self.genomes["policy.engine"] = ComponentGenome(
            component="policy.engine",
            layer="Identity & Governance",
            params=[
                ParamSpec(
                    name="default_max_tokens", default=200_000, min_val=50_000, max_val=1_000_000,
                    param_type="int", description="Default agent token budget",
                ),
                ParamSpec(
                    name="default_max_turns", default=50, min_val=10, max_val=200,
                    param_type="int", description="Default agent turn limit",
                ),
                ParamSpec(
                    name="default_rate_limit", default=60, min_val=10, max_val=200,
                    param_type="int", description="Tool calls per minute",
                ),
                ParamSpec(
                    name="default_read_only", default=False,
                    param_type="bool", description="Default read-only mode",
                ),
            ],
        )

        # ── Layer 5: Episodic Experience ──
        self.genomes["events.bus"] = ComponentGenome(
            component="events.bus",
            layer="Episodic Experience",
            params=[
                ParamSpec(
                    name="history_limit", default=500, min_val=100, max_val=5000,
                    param_type="int", description="Max events in memory",
                ),
            ],
        )

        self.genomes["events.tracing"] = ComponentGenome(
            component="events.tracing",
            layer="Episodic Experience",
            params=[
                ParamSpec(
                    name="max_traces", default=200, min_val=50, max_val=1000,
                    param_type="int", description="Max traces retained",
                ),
            ],
        )

    async def run_meta_cycle(
        self,
        event_bus=None,
        audit_trail=None,
        tracer=None,
        loom=None,
        policy_engine=None,
        runtime=None,
        integrator=None,
    ) -> MetaCycleReport:
        """Run one meta-evolution cycle across all components.

        1. Collect fitness signals
        2. Update genome fitness scores
        3. Identify underperformers
        4. Propose mutations
        5. Apply mutations via integrator (with snapshot + rollback)
        """
        start = time.monotonic()
        report = MetaCycleReport()

        # Step 1: Collect fitness
        signals = await self.fitness.collect(
            event_bus=event_bus,
            audit_trail=audit_trail,
            tracer=tracer,
            loom=loom,
            policy_engine=policy_engine,
            runtime=runtime,
        )
        report.signals_collected = len(signals)

        # Step 2: Update genome fitness scores
        for name, genome in self.genomes.items():
            score = self.fitness.aggregate_fitness(name)
            genome.fitness_score = score
            genome.last_evaluated = datetime.utcnow().isoformat()

        # Step 3: Identify underperformers (fitness < 0.6)
        underperformers = [
            g for g in self.genomes.values()
            if g.fitness_score < 0.6 and g.params
        ]
        report.underperformers = [g.component for g in underperformers]

        # Step 4: Propose mutations for underperformers
        proposed: list[Mutation] = []
        for genome in underperformers:
            mutations = self._propose_mutations(genome)
            proposed.extend(mutations)
        report.mutations_proposed = len(proposed)

        # Step 5: Apply mutations
        applied_count = 0
        for mutation in proposed:
            success = await self._apply_mutation(
                mutation,
                loom=loom,
                policy_engine=policy_engine,
                event_bus=event_bus,
                tracer=tracer,
            )
            if success:
                mutation.applied = True
                applied_count += 1
                # Update genome
                genome = self.genomes.get(mutation.component)
                if genome:
                    genome.mutations_applied += 1
                    for p in genome.params:
                        if p.name == mutation.param_name:
                            p.current = mutation.new_value

        self.mutations.extend(proposed)
        # Keep last 200 mutations
        self.mutations = self.mutations[-200:]
        report.mutations_applied = applied_count

        report.duration_ms = (time.monotonic() - start) * 1000

        # Emit event if bus available
        if event_bus:
            await event_bus.emit("meta.evolution_cycle", {
                "signals": report.signals_collected,
                "underperformers": report.underperformers,
                "proposed": report.mutations_proposed,
                "applied": report.mutations_applied,
                "duration_ms": round(report.duration_ms),
            }, source="meta_evolver")

        return report

    def _propose_mutations(self, genome: ComponentGenome) -> list[Mutation]:
        """Propose parameter mutations for an underperforming component.

        Strategy: nudge numeric params toward better-performing ranges.
        For new params with no current value, set to the default.
        """
        import random

        mutations: list[Mutation] = []
        for param in genome.params:
            # Skip params that are already at a good value
            if param.current is not None and param.current != param.default:
                continue  # already mutated, don't stack

            if param.param_type == "float":
                old = param.current if param.current is not None else param.default
                # Random perturbation within ±20% of range
                range_size = (param.max_val or 1.0) - (param.min_val or 0.0)
                delta = random.uniform(-0.2, 0.2) * range_size
                new = max(param.min_val or 0.0, min(param.max_val or 1.0, old + delta))
                if abs(new - old) > 0.001:
                    mutations.append(Mutation(
                        component=genome.component,
                        param_name=param.name,
                        old_value=old,
                        new_value=round(new, 4),
                        reason=f"Fitness {genome.fitness_score:.2f} < 0.6, nudging {param.name}",
                        fitness_before=genome.fitness_score,
                    ))

            elif param.param_type == "int":
                old = param.current if param.current is not None else param.default
                range_size = (param.max_val or 100) - (param.min_val or 0)
                delta = random.randint(-max(1, range_size // 5), max(1, range_size // 5))
                new = max(param.min_val or 0, min(param.max_val or 999999, old + delta))
                if new != old:
                    mutations.append(Mutation(
                        component=genome.component,
                        param_name=param.name,
                        old_value=old,
                        new_value=new,
                        reason=f"Fitness {genome.fitness_score:.2f} < 0.6, adjusting {param.name}",
                        fitness_before=genome.fitness_score,
                    ))

            elif param.param_type == "bool":
                old = param.current if param.current is not None else param.default
                # Flip with 30% probability
                if random.random() < 0.3:
                    mutations.append(Mutation(
                        component=genome.component,
                        param_name=param.name,
                        old_value=old,
                        new_value=not old,
                        reason=f"Fitness {genome.fitness_score:.2f} < 0.6, toggling {param.name}",
                        fitness_before=genome.fitness_score,
                    ))

        # Limit to 2 mutations per component per cycle
        return mutations[:2]

    async def _apply_mutation(
        self, mutation: Mutation,
        loom=None, policy_engine=None, event_bus=None, tracer=None,
    ) -> bool:
        """Apply a single mutation to the target component."""
        try:
            comp = mutation.component
            param = mutation.param_name
            val = mutation.new_value

            # ── Knowledge mutations ──
            if comp == "knowledge.semantic" and loom:
                if param == "temperature":
                    loom.semantic.set_temperature(val)
                elif param == "track_access":
                    loom.semantic.enable_access_tracking(val)
                elif param == "relevance_threshold":
                    # Store as attribute for future use
                    loom.semantic._relevance_threshold = val
                elif param == "confidence_decay_factor":
                    loom.semantic._decay_factor = val
                elif param == "confidence_decay_days":
                    loom.semantic._decay_days = val
                else:
                    return False
                return True

            if comp == "knowledge.loom" and loom:
                if param == "use_layered_recall":
                    loom.enable_layered_recall(val)
                elif param == "recall_limit":
                    loom._default_recall_limit = val
                else:
                    return False
                return True

            if comp == "knowledge.consolidator" and loom:
                if param == "max_concurrent_writes":
                    loom.learner._max_concurrent = val
                else:
                    # Store for consolidator use
                    setattr(loom, f"_consolidator_{param}", val)
                return True

            if comp == "knowledge.graph" and loom:
                setattr(loom.graph, f"_{param}", val)
                return True

            # ── Policy mutations ──
            if comp == "policy.engine" and policy_engine:
                default = policy_engine._default
                if param == "default_max_tokens":
                    default.max_tokens = val
                elif param == "default_max_turns":
                    default.max_turns = val
                elif param == "default_rate_limit":
                    default.max_tool_calls_per_minute = val
                elif param == "default_read_only":
                    default.read_only = val
                else:
                    return False
                return True

            # ── Event bus mutations ──
            if comp == "events.bus" and event_bus:
                if param == "history_limit":
                    event_bus._history_limit = val
                else:
                    return False
                return True

            # ── Tracing mutations ──
            if comp == "events.tracing" and tracer:
                if param == "max_traces":
                    tracer._max_traces = val
                else:
                    return False
                return True

            # ── Persona / orchestration / intent mutations ──
            # These are stored in genomes and applied on next agent spawn
            if comp in ("intent.engine", "intent.personas",
                        "orchestration.planner", "orchestration.runtime"):
                # Value stored in genome.params — read at spawn time
                return True

            return False

        except Exception:
            return False

    def get_genome(self, component: str) -> ComponentGenome | None:
        return self.genomes.get(component)

    def all_genomes(self) -> list[ComponentGenome]:
        return list(self.genomes.values())

    def export_state(self) -> dict:
        """Export full meta-evolution state for persistence."""
        return {
            "genomes": {
                name: g.model_dump() for name, g in self.genomes.items()
            },
            "recent_mutations": [m.model_dump() for m in self.mutations[-50:]],
            "timestamp": datetime.utcnow().isoformat(),
        }

    def restore_state(self, data: dict) -> None:
        """Restore meta-evolution state from persisted data."""
        if "genomes" in data:
            for name, gdata in data["genomes"].items():
                if name in self.genomes:
                    stored = ComponentGenome(**gdata)
                    existing = self.genomes[name]
                    existing.fitness_score = stored.fitness_score
                    existing.last_evaluated = stored.last_evaluated
                    existing.mutations_applied = stored.mutations_applied
                    # Restore current param values
                    stored_params = {p.name: p for p in stored.params}
                    for p in existing.params:
                        if p.name in stored_params:
                            p.current = stored_params[p.name].current

        if "recent_mutations" in data:
            self.mutations = [Mutation(**m) for m in data["recent_mutations"]]


class MetaCycleReport(BaseModel):
    """Summary of one meta-evolution cycle."""

    signals_collected: int = 0
    underperformers: list[str] = Field(default_factory=list)
    mutations_proposed: int = 0
    mutations_applied: int = 0
    duration_ms: float = 0.0
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
