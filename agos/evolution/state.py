"""Evolution state persistence — save/load/export evolved parameters.

Captures the runtime parameters modified by integration strategies,
persists them to .agos/evolution_state.json, and restores them on boot.

Includes ALMA-inspired DesignArchive for tracking strategy lineage and
softmax-based selection pressure.
"""

from __future__ import annotations

import logging
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import BaseModel, Field

from agos.types import new_id

if TYPE_CHECKING:
    from agos.knowledge.manager import TheLoom

logger = logging.getLogger(__name__)


class AppliedStrategy(BaseModel):
    """Record of a single applied strategy."""

    name: str
    module: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    source_papers: list[dict[str, str]] = Field(default_factory=list)
    sandbox_passed: bool = False
    health_check_passed: bool = True
    applied_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    applied_count: int = 1


class DiscoveredPattern(BaseModel):
    """A code pattern discovered during evolution."""

    name: str
    module: str
    code_snippet: str = ""
    sandbox_output: str = ""
    source_paper: str = ""


# ── ALMA-inspired Design Archive ─────────────────────────────────


class DesignEntry(BaseModel):
    """A single design in the archive — tracks lineage for iterative improvement."""

    id: str = Field(default_factory=new_id)
    strategy_name: str
    module: str  # target agos module (e.g. "knowledge.semantic")
    code_hash: str = ""
    code_snippet: str = ""  # the actual pattern code (truncated for persistence)
    fitness_scores: list[float] = Field(default_factory=list)  # history
    current_fitness: float = 0.0
    generation: int = 0  # 0 = original, 1+ = iterated children
    parent_id: str = ""  # lineage tracking
    source_paper: str = ""
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class DesignArchive:
    """ALMA-style population archive with softmax selection pressure.

    Maintains a bounded set of designs. High-fitness designs are
    sampled more often (softmax) but all have non-zero probability,
    encouraging open-ended exploration.
    """

    def __init__(self, max_size: int = 50, temperature: float = 0.3) -> None:
        self.entries: list[DesignEntry] = []
        self.max_size = max_size
        self.temperature = temperature

    def add(self, entry: DesignEntry) -> None:
        """Add a design, evicting lowest-fitness if over capacity."""
        self.entries.append(entry)
        if len(self.entries) > self.max_size:
            self.entries.sort(key=lambda e: e.current_fitness)
            self.entries.pop(0)  # remove lowest fitness

    def sample(self, n: int) -> list[DesignEntry]:
        """ALMA softmax sampling: P(d) ~ exp(fitness / temp).

        Weighted random without replacement.
        """
        if not self.entries or n <= 0:
            return []
        n = min(n, len(self.entries))

        # Compute softmax weights
        temp = max(self.temperature, 0.01)  # prevent div-by-zero
        scores = [e.current_fitness for e in self.entries]
        max_score = max(scores) if scores else 0
        # Subtract max for numerical stability
        weights = [math.exp((s - max_score) / temp) for s in scores]
        total = sum(weights)
        if total == 0:
            # Uniform fallback
            return random.sample(self.entries, n)

        probs = [w / total for w in weights]

        # Weighted sample without replacement
        indices = list(range(len(self.entries)))
        selected: list[int] = []
        remaining_probs = list(probs)
        for _ in range(n):
            r = random.random() * sum(remaining_probs)
            cumulative = 0.0
            for i, idx in enumerate(indices):
                if idx in selected:
                    continue
                cumulative += remaining_probs[i]
                if cumulative >= r:
                    selected.append(idx)
                    remaining_probs[i] = 0.0
                    break
            else:
                # Fallback: pick first unselected
                for i, idx in enumerate(indices):
                    if idx not in selected:
                        selected.append(idx)
                        remaining_probs[i] = 0.0
                        break

        return [self.entries[i] for i in selected]

    def best(self, n: int = 5) -> list[DesignEntry]:
        """Top N designs by fitness."""
        return sorted(self.entries, key=lambda e: e.current_fitness, reverse=True)[:n]

    def by_module(self, module: str) -> list[DesignEntry]:
        """Filter designs by target module."""
        return [e for e in self.entries if e.module == module]

    def update_fitness(self, design_id: str, fitness: float) -> None:
        """Update a design's fitness score."""
        for entry in self.entries:
            if entry.id == design_id:
                entry.fitness_scores.append(fitness)
                entry.current_fitness = fitness
                return

    def to_dict(self) -> dict:
        """Serialize for persistence."""
        return {
            "max_size": self.max_size,
            "temperature": self.temperature,
            "entries": [e.model_dump() for e in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict) -> DesignArchive:
        """Restore from persisted data."""
        archive = cls(
            max_size=data.get("max_size", 50),
            temperature=data.get("temperature", 0.3),
        )
        for entry_data in data.get("entries", []):
            archive.entries.append(DesignEntry(**entry_data))
        return archive


class EvalTask(BaseModel):
    """A concrete evaluation task with known correct answers.

    Run in sandbox to produce real fitness scores instead of proxy signals.
    """

    component: str  # target component (e.g. "knowledge.semantic")
    name: str
    test_code: str  # Python code to execute in sandbox
    expected_output: str = ""  # substring expected in output
    weight: float = 1.0  # importance weight for fitness blending


class EvolutionStateData(BaseModel):
    """The full persisted evolution state."""

    instance_id: str = Field(default_factory=new_id)
    agos_version: str = "0.1.0"
    last_saved: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    cycles_completed: int = 0
    strategies_applied: list[AppliedStrategy] = Field(default_factory=list)
    discovered_patterns: list[DiscoveredPattern] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)
    # Meta-evolution state (ALMA-style all-component evolution)
    meta_evolution: dict[str, Any] = Field(default_factory=dict)
    meta_cycles_completed: int = 0
    # ALMA design archive (persisted across restarts)
    design_archive: dict[str, Any] = Field(default_factory=dict)


class EvolutionState:
    """Manages persistence of evolution state to disk.

    save_path defaults to .agos/evolution_state.json
    """

    def __init__(self, save_path: Path | str | None = None) -> None:
        self._path = Path(save_path) if save_path else Path(".agos/evolution_state.json")
        self._data = EvolutionStateData()

    @property
    def data(self) -> EvolutionStateData:
        return self._data

    # ── Capture live parameters ──────────────────────────────────

    def capture_parameters(self, loom: TheLoom) -> dict[str, Any]:
        """Snapshot all evolved parameters from live objects."""
        params: dict[str, Any] = {}
        params["semantic.temperature"] = loom.semantic._temperature
        params["semantic.track_access"] = loom.semantic._track_access
        params["loom.use_layered_recall"] = loom._use_layered_recall
        params["loom.layers"] = [
            {"name": ly.name, "priority": ly.priority, "enabled": ly.enabled}
            for ly in loom._layers
        ]
        return params

    # ── Record events ────────────────────────────────────────────

    def record_integration(
        self,
        strategy_name: str,
        module: str,
        parameters: dict[str, Any] | None = None,
        source_papers: list[dict[str, str]] | None = None,
        sandbox_passed: bool = False,
    ) -> None:
        """Record that a strategy was applied (dedupes by name)."""
        for existing in self._data.strategies_applied:
            if existing.name == strategy_name:
                existing.applied_count += 1
                if parameters:
                    existing.parameters = parameters
                existing.applied_at = datetime.utcnow().isoformat()
                if source_papers:
                    existing.source_papers.extend(source_papers)
                return
        self._data.strategies_applied.append(
            AppliedStrategy(
                name=strategy_name,
                module=module,
                parameters=parameters or {},
                source_papers=source_papers or [],
                sandbox_passed=sandbox_passed,
            )
        )

    def record_pattern(
        self,
        name: str,
        module: str,
        code_snippet: str = "",
        sandbox_output: str = "",
        source_paper: str = "",
    ) -> None:
        """Record a discovered code pattern."""
        # Avoid duplicates
        for existing in self._data.discovered_patterns:
            if existing.name == name and existing.module == module:
                return
        self._data.discovered_patterns.append(
            DiscoveredPattern(
                name=name,
                module=module,
                code_snippet=code_snippet,
                sandbox_output=sandbox_output,
                source_paper=source_paper,
            )
        )

    def increment_cycle(self) -> None:
        self._data.cycles_completed += 1

    # ── Save / Load ─────────────────────────────────────────────

    def save(self, loom: TheLoom | None = None) -> None:
        """Persist current state to disk."""
        if loom is not None:
            self._data.parameters = self.capture_parameters(loom)
        self._data.last_saved = datetime.utcnow().isoformat()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            self._data.model_dump_json(indent=2), encoding="utf-8"
        )
        logger.info("Evolution state saved to %s", self._path)

    def load(self) -> bool:
        """Load state from disk. Returns True if loaded successfully."""
        if not self._path.exists():
            return False
        try:
            raw = self._path.read_text(encoding="utf-8")
            self._data = EvolutionStateData.model_validate_json(raw)
            logger.info(
                "Loaded evolution state: %d strategies, %d cycles",
                len(self._data.strategies_applied),
                self._data.cycles_completed,
            )
            return True
        except Exception as e:
            logger.warning("Failed to load evolution state: %s", e)
            return False

    def restore_parameters(self, loom: TheLoom) -> list[str]:
        """Re-apply persisted parameters to live objects."""
        changes: list[str] = []
        params = self._data.parameters
        if not params:
            return changes

        if "semantic.temperature" in params:
            temp = float(params["semantic.temperature"])
            if temp > 0 and temp != loom.semantic._temperature:
                loom.semantic.set_temperature(temp)
                changes.append(f"Restored semantic temperature={temp}")

        if "semantic.track_access" in params:
            tracking = bool(params["semantic.track_access"])
            if tracking and not loom.semantic._track_access:
                loom.semantic.enable_access_tracking(tracking)
                changes.append(f"Restored access tracking={tracking}")

        if "loom.use_layered_recall" in params:
            layered = bool(params["loom.use_layered_recall"])
            if layered and not loom._use_layered_recall:
                loom.enable_layered_recall(layered)
                changes.append(f"Restored layered recall={layered}")

        if "loom.layers" in params and params["loom.layers"] and not loom._layers:
            for ld in params["loom.layers"]:
                name = ld.get("name", "")
                priority = ld.get("priority", 0)
                if name == "semantic":
                    loom.add_layer("semantic", loom.semantic, priority=priority)
                elif name == "episodic":
                    loom.add_layer("episodic", loom.episodic, priority=priority)
                changes.append(f"Restored layer '{name}' (priority={priority})")

        return changes

    # ── Meta-evolution state ────────────────────────────────────

    def save_meta_state(self, meta_evolver) -> None:
        """Persist meta-evolution genomes and mutations."""
        self._data.meta_evolution = meta_evolver.export_state()
        self._data.meta_cycles_completed = sum(
            g.mutations_applied for g in meta_evolver.all_genomes()
        )

    def restore_meta_state(self, meta_evolver) -> int:
        """Restore meta-evolution state. Returns count of restored genomes."""
        if not self._data.meta_evolution:
            return 0
        meta_evolver.restore_state(self._data.meta_evolution)
        restored = sum(
            1 for g in meta_evolver.all_genomes()
            if g.mutations_applied > 0
        )
        logger.info("Restored meta-evolution: %d genomes with mutations", restored)
        return restored

    # ── Design Archive ─────────────────────────────────────────────

    def save_design_archive(self, archive: DesignArchive) -> None:
        """Persist the design archive into state data."""
        self._data.design_archive = archive.to_dict()

    def restore_design_archive(self) -> DesignArchive:
        """Restore design archive from persisted state."""
        if self._data.design_archive:
            return DesignArchive.from_dict(self._data.design_archive)
        return DesignArchive()

    # ── Export for community contribution ────────────────────────

    def export_contribution(self, evolved_dir: Path | None = None) -> dict:
        """Export state as a community contribution dict.

        Includes actual evolved code files so other instances can
        use them directly without re-discovering the same papers.
        """
        # Collect evolved .py files from disk
        evolved_files: dict[str, str] = {}
        d = evolved_dir or Path(".agos/evolved")
        if d.exists():
            for py_file in sorted(d.glob("*.py")):
                if py_file.name.startswith("_"):
                    continue
                try:
                    evolved_files[py_file.name] = py_file.read_text(encoding="utf-8")
                except Exception:
                    pass

        return {
            "instance_id": self._data.instance_id,
            "agos_version": self._data.agos_version,
            "contributed_at": datetime.utcnow().isoformat(),
            "cycles_completed": self._data.cycles_completed,
            "strategies_applied": [
                s.model_dump() for s in self._data.strategies_applied
            ],
            "discovered_patterns": [
                p.model_dump() for p in self._data.discovered_patterns
            ],
            "meta_evolution": self._data.meta_evolution,
            "meta_cycles_completed": self._data.meta_cycles_completed,
            "evolved_code": evolved_files,
            "design_archive": self._data.design_archive,
        }
