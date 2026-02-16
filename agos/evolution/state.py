"""Evolution state persistence — save/load/export evolved parameters.

Captures the runtime parameters modified by integration strategies,
persists them to .agos/evolution_state.json, and restores them on boot.
"""

from __future__ import annotations

import json
import logging
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


class EvolutionStateData(BaseModel):
    """The full persisted evolution state."""

    instance_id: str = Field(default_factory=new_id)
    agos_version: str = "0.1.0"
    last_saved: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    cycles_completed: int = 0
    strategies_applied: list[AppliedStrategy] = Field(default_factory=list)
    discovered_patterns: list[DiscoveredPattern] = Field(default_factory=list)
    parameters: dict[str, Any] = Field(default_factory=dict)


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

    # ── Export for community contribution ────────────────────────

    def export_contribution(self) -> dict:
        """Export state as a community contribution dict."""
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
        }
