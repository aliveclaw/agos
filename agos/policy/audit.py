"""Audit Trail — immutable log of every agent action.

Every tool call, policy check, and agent lifecycle event gets
recorded here. The audit log is append-only and timestamped.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

import aiosqlite
from pydantic import BaseModel, Field

from agos.types import new_id


class AuditEntry(BaseModel):
    """A single audit log entry."""

    id: str = Field(default_factory=new_id)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent_id: str = ""
    agent_name: str = ""
    action: str = ""  # "tool_call", "policy_check", "state_change", etc.
    detail: str = ""
    tool_name: str = ""
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    success: bool = True
    policy_violation: str = ""


class AuditTrail:
    """Append-only audit log backed by SQLite."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._entries: list[AuditEntry] = []
        self._lock = asyncio.Lock()
        self._db: aiosqlite.Connection | None = None

    async def initialize(self) -> None:
        """Create the audit table if needed."""
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("""
            CREATE TABLE IF NOT EXISTS audit_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                agent_id TEXT,
                agent_name TEXT,
                action TEXT NOT NULL,
                detail TEXT,
                tool_name TEXT,
                arguments TEXT,
                result TEXT,
                success INTEGER DEFAULT 1,
                policy_violation TEXT
            )
        """)
        await self._db.commit()

    async def record(self, entry: AuditEntry) -> None:
        """Record an audit entry (immutable append)."""
        async with self._lock:
            self._entries.append(entry)
            if self._db:
                await self._db.execute(
                    """INSERT INTO audit_log
                       (id, timestamp, agent_id, agent_name, action,
                        detail, tool_name, arguments, result, success,
                        policy_violation)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        entry.id,
                        entry.timestamp.isoformat(),
                        entry.agent_id,
                        entry.agent_name,
                        entry.action,
                        entry.detail,
                        entry.tool_name,
                        str(entry.arguments),
                        entry.result,
                        int(entry.success),
                        entry.policy_violation,
                    ),
                )
                await self._db.commit()

    async def log_tool_call(
        self,
        agent_id: str,
        agent_name: str,
        tool_name: str,
        arguments: dict,
        result: str = "",
        success: bool = True,
    ) -> AuditEntry:
        """Convenience: log a tool call."""
        entry = AuditEntry(
            agent_id=agent_id,
            agent_name=agent_name,
            action="tool_call",
            detail=f"Called {tool_name}",
            tool_name=tool_name,
            arguments=arguments,
            result=result[:500],
            success=success,
        )
        await self.record(entry)
        return entry

    async def log_policy_violation(
        self,
        agent_id: str,
        agent_name: str,
        tool_name: str,
        violation: str,
    ) -> AuditEntry:
        """Convenience: log a policy violation."""
        entry = AuditEntry(
            agent_id=agent_id,
            agent_name=agent_name,
            action="policy_violation",
            detail=f"Blocked: {tool_name}",
            tool_name=tool_name,
            success=False,
            policy_violation=violation,
        )
        await self.record(entry)
        return entry

    async def log_state_change(
        self,
        agent_id: str,
        agent_name: str,
        from_state: str,
        to_state: str,
    ) -> AuditEntry:
        """Convenience: log an agent state transition."""
        entry = AuditEntry(
            agent_id=agent_id,
            agent_name=agent_name,
            action="state_change",
            detail=f"{from_state} -> {to_state}",
        )
        await self.record(entry)
        return entry

    async def query(
        self,
        agent_id: str = "",
        action: str = "",
        limit: int = 50,
    ) -> list[AuditEntry]:
        """Query the audit log with filters."""
        # In-memory fast path
        results = self._entries

        if agent_id:
            results = [e for e in results if e.agent_id == agent_id]
        if action:
            results = [e for e in results if e.action == action]

        # Most recent first
        results = sorted(results, key=lambda e: e.timestamp, reverse=True)
        return results[:limit]

    async def count(self) -> int:
        """Total number of audit entries."""
        return len(self._entries)

    async def violations(self, limit: int = 50) -> list[AuditEntry]:
        """Get recent policy violations."""
        return await self.query(action="policy_violation", limit=limit)

    def __repr__(self) -> str:
        return f"AuditTrail(entries={len(self._entries)})"
