"""Live fleet status monitor for AGOS multi-node deployment.

Polls all nodes every N seconds and displays a rolling status table.

Usage:
    python scripts/monitor.py
    python scripts/monitor.py --interval 5
    python scripts/monitor.py --nodes http://localhost:8421,http://localhost:8422
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

import httpx

DEFAULT_NODES = [
    ("http://localhost:8421", "knowledge"),
    ("http://localhost:8422", "intent"),
    ("http://localhost:8423", "orchestration"),
    ("http://localhost:8424", "policy"),
    ("http://localhost:8425", "general"),
]


def poll_node(url: str, timeout: float = 5.0) -> dict:
    """Poll a single node for status."""
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{url}/api/status")
            resp.raise_for_status()
            data = resp.json()
            return {
                "online": True,
                "role": data.get("node_role", "?"),
                "cycles": data.get("evolution_cycles", 0),
                "agents_running": data.get("agents_running", 0),
                "agents_total": data.get("agents_total", 0),
                "uptime": data.get("uptime_s", 0),
                "audit": data.get("audit_entries", 0),
                "knowledge": data.get("knowledge_available", False),
            }
    except Exception:
        return {"online": False}


def format_uptime(seconds: int) -> str:
    """Format seconds into human-readable uptime."""
    if seconds < 60:
        return f"{seconds}s"
    elif seconds < 3600:
        return f"{seconds // 60}m{seconds % 60}s"
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f"{h}h{m}m"


def clear_screen() -> None:
    """Clear terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


def display(nodes: list[tuple[str, str]], results: list[dict], iteration: int) -> None:
    """Display the status table."""
    clear_screen()
    now = datetime.utcnow().strftime("%H:%M:%S UTC")
    print(f"  AGOS Fleet Monitor  |  {now}  |  Refresh #{iteration}")
    print("=" * 90)
    print(f"  {'Node':<22} {'Role':<15} {'Status':<10} {'Cycles':<8} {'Agents':<10} {'Audit':<8} {'Uptime':<10}")
    print("-" * 90)

    online_count = 0
    total_cycles = 0

    for (url, expected_role), result in zip(nodes, results):
        short_url = url.replace("http://localhost:", ":")
        if result["online"]:
            online_count += 1
            role = result["role"]
            cycles = result["cycles"]
            total_cycles += cycles
            agents = f"{result['agents_running']}/{result['agents_total']}"
            audit = result["audit"]
            uptime = format_uptime(result["uptime"])
            status = "ONLINE"
            print(f"  {short_url:<22} {role:<15} {status:<10} {cycles:<8} {agents:<10} {audit:<8} {uptime:<10}")
        else:
            print(f"  {short_url:<22} {expected_role:<15} {'OFFLINE':<10} {'-':<8} {'-':<10} {'-':<8} {'-':<10}")

    print("-" * 90)
    print(f"  Fleet: {online_count}/{len(nodes)} online  |  Total evolution cycles: {total_cycles}")
    print("=" * 90)
    print("\n  Press Ctrl+C to stop monitoring")


def main() -> None:
    parser = argparse.ArgumentParser(description="Live AGOS fleet monitor")
    parser.add_argument("--interval", type=int, default=10, help="Poll interval in seconds (default: 10)")
    parser.add_argument("--nodes", type=str, default=None, help="Comma-separated node URLs")
    args = parser.parse_args()

    if args.nodes:
        urls = [u.strip().rstrip("/") for u in args.nodes.split(",") if u.strip()]
        nodes = [(u, "?") for u in urls]
    else:
        nodes = DEFAULT_NODES

    print(f"Monitoring {len(nodes)} AGOS nodes (every {args.interval}s)...")
    print("Press Ctrl+C to stop.\n")

    iteration = 0
    try:
        while True:
            iteration += 1
            results = [poll_node(url) for url, _ in nodes]
            display(nodes, results, iteration)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


if __name__ == "__main__":
    main()
