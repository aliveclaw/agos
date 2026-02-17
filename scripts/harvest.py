"""Harvest evolution learnings from all running AGOS fleet nodes.

Connects to each node's API, pulls evolution state and meta-evolution data,
aggregates into a consolidated report, and saves to reports/ directory.

Usage:
    python scripts/harvest.py
    python scripts/harvest.py --nodes http://localhost:8421,http://localhost:8422
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import httpx

DEFAULT_NODES = [
    "http://localhost:8421",
    "http://localhost:8422",
    "http://localhost:8423",
    "http://localhost:8424",
    "http://localhost:8425",
]


def harvest_node(url: str, timeout: float = 10.0) -> dict:
    """Pull evolution data from a single node."""
    result = {"url": url, "status": "error", "error": None}

    try:
        with httpx.Client(timeout=timeout) as client:
            # Get node status
            resp = client.get(f"{url}/api/status")
            resp.raise_for_status()
            result["status_data"] = resp.json()
            result["node_role"] = result["status_data"].get("node_role", "unknown")

            # Get evolution state
            resp = client.get(f"{url}/api/evolution/state")
            resp.raise_for_status()
            result["evolution"] = resp.json()

            # Get meta-evolution state
            resp = client.get(f"{url}/api/evolution/meta")
            resp.raise_for_status()
            result["meta"] = resp.json()

            # Get A2A agent card
            resp = client.get(f"{url}/.well-known/agent-card.json")
            resp.raise_for_status()
            result["agent_card"] = resp.json()

            result["status"] = "ok"
    except httpx.ConnectError:
        result["error"] = "Connection refused (node not running?)"
    except Exception as e:
        result["error"] = str(e)[:200]

    return result


def print_summary(nodes: list[dict]) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 75)
    print("  AGOS Fleet Evolution Harvest Report")
    print("  " + datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
    print("=" * 75)

    total_cycles = 0
    total_strategies = 0
    total_patterns = 0

    for node in nodes:
        url = node["url"]
        if node["status"] != "ok":
            print(f"\n  {url}  [OFFLINE] {node.get('error', '')}")
            continue

        role = node.get("node_role", "?")
        evo = node.get("evolution", {})
        cycles = evo.get("cycles_completed", 0)
        strategies = evo.get("strategies_applied", [])
        patterns = evo.get("discovered_patterns", [])

        total_cycles += cycles
        total_strategies += len(strategies)
        total_patterns += len(patterns)

        print(f"\n  {url}  [{role.upper()}]")
        print(f"    Cycles: {cycles}  |  Strategies: {len(strategies)}  |  Patterns: {len(patterns)}")
        print(f"    Uptime: {node['status_data'].get('uptime_s', 0)}s")

        if strategies:
            print("    Applied strategies:")
            for s in strategies[:5]:
                name = s.get("name", "?")[:50]
                module = s.get("module", "?")
                print(f"      - {name} ({module})")
            if len(strategies) > 5:
                print(f"      ... and {len(strategies) - 5} more")

        if patterns:
            print("    Discovered patterns:")
            for p in patterns[:3]:
                name = p.get("name", "?")[:50]
                print(f"      - {name}")
            if len(patterns) > 3:
                print(f"      ... and {len(patterns) - 3} more")

        # Meta-evolution summary
        meta = node.get("meta", {}) or {}
        genomes = meta.get("genomes", {}) if isinstance(meta, dict) else {}
        if isinstance(genomes, dict):
            genome_items = list(genomes.values())
        elif isinstance(genomes, list):
            genome_items = genomes
        else:
            genome_items = []
        if genome_items:
            mutated = sum(1 for g in genome_items if isinstance(g, dict) and g.get("mutations_applied", 0) > 0)
            print(f"    Meta-evolution: {mutated}/{len(genome_items)} components mutated")

    print("\n" + "-" * 75)
    online = sum(1 for n in nodes if n["status"] == "ok")
    print(f"  Fleet: {online}/{len(nodes)} nodes online")
    print(f"  Total: {total_cycles} cycles | {total_strategies} strategies | {total_patterns} patterns")
    print("=" * 75 + "\n")


def save_report(nodes: list[dict], output_dir: Path) -> Path:
    """Save consolidated report to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"evolution_harvest_{timestamp}.json"

    report = {
        "harvested_at": datetime.utcnow().isoformat(),
        "nodes": nodes,
        "summary": {
            "total_nodes": len(nodes),
            "online_nodes": sum(1 for n in nodes if n["status"] == "ok"),
            "total_cycles": sum(
                n.get("evolution", {}).get("cycles_completed", 0)
                for n in nodes if n["status"] == "ok"
            ),
            "total_strategies": sum(
                len(n.get("evolution", {}).get("strategies_applied", []))
                for n in nodes if n["status"] == "ok"
            ),
            "total_patterns": sum(
                len(n.get("evolution", {}).get("discovered_patterns", []))
                for n in nodes if n["status"] == "ok"
            ),
        },
    }

    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Harvest evolution learnings from AGOS fleet")
    parser.add_argument(
        "--nodes", type=str, default=None,
        help="Comma-separated node URLs (default: localhost:8421-8425)",
    )
    parser.add_argument(
        "--output", type=str, default="reports",
        help="Output directory for harvest reports (default: reports/)",
    )
    args = parser.parse_args()

    node_urls = args.nodes.split(",") if args.nodes else DEFAULT_NODES
    node_urls = [u.strip().rstrip("/") for u in node_urls if u.strip()]

    print(f"Harvesting from {len(node_urls)} nodes...")

    nodes = []
    for url in node_urls:
        print(f"  Connecting to {url}...", end=" ", flush=True)
        result = harvest_node(url)
        status = "OK" if result["status"] == "ok" else "FAIL"
        print(status)
        nodes.append(result)

    print_summary(nodes)

    output_path = save_report(nodes, Path(args.output))
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
