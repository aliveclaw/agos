"""IronClaw — Security-focused AI agent.

Simulates a security-hardened agent workload on AGOS:
- Scans for vulnerabilities in running processes
- Monitors file system access patterns
- Detects anomalous token consumption
- Enforces access control policies
- Reports security events to stdout for OS monitoring
"""

import hashlib
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────
SCAN_INTERVAL = 12  # seconds between security scans
MAX_TOKEN_RATE = 500  # tokens/minute alert threshold

# ── State ──────────────────────────────────────────────────────
total_tokens = 0
scan_count = 0
alerts_raised = 0
start_time = time.time()
audit_log: list[dict] = []
file_access_history: dict[str, int] = {}
token_history: list[tuple[float, int]] = []  # (timestamp, tokens)


def emit(event: dict) -> None:
    """Emit structured event to stdout."""
    event["agent"] = "ironclaw"
    event["uptime_s"] = int(time.time() - start_time)
    event["tokens_total"] = total_tokens
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        event["memory_mb"] = round(usage.ru_maxrss / 1024, 1)
    except (ImportError, Exception):
        pass
    print(json.dumps(event), flush=True)


# ── Security scan tasks ────────────────────────────────────────

def scan_process_table() -> list[str]:
    """Scan /proc for suspicious processes."""
    findings = []
    proc_dir = Path("/proc")
    if not proc_dir.exists():
        return ["proc filesystem not available (non-Linux)"]

    process_count = 0
    high_memory = []

    for entry in proc_dir.iterdir():
        if not entry.name.isdigit():
            continue
        process_count += 1
        try:
            status = (entry / "status").read_text()
            for line in status.split("\n"):
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    if rss_kb > 200_000:  # >200MB
                        name_line = [l for l in status.split("\n") if l.startswith("Name:")]
                        proc_name = name_line[0].split(":")[1].strip() if name_line else "unknown"
                        high_memory.append(f"{proc_name}({entry.name}): {rss_kb//1024}MB")
        except (PermissionError, FileNotFoundError, ValueError):
            continue

    findings.append(f"processes_scanned: {process_count}")
    if high_memory:
        findings.append(f"high_memory_processes: {', '.join(high_memory[:5])}")
        return findings + [f"ALERT: {len(high_memory)} processes using >200MB RAM"]

    return findings


def scan_file_access() -> list[str]:
    """Monitor key file access patterns."""
    findings = []
    sensitive_paths = [
        "/etc/passwd", "/etc/shadow", "/etc/hosts",
        "/app/.agos/agos.db", "/app/.agos/knowledge.db",
        "/root/.ssh", "/tmp",
    ]

    for fpath in sensitive_paths:
        try:
            if os.path.exists(fpath):
                stat = os.stat(fpath)
                mtime = stat.st_mtime
                prev = file_access_history.get(fpath, mtime)
                if mtime != prev:
                    findings.append(f"ALERT: {fpath} modified since last scan")
                file_access_history[fpath] = mtime
        except (PermissionError, OSError):
            continue

    # Check /tmp for unexpected growth
    try:
        tmp_files = list(Path("/tmp").iterdir())
        if len(tmp_files) > 100:
            findings.append(f"WARNING: /tmp has {len(tmp_files)} files")
    except Exception:
        pass

    return findings


def scan_token_rates() -> list[str]:
    """Check for token consumption anomalies."""
    findings = []
    now = time.time()

    # Simulate token usage for this scan
    tokens_used = random.randint(20, 80)
    global total_tokens
    total_tokens += tokens_used
    token_history.append((now, tokens_used))

    # Keep last 5 minutes of history
    cutoff = now - 300
    token_history[:] = [(t, n) for t, n in token_history if t > cutoff]

    # Calculate rate
    if len(token_history) > 1:
        window = now - token_history[0][0]
        if window > 0:
            total_in_window = sum(n for _, n in token_history)
            rate_per_minute = total_in_window / (window / 60)
            if rate_per_minute > MAX_TOKEN_RATE:
                findings.append(f"ALERT: token_rate {rate_per_minute:.0f}/min exceeds threshold {MAX_TOKEN_RATE}/min")

    return findings


def scan_network_exposure() -> list[str]:
    """Check for unexpected network listeners."""
    findings = []
    try:
        tcp = Path("/proc/net/tcp")
        if tcp.exists():
            lines = tcp.read_text().strip().split("\n")[1:]  # Skip header
            listeners = [l for l in lines if "0A" in l.split()[3:4]]  # LISTEN state
            findings.append(f"tcp_listeners: {len(listeners)}")
            if len(listeners) > 10:
                findings.append(f"WARNING: {len(listeners)} TCP listeners (expected <10)")
    except Exception:
        pass
    return findings


def compute_audit_hash(entries: list[dict]) -> str:
    """Hash-chain the audit log for tamper detection."""
    chain = ""
    for entry in entries:
        data = json.dumps(entry, sort_keys=True)
        chain = hashlib.sha256((chain + data).encode()).hexdigest()
    return chain


# ── Main scan loop ─────────────────────────────────────────────

def run_scan():
    global scan_count, alerts_raised
    scan_count += 1

    all_findings: list[str] = []
    alerts: list[str] = []

    # Run all security scans
    for scan_name, scan_fn in [
        ("process_table", scan_process_table),
        ("file_access", scan_file_access),
        ("token_rates", scan_token_rates),
        ("network_exposure", scan_network_exposure),
    ]:
        findings = scan_fn()
        scan_alerts = [f for f in findings if f.startswith("ALERT:") or f.startswith("WARNING:")]
        alerts.extend(scan_alerts)
        all_findings.extend(findings)

    alerts_raised += len(alerts)

    # Log to audit
    audit_entry = {
        "scan_id": scan_count,
        "timestamp": time.time(),
        "findings": len(all_findings),
        "alerts": len(alerts),
    }
    audit_log.append(audit_entry)
    if len(audit_log) > 100:
        audit_log[:] = audit_log[-100:]

    emit({
        "event": "scan_completed",
        "scan_id": scan_count,
        "findings_count": len(all_findings),
        "alerts_count": len(alerts),
        "alerts": alerts[:5],
        "total_alerts": alerts_raised,
        "audit_hash": compute_audit_hash(audit_log[-10:]),
        "tokens_used": random.randint(20, 80),
    })


def main():
    emit({
        "event": "started",
        "pid": os.getpid(),
        "description": "Security-focused AI agent",
        "scan_interval_s": SCAN_INTERVAL,
    })

    running = True

    def handle_signal(sig, frame):
        nonlocal running
        running = False
        emit({
            "event": "shutting_down",
            "total_scans": scan_count,
            "total_alerts": alerts_raised,
        })
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    while running:
        try:
            run_scan()
        except Exception as e:
            emit({"event": "error", "error": str(e)[:200]})
        time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
