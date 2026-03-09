"""
agent_lifecycle.py — Shared agent worker management
====================================================
Functions for starting, stopping, and finding agent worker processes and logs.
Used by both app.py and test_browser.py.
"""

import glob
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("callkaro.lifecycle")
logger.setLevel(logging.INFO)

# Log to stderr — the standard approach for containerized apps.
# Kubernetes captures stderr as pod logs, queryable via `kubectl logs`.
# No file handler needed: logs inside a pod are ephemeral anyway.
_stderr_handler = logging.StreamHandler(sys.stderr)
_stderr_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S",
))
logger.addHandler(_stderr_handler)

_agent_proc = None
_watchdog_stop = threading.Event()
_restart_count = 0
_last_spawn_time = None  # UTC datetime when the worker was last started

# Use "start" in Docker/production, "dev" for local development.
# Detect containers via /.dockerenv (Docker) or /run/.containerenv (Podman)
# or the presence of a cgroup hint.
def _in_container():
    if os.path.exists("/.dockerenv") or os.path.exists("/run/.containerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "r") as f:
            cgroup_text = f.read()
        return any(k in cgroup_text for k in ("docker", "kubepods", "containerd"))
    except (FileNotFoundError, PermissionError):
        return False

_AGENT_MODE = "start" if _in_container() else "dev"


def kill_old_agents():
    """Kill any existing agent_worker.py processes."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "agent_worker.py"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        # pgrep not available (e.g. slim Docker images) — skip cleanup
        return
    pids = result.stdout.strip().split("\n")
    my_pid = str(os.getpid())
    for pid in pids:
        pid = pid.strip()
        if pid and pid != my_pid:
            print(f"  Killing old agent worker (PID {pid})")
            try:
                os.kill(int(pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
    if any(p.strip() and p.strip() != my_pid for p in pids):
        time.sleep(1)


def _log_event(event: str, **fields):
    """Emit a structured JSON log line for lifecycle events."""
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "component": "lifecycle",
        "event": event,
        **fields,
    }
    logger.info(json.dumps(entry, default=str))


def _spawn_worker():
    """Spawn the agent_worker.py subprocess and return it."""
    global _last_spawn_time
    python = sys.executable
    script = Path(__file__).parent / "agent_worker.py"
    proc = subprocess.Popen(
        [python, str(script), _AGENT_MODE],
        cwd=str(Path(__file__).parent),
        stdout=sys.stderr,
        stderr=sys.stderr,
    )
    _last_spawn_time = datetime.now(timezone.utc)
    print(f"  Agent worker started (PID {proc.pid}, mode={_AGENT_MODE})")
    _log_event("worker_started", pid=proc.pid, mode=_AGENT_MODE)
    return proc


def _watchdog_loop():
    """Monitor the agent worker and restart it if it dies unexpectedly.

    Runs in a daemon thread. Checks every 5 seconds whether the subprocess
    is still alive. If it has exited, waits a short backoff and respawns.
    """
    global _agent_proc, _restart_count
    max_backoff = 30  # seconds

    while not _watchdog_stop.is_set():
        _watchdog_stop.wait(5)  # check every 5s
        if _watchdog_stop.is_set():
            break

        if _agent_proc and _agent_proc.poll() is not None:
            exit_code = _agent_proc.returncode
            _restart_count += 1
            backoff = min(5 * _restart_count, max_backoff)
            print(f"  [watchdog] Agent worker exited (code={exit_code}), "
                  f"restarting in {backoff}s (restart #{_restart_count})...")
            _log_event("worker_exited", exit_code=exit_code,
                       restart_number=_restart_count, backoff_seconds=backoff)
            _watchdog_stop.wait(backoff)
            if _watchdog_stop.is_set():
                break
            _agent_proc = _spawn_worker()
            time.sleep(3)  # give it time to register with LiveKit


def start_agent_worker():
    """Start agent worker and a watchdog thread that auto-restarts on crash."""
    global _agent_proc
    _agent_proc = _spawn_worker()
    time.sleep(3)

    # Start watchdog as daemon thread — dies with main process
    watchdog = threading.Thread(target=_watchdog_loop, daemon=True)
    watchdog.start()


def cleanup_agent():
    """Terminate agent worker and stop watchdog on exit."""
    global _agent_proc
    _watchdog_stop.set()
    if _agent_proc and _agent_proc.poll() is None:
        print(f"\n  Stopping agent worker (PID {_agent_proc.pid})")
        _log_event("worker_stopping", pid=_agent_proc.pid)
        _agent_proc.terminate()
        try:
            _agent_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _agent_proc.kill()
            _log_event("worker_killed", pid=_agent_proc.pid,
                       reason="graceful shutdown timed out")


def agent_health() -> dict:
    """Return health status of the agent worker subprocess.

    Returns a dict with:
      - status: "healthy", "dead", or "not_started"
      - pid: worker PID (if started)
      - uptime_seconds: seconds since last spawn (if alive)
      - restart_count: total watchdog restarts since server start
      - mode: "dev" or "start"
    """
    now = datetime.now(timezone.utc)

    if _agent_proc is None:
        return {"status": "not_started", "pid": None,
                "uptime_seconds": 0, "restart_count": _restart_count,
                "mode": _AGENT_MODE}

    alive = _agent_proc.poll() is None
    uptime = 0
    if alive and _last_spawn_time:
        uptime = int((now - _last_spawn_time).total_seconds())

    return {
        "status": "healthy" if alive else "dead",
        "pid": _agent_proc.pid,
        "exit_code": None if alive else _agent_proc.returncode,
        "uptime_seconds": uptime,
        "restart_count": _restart_count,
        "mode": _AGENT_MODE,
    }


def find_agent_log():
    """Find the most recent LiveKit agent log file."""
    patterns = [
        "/tmp/livekit-agents-*.log",
        "/private/tmp/livekit-agents-*.log",
        os.path.expanduser("~/.livekit/agents/*.log"),
    ]
    task_dir = "/private/tmp/claude-501/-Users-dg-Documents-lab-hyperlocal-discovery/tasks"
    if os.path.isdir(task_dir):
        outputs = sorted(Path(task_dir).glob("*.output"), key=lambda p: p.stat().st_mtime, reverse=True)
        for f in outputs:
            try:
                content = f.read_text(errors="replace")
                if "price-agent" in content or "livekit.agents" in content:
                    return str(f)
            except Exception:
                continue
    for pat in patterns:
        files = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
        if files:
            return files[0]
    return None
