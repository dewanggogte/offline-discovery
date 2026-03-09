"""Tests for agent_lifecycle.py — subprocess management, watchdog, container detection, health, and structured logging."""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from unittest import mock

import pytest

sys.path.insert(0, str(os.path.dirname(os.path.dirname(__file__))))

import agent_lifecycle


# ===================================================================
# A. Container detection
# ===================================================================
class TestContainerDetection:
    """_in_container() should detect Docker, Podman, and Kubernetes."""

    def test_detects_dockerenv(self, tmp_path):
        with mock.patch("os.path.exists") as mock_exists:
            mock_exists.side_effect = lambda p: p == "/.dockerenv"
            assert agent_lifecycle._in_container() is True

    def test_detects_containerenv(self, tmp_path):
        with mock.patch("os.path.exists") as mock_exists:
            mock_exists.side_effect = lambda p: p == "/run/.containerenv"
            assert agent_lifecycle._in_container() is True

    def test_detects_docker_cgroup(self, tmp_path):
        cgroup_file = tmp_path / "cgroup"
        cgroup_file.write_text("12:memory:/docker/abc123\n")
        with mock.patch("os.path.exists", return_value=False):
            with mock.patch("builtins.open", mock.mock_open(read_data="12:memory:/docker/abc123\n")):
                assert agent_lifecycle._in_container() is True

    def test_detects_kubepods_cgroup(self):
        cgroup_data = "11:memory:/kubepods/burstable/pod-xyz\n"
        with mock.patch("agent_lifecycle.os.path.exists", return_value=False):
            with mock.patch("builtins.open", mock.mock_open(read_data=cgroup_data)):
                assert agent_lifecycle._in_container() is True

    def test_detects_containerd_cgroup(self):
        cgroup_data = "0::/system.slice/containerd.service\n"
        with mock.patch("agent_lifecycle.os.path.exists", return_value=False):
            with mock.patch("builtins.open", mock.mock_open(read_data=cgroup_data)):
                assert agent_lifecycle._in_container() is True

    def test_false_on_host_machine(self):
        with mock.patch("os.path.exists", return_value=False):
            with mock.patch("builtins.open", side_effect=FileNotFoundError):
                assert agent_lifecycle._in_container() is False

    def test_false_on_permission_error(self):
        with mock.patch("os.path.exists", return_value=False):
            with mock.patch("builtins.open", side_effect=PermissionError):
                assert agent_lifecycle._in_container() is False

    def test_agent_mode_is_dev_locally(self):
        """On the local machine (no container), mode should be 'dev'."""
        assert agent_lifecycle._AGENT_MODE == "dev"


# ===================================================================
# B. Spawn worker
# ===================================================================
class TestSpawnWorker:
    """_spawn_worker() should launch agent_worker.py with correct args."""

    def test_spawns_with_correct_mode(self):
        with mock.patch("subprocess.Popen") as mock_popen:
            mock_proc = mock.MagicMock()
            mock_proc.pid = 12345
            mock_popen.return_value = mock_proc

            proc = agent_lifecycle._spawn_worker()

            args = mock_popen.call_args[0][0]
            assert args[0] == sys.executable
            assert "agent_worker.py" in args[1]
            assert args[2] == agent_lifecycle._AGENT_MODE
            assert proc.pid == 12345

    def test_uses_project_directory_as_cwd(self):
        with mock.patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = mock.MagicMock(pid=1)
            agent_lifecycle._spawn_worker()

            cwd = mock_popen.call_args[1]["cwd"]
            assert "hyperlocal-discovery" in cwd or os.path.isdir(cwd)

    def test_sets_last_spawn_time(self):
        """_spawn_worker should record when the worker was last started."""
        agent_lifecycle._last_spawn_time = None
        with mock.patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = mock.MagicMock(pid=1)
            before = datetime.now(timezone.utc)
            agent_lifecycle._spawn_worker()
            after = datetime.now(timezone.utc)

        assert agent_lifecycle._last_spawn_time is not None
        assert before <= agent_lifecycle._last_spawn_time <= after


# ===================================================================
# C. Watchdog thread
# ===================================================================
class TestWatchdog:
    """The watchdog should restart the agent worker when it exits."""

    def setup_method(self):
        """Reset module-level state before each test."""
        agent_lifecycle._watchdog_stop.clear()
        agent_lifecycle._agent_proc = None
        agent_lifecycle._restart_count = 0

    def teardown_method(self):
        """Ensure watchdog is stopped after each test."""
        agent_lifecycle._watchdog_stop.set()
        time.sleep(0.1)

    def test_watchdog_restarts_dead_process(self):
        """When the agent process exits, watchdog should respawn it."""
        dead_proc = mock.MagicMock()
        dead_proc.poll.return_value = 1  # exited with code 1
        dead_proc.returncode = 1
        agent_lifecycle._agent_proc = dead_proc

        respawn_count = 0

        def mock_spawn():
            nonlocal respawn_count
            respawn_count += 1
            new_proc = mock.MagicMock()
            new_proc.poll.return_value = None  # alive
            new_proc.pid = 99999
            agent_lifecycle._agent_proc = new_proc
            return new_proc

        fast_event = threading.Event()
        agent_lifecycle._watchdog_stop = fast_event
        wait_calls = 0

        def fast_wait(timeout=None):
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls >= 3:
                fast_event.set()
                return True
            return False

        fast_event.wait = fast_wait

        with mock.patch.object(agent_lifecycle, "_spawn_worker", side_effect=mock_spawn):
            with mock.patch.object(agent_lifecycle, "time"):
                t = threading.Thread(target=agent_lifecycle._watchdog_loop, daemon=True)
                t.start()
                t.join(timeout=5)

        assert respawn_count >= 1, "Watchdog should have respawned the dead process"

    def test_watchdog_increments_restart_count(self):
        """Each restart should increment the global _restart_count."""
        dead_proc = mock.MagicMock()
        dead_proc.poll.return_value = 1
        dead_proc.returncode = 1
        agent_lifecycle._agent_proc = dead_proc

        def mock_spawn():
            new_proc = mock.MagicMock()
            new_proc.poll.return_value = None
            new_proc.pid = 99999
            agent_lifecycle._agent_proc = new_proc
            return new_proc

        fast_event = threading.Event()
        agent_lifecycle._watchdog_stop = fast_event
        wait_calls = 0

        def fast_wait(timeout=None):
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls >= 3:
                fast_event.set()
                return True
            return False

        fast_event.wait = fast_wait

        with mock.patch.object(agent_lifecycle, "_spawn_worker", side_effect=mock_spawn):
            with mock.patch.object(agent_lifecycle, "time"):
                t = threading.Thread(target=agent_lifecycle._watchdog_loop, daemon=True)
                t.start()
                t.join(timeout=5)

        assert agent_lifecycle._restart_count >= 1

    def test_watchdog_does_not_restart_alive_process(self):
        """When the agent process is alive, watchdog should not respawn."""
        alive_proc = mock.MagicMock()
        alive_proc.poll.return_value = None  # still running
        agent_lifecycle._agent_proc = alive_proc

        respawn_count = 0

        def mock_spawn():
            nonlocal respawn_count
            respawn_count += 1
            return mock.MagicMock(pid=1)

        with mock.patch.object(agent_lifecycle, "_spawn_worker", side_effect=mock_spawn):
            t = threading.Thread(target=agent_lifecycle._watchdog_loop, daemon=True)
            t.start()
            time.sleep(0.3)
            agent_lifecycle._watchdog_stop.set()
            t.join(timeout=2)

        assert respawn_count == 0, "Watchdog should NOT respawn a living process"

    def test_watchdog_stops_cleanly(self):
        """Setting _watchdog_stop should exit the loop."""
        alive_proc = mock.MagicMock()
        alive_proc.poll.return_value = None
        agent_lifecycle._agent_proc = alive_proc

        t = threading.Thread(target=agent_lifecycle._watchdog_loop, daemon=True)
        t.start()
        agent_lifecycle._watchdog_stop.set()
        t.join(timeout=3)
        assert not t.is_alive(), "Watchdog thread should have exited"


# ===================================================================
# D. start_agent_worker integration
# ===================================================================
class TestStartAgentWorker:
    """start_agent_worker() should spawn worker + start watchdog."""

    def teardown_method(self):
        agent_lifecycle._watchdog_stop.set()
        time.sleep(0.1)

    def test_starts_worker_and_watchdog(self):
        with mock.patch.object(agent_lifecycle, "_spawn_worker") as mock_spawn:
            mock_spawn.return_value = mock.MagicMock(pid=123)
            with mock.patch("time.sleep"):
                agent_lifecycle._watchdog_stop = threading.Event()
                agent_lifecycle.start_agent_worker()

        assert agent_lifecycle._agent_proc is not None
        assert agent_lifecycle._agent_proc.pid == 123
        agent_lifecycle._watchdog_stop.set()


# ===================================================================
# E. cleanup_agent
# ===================================================================
class TestCleanupAgent:
    """cleanup_agent() should stop watchdog and terminate worker."""

    def test_terminates_running_process(self):
        proc = mock.MagicMock()
        proc.poll.return_value = None  # still running
        proc.wait.return_value = 0
        agent_lifecycle._agent_proc = proc
        agent_lifecycle._watchdog_stop = threading.Event()

        agent_lifecycle.cleanup_agent()

        proc.terminate.assert_called_once()
        assert agent_lifecycle._watchdog_stop.is_set()

    def test_kills_on_timeout(self):
        proc = mock.MagicMock()
        proc.poll.return_value = None
        proc.wait.side_effect = subprocess.TimeoutExpired(cmd="agent", timeout=5)
        agent_lifecycle._agent_proc = proc
        agent_lifecycle._watchdog_stop = threading.Event()

        agent_lifecycle.cleanup_agent()

        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()

    def test_noop_when_already_dead(self):
        proc = mock.MagicMock()
        proc.poll.return_value = 0  # already exited
        agent_lifecycle._agent_proc = proc
        agent_lifecycle._watchdog_stop = threading.Event()

        agent_lifecycle.cleanup_agent()

        proc.terminate.assert_not_called()


# ===================================================================
# F. kill_old_agents
# ===================================================================
class TestKillOldAgents:
    """kill_old_agents() edge cases."""

    def test_handles_missing_pgrep(self):
        """On slim Docker images, pgrep may not exist — should not crash."""
        with mock.patch("subprocess.run", side_effect=FileNotFoundError):
            agent_lifecycle.kill_old_agents()  # should not raise

    def test_handles_no_processes(self):
        """When no agent_worker.py processes are running."""
        result = mock.MagicMock()
        result.stdout = "\n"
        with mock.patch("subprocess.run", return_value=result):
            agent_lifecycle.kill_old_agents()  # should not raise

    def test_kills_other_pids(self):
        """Should SIGTERM other agent_worker.py processes, not itself."""
        result = mock.MagicMock()
        result.stdout = "1111\n2222\n"
        with mock.patch("subprocess.run", return_value=result):
            with mock.patch("os.getpid", return_value=1111):
                with mock.patch("os.kill") as mock_kill:
                    with mock.patch("time.sleep"):
                        agent_lifecycle.kill_old_agents()

                    mock_kill.assert_called_once_with(2222, signal.SIGTERM)


# ===================================================================
# G. agent_health
# ===================================================================
class TestAgentHealth:
    """agent_health() should report worker status for the /healthz endpoint."""

    def test_not_started(self):
        agent_lifecycle._agent_proc = None
        agent_lifecycle._restart_count = 0
        h = agent_lifecycle.agent_health()
        assert h["status"] == "not_started"
        assert h["pid"] is None
        assert h["restart_count"] == 0
        assert h["mode"] == agent_lifecycle._AGENT_MODE

    def test_healthy_worker(self):
        proc = mock.MagicMock()
        proc.poll.return_value = None  # alive
        proc.pid = 5555
        agent_lifecycle._agent_proc = proc
        agent_lifecycle._restart_count = 0
        agent_lifecycle._last_spawn_time = datetime.now(timezone.utc)

        h = agent_lifecycle.agent_health()
        assert h["status"] == "healthy"
        assert h["pid"] == 5555
        assert h["uptime_seconds"] >= 0
        assert "exit_code" not in h or h["exit_code"] is None

    def test_dead_worker(self):
        proc = mock.MagicMock()
        proc.poll.return_value = 137  # killed by OOM
        proc.returncode = 137
        proc.pid = 6666
        agent_lifecycle._agent_proc = proc
        agent_lifecycle._restart_count = 3

        h = agent_lifecycle.agent_health()
        assert h["status"] == "dead"
        assert h["exit_code"] == 137
        assert h["restart_count"] == 3
        assert h["uptime_seconds"] == 0

    def test_uptime_increases(self):
        proc = mock.MagicMock()
        proc.poll.return_value = None
        proc.pid = 7777
        agent_lifecycle._agent_proc = proc
        agent_lifecycle._last_spawn_time = datetime(2020, 1, 1, tzinfo=timezone.utc)

        h = agent_lifecycle.agent_health()
        assert h["uptime_seconds"] > 0  # years of uptime

    def test_restart_count_reflects_watchdog_restarts(self):
        agent_lifecycle._agent_proc = mock.MagicMock()
        agent_lifecycle._agent_proc.poll.return_value = None
        agent_lifecycle._restart_count = 42
        agent_lifecycle._last_spawn_time = datetime.now(timezone.utc)

        h = agent_lifecycle.agent_health()
        assert h["restart_count"] == 42


# ===================================================================
# H. Structured logging
# ===================================================================
class TestStructuredLogging:
    """_log_event() should emit JSON log lines."""

    def test_log_event_emits_json(self, caplog):
        """Log entries should be valid JSON with required fields."""
        import logging
        with caplog.at_level(logging.INFO, logger="callkaro.lifecycle"):
            agent_lifecycle._log_event("test_event", foo="bar", count=42)

        assert len(caplog.records) == 1
        entry = json.loads(caplog.records[0].message)
        assert entry["component"] == "lifecycle"
        assert entry["event"] == "test_event"
        assert entry["foo"] == "bar"
        assert entry["count"] == 42
        # ts should be a valid ISO timestamp
        datetime.fromisoformat(entry["ts"])

    def test_spawn_emits_worker_started(self, caplog):
        """_spawn_worker should log a worker_started event."""
        import logging
        with caplog.at_level(logging.INFO, logger="callkaro.lifecycle"):
            with mock.patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value = mock.MagicMock(pid=1234)
                agent_lifecycle._spawn_worker()

        started_logs = [r for r in caplog.records
                        if "worker_started" in r.message]
        assert len(started_logs) == 1
        entry = json.loads(started_logs[0].message)
        assert entry["pid"] == 1234
        assert entry["mode"] == agent_lifecycle._AGENT_MODE

    def test_watchdog_restart_emits_worker_exited(self, caplog):
        """When watchdog detects a crash, it should log worker_exited."""
        import logging

        dead_proc = mock.MagicMock()
        dead_proc.poll.return_value = 1
        dead_proc.returncode = 1
        agent_lifecycle._agent_proc = dead_proc
        agent_lifecycle._restart_count = 0

        fast_event = threading.Event()
        agent_lifecycle._watchdog_stop = fast_event
        wait_calls = 0

        def fast_wait(timeout=None):
            nonlocal wait_calls
            wait_calls += 1
            if wait_calls >= 3:
                fast_event.set()
                return True
            return False

        fast_event.wait = fast_wait

        def mock_spawn():
            new_proc = mock.MagicMock()
            new_proc.poll.return_value = None
            new_proc.pid = 9999
            agent_lifecycle._agent_proc = new_proc
            return new_proc

        with caplog.at_level(logging.INFO, logger="callkaro.lifecycle"):
            with mock.patch.object(agent_lifecycle, "_spawn_worker", side_effect=mock_spawn):
                with mock.patch.object(agent_lifecycle, "time"):
                    t = threading.Thread(target=agent_lifecycle._watchdog_loop, daemon=True)
                    t.start()
                    t.join(timeout=5)

        exited_logs = [r for r in caplog.records
                       if "worker_exited" in r.message]
        assert len(exited_logs) >= 1
        entry = json.loads(exited_logs[0].message)
        assert entry["exit_code"] == 1
        assert entry["restart_number"] >= 1
        assert "backoff_seconds" in entry


# ===================================================================
# I. Integration test with real subprocess
# ===================================================================
class TestWatchdogIntegration:
    """End-to-end test: spawn a real subprocess, kill it, verify watchdog restarts it.

    Uses a tiny Python script instead of agent_worker.py to avoid LiveKit dependencies.
    """

    def setup_method(self):
        agent_lifecycle._watchdog_stop = threading.Event()
        agent_lifecycle._restart_count = 0

    def teardown_method(self):
        agent_lifecycle._watchdog_stop.set()
        time.sleep(0.2)
        # Clean up any leftover child process
        if agent_lifecycle._agent_proc and agent_lifecycle._agent_proc.poll() is None:
            agent_lifecycle._agent_proc.terminate()
            agent_lifecycle._agent_proc.wait(timeout=3)

    def test_watchdog_restarts_crashed_subprocess(self, tmp_path):
        """Spawn a real process that exits immediately, verify watchdog respawns it."""
        # Write a tiny script: first invocation exits with code 42,
        # subsequent invocations sleep forever (simulating a healthy worker).
        marker = tmp_path / "started_count"
        marker.write_text("0")

        script = tmp_path / "fake_worker.py"
        script.write_text(f"""
import sys, time
marker = "{marker}"
with open(marker, "r") as f:
    count = int(f.read().strip())
with open(marker, "w") as f:
    f.write(str(count + 1))
if count == 0:
    # First run: crash immediately
    sys.exit(42)
else:
    # Second run: stay alive
    time.sleep(60)
""")

        # Patch _spawn_worker to use our fake script instead of agent_worker.py
        def spawn_fake():
            proc = subprocess.Popen(
                [sys.executable, str(script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            agent_lifecycle._last_spawn_time = datetime.now(timezone.utc)
            agent_lifecycle._agent_proc = proc
            return proc

        agent_lifecycle._agent_proc = None

        with mock.patch.object(agent_lifecycle, "_spawn_worker", side_effect=spawn_fake):
            # First spawn: the process will exit(42) immediately
            agent_lifecycle._agent_proc = spawn_fake()

            # Run watchdog with real timing but short intervals
            original_wait = agent_lifecycle._watchdog_stop.wait

            def fast_wait(timeout=None):
                # Speed up: poll every 0.2s, backoff 0.5s
                return original_wait(min(timeout or 0.2, 0.5))

            agent_lifecycle._watchdog_stop.wait = fast_wait

            t = threading.Thread(target=agent_lifecycle._watchdog_loop, daemon=True)
            t.start()

            # Wait for the watchdog to detect the crash and respawn
            deadline = time.time() + 10
            while time.time() < deadline:
                count = int(marker.read_text().strip())
                if count >= 2:
                    break
                time.sleep(0.2)

            agent_lifecycle._watchdog_stop.set()
            t.join(timeout=3)

        # The fake script should have been started at least twice
        final_count = int(marker.read_text().strip())
        assert final_count >= 2, f"Expected >=2 starts, got {final_count}"
        assert agent_lifecycle._restart_count >= 1

        # The second process should be alive (sleeping)
        assert agent_lifecycle._agent_proc.poll() is None, "Respawned process should be alive"

    def test_healthy_subprocess_not_restarted(self, tmp_path):
        """A subprocess that stays alive should not be restarted."""
        marker = tmp_path / "started_count"
        marker.write_text("0")

        script = tmp_path / "healthy_worker.py"
        script.write_text(f"""
import time
marker = "{marker}"
with open(marker, "r") as f:
    count = int(f.read().strip())
with open(marker, "w") as f:
    f.write(str(count + 1))
time.sleep(60)
""")

        proc = subprocess.Popen(
            [sys.executable, str(script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        agent_lifecycle._agent_proc = proc
        agent_lifecycle._last_spawn_time = datetime.now(timezone.utc)

        original_wait = agent_lifecycle._watchdog_stop.wait

        def fast_wait(timeout=None):
            return original_wait(min(timeout or 0.2, 0.3))

        agent_lifecycle._watchdog_stop.wait = fast_wait

        t = threading.Thread(target=agent_lifecycle._watchdog_loop, daemon=True)
        t.start()
        time.sleep(1.5)  # let watchdog run several cycles
        agent_lifecycle._watchdog_stop.set()
        t.join(timeout=3)

        # Should have started exactly once — no restarts
        final_count = int(marker.read_text().strip())
        assert final_count == 1, f"Healthy process should not be restarted, started {final_count} times"
        assert agent_lifecycle._restart_count == 0
