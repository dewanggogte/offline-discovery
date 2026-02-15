"""
test_browser.py — Browser-Based Voice Agent Test
=================================================
Talk to the AC price enquiry agent from your browser (WebRTC).
No SIP trunk or phone calls needed.

Usage:
  1. Start this server:  python test_browser.py
     (automatically kills old agent workers and starts a fresh one)
  2. Open http://localhost:8080 in your browser
  3. Click "Start Conversation" and allow microphone access
"""

import asyncio
import atexit
import glob
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from dotenv import load_dotenv

# Import dashboard metrics parsers
from dashboard import parse_transcripts, parse_logs, compute_metrics, run_tests

load_dotenv(".env.local")

LIVEKIT_URL = os.environ["LIVEKIT_URL"]
LIVEKIT_API_KEY = os.environ["LIVEKIT_API_KEY"]
LIVEKIT_API_SECRET = os.environ["LIVEKIT_API_SECRET"]

PORT = 8080

# ---------------------------------------------------------------------------
# HTML page served at GET /
# ---------------------------------------------------------------------------
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>AC Price Agent</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f172a; color: #e2e8f0;
      min-height: 100vh; padding: 0;
    }

    /* ---- Tab bar ---- */
    .tab-bar {
      display: flex; background: #1e293b; border-bottom: 1px solid #334155;
      position: sticky; top: 0; z-index: 100;
    }
    .tab-btn {
      padding: .75rem 1.5rem; border: none; background: none; color: #64748b;
      font-size: .9rem; font-weight: 600; cursor: pointer; transition: .15s;
      border-bottom: 2px solid transparent;
    }
    .tab-btn:hover { color: #e2e8f0; }
    .tab-btn.active { color: #e2e8f0; border-bottom-color: #3b82f6; }
    .tab-content { display: none; }
    .tab-content.active { display: block; }

    /* ---- Voice tab ---- */
    .voice-wrap { display: flex; justify-content: center; padding: 2rem 1rem; }
    .card {
      background: #1e293b; border-radius: 12px; padding: 2rem;
      max-width: 540px; width: 100%; box-shadow: 0 4px 24px rgba(0,0,0,.4);
    }
    h1 { font-size: 1.4rem; margin-bottom: .25rem; }
    .subtitle { color: #94a3b8; font-size: .85rem; margin-bottom: 1.5rem; }
    .btn {
      display: inline-block; padding: .75rem 1.5rem; border: none; border-radius: 8px;
      font-size: 1rem; font-weight: 600; cursor: pointer; transition: .15s;
      width: 100%; margin-bottom: .5rem;
    }
    .btn:disabled { opacity: .4; cursor: not-allowed; }
    .btn-start { background: #22c55e; color: #fff; }
    .btn-start:hover:not(:disabled) { background: #16a34a; }
    .btn-end { background: #ef4444; color: #fff; }
    .btn-end:hover:not(:disabled) { background: #dc2626; }
    #status {
      margin-top: 1rem; padding: .75rem; border-radius: 8px;
      background: #0f172a; font-size: .85rem; min-height: 2.5rem;
      line-height: 1.5;
    }
    .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
           margin-right: 6px; vertical-align: middle; }
    .dot-idle { background: #64748b; }
    .dot-connecting { background: #f59e0b; animation: pulse 1s infinite; }
    .dot-connected { background: #22c55e; }
    .dot-error { background: #ef4444; }
    @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: .4; } }
    .viz-row { display: flex; gap: 1rem; margin-top: 1rem; }
    .viz-box { flex: 1; background: #0f172a; border-radius: 8px; padding: .5rem; text-align: center; }
    .viz-label { font-size: .7rem; color: #64748b; margin-bottom: .25rem; text-transform: uppercase; letter-spacing: .05em; }
    .viz-box canvas { width: 100%; height: 48px; display: block; border-radius: 4px; }
    #log {
      margin-top: 1rem; background: #0f172a; border-radius: 8px;
      padding: .5rem .75rem; max-height: 180px; overflow-y: auto;
      font-family: "SF Mono", "Fira Code", monospace; font-size: .7rem;
      line-height: 1.6; color: #94a3b8;
    }
    #log:empty { display: none; }
    .log-event { color: #38bdf8; }
    .log-error { color: #f87171; }
    .log-info { color: #a3e635; }

    /* ---- Dashboard tab ---- */
    .dash-wrap { padding: 20px; max-width: 1200px; margin: 0 auto; }
    .dash-wrap h2 { font-size: 1.1rem; margin-bottom: 12px; color: #94a3b8; font-weight: 500; }
    .dgrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 20px; }
    .dcard { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }
    .dcard-wide { grid-column: 1 / -1; }
    .stat { font-size: 2rem; font-weight: 700; color: #f8fafc; }
    .stat-label { font-size: .85rem; color: #64748b; margin-top: 4px; }
    .stat-row { display: flex; gap: 24px; margin-top: 12px; }
    .stat-item { text-align: center; }
    .stat-item .value { font-size: 1.2rem; font-weight: 600; color: #cbd5e1; }
    .stat-item .label { font-size: .75rem; color: #64748b; }
    .badge { display: inline-block; padding: 4px 12px; border-radius: 6px; font-size: .85rem; font-weight: 600; }
    .badge-green { background: #166534; color: #86efac; }
    .badge-red { background: #991b1b; color: #fca5a5; }
    .badge-yellow { background: #854d0e; color: #fde047; }
    .dash-wrap table { width: 100%; border-collapse: collapse; font-size: .85rem; }
    .dash-wrap th { text-align: left; padding: 8px; color: #64748b; border-bottom: 1px solid #334155; }
    .dash-wrap td { padding: 8px; border-bottom: 1px solid #1e293b; }
    .dash-wrap pre { background: #0f172a; padding: 12px; border-radius: 8px; font-size: .8rem; overflow-x: auto; max-height: 300px; overflow-y: auto; color: #94a3b8; }
    .dash-wrap canvas { max-height: 200px; }
    .dash-loading { text-align: center; padding: 3rem; color: #64748b; }
    .dash-refresh {
      background: #334155; color: #e2e8f0; border: none; padding: 6px 14px;
      border-radius: 6px; cursor: pointer; font-size: .8rem; margin-bottom: 16px;
    }
    .dash-refresh:hover { background: #475569; }
  </style>
</head>
<body>

  <!-- Tab bar -->
  <div class="tab-bar">
    <button class="tab-btn active" onclick="switchTab('voice')">Voice Test</button>
    <button class="tab-btn" onclick="switchTab('dashboard')">Dashboard</button>
  </div>

  <!-- Voice Test Tab -->
  <div id="tab-voice" class="tab-content active">
    <div class="voice-wrap">
      <div class="card">
        <h1>AC Price Agent</h1>
        <p class="subtitle">Browser voice test — speak Hindi/Hinglish</p>

        <button class="btn btn-start" id="btnStart" onclick="startConversation()">
          Start Conversation
        </button>
        <button class="btn btn-end" id="btnEnd" onclick="endConversation()" disabled>
          End Conversation
        </button>

        <div id="status"><span class="dot dot-idle"></span>Ready</div>

        <div class="viz-row">
          <div class="viz-box">
            <div class="viz-label">Your Mic</div>
            <canvas id="micViz"></canvas>
          </div>
          <div class="viz-box">
            <div class="viz-label">Agent Audio</div>
            <canvas id="agentViz"></canvas>
          </div>
        </div>

        <div id="log"></div>

        <details style="margin-top:1rem">
          <summary style="cursor:pointer; color:#64748b; font-size:.8rem;">Agent Worker Logs (click to load)</summary>
          <div style="margin-top:.5rem; display:flex; gap:.5rem;">
            <button onclick="loadAgentLogs()" style="background:#334155; color:#e2e8f0; border:none; border-radius:4px; padding:.3rem .75rem; cursor:pointer; font-size:.7rem;">Refresh</button>
            <label style="color:#64748b; font-size:.7rem; display:flex; align-items:center; gap:.3rem;">
              <input type="checkbox" id="autoRefresh" onchange="toggleAutoRefresh()"> Auto-refresh
            </label>
          </div>
          <pre id="agentLogs" style="margin-top:.5rem; background:#020617; border-radius:8px; padding:.5rem; max-height:300px; overflow:auto; font-size:.65rem; color:#94a3b8; line-height:1.4; white-space:pre-wrap; word-break:break-all;"></pre>
        </details>
      </div>
    </div>
  </div>

  <!-- Dashboard Tab -->
  <div id="tab-dashboard" class="tab-content">
    <div class="dash-wrap">
      <button class="dash-refresh" onclick="loadDashboard()">Refresh Dashboard</button>
      <div id="dash-content"><div class="dash-loading">Click the Dashboard tab to load metrics...</div></div>
    </div>
  </div>

  <script src="https://cdn.jsdelivr.net/npm/livekit-client/dist/livekit-client.umd.js"></script>
  <script>
    /* ---- Tab switching ---- */
    let dashLoaded = false;
    function switchTab(name) {
      document.querySelectorAll('.tab-btn').forEach((b, i) => {
        b.classList.toggle('active', (name === 'voice' && i === 0) || (name === 'dashboard' && i === 1));
      });
      document.getElementById('tab-voice').classList.toggle('active', name === 'voice');
      document.getElementById('tab-dashboard').classList.toggle('active', name === 'dashboard');
      if (name === 'dashboard' && !dashLoaded) loadDashboard();
    }

    /* ---- Dashboard ---- */
    let ttftChartInst = null, tokenChartInst = null;
    async function loadDashboard() {
      const el = document.getElementById('dash-content');
      el.innerHTML = '<div class="dash-loading">Loading metrics & running tests...</div>';
      try {
        const resp = await fetch('/api/metrics');
        const d = await resp.json();
        dashLoaded = true;
        renderDashboard(d, el);
      } catch(e) {
        el.innerHTML = '<div class="dash-loading" style="color:#f87171">Failed to load: ' + e.message + '</div>';
      }
    }

    function renderDashboard(d, el) {
      const m = d.metrics;
      const t = d.tests;
      const transcripts = d.transcripts;
      const tRows = transcripts.slice(-10).map(tr =>
        `<tr><td>${tr._filename}</td><td>${tr.store_name}</td><td>${tr._total_messages}</td><td>${tr._duration_seconds}s</td><td>${tr.phone}</td></tr>`
      ).join('') || '<tr><td colspan="5" style="color:#64748b">No transcripts</td></tr>';

      const testOut = (t.output || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

      el.innerHTML = `
        <div class="dgrid">
          <div class="dcard">
            <h2>Test Results</h2>
            <div style="display:flex;gap:12px;align-items:center;">
              <span class="badge badge-green">${t.passed} passed</span>
              <span class="badge ${t.failed > 0 ? 'badge-red' : 'badge-green'}">${t.failed} failed</span>
              <span class="badge badge-yellow">${t.skipped} skipped</span>
            </div>
          </div>
          <div class="dcard">
            <h2>Calls Recorded</h2>
            <div class="stat">${m.total_calls}</div>
            <div class="stat-label">Total conversations</div>
          </div>
          <div class="dcard">
            <h2>Errors</h2>
            <div class="stat" style="color:${m.total_errors > 0 ? '#ef4444' : '#22c55e'}">${m.total_errors}</div>
            <div class="stat-label">Across all log files</div>
          </div>
        </div>
        <div class="dgrid">
          <div class="dcard">
            <h2>Time to First Token (TTFT)</h2>
            <div class="stat">${m.ttft.avg}s</div>
            <div class="stat-label">Average</div>
            <div class="stat-row">
              <div class="stat-item"><div class="value">${m.ttft.p50}s</div><div class="label">P50</div></div>
              <div class="stat-item"><div class="value">${m.ttft.p95}s</div><div class="label">P95</div></div>
              <div class="stat-item"><div class="value">${m.ttft.min}s</div><div class="label">Min</div></div>
              <div class="stat-item"><div class="value">${m.ttft.max}s</div><div class="label">Max</div></div>
            </div>
          </div>
          <div class="dcard">
            <h2>LLM Response Duration</h2>
            <div class="stat">${m.llm_duration.avg}s</div>
            <div class="stat-label">Average</div>
            <div class="stat-row">
              <div class="stat-item"><div class="value">${m.llm_duration.p50}s</div><div class="label">P50</div></div>
              <div class="stat-item"><div class="value">${m.llm_duration.p95}s</div><div class="label">P95</div></div>
            </div>
          </div>
          <div class="dcard">
            <h2>Turn Latency</h2>
            <div class="stat">${m.turn_latency.avg}s</div>
            <div class="stat-label">User speech to agent response</div>
            <div class="stat-row">
              <div class="stat-item"><div class="value">${m.turn_latency.p50}s</div><div class="label">P50</div></div>
              <div class="stat-item"><div class="value">${m.turn_latency.p95}s</div><div class="label">P95</div></div>
            </div>
          </div>
        </div>
        <div class="dgrid">
          <div class="dcard"><h2>TTFT Distribution</h2><canvas id="dashTtft"></canvas></div>
          <div class="dcard"><h2>Token Usage per Turn</h2><canvas id="dashTokens"></canvas></div>
        </div>
        <div class="dgrid">
          <div class="dcard">
            <h2>Conversation Duration</h2>
            <div class="stat">${m.conv_duration.avg}s</div>
            <div class="stat-label">Average call length</div>
          </div>
          <div class="dcard">
            <h2>Messages per Call</h2>
            <div class="stat">${m.msg_counts.avg}</div>
            <div class="stat-label">Average</div>
          </div>
          <div class="dcard">
            <h2>Token Efficiency</h2>
            <div class="stat">${m.prompt_tokens.avg}</div>
            <div class="stat-label">Avg prompt tokens/turn</div>
            <div class="stat-row">
              <div class="stat-item"><div class="value">${m.completion_tokens.avg}</div><div class="label">Avg completion</div></div>
              <div class="stat-item"><div class="value">${m.prompt_tokens.max}</div><div class="label">Max prompt</div></div>
            </div>
          </div>
        </div>
        <div class="dgrid">
          <div class="dcard dcard-wide">
            <h2>Recent Transcripts</h2>
            <table><tr><th>File</th><th>Store</th><th>Messages</th><th>Duration</th><th>Channel</th></tr>${tRows}</table>
          </div>
        </div>
        <div class="dgrid">
          <div class="dcard dcard-wide">
            <h2>Test Output</h2>
            <pre>${testOut}</pre>
          </div>
        </div>
      `;

      // Render charts
      const ttftData = m.all_ttfts || [];
      if (ttftData.length > 0) {
        if (ttftChartInst) ttftChartInst.destroy();
        ttftChartInst = new Chart(document.getElementById('dashTtft'), {
          type: 'bar',
          data: { labels: ttftData.map((_,i) => 'T'+(i+1)), datasets: [{ label: 'TTFT (s)', data: ttftData, backgroundColor: '#3b82f6', borderRadius: 4 }] },
          options: { responsive: true, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, grid: { color: '#1e293b' }, ticks: { color: '#64748b' } }, x: { grid: { display: false }, ticks: { color: '#64748b', maxRotation: 0, autoSkip: true, maxTicksLimit: 20 } } } }
        });
      }
      const pt = m.all_prompt_tokens || [], ct = m.all_completion_tokens || [];
      if (pt.length > 0) {
        if (tokenChartInst) tokenChartInst.destroy();
        tokenChartInst = new Chart(document.getElementById('dashTokens'), {
          type: 'bar',
          data: { labels: pt.map((_,i) => 'T'+(i+1)), datasets: [
            { label: 'Prompt', data: pt, backgroundColor: '#6366f1', borderRadius: 4 },
            { label: 'Completion', data: ct, backgroundColor: '#22c55e', borderRadius: 4 }
          ]},
          options: { responsive: true, plugins: { legend: { labels: { color: '#94a3b8' } } }, scales: { y: { stacked: true, beginAtZero: true, grid: { color: '#1e293b' }, ticks: { color: '#64748b' } }, x: { stacked: true, grid: { display: false }, ticks: { color: '#64748b', maxRotation: 0, autoSkip: true, maxTicksLimit: 20 } } } }
        });
      }
    }

    /* ---- Voice test ---- */
    const { Room, RoomEvent, Track, ParticipantEvent } = LivekitClient;
    let room = null;
    let micAnalyser = null, agentAnalyser = null;
    let vizRAF = null;
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();

    function setStatus(text, state) {
      document.getElementById('status').innerHTML =
        `<span class="dot dot-${state}"></span>${text}`;
    }

    function log(msg, cls = 'log-event') {
      const el = document.getElementById('log');
      const ts = new Date().toLocaleTimeString('en', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
      el.innerHTML += `<div class="${cls}"><span style="color:#475569">${ts}</span> ${msg}</div>`;
      el.scrollTop = el.scrollHeight;
    }

    function createAnalyser(mediaStream) {
      const src = audioCtx.createMediaStreamSource(mediaStream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      src.connect(analyser);
      return analyser;
    }

    function drawBars(canvas, analyser, color) {
      if (!analyser) return;
      const ctx = canvas.getContext('2d');
      const W = canvas.width, H = canvas.height;
      const buf = new Uint8Array(analyser.frequencyBinCount);
      analyser.getByteFrequencyData(buf);
      ctx.clearRect(0, 0, W, H);
      const bars = 32, step = Math.floor(buf.length / bars), barW = W / bars - 1;
      for (let i = 0; i < bars; i++) {
        const v = buf[i * step] / 255, h = Math.max(2, v * H);
        ctx.fillStyle = v > 0.05 ? color : '#1e293b';
        ctx.fillRect(i * (barW + 1), H - h, barW, h);
      }
    }

    function vizLoop() {
      drawBars(document.getElementById('micViz'), micAnalyser, '#22c55e');
      drawBars(document.getElementById('agentViz'), agentAnalyser, '#38bdf8');
      vizRAF = requestAnimationFrame(vizLoop);
    }

    function initCanvasSize() {
      for (const id of ['micViz', 'agentViz']) {
        const c = document.getElementById(id);
        c.width = c.offsetWidth * devicePixelRatio;
        c.height = c.offsetHeight * devicePixelRatio;
      }
    }

    async function startConversation() {
      const btnStart = document.getElementById('btnStart');
      const btnEnd = document.getElementById('btnEnd');
      btnStart.disabled = true;
      document.getElementById('log').innerHTML = '';
      setStatus('Requesting token...', 'connecting');
      log('Requesting token...');
      try {
        const resp = await fetch('/api/token');
        if (!resp.ok) throw new Error(await resp.text());
        const data = await resp.json();
        log(`Room: ${data.room}`, 'log-info');
        room = new Room({ audioCaptureDefaults: { echoCancellation: true, noiseSuppression: true }, audioOutput: { deviceId: 'default' } });
        room.on(RoomEvent.TrackSubscribed, (track, pub, participant) => {
          log(`Track subscribed: ${track.kind} from ${participant.identity}`);
          if (track.kind === Track.Kind.Audio) {
            const el = track.attach(); el.id = 'agent-audio'; document.body.appendChild(el);
            agentAnalyser = createAnalyser(new MediaStream([track.mediaStreamTrack]));
            setStatus('Agent connected — speak now!', 'connected');
            log('Agent audio playing', 'log-info');
          }
        });
        room.on(RoomEvent.TrackUnsubscribed, (track, pub, p) => log(`Track unsubscribed: ${track.kind} from ${p.identity}`));
        room.on(RoomEvent.ParticipantConnected, p => log(`Participant joined: ${p.identity}`));
        room.on(RoomEvent.ParticipantDisconnected, p => log(`Participant left: ${p.identity}`));
        room.on(RoomEvent.Disconnected, reason => { log(`Disconnected: ${reason}`, 'log-error'); setStatus('Disconnected', 'idle'); btnStart.disabled = false; btnEnd.disabled = true; cleanup(); });
        room.on(RoomEvent.AudioPlaybackStatusChanged, () => log(`Audio playback allowed: ${room.canPlaybackAudio}`));
        setStatus('Connecting to LiveKit...', 'connecting');
        log('Connecting to LiveKit...');
        await audioCtx.resume();
        await room.connect(data.url, data.token);
        log(`Connected as ${room.localParticipant.identity}`, 'log-info');
        await room.localParticipant.setMicrophoneEnabled(true);
        log('Microphone enabled', 'log-info');
        initCanvasSize();
        const micTrack = room.localParticipant.getTrackPublication(Track.Source.Microphone);
        if (micTrack && micTrack.track) micAnalyser = createAnalyser(new MediaStream([micTrack.track.mediaStreamTrack]));
        vizLoop();
        setStatus('Connected — waiting for agent...', 'connecting');
        btnEnd.disabled = false;
      } catch (err) {
        console.error(err); log(`Error: ${err.message}`, 'log-error');
        setStatus('Error: ' + err.message, 'error'); btnStart.disabled = false; btnEnd.disabled = true;
      }
    }

    async function endConversation() {
      if (room) { log('Ending conversation...'); await room.disconnect(); cleanup(); }
      document.getElementById('btnStart').disabled = false;
      document.getElementById('btnEnd').disabled = true;
      setStatus('Ended', 'idle');
    }

    function cleanup() {
      const el = document.getElementById('agent-audio'); if (el) el.remove();
      room = null; micAnalyser = null; agentAnalyser = null;
      if (vizRAF) { cancelAnimationFrame(vizRAF); vizRAF = null; }
    }

    /* ---- Agent logs viewer ---- */
    let autoRefreshId = null;
    async function loadAgentLogs() {
      const el = document.getElementById('agentLogs');
      try {
        const resp = await fetch('/api/logs?n=150');
        const data = await resp.json();
        el.innerHTML = data.lines.map(l => {
          let cls = 'color:#64748b';
          if (/ERROR|error|ERRO/i.test(l)) cls = 'color:#f87171';
          else if (/WARN/i.test(l)) cls = 'color:#fbbf24';
          else if (/INFO/i.test(l)) cls = 'color:#94a3b8';
          else if (/user_transcript|transcript/i.test(l)) cls = 'color:#a3e635';
          else if (/TTS|tts|bulbul/i.test(l)) cls = 'color:#38bdf8';
          return `<span style="${cls}">${l.replace(/</g,'&lt;')}</span>`;
        }).join('\n');
        el.scrollTop = el.scrollHeight;
        if (data.file) el.title = data.file;
      } catch(e) { el.textContent = 'Failed to fetch logs: ' + e.message; }
    }
    function toggleAutoRefresh() {
      if (document.getElementById('autoRefresh').checked) {
        loadAgentLogs(); autoRefreshId = setInterval(loadAgentLogs, 3000);
      } else { clearInterval(autoRefreshId); autoRefreshId = null; }
    }
    document.querySelector('details').addEventListener('toggle', e => { if (e.target.open) loadAgentLogs(); });
  </script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------
def find_agent_log():
    """Find the most recent LiveKit agent log file."""
    # LiveKit agents write logs to /tmp or the system temp dir
    patterns = [
        "/tmp/livekit-agents-*.log",
        "/private/tmp/livekit-agents-*.log",
        os.path.expanduser("~/.livekit/agents/*.log"),
    ]
    # Also check the background task output files from our session
    task_dir = "/private/tmp/claude-501/-Users-dg-Documents-lab-hyperlocal-discovery/tasks"
    if os.path.isdir(task_dir):
        outputs = sorted(Path(task_dir).glob("*.output"), key=lambda p: p.stat().st_mtime, reverse=True)
        for f in outputs:
            try:
                content = f.read_text(errors="replace")
                if "ac-price" in content or "livekit.agents" in content:
                    return str(f)
            except Exception:
                continue
    for pat in patterns:
        files = sorted(glob.glob(pat), key=os.path.getmtime, reverse=True)
        if files:
            return files[0]
    return None


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self._serve_html()
        elif self.path == "/api/token":
            self._serve_token()
        elif self.path == "/api/metrics":
            self._serve_metrics()
        elif self.path.startswith("/api/logs"):
            self._serve_logs()
        else:
            self.send_error(404)

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode())

    def _serve_token(self):
        try:
            token_data = asyncio.run(create_token_and_dispatch())
            body = json.dumps(token_data).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(f"Token error: {e}".encode())

    def _serve_metrics(self):
        """Return dashboard metrics (transcripts, logs, test results) as JSON."""
        try:
            transcripts = parse_transcripts()
            log_data = parse_logs()
            metrics = compute_metrics(transcripts, log_data)
            tests = run_tests()
            body = json.dumps({
                "metrics": metrics,
                "tests": tests,
                "transcripts": transcripts,
            }, default=str).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _serve_logs(self):
        """Return the last N lines of agent worker logs as JSON."""
        from urllib.parse import urlparse, parse_qs
        params = parse_qs(urlparse(self.path).query)
        n = int(params.get("n", ["200"])[0])

        log_file = find_agent_log()
        if not log_file:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"file": None, "lines": ["No agent log file found. Is the agent worker running?"]}).encode())
            return

        try:
            with open(log_file, "r", errors="replace") as f:
                all_lines = f.readlines()
            lines = [l.rstrip() for l in all_lines[-n:]]
        except Exception as e:
            lines = [f"Error reading log: {e}"]

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps({"file": log_file, "lines": lines}).encode())

    def log_message(self, format, *args):
        # Friendlier log format
        print(f"  [{self.address_string()}] {format % args}")


# ---------------------------------------------------------------------------
# Token generation + agent dispatch
# ---------------------------------------------------------------------------
async def create_token_and_dispatch():
    from livekit.api import LiveKitAPI, AccessToken, VideoGrants
    from livekit.protocol.agent_dispatch import CreateAgentDispatchRequest

    room_name = f"browser-test-{uuid.uuid4().hex[:8]}"
    user_identity = f"user-{uuid.uuid4().hex[:6]}"

    # 1. Create a user token for the browser participant
    token = (
        AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
        .with_identity(user_identity)
        .with_name("Browser User")
        .with_grants(VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
        ))
        .to_jwt()
    )

    # 2. Dispatch the agent into the room
    lk = LiveKitAPI()
    try:
        await lk.agent_dispatch.create_dispatch(
            CreateAgentDispatchRequest(
                agent_name="ac-price-agent",
                room=room_name,
                metadata=json.dumps({
                    "store_name": "Browser Test",
                    "ac_model": "Samsung 1.5 Ton 5 Star Inverter Split AC",
                    "nearby_area": "Koramangala 4th Block",
                }),
            )
        )
    finally:
        await lk.aclose()

    return {
        "token": token,
        "url": LIVEKIT_URL,
        "room": room_name,
    }


# ---------------------------------------------------------------------------
# Agent worker lifecycle management
# ---------------------------------------------------------------------------
_agent_proc = None

def _kill_old_agents():
    """Kill any existing agent_worker.py processes."""
    import subprocess
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
    # Brief wait for cleanup
    if any(p.strip() and p.strip() != my_pid for p in pids):
        time.sleep(1)


def _start_agent_worker():
    """Start a new agent_worker.py dev process in the background."""
    global _agent_proc
    # Use the same Python interpreter as this process
    python = sys.executable
    script = Path(__file__).parent / "agent_worker.py"
    _agent_proc = subprocess.Popen(
        [python, str(script), "dev"],
        cwd=str(Path(__file__).parent),
        # Let agent worker inherit stdout/stderr so logs are visible
        stdout=sys.stderr,
        stderr=sys.stderr,
    )
    print(f"  Agent worker started (PID {_agent_proc.pid})")
    # Give it a moment to register with LiveKit
    time.sleep(3)


def _cleanup_agent():
    """Terminate agent worker on exit."""
    global _agent_proc
    if _agent_proc and _agent_proc.poll() is None:
        print(f"\n  Stopping agent worker (PID {_agent_proc.pid})")
        _agent_proc.terminate()
        try:
            _agent_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _agent_proc.kill()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n  AC Price Agent — Browser Test")
    print(f"  {'=' * 40}")

    # Auto-manage agent worker
    _kill_old_agents()
    _start_agent_worker()
    atexit.register(_cleanup_agent)

    print(f"  Server:  http://localhost:{PORT}")
    print(f"  LiveKit: {LIVEKIT_URL}")
    print()

    server = HTTPServer(("", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down.")
        server.server_close()
