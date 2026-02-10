"""
test_browser.py — Browser-Based Voice Agent Test
=================================================
Talk to the AC price enquiry agent from your browser (WebRTC).
No SIP trunk or phone calls needed.

Usage:
  1. Ensure the agent worker is running:  python agent_worker.py dev
  2. Start this server:                   python test_browser.py
  3. Open http://localhost:8080 in your browser
  4. Click "Start Conversation" and allow microphone access
"""

import asyncio
import glob
import json
import os
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

from dotenv import load_dotenv

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
  <title>AC Price Agent — Browser Test</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: #0f172a; color: #e2e8f0;
      display: flex; justify-content: center; align-items: center;
      min-height: 100vh; padding: 1rem;
    }
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

    /* Audio visualizers */
    .viz-row { display: flex; gap: 1rem; margin-top: 1rem; }
    .viz-box {
      flex: 1; background: #0f172a; border-radius: 8px; padding: .5rem;
      text-align: center;
    }
    .viz-label { font-size: .7rem; color: #64748b; margin-bottom: .25rem; text-transform: uppercase; letter-spacing: .05em; }
    canvas { width: 100%; height: 48px; display: block; border-radius: 4px; }

    /* Event log */
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
  </style>
</head>
<body>
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

  <script src="https://cdn.jsdelivr.net/npm/livekit-client/dist/livekit-client.umd.js"></script>
  <script>
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

    /* ---- Audio visualization ---- */
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

      const bars = 32;
      const step = Math.floor(buf.length / bars);
      const barW = W / bars - 1;
      for (let i = 0; i < bars; i++) {
        const v = buf[i * step] / 255;
        const h = Math.max(2, v * H);
        ctx.fillStyle = v > 0.05 ? color : '#1e293b';
        ctx.fillRect(i * (barW + 1), H - h, barW, h);
      }
    }

    function vizLoop() {
      const micCanvas = document.getElementById('micViz');
      const agentCanvas = document.getElementById('agentViz');
      drawBars(micCanvas, micAnalyser, '#22c55e');
      drawBars(agentCanvas, agentAnalyser, '#38bdf8');
      vizRAF = requestAnimationFrame(vizLoop);
    }

    function initCanvasSize() {
      for (const id of ['micViz', 'agentViz']) {
        const c = document.getElementById(id);
        c.width = c.offsetWidth * devicePixelRatio;
        c.height = c.offsetHeight * devicePixelRatio;
      }
    }

    /* ---- Main ---- */
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

        room = new Room({
          audioCaptureDefaults: { echoCancellation: true, noiseSuppression: true },
          audioOutput: { deviceId: 'default' },
        });

        /* Track events */
        room.on(RoomEvent.TrackSubscribed, (track, pub, participant) => {
          log(`Track subscribed: ${track.kind} from ${participant.identity}`);
          if (track.kind === Track.Kind.Audio) {
            const el = track.attach();
            el.id = 'agent-audio';
            document.body.appendChild(el);
            // Visualize agent audio
            const stream = new MediaStream([track.mediaStreamTrack]);
            agentAnalyser = createAnalyser(stream);
            setStatus('Agent connected — speak now!', 'connected');
            log('Agent audio playing', 'log-info');
          }
        });

        room.on(RoomEvent.TrackUnsubscribed, (track, pub, participant) => {
          log(`Track unsubscribed: ${track.kind} from ${participant.identity}`);
        });

        room.on(RoomEvent.ParticipantConnected, (p) => {
          log(`Participant joined: ${p.identity}`);
        });

        room.on(RoomEvent.ParticipantDisconnected, (p) => {
          log(`Participant left: ${p.identity}`);
        });

        room.on(RoomEvent.Disconnected, (reason) => {
          log(`Disconnected: ${reason}`, 'log-error');
          setStatus('Disconnected', 'idle');
          btnStart.disabled = false;
          btnEnd.disabled = true;
          cleanup();
        });

        room.on(RoomEvent.AudioPlaybackStatusChanged, () => {
          log(`Audio playback allowed: ${room.canPlaybackAudio}`);
        });

        setStatus('Connecting to LiveKit...', 'connecting');
        log('Connecting to LiveKit...');
        await audioCtx.resume();
        await room.connect(data.url, data.token);
        log(`Connected as ${room.localParticipant.identity}`, 'log-info');

        await room.localParticipant.setMicrophoneEnabled(true);
        log('Microphone enabled', 'log-info');

        // Visualize local mic
        initCanvasSize();
        const micTrack = room.localParticipant.getTrackPublication(Track.Source.Microphone);
        if (micTrack && micTrack.track) {
          const stream = new MediaStream([micTrack.track.mediaStreamTrack]);
          micAnalyser = createAnalyser(stream);
        }
        vizLoop();

        setStatus('Connected — waiting for agent...', 'connecting');
        btnEnd.disabled = false;
      } catch (err) {
        console.error(err);
        log(`Error: ${err.message}`, 'log-error');
        setStatus('Error: ' + err.message, 'error');
        btnStart.disabled = false;
        btnEnd.disabled = true;
      }
    }

    async function endConversation() {
      if (room) {
        log('Ending conversation...');
        await room.disconnect();
        cleanup();
      }
      document.getElementById('btnStart').disabled = false;
      document.getElementById('btnEnd').disabled = true;
      setStatus('Ended', 'idle');
    }

    function cleanup() {
      const el = document.getElementById('agent-audio');
      if (el) el.remove();
      room = null;
      micAnalyser = null;
      agentAnalyser = null;
      if (vizRAF) { cancelAnimationFrame(vizRAF); vizRAF = null; }
    }

    /* ---- Agent logs viewer ---- */
    let autoRefreshId = null;
    async function loadAgentLogs() {
      const el = document.getElementById('agentLogs');
      try {
        const resp = await fetch('/api/logs?n=150');
        const data = await resp.json();
        // Color-code log lines
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
        loadAgentLogs();
        autoRefreshId = setInterval(loadAgentLogs, 3000);
      } else {
        clearInterval(autoRefreshId);
        autoRefreshId = null;
      }
    }
    // Load logs when details is opened
    document.querySelector('details').addEventListener('toggle', e => {
      if (e.target.open) loadAgentLogs();
    });
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
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n  AC Price Agent — Browser Test")
    print(f"  {'=' * 40}")
    print(f"  Server:  http://localhost:{PORT}")
    print(f"  LiveKit: {LIVEKIT_URL}")
    print(f"\n  Make sure the agent worker is running:")
    print(f"    python agent_worker.py dev")
    print()

    server = HTTPServer(("", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Shutting down.")
        server.server_close()
