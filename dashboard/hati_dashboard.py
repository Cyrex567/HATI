"""HATI Mission Dashboard -- local control panel for the HATI pipeline.

Stdlib-only web server (no pip installs needed): serves the dashboard UI,
streams live job output over Server-Sent Events, and launches the actual
pipeline scripts (consensus census, kinematics benchmark, sweep query, ISIS
ingestion) as subprocesses. Runs identically on the laptop (UI only -- don't
click the heavy jobs) and on the GPU box (click everything).

Usage:
    python dashboard/hati_dashboard.py            # serve + open browser
    python dashboard/hati_dashboard.py --no-browser --port 8737

PyInstaller-aware: when frozen, static assets resolve via sys._MEIPASS and
the project root is the executable's directory (see build_exe.bat).
"""
from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from shutil import which

MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css",
        ".png": "image/png", ".jpg": "image/jpeg", ".svg": "image/svg+xml",
        ".json": "application/json", ".ico": "image/x-icon", ".csv": "text/csv",
        ".txt": "text/plain; charset=utf-8"}


def static_dir() -> Path:
    m = getattr(sys, "_MEIPASS", None)
    return Path(m) / "static" if m else Path(__file__).resolve().parent / "static"


def project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


ROOT = project_root()
OUT = ROOT / "output" / "athena"

# job name -> script + args (all relative to project root, run with this python)
JOBS = {
    "sweep":      ["scripts/solar_sweep_query.py", "--lat", "-84.7906", "--lon", "29.1957",
                   "--halfwidth-km", "3", "--pt", "EDRNAC4", "--max-frames", "2000"],
    "mare-v3":    ["scripts/mare_control_v3.py"],
    "consensus":  ["scripts/shadow_consensus.py"],
    "kinematics": ["scripts/shadow_kinematics.py"],
    "ingest-dry": ["scripts/ingest_sweep.py", "--n", "10"],
    "ingest":     ["scripts/ingest_sweep.py", "--n", "10", "--execute"],
}
HEAVY = {"consensus", "kinematics", "ingest", "mare-v3"}   # flagged in the UI, not blocked


class JobManager:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.proc: subprocess.Popen | None = None
        self.name: str | None = None
        self.buf: deque[str] = deque(maxlen=4000)
        self.subs: list[queue.Queue] = []

    def publish(self, ev: dict) -> None:
        if ev.get("t") == "log":
            self.buf.append(ev["line"])
        with self.lock:
            for q in list(self.subs):
                try:
                    q.put_nowait(ev)
                except queue.Full:
                    pass

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=2000)
        with self.lock:
            self.subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, name: str) -> tuple[bool, str]:
        if name not in JOBS:
            return False, f"unknown job '{name}'"
        with self.lock:
            if self.running():
                return False, f"job '{self.name}' is already running"
            script = ROOT / JOBS[name][0]
            if not script.exists():
                return False, f"script not found: {script}"
            cmd = [sys.executable, "-u", str(script)] + JOBS[name][1:]
            self.proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT, text=True,
                                         encoding="utf-8", errors="replace", bufsize=1)
            self.name = name
        self.publish({"t": "status", "job": name, "state": "running"})
        self.publish({"t": "log", "line": f"[hati] started job '{name}': {' '.join(cmd[1:])}"})
        threading.Thread(target=self._pump, daemon=True).start()
        return True, "started"

    def _pump(self) -> None:
        p, name = self.proc, self.name
        assert p and p.stdout
        for line in p.stdout:
            self.publish({"t": "log", "line": line.rstrip("\n")})
        rc = p.wait()
        state = "done" if rc == 0 else "fail"
        self.publish({"t": "log", "line": f"[hati] job '{name}' finished (exit {rc})"})
        self.publish({"t": "status", "job": name, "state": state, "rc": rc})

    def stop(self) -> str:
        with self.lock:
            if not self.running():
                return "no job running"
            self.proc.terminate()
        self.publish({"t": "log", "line": f"[hati] job '{self.name}' terminated by user"})
        return "terminated"


JM = JobManager()


def read_sweep() -> tuple[str | None, list[dict]]:
    csvs = sorted(OUT.glob("solar_sweep_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        return None, []
    rows: list[dict] = []
    lines = csvs[0].read_text(encoding="utf-8", errors="replace").splitlines()
    for ln in lines[1:]:
        f = ln.split(",")
        if len(f) < 8:
            continue
        try:
            rows.append({"pid": f[0], "elev": float(f[3]) if f[3] else None,
                         "az": float(f[6]) if f[6] else None,
                         "year": int(f[1][:4]) if f[1][:4].isdigit() else None})
        except ValueError:
            continue
    return csvs[0].name, rows


def state_json() -> dict:
    csv_name, sweep = read_sweep()
    figs = [p.name for p in sorted(OUT.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)]
    isis = which("lronac2isis") is not None
    return {
        "sweep": sweep, "sweepCsv": csv_name, "figures": figs,
        "checks": {
            "isis": isis,
            "dtm": (ROOT / "data/athena/NAC_DTM_NOBILE03.TIF").exists(),
            "orthos": len(list((ROOT / "data/athena").glob("NAC_DTM_NOBILE03_M*_90CM.IMG"))),
            "manifest": (ROOT / "data/sweep/manifest.json").exists(),
            "platform": sys.platform, "python": sys.version.split()[0],
        },
        "job": {"name": JM.name, "running": JM.running()},
        "heavy": sorted(HEAVY),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json") -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path) -> None:
        if not path.is_file():
            self._send(404, b'{"error":"not found"}')
            return
        self._send(200, path.read_bytes(), MIME.get(path.suffix.lower(), "application/octet-stream"))

    def do_GET(self) -> None:
        p = self.path.split("?")[0]
        if p in ("/", "/index.html"):
            self._file(static_dir() / "index.html")
        elif p == "/api/state":
            self._send(200, json.dumps(state_json()).encode())
        elif p == "/api/events":
            self._sse()
        elif p.startswith("/figs/"):
            name = Path(p[len("/figs/"):]).name          # no traversal
            if name.endswith((".png", ".csv")):
                self._file(OUT / name)
            else:
                self._send(403, b'{"error":"forbidden"}')
        elif p.startswith("/assets/") or p.startswith("/static/"):
            rel = p.split("/", 2)[2] if p.startswith("/static/") else "assets/" + Path(p[8:]).name
            f = (static_dir() / rel).resolve()
            if static_dir().resolve() in f.parents or f == static_dir().resolve():
                self._file(f)
            else:
                self._send(403, b'{"error":"forbidden"}')
        else:
            self._send(404, b'{"error":"not found"}')

    def do_POST(self) -> None:
        p, _, q = self.path.partition("?")
        if p == "/api/run":
            job = dict(kv.split("=") for kv in q.split("&") if "=" in kv).get("job", "")
            ok, msg = JM.start(job)
            self._send(200 if ok else 409, json.dumps({"ok": ok, "msg": msg}).encode())
        elif p == "/api/stop":
            self._send(200, json.dumps({"ok": True, "msg": JM.stop()}).encode())
        else:
            self._send(404, b'{"error":"not found"}')

    def _sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        q = JM.subscribe()
        try:
            for line in list(JM.buf)[-200:]:              # replay recent log
                self.wfile.write(f"data: {json.dumps({'t': 'log', 'line': line})}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    ev = q.get(timeout=15)
                    self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                except queue.Empty:
                    self.wfile.write(b": hb\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionError, OSError):
            pass
        finally:
            JM.unsubscribe(q)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8737)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    print(f"HATI dashboard -> {url}   (root: {ROOT})")
    if not args.no_browser:
        threading.Timer(0.6, webbrowser.open, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nHATI dashboard stopped.")


if __name__ == "__main__":
    main()
