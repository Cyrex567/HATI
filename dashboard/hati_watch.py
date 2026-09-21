"""Read-only browser observer for a HATI saturation campaign.

No job manager, shell endpoint, write route or pipeline controls. The server
binds only to loopback and reads the selected run directory.
"""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sys
import threading
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent/'watch'


def safe_file(root, relative):
    p = (root/relative).resolve()
    if not p.is_relative_to(root.resolve()) or not p.is_file():
        raise FileNotFoundError(relative)
    return p


def read_json(path, default=None, limit=4*1024*1024):
    try:
        if path.stat().st_size > limit:
            return default
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def tail(path, limit=18000):
    try:
        with path.open('rb') as f:
            f.seek(max(0, path.stat().st_size-limit))
            return f.read(limit).decode('utf-8', errors='replace')
    except OSError:
        return ''


class WatchStore:
    def __init__(self, folder):
        self.root = Path(folder).resolve()
        self.lock = threading.Lock()
        self.input_stamp = None
        self.input_metadata = None
        self.images = {}
        self.input_error = None

    def inputs(self):
        meta = read_json(self.root/'live/inputs.json')
        if meta:
            return meta
        # Existing campaigns can be watched without touching their output files.
        bundle = self.root/'inputs/hati_diagnostic_bundle.zip'
        if not bundle.is_file():
            return None
        stamp = (bundle.stat().st_size, bundle.stat().st_mtime_ns)
        with self.lock:
            if stamp != self.input_stamp:
                self.input_stamp = stamp
                try:
                    sys.path.insert(0, str(ROOT/'scripts'))
                    from review_shadow_bundle import read_bundle
                    from live_feedback import render_inputs
                    data, run, *_ = read_bundle(bundle)
                    self.input_metadata, self.images = render_inputs(data, run)
                    self.input_error = None
                except Exception as exc:
                    self.input_error = str(exc); self.input_metadata = None; self.images = {}
        return self.input_metadata

    def state(self, selected=None):
        campaign = read_json(self.root/'campaign.json', {})
        stages = campaign.get('stages', [])
        active = next((r for r in stages if r['status'] == 'RUNNING'), None)
        chosen = next((r for r in stages if r['id'] == selected), None) if selected else None
        chosen = chosen or active or next((r for r in reversed(stages) if r['status'] != 'PENDING'), None)
        stage = chosen['id'] if chosen else None
        def under(relative):
            try:
                return safe_file(self.root, relative)
            except FileNotFoundError:
                return None
        snapshot = under(f'live/{stage}.json') if stage else None
        log = under(f'logs/{stage}.log') if stage else None
        heartbeat = read_json(self.root/'live/runtime.json', {})
        now = datetime.now(timezone.utc)
        try:
            age = max(0, (now-datetime.fromisoformat(heartbeat['updated'])).total_seconds())
        except (KeyError, ValueError, TypeError):
            age = None
        health = ('live' if age is not None and age < 10 else 'stale') if heartbeat.get('state') == 'running' else heartbeat.get('state', 'no_heartbeat')
        artifacts = []
        for p in sorted((self.root/'stages').rglob('*.png')):
            if p.resolve().is_relative_to(self.root):
                artifacts.append(dict(path=p.relative_to(self.root).as_posix(), version=p.stat().st_mtime_ns,
                                      title=p.stem.replace('_', ' ')))
        return dict(run_name=self.root.name, stages=stages, selected_stage=stage,
                    active_stage=active['id'] if active else None, health=health, heartbeat_age_seconds=age,
                    snapshot=read_json(snapshot) if snapshot else None,
                    terrain=(read_json(self.root/'live/maps.json', {}) or {}).get('terrain'),
                    log=tail(log) if log else '', inputs=self.inputs(), input_error=self.input_error,
                    artifacts=artifacts, verdict=campaign.get('verdict'), server_time=now.isoformat())


def make_handler(store):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, payload, mime='application/json; charset=utf-8'):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            if self.command != 'HEAD':
                try:
                    self.wfile.write(payload)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        def do_GET(self):
            parsed = urlsplit(self.path)
            path = unquote(parsed.path)
            try:
                if path == '/api/state':
                    selected = parse_qs(parsed.query).get('stage', [None])[0]
                    self.respond(200, json.dumps(store.state(selected), allow_nan=False).encode('utf-8'))
                elif path.startswith('/input/'):
                    name = path.removeprefix('/input/')
                    metadata = store.inputs()
                    names = {r['image'] for r in metadata.get('frames', [])} if metadata else set()
                    if name not in names:
                        raise FileNotFoundError(name)
                    p = store.root/'live'/name
                    payload = safe_file(store.root, 'live/'+name).read_bytes() if p.exists() else store.images[name]
                    self.respond(200, payload, 'image/png')
                elif path.startswith('/artifact/'):
                    p = safe_file(store.root, path.removeprefix('/artifact/'))
                    if p.suffix.lower() not in ('.png', '.json', '.csv', '.txt', '.md'):
                        raise FileNotFoundError(path)
                    mime = 'image/png' if p.suffix == '.png' else 'text/plain; charset=utf-8'
                    self.respond(200, p.read_bytes(), mime)
                elif path == '/logo.png':
                    self.respond(200, (ROOT/'dashboard/static/assets/hati_logo.png').read_bytes(), 'image/png')
                else:
                    name = {'/': 'index.html', '/watch.js': 'watch.js', '/watch.css': 'watch.css'}.get(path)
                    if name is None:
                        raise FileNotFoundError(path)
                    p = safe_file(STATIC, name)
                    self.respond(200, p.read_bytes(), mimetypes.guess_type(p.name)[0]+'; charset=utf-8')
            except (FileNotFoundError, KeyError, OSError):
                self.respond(404, b'{"error":"Not found"}')

        def do_HEAD(self):
            self.do_GET()

        def do_POST(self):
            self.respond(405, b'{"error":"HATI Watch is read-only"}')

        do_PUT = do_DELETE = do_PATCH = do_POST

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', type=Path, required=True, help='campaign output folder; may still be running')
    ap.add_argument('--port', type=int, default=8765)
    args = ap.parse_args()
    if not 1 <= args.port <= 65535:
        ap.error('port must be between 1 and 65535')
    store = WatchStore(args.run_dir)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(store))
    print(f'HATI Watch: http://localhost:{args.port}\nWatching: {store.root}\nRead-only. Closing this viewer leaves the calculation running.', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
