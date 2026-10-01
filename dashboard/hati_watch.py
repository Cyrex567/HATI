"""Read-only browser observer for a HATI saturation campaign.

No job manager, shell endpoint, write route or pipeline controls. The server
binds only to loopback and reads the selected run directory.
"""
import argparse
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import sys
import threading
from urllib.parse import parse_qs, unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent/'watch'


def read_bytes(path):
    # Read and close at once: on Windows an open handle blocks the writers' rename,
    # and they retry only briefly.
    with open(path, 'rb') as f:
        return f.read()


def safe_file(root, relative):
    p = (root/relative).resolve()
    if not p.is_relative_to(root.resolve()) or not p.is_file():
        raise FileNotFoundError(relative)
    return p


def read_json(path, default=None, limit=4*1024*1024):
    try:
        if path.stat().st_size > limit:
            return default
        return json.loads(read_bytes(path).decode('utf-8'))
    except (OSError, ValueError):
        return default


def stage_titles():
    try:
        sys.path.insert(0, str(ROOT/'scripts'))
        from run_saturation_campaign import STAGES
        return dict(STAGES)
    except Exception:
        return {}


def when(value):
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


# A stage started directly, without the campaign runner, has no campaign.json or
# heartbeat. Its live snapshot and log still show whether it is working.
QUIET_SECONDS = 900


def natural(stage_id):
    digits = ''.join(ch for ch in stage_id if ch.isdigit())
    return (0 if stage_id == 'maps' else 1, int(digits) if digits else 0, stage_id)


# Supplementary per-stage files the stage reports draw on, beside result.json.
EXTRAS = {'T14': {'injection': 'injection.json', 'real_rocks': 'real_rocks.json'}}
STAGE_ID = re.compile(r'[A-Za-z0-9_-]{1,40}')


def tail(path, limit=18000):
    try:
        with open(path, 'rb') as f:
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
        self.titles = None

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

    def activity(self, stage):
        """Latest sign of work for one stage: its snapshot time, snapshot file or log."""
        times = []
        snapshot = read_json(self.root/'live'/f'{stage}.json', {}) or {}
        for value in (snapshot.get('updated'), (snapshot.get('sizing') or {}).get('updated')):
            if when(value):
                times.append(when(value))
        for p in (self.root/'live'/f'{stage}.json', self.root/'logs'/f'{stage}.log'):
            if p.is_file():
                times.append(datetime.fromtimestamp(p.stat().st_mtime, timezone.utc))
        return max(times) if times else None

    def inferred(self, now):
        """Stages and a heartbeat from live snapshots, logs and results, for runs without the runner."""
        if self.titles is None:
            self.titles = stage_titles()
        titles = self.titles
        ids = {p.stem for p in (self.root/'live').glob('*.json') if p.stem not in ('inputs', 'runtime', 'maps')}
        ids |= {p.parent.name for p in (self.root/'stages').glob('*/result.json')}
        ids |= {p.stem for p in (self.root/'logs').glob('*.log')}
        rows, seen_by = [], {}
        for stage in sorted(ids, key=natural):
            seen = self.activity(stage)
            result_path = self.root/'stages'/stage/'result.json'
            result = read_json(result_path, {}) or {}
            finished = datetime.fromtimestamp(result_path.stat().st_mtime, timezone.utc) if result_path.is_file() else None
            # A result older than the stage's latest activity belongs to an earlier run of it.
            if result.get('status') and (seen is None or finished >= seen-timedelta(seconds=60)):
                status, reason = result['status'], result.get('reason', '')
            elif seen is not None and (now-seen).total_seconds() < QUIET_SECONDS:
                status, reason = 'RUNNING', ''
            else:
                status, reason = 'STALE', 'No result and no recent activity.'
            rows.append(dict(id=stage, title=titles.get(stage, stage), status=status, reason=reason))
            if seen:
                seen_by[stage] = seen
        running = [r['id'] for r in rows if r['status'] == 'RUNNING']
        state = 'running' if running else 'stale' if any(r['status'] == 'STALE' for r in rows) else 'finished'
        # Liveness follows the running stage; a finished stage's files say nothing about it.
        times = [seen_by[s] for s in (running or seen_by) if s in seen_by]
        heartbeat = dict(state=state, updated=max(times).isoformat()) if times else {}
        return rows, heartbeat

    def state(self, selected=None):
        campaign = read_json(self.root/'campaign.json', {})
        stages = campaign.get('stages', [])
        now = datetime.now(timezone.utc)
        inferred = not stages
        if inferred:
            stages, heartbeat = self.inferred(now)
        else:
            heartbeat = read_json(self.root/'live/runtime.json', {})
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
        try:
            age = max(0, (now-datetime.fromisoformat(heartbeat['updated'])).total_seconds())
        except (KeyError, ValueError, TypeError):
            age = None
        if heartbeat.get('state') != 'running':
            health = heartbeat.get('state', 'no_heartbeat')
        elif not inferred:
            health = 'live' if age is not None and age < 10 else 'stale'
        else:
            # Without a runner heartbeat, silence during one long step is not failure.
            health = 'live' if age is not None and age < 30 else 'quiet' if age is not None and age < QUIET_SECONDS else 'stale'
        artifacts = []
        for p in sorted((self.root/'stages').rglob('*.png')):
            if p.resolve().is_relative_to(self.root):
                artifacts.append(dict(path=p.relative_to(self.root).as_posix(), version=p.stat().st_mtime_ns,
                                      title=p.stem.replace('_', ' ')))
        # One version per finished stage, so the page fetches a report only when it changes.
        results = {}
        for p in (self.root/'stages').glob('*/result.json'):
            try:
                results[p.parent.name] = str(p.stat().st_mtime_ns)
            except OSError:
                pass
        return dict(run_name=self.root.name, stages=stages, selected_stage=stage, results=results,
                    active_stage=active['id'] if active else None, health=health, heartbeat_age_seconds=age,
                    heartbeat_source='inferred' if inferred else 'runner',
                    snapshot=read_json(snapshot) if snapshot else None,
                    terrain=(read_json(self.root/'live/maps.json', {}) or {}).get('terrain'),
                    log=tail(log) if log else '', inputs=self.inputs(), input_error=self.input_error,
                    artifacts=artifacts, verdict=campaign.get('verdict'), server_time=now.isoformat())

    def result(self, stage):
        """One stage's saved result for its report, without the configuration echo, plus its extras."""
        if not STAGE_ID.fullmatch(stage or ''):
            raise FileNotFoundError(stage)
        folder = self.root/'stages'/stage
        data = read_json(safe_file(self.root, f'stages/{stage}/result.json'))
        if not isinstance(data, dict):
            raise FileNotFoundError(stage)
        data = {k: v for k, v in data.items() if k not in ('configuration', 'provenance')}
        extras = {}
        for key, name in EXTRAS.get(stage, {}).items():
            if (folder/name).is_file():
                extras[key] = read_json(safe_file(self.root, f'stages/{stage}/{name}'))
        return dict(stage=stage, result=data, extras=extras,
                    version=str((folder/'result.json').stat().st_mtime_ns))


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
                elif path == '/api/result':
                    stage = parse_qs(parsed.query).get('stage', [''])[0]
                    self.respond(200, json.dumps(store.result(stage), allow_nan=False).encode('utf-8'))
                elif path.startswith('/input/'):
                    name = path.removeprefix('/input/')
                    metadata = store.inputs()
                    names = {r['image'] for r in metadata.get('frames', [])} if metadata else set()
                    if name not in names:
                        raise FileNotFoundError(name)
                    p = store.root/'live'/name
                    payload = read_bytes(safe_file(store.root, 'live/'+name)) if p.exists() else store.images[name]
                    self.respond(200, payload, 'image/png')
                elif path.startswith('/artifact/'):
                    p = safe_file(store.root, path.removeprefix('/artifact/'))
                    if p.suffix.lower() not in ('.png', '.json', '.csv', '.txt', '.md'):
                        raise FileNotFoundError(path)
                    mime = 'image/png' if p.suffix == '.png' else 'text/plain; charset=utf-8'
                    self.respond(200, read_bytes(p), mime)
                elif path == '/logo.png':
                    self.respond(200, (ROOT/'dashboard/static/assets/hati_logo.png').read_bytes(), 'image/png')
                else:
                    name = {'/': 'index.html', '/watch.js': 'watch.js', '/stages.js': 'stages.js',
                            '/watch.css': 'watch.css'}.get(path)
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
