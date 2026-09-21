"""Sequential, resumable saturation tests with an always-written review ZIP.

No ingestion, network requests or ISIS commands. Scientific workers run in
separate processes. A failed stage cannot be reported as a scientific pass.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent.parent
STAGES = [('maps', 'Replay terrain, shadow and fused maps'),
          ('T1', 'Physical response and signed frame contributions'),
          ('T2', 'Matched frame ablations'), ('T3', 'Regional geometry stress'),
          ('T4', 'Independent thermal comparison'), ('T5', 'Expanded width profiles'),
          ('T6', 'Height profiles and endpoint support'),
          ('T7', 'Local registration and planted shifts'),
          ('T8', 'Held-out annotated scenes')]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.part')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def utc():
    return datetime.now(timezone.utc).isoformat()


def capture(command):
    p = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)


def run_logged(command, log):
    """One child at a time; logs survive failures and Ctrl-C."""
    with Path(log).open('w', encoding='utf-8') as stream:
        stream.write(json.dumps(command)+'\n'); stream.flush()
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   encoding='utf-8', errors='replace', bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line); stream.flush()
            return process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
            raise


def result_record(stage, title, code, elapsed, folder):
    artifact = folder/'result.json'
    if code == 0 and artifact.exists():
        result = json.loads(artifact.read_text(encoding='utf-8'))
        if result.get('status') not in ('COMPLETE', 'PARTIAL', 'BLOCKED'):
            raise ValueError('worker returned an invalid scientific status')
        status, reason = result['status'], result.get('reason', '')
    else:
        status, reason = 'FAILED', f'Worker exited {code}; see logs/{stage}.log.'
    return dict(id=stage, title=title, status=status, reason=reason,
                exit_code=code, elapsed_seconds=round(elapsed, 2), finished=utc())


def verdict(records):
    failed = [r['id'] for r in records if r['status'] in ('FAILED', 'INTERRUPTED')]
    missing = [r['id'] for r in records if r['status'] in ('BLOCKED', 'PENDING', 'RUNNING')]
    partial = [r['id'] for r in records if r['status'] == 'PARTIAL']
    science = 'INCONCLUSIVE'
    if failed:
        execution = 'FAILED'
    elif missing:
        execution = 'INCOMPLETE'
    else:
        execution = 'COMPLETE'
    if not failed and not missing and not partial:
        science = 'READY_FOR_SCIENTIFIC_REVIEW'
    return dict(execution=execution, scientific_verdict=science,
                operational_verdict='WITHHOLD_OPERATIONAL_USE', failed=failed,
                missing=missing, partial=partial,
                interpretation='A completed diagnostic is not a validated detector. H1 and H2 may coexist. '
                'No score, warning or software pass establishes landing clearance or a mission-loss bound.')


def evidence_summary(output, records):
    """Describe the saved measurements without inventing a binary H1/H2 test."""
    findings = []
    complete = {r['id'] for r in records if r['status'] in ('COMPLETE', 'PARTIAL')}
    def read(stage, name='result.json'):
        path = output/'stages'/stage/name
        return json.loads(path.read_text(encoding='utf-8')) if stage in complete and path.exists() else {}
    base = read('T1', 'baseline/summary.json')
    fraction = base.get('exceedance_fraction_assessed')
    if fraction is not None:
        findings.append(f'Baseline: {fraction:.1%} of assessed pixels exceed the unchanged raw-score threshold; '
                        f'{base["cells_assessed"]} cells assessed. This is not a rock-occupancy fraction.')
    controls = read('T1').get('controls', [])
    planted = [r for r in controls if r.get('kind') == 'caster' and r['status'] == 'assessed']
    static = [r for r in controls if r.get('kind') == 'static' and r['status'] == 'assessed']
    if planted:
        findings.append(f'Independent caster controls: {sum(bool(r["recovered_within_2px"]) for r in planted)}/{len(planted)} '
                        'assessed trials recover a warning root within two pixels. Check recovery by height, location and support in T1.')
    if static:
        findings.append(f'Static no-caster controls: {sum(r["warning_roots"] > 0 for r in static)}/{len(static)} '
                        'assessed trials contain warning roots. These synthetic cases do not calibrate the lunar false-alarm rate.')
    for row in read('T2').get('comparisons', []):
        delta = row.get('median_paired_score_change')
        if delta is not None:
            findings.append(f'Frame test {row["group"]}: median matched-root score change {delta:+.3f}, '
                            f'with {row["matched_roots"]} matched roots. Compare its paired injection controls before attributing the change.')
    for row in read('T3').get('comparisons', []):
        lower = row.get('fraction_second_lower')
        if lower is not None:
            findings.append(f'Geometry shift {row["shift"]}: correct geometry scores higher at {lower:.1%} '
                            f'of {row["matched_roots"]} matched roots. This does not uniquely identify unresolved casters.')
    fraction = read('T5').get('widest_selected_fraction')
    if fraction is not None:
        findings.append(f'Expanded width bank: {fraction:.1%} of deduplicated warning roots choose the new widest bin.')
    height = read('T6')
    if height:
        findings.append(f'Height diagnostics: {height.get("real_assessed", 0)} of {height.get("real_profiles", 0)} '
                        'support/position profiles assessed. Confidence levels and observed-continuation validation remain unresolved.')
    reg = read('T7')
    if reg:
        findings.append(f'Registration: {reg.get("measured_real_tiles", 0)} real tile measurements across the declared tile sizes; '
                        f'{reg.get("planted_measured", 0)} measured planted-shift trials. These are apparent offsets, not a regional error sigma.')
    thermal = read('T4')
    if thermal.get('descriptive_spearman') is not None:
        findings.append(f'Thermal comparison: descriptive Spearman coefficient {thermal["descriptive_spearman"]:.3f} '
                        f'across {thermal["supported_footprints"]} supported footprints. Overlap and effective resolution limit interpretation.')
    held = read('T8')
    if held.get('scenes'):
        count = sum(s['meets_declared_targets'] for s in held['scenes'])
        findings.append(f'Held-out scenes: {count}/{len(held["scenes"])} meet the declared research recall/false-alarm targets. '
                        'Annotations, unknown coverage and scene dependence still require review.')
    return findings


def package(output, records, provenance):
    """Single ordinary ZIP, relative paths, per-file hashes and readable entrypoint."""
    decision = verdict(records)
    decision['findings'] = evidence_summary(output, records)
    write_json(output/'verdict.json', decision)
    write_json(output/'campaign.json', dict(provenance=provenance, stages=records, verdict=decision))
    lines = ['# HATI saturation campaign', '', f"Execution: **{decision['execution']}**.",
             f"Scientific verdict: **{decision['scientific_verdict']}**.",
             f"Operational decision: **{decision['operational_verdict']}**.", '',
             decision['interpretation'], '', '| Stage | Status | Result |', '|---|---|---|']
    for r in records:
        lines.append(f"| {r['id']}: {r['title']} | {r['status']} | {r.get('reason', '').replace('|', '/')} |")
    lines += ['', '## Measured findings', '', *['- '+f for f in decision['findings']], '',
              'Unzip the entire folder and open START_HERE.html. The stage folders contain results, '
              'arrays, CSV tables and figures; logs contain the complete process output.',
              'PARTIAL and BLOCKED stages remain unresolved evidence, never passes.', '']
    (output/'VERDICT.md').write_text('\n'.join(lines), encoding='utf-8')
    with (output/'results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id', 'title', 'status', 'reason', 'elapsed_seconds'], extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)
    e = html.escape
    body = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>HATI campaign results</title>',
            '<style>body{max-width:1100px;margin:36px auto;padding:0 24px;font:16px/1.5 system-ui;color:#13283b}'
            'h1{border-bottom:4px solid #db334a}table{border-collapse:collapse;width:100%}'
            'td,th{padding:10px;text-align:left;border-bottom:1px solid #ddd}img{max-width:100%}'
            '.status{font-weight:bold}pre{white-space:pre-wrap;background:#f0f4f8;padding:16px}</style>',
            '<h1>HATI saturation campaign</h1>',
            f'<p class="status">Execution: {e(decision["execution"])} | Science: {e(decision["scientific_verdict"])}</p>',
            f'<p>{e(decision["interpretation"])}</p>',
            '<p><a href="VERDICT.md">Verdict</a> | <a href="results.csv">Result table</a> | '
            '<a href="campaign.json">Full run record</a> | <a href="MANIFEST.sha256">File checksums</a></p>',
            '<table><tr><th>Stage</th><th>Status</th><th>Finding</th></tr>']
    for r in records:
        stage = r['id']
        link = f'stages/{stage}/result.json' if (output/'stages'/stage/'result.json').exists() else f'logs/{stage}.log'
        body.append(f'<tr><td><a href="{e(link)}">{e(stage+": "+r["title"])}</a></td>'
                    f'<td>{e(r["status"])}</td><td>{e(r.get("reason", ""))}</td></tr>')
    body.append('</table><h2>Measured findings</h2><ul>')
    body.extend('<li>'+e(f)+'</li>' for f in decision['findings'])
    body.append('</ul><p>Open each stage result for denominators, controls, support and unresolved requirements.</p>')
    for p in sorted((output/'stages').rglob('*.png')):
        rel = p.relative_to(output).as_posix()
        body.append(f'<h2>{e(rel)}</h2><a href="{e(rel)}"><img src="{e(rel)}" alt="{e(rel)}"></a>')
    body.append('</html>')
    (output/'START_HERE.html').write_text('\n'.join(body), encoding='utf-8')
    files = sorted(p for p in output.rglob('*') if p.is_file() and p.name != 'MANIFEST.sha256' and not p.name.endswith('.part'))
    (output/'MANIFEST.sha256').write_text(''.join(f'{digest(p)}  {p.relative_to(output).as_posix()}\n' for p in files), encoding='utf-8')
    archive = output.with_name(output.name+'_results.zip')
    temporary = archive.with_suffix('.zip.part')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, allowZip64=True) as z:
        for p in [*files, output/'MANIFEST.sha256']:
            z.write(p, (Path(output.name)/p.relative_to(output)).as_posix())
    with zipfile.ZipFile(temporary) as z:
        bad = z.testzip()
        if bad:
            raise ValueError(f'ZIP verification failed: {bad}')
    temporary.replace(archive)
    archive.with_suffix('.zip.sha256').write_text(f'{digest(archive)}  {archive.name}\n', encoding='ascii')
    print(f'\nSEND THIS FILE: {archive}', flush=True)
    return archive


def stage_cached(output, record):
    if record['status'] not in ('COMPLETE', 'PARTIAL', 'BLOCKED'):
        return False
    expected = record.get('artifact_sha256', {})
    return bool(expected) and all((output/p).is_file() and digest(output/p) == sha for p, sha in expected.items())


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle', type=Path, required=True, help='verified hati_diagnostic_bundle.zip from the saved run')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--config', type=Path, default=ROOT/'configs/saturation_campaign.json')
    ap.add_argument('--dem', type=Path, help='native DEM; defaults to the existing Athena asset path')
    ap.add_argument('--thermal', type=Path, help='independent footprint CSV; see campaign runbook')
    ap.add_argument('--held-out', type=Path, help='independent annotated scene manifest')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--export-dir', type=Path, help='also copy ZIP/checksum here, e.g. a Windows /mnt/c/... folder')
    args = ap.parse_args()
    output = (args.output or ROOT/'output/athena/saturation_campaign'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        ap.error('output is not empty; use --resume with identical inputs or choose a new folder')
    if args.resume and not (output/'campaign.json').exists():
        ap.error('--resume requires an existing campaign.json')
    # Keep numerical libraries deterministic and avoid nested BLAS oversubscription.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[key] = '1'
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONHASHSEED'] = '0'
    sources = [*sorted((ROOT/'src').rglob('*.py')), *sorted((ROOT/'scripts').glob('*.py')),
               *sorted((ROOT/'scripts').glob('*.sh')), *sorted((ROOT/'tests').glob('test_*.py'))]
    inputs = dict(bundle=str(args.bundle.resolve()), bundle_sha256=digest(args.bundle), config_sha256=digest(args.config))
    for key in ('dem', 'thermal', 'held_out'):
        p = getattr(args, key)
        inputs[key] = dict(path=str(p.resolve()), sha256=digest(p)) if p else None
    # Resolve/hash the default DEM too: a changed raster must invalidate resume.
    if args.dem is None:
        default_dem = ROOT/'data/athena/NAC_DTM_NOBILE03.TIF'
        if default_dem.is_file():
            args.dem = default_dem
            inputs['dem'] = dict(path=str(args.dem.resolve()), sha256=digest(args.dem))
    if args.held_out:
        manifest = json.loads(args.held_out.read_text(encoding='utf-8'))
        inputs['held_out_payloads'] = [{k: digest(args.held_out.parent/scene[k]) for k in ('bundle', 'labels')}
                                      for scene in manifest['scenes']]
    environment = capture([sys.executable, '-m', 'pip', 'freeze'])
    provenance = dict(started=utc(), inputs=inputs, packages=sorted(environment['stdout'].splitlines()),
                      source_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in sources},
                      revision=capture(['git', 'rev-parse', 'HEAD'])['stdout'].strip(),
                      working_tree=capture(['git', 'status', '--short'])['stdout'],
                      python=sys.version, platform=platform.platform(),
                      configuration=json.loads(args.config.read_text(encoding='utf-8')),
                      sequential=True, numerical_threads=1)
    records = [dict(id='software-'+p.stem[5:], title=p.name, status='PENDING', reason='Not run yet')
               for p in sorted((ROOT/'tests').glob('test_*.py'))]
    records += [dict(id=s, title=t, status='PENDING', reason='Not run yet') for s, t in STAGES]
    if args.resume:
        saved = json.loads((output/'campaign.json').read_text(encoding='utf-8'))
        for k in ('inputs', 'source_sha256', 'python', 'platform', 'configuration', 'packages'):
            if saved['provenance'][k] != provenance[k]:
                ap.error(f'resume provenance differs: {k}; use a new output folder')
        records = saved['stages']; provenance = saved['provenance']
    output.mkdir(parents=True, exist_ok=True)
    for sub in ('logs', 'stages', 'inputs'):
        (output/sub).mkdir(exist_ok=True)
    shutil.copy2(args.config, output/'inputs/campaign_config.json')
    shutil.copy2(args.bundle, output/'inputs/hati_diagnostic_bundle.zip')
    with zipfile.ZipFile(output/'inputs/source_code.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sources:
            archive.write(p, p.relative_to(ROOT).as_posix())
    write_json(output/'inputs/environment.json', environment)
    software_ok = True
    current = None
    try:
        for i, record in enumerate(records):
            current = record
            stage = record['id']
            if args.resume and stage_cached(output, record):
                print(f'Reusing verified stage {stage}', flush=True)
                continue
            record.update(status='RUNNING', reason='Running', started=utc())
            write_json(output/'campaign.json', dict(provenance=provenance, stages=records))
            print(f'\n[{i+1}/{len(records)}] {stage}: {record["title"]}', flush=True)
            start = time.monotonic()
            folder = output/'stages'/stage
            folder.mkdir(exist_ok=True)
            if stage.startswith('software-'):
                command = [sys.executable, str(ROOT/'tests'/record['title'])]
                code = run_logged(command, output/'logs'/f'{stage}.log')
                record.update(status='COMPLETE' if code == 0 else 'FAILED', exit_code=code,
                              reason='Software checks passed' if code == 0 else 'Software checks failed',
                              elapsed_seconds=round(time.monotonic()-start, 2))
                software_ok &= code == 0
            elif not software_ok:
                record.update(status='BLOCKED', reason='A software check failed; science results withheld')
            elif stage in ('T2', 'T3', 'T4', 'T5') and not any(r['id'] == 'T1' and r['status'] in ('COMPLETE', 'PARTIAL') for r in records):
                record.update(status='BLOCKED', reason='T1 baseline prerequisite failed or is unavailable')
            else:
                command = [sys.executable, str(ROOT/'scripts/saturation_experiments.py'),
                           '--stage', stage, '--bundle', str(args.bundle.resolve()),
                           '--config', str(args.config.resolve()), '--output', str(folder),
                           '--campaign', str(output)]
                for key in ('dem', 'thermal', 'held_out'):
                    value = getattr(args, key.replace('-', '_'))
                    if value:
                        command.extend(['--'+key.replace('_', '-'), str(value.resolve())])
                code = run_logged(command, output/'logs'/f'{stage}.log')
                record.update(result_record(stage, record['title'], code, time.monotonic()-start, folder))
            artifacts = [p for p in folder.rglob('*') if p.is_file() and not p.name.endswith('.part')]
            log = output/'logs'/f'{stage}.log'
            if log.exists():
                artifacts.append(log)
            record['artifact_sha256'] = {p.relative_to(output).as_posix(): digest(p) for p in artifacts}
            record['finished'] = utc()
            write_json(output/'campaign.json', dict(provenance=provenance, stages=records))
    except KeyboardInterrupt:
        if current:
            current.update(status='INTERRUPTED', reason='Interrupted by operator; partial files retained')
        print('Interrupted; packaging completed results.', flush=True)
    except Exception as exc:
        if current:
            current.update(status='FAILED', reason=f'{type(exc).__name__}: {exc}')
        print(f'Campaign error: {exc}', file=sys.stderr)
    finally:
        archive = package(output, records, provenance)
        if args.export_dir:
            try:
                args.export_dir.mkdir(parents=True, exist_ok=True)
                for p in (archive, archive.with_suffix('.zip.sha256')):
                    dest = args.export_dir/p.name
                    if dest.resolve() != p.resolve():
                        shutil.copy2(p, dest)
                print(f'WINDOWS/EXPORT COPY: {args.export_dir/archive.name}', flush=True)
            except OSError as exc:
                print(f'Export copy failed ({exc}); the original verified ZIP remains at {archive}', file=sys.stderr)
    return 1 if verdict(records)['failed'] else 2 if verdict(records)['missing'] else 0


if __name__ == '__main__':
    sys.exit(main())
