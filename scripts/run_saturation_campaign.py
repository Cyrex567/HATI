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
import signal
import subprocess
import sys
import time
import zipfile
from live_feedback import Heartbeat

ROOT = Path(__file__).resolve().parent.parent
STAGES = [('maps', 'Replay terrain, shadow and fused maps'),
          ('T1', 'Physical response and signed frame contributions'),
          ('T2', 'Matched frame ablations'), ('T3', 'Regional geometry stress'),
          ('T4', 'Independent thermal comparison'), ('T5', 'Expanded width profiles'),
          ('T6', 'Height profiles and endpoint support'),
          ('T7', 'Local registration and planted shifts'),
          ('T8', 'Held-out annotated scenes'),
          ('T9', 'Adaptive context and joint dimension refinement'),
          ('T10', 'Independent 3D rocks and adaptive controls'),
          ('T11', 'Withheld illumination and changing-background alternatives'),
          ('T12', 'Residual noise scale and controls at the measured level'),
          ('T13', 'Compact, extended and depression models compete'),
          ('T14', 'Shape from shading as a structural null'),
          ('T16', 'No-caster scenes through the full adaptive procedure')]


def select_stages(text):
    """Parse --stages; returns the chosen science stages in their declared order."""
    known = [s for s, _ in STAGES]
    if not text:
        return known
    chosen = [s.strip() for s in text.split(',') if s.strip()]
    unknown = [s for s in chosen if s not in known]
    if unknown or not chosen:
        raise ValueError(f'unknown stages {unknown}; choose from {",".join(known)}')
    return [s for s in known if s in chosen]


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
                                   encoding='utf-8', errors='replace', bufsize=1,
                                   start_new_session=os.name != 'nt',
                                   creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == 'nt' else 0)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                stream.write(line); stream.flush()
            return process.wait()
        except BaseException:
            # T9 has child workers: interrupt the whole owned process tree.
            # Never leave them consuming resources after the campaign packages.
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
            else:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
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
    structured = [r for r in controls if r.get('kind') == 'structured_null' and r['status'] == 'assessed']
    if structured:
        findings.append(f'Changing no-caster controls: {sum(r["warning_roots"] > 0 for r in structured)}/{len(structured)} '
                        'assessed trials contain warning roots. These are a direct test of background-model selectivity.')
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
    adaptive = read('T9')
    if adaptive:
        findings.append(f'Adaptive context: {adaptive.get("processed", 0)}/{adaptive.get("requested", 0)} requested cells processed; '
                        f'{adaptive.get("unprocessed", 0)} remain queued. States: {adaptive.get("state_counts", {})}. '
                        'Equivalent dimensions and scores are experimental; baseline fusion is unchanged.')
    for row in read('T10').get('controls', []):
        findings.append(f'3D/control {row["kind"]}: {row["trials_with_warning"]}/{row["trials"]} trials contain adaptive warnings in fixed ROIs; '
                        f'{row.get("trials_with_context_supported", 0)} reach context-supported status; '
                        f'{row["dimension_trials"]} trials provide central dimension estimates; '
                        f'{row.get("trials_with_unresolved_cells", 0)} contain unresolved cells. These are ROI diagnostics, not field false-alarm rates.')
    prediction = read('T11')
    if prediction:
        findings.append(f'Withheld illumination: {sum(r["assessed"] for r in prediction.get("predictions", []))}/{prediction.get("trials", 0)} '
                        'declared trials assessed. Predictive errors compare correct geometry, rotated geometry and static background at fixed scales.')
    noise = read('T12')
    if noise.get('measured_pooled_sigma') is not None:
        findings.append(f'Residual scale: {noise["measured_pooled_sigma"]:.4f} (plane null) against an assumed {noise["assumed_sigma"]:.4f}, '
                        f'{noise["ratio_to_assumed"]:.2f} times, over {noise["patches"]} patches chosen by support, illumination and slope. '
                        'It includes unmodelled structure, so it bounds independent noise from above.')
        rescaled = noise.get('rescaled_baseline')
        if rescaled:
            findings.append(f'If the whole excess were noise, baseline exceedance would move from {rescaled["fraction_above_threshold_assumed"]:.1%} '
                            f'to {rescaled["fraction_above_threshold_rescaled"]:.1%} (median score {rescaled["median_score_assumed"]:.1f} to '
                            f'{rescaled["median_score_rescaled"]:.1f}). Upper-bound arithmetic, not a corrected map.')
        relief = noise.get('relief_consistency') or {}
        if relief.get('available'):
            findings.append(f'Sun-consistent shading of a shared slope field explains {relief["explained_true_geometry"]:.1%} of the residual '
                            f'with the measured geometry, against {relief["explained_shuffled_median"]:.1%} (median) when the '
                            f'{relief["shuffled_assignments"]} reassignments of Sun directions to frames are tried; the best any two-component '
                            f'model reaches is {relief["explained_best_rank2"]:.1%}. After removing it the residual scale is '
                            f'{relief["relief_corrected_sigma"]:.4f}, still an upper bound on independent noise.')
        for p in noise.get('control_passes', []):
            parts = [f'{s["kind"]}{" "+str(s["height_m"])+" m" if s["kind"] == "caster" else ""} '
                     f'{s["trials_with_warning_roots"]}/{s["assessed"]}' for s in p['scenarios']]
            findings.append(f'Controls ({noise.get("control_generator", "control")} generator), rendered sigma {p["render_noise"]:.4f} '
                            f'and model sigma {p["model_noise"]:.4f}: ' + '; '.join(parts) + ' trials with warning roots.')
    compete = read('T13')
    if compete.get('margins'):
        fired = compete.get('relief_scenes_with_warnings', {})
        findings.append(f'Relief with no caster: {fired.get("with_warnings", 0)}/{fired.get("scenes", 0)} generated mound, bowl and ripple '
                        'scenes produce warning roots in the unchanged detector.')
        for truth, row in compete.get('evaluation_confusion', {}).items():
            total = sum(row.values())
            findings.append(f'Held-out simulated {truth} scenes: ' + ', '.join(f'{v}/{total} {k.replace("_", " ")}' for k, v in row.items()) + '.')
        for group, row in (compete.get('athena_summary') or {}).items():
            if row.get('cells'):
                findings.append(f'Athena cells ({group.replace("_", " ")}, {row["cells"]}): ' + ', '.join(
                    f'{row[c]:.1%} {c.replace("_", " ")}' for c in ('rock_like', 'relief_like', 'ambiguous', 'none')) +
                    '. Research labels from withheld-frame prediction; relief-like goes to the terrain module as a slope hazard.')
        signs = ((compete.get('athena_summary') or {}).get('all_sampled') or {}).get('relief_signs') or {}
        if signs.get('protrusion') is not None:
            findings.append(f'Relief-like Athena cells by sign: {signs["protrusion"]:.1%} protrusion, {signs["depression"]:.1%} depression, '
                            f'{signs["undetermined"]:.1%} undetermined (extended against depression model, withheld-frame prediction).')
        for kind, row in compete.get('evaluation_sign_confusion', {}).items():
            findings.append(f'Held-out simulated {kind}s called relief-like: ' + ', '.join(f'{v} {s}' for s, v in row.items()) + '.')
        cell = compete.get('touchdown_cell')
        if cell:
            sign = f', {cell["relief_sign"]}' if cell.get('relief_sign') else ''
            findings.append(f'Cell nearest the touchdown: {str(cell["label"]).replace("_", " ")}{sign} (baseline score {cell["baseline_score"]:.1f}).')
    sfs = read('T14')
    if sfs.get('sfs'):
        e = sfs.get('exceedance', {})
        findings.append(f'Shape from shading explains {sfs["sfs"]["explained_fraction"]:.1%} of the frame-to-frame brightness variation; '
                        f'metre-scale slope median {sfs["slope_deg"]["median"]:.1f} deg, 90th percentile {sfs["slope_deg"]["p90"]:.1f} deg.')
        if e.get('after_assumed_sigma') is not None:
            before = f'from {e["before_assumed_sigma"]:.1%} ' if e.get('before_assumed_sigma') is not None else ''
            findings.append(f'With relief shading removed, exceedance moves {before}to {e["after_assumed_sigma"]:.1%} at the assumed sigma and '
                            f'{e["after_measured_sigma"]:.1%} at the residual scale measured after correction '
                            f'({sfs["residual_scale_after"]["pooled_sigma"] or float("nan"):.4f}).')
        for height, row in sfs.get('injection_recovery', {}).items():
            m, a, u = row['corrected_measured'], row['corrected_assumed'], row['original_assumed']
            findings.append(f'Injected {height} rocks recovered on quiet sites: {m["recovered"]}/{m["quiet_sites"]} after the relief correction '
                            f'at the residual scale measured after it, {a["recovered"]}/{a["quiet_sites"]} at the assumed sigma, '
                            f'{u["recovered"]}/{u["quiet_sites"]} without the correction.')
    for row in read('T16').get('summaries', []):
        label = row['kind'] if row['height_m'] is None else f'{row["kind"]} {row["height_m"]} m'
        gate = '' if row['height_m'] is not None else (' (within the declared gate)' if row['within_declared_gate'] else ' (above the declared gate)')
        findings.append(f'Adaptive null {row["noise_pass"]}, {label}: {row["trials_with_context_supported"]}/{row["trials"]} trials reach '
                        f'context-supported status{gate}; {row["trials_with_adaptive_warning"]} contain adaptive warnings.')
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
    ap.add_argument('--rock-catalog', type=Path, help='verified local OBJ manifest; defaults to the bundled Apollo shape proxies')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--export-dir', type=Path, help='also copy ZIP/checksum here, e.g. a Windows /mnt/c/... folder')
    ap.add_argument('--stages', help='comma-separated subset of science stages, e.g. T1,T12,T16; '
                    'software checks always run. Default: every stage')
    args = ap.parse_args()
    try:
        selected = select_stages(args.stages)
    except ValueError as exc:
        ap.error(str(exc))
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
               *sorted((ROOT/'scripts').glob('*.sh')), *sorted((ROOT/'tests').glob('test_*.py')),
               ROOT/'dashboard/hati_watch.py', *sorted((ROOT/'dashboard/watch').glob('*')),
               ROOT/'dashboard/static/assets/hati_logo.png']
    inputs = dict(bundle=str(args.bundle.resolve()), bundle_sha256=digest(args.bundle), config_sha256=digest(args.config))
    if args.rock_catalog is None:
        default_catalog = ROOT/'data/rock_shapes/apollo_proxy_v1/catalog.json'
        if default_catalog.is_file():
            args.rock_catalog = default_catalog
    for key in ('dem', 'thermal', 'held_out', 'rock_catalog'):
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
    catalog_payloads = []
    if args.rock_catalog:
        sys.path.insert(0, str(ROOT))
        from src.hati_core.rock_scenes import load_catalog
        load_catalog(args.rock_catalog)  # validate hashes, path containment and split identities
        catalog = json.loads(args.rock_catalog.read_text(encoding='utf-8'))
        catalog_payloads = [(args.rock_catalog.parent/row['path']).resolve() for row in catalog['meshes']]
        inputs['rock_payloads'] = {p.relative_to(args.rock_catalog.parent.resolve()).as_posix(): digest(p) for p in catalog_payloads}
    environment = capture([sys.executable, '-m', 'pip', 'freeze'])
    provenance = dict(started=utc(), inputs=inputs, packages=sorted(environment['stdout'].splitlines()),
                      source_sha256={p.relative_to(ROOT).as_posix(): digest(p) for p in sources},
                      revision=capture(['git', 'rev-parse', 'HEAD'])['stdout'].strip(),
                      working_tree=capture(['git', 'status', '--short'])['stdout'],
                      python=sys.version, platform=platform.platform(),
                      configuration=json.loads(args.config.read_text(encoding='utf-8')),
                      sequential=True, numerical_threads=1, selected_stages=selected)
    records = [dict(id='software-'+p.stem[5:], title=p.name, status='PENDING', reason='Not run yet')
               for p in sorted((ROOT/'tests').glob('test_*.py'))]
    records += [dict(id=s, title=t, status='PENDING', reason='Not run yet') for s, t in STAGES if s in selected]
    if args.resume:
        saved = json.loads((output/'campaign.json').read_text(encoding='utf-8'))
        for k in ('inputs', 'source_sha256', 'python', 'platform', 'configuration', 'packages'):
            if saved['provenance'][k] != provenance[k]:
                ap.error(f'resume provenance differs: {k}; use a new output folder')
        records = saved['stages']; provenance = saved['provenance']
        if args.stages and provenance.get('selected_stages') and provenance['selected_stages'] != selected:
            print(f'Resuming with the original stage selection {provenance["selected_stages"]}; --stages is ignored on resume.', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    for sub in ('logs', 'stages', 'inputs'):
        (output/sub).mkdir(exist_ok=True)
    shutil.copy2(args.config, output/'inputs/campaign_config.json')
    shutil.copy2(args.bundle, output/'inputs/hati_diagnostic_bundle.zip')
    if args.rock_catalog:
        dest = output/'inputs/rock_catalog'; dest.mkdir(exist_ok=True)
        shutil.copy2(args.rock_catalog, dest/'catalog.json')
        for p in catalog_payloads:
            target = dest/p.relative_to(args.rock_catalog.parent.resolve())
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, target)
    with zipfile.ZipFile(output/'inputs/source_code.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in sources:
            archive.write(p, p.relative_to(ROOT).as_posix())
    write_json(output/'inputs/environment.json', environment)
    software_ok = True
    current = None
    heartbeat = Heartbeat(output)
    heartbeat.start()
    try:
        for i, record in enumerate(records):
            current = record
            stage = record['id']
            heartbeat.stage = stage
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
            elif stage in ('T2', 'T3', 'T4', 'T5', 'T9') and not any(r['id'] == 'T1' and r['status'] in ('COMPLETE', 'PARTIAL') for r in records):
                record.update(status='BLOCKED', reason='T1 baseline prerequisite failed or is unavailable')
            else:
                command = [sys.executable, str(ROOT/'scripts/saturation_experiments.py'),
                           '--stage', stage, '--bundle', str(args.bundle.resolve()),
                           '--config', str(args.config.resolve()), '--output', str(folder),
                           '--campaign', str(output)]
                for key in ('dem', 'thermal', 'held_out', 'rock_catalog'):
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
        heartbeat.close('interrupted' if current and current['status'] == 'INTERRUPTED' else 'finished')
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
