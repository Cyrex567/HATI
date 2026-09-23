"""Best-effort, bounded display snapshots. Never used as scientific inputs.

Only this writer creates observations. The watch server reads them and cannot
send commands to the computation. Sampling/rounding here is for display only.
"""
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from datetime import datetime, timezone


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix=path.stem+'-', suffix='.part', delete=False) as f:
            name = f.name
            json.dump(value, f, allow_nan=False, separators=(',', ':'))
        os.replace(name, path)
    finally:
        if name and Path(name).exists():
            Path(name).unlink()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


class Heartbeat:
    def __init__(self, output):
        self.path = Path(output)/'live/runtime.json'
        self.stop = threading.Event()
        self.stage = None
        self.thread = None

    def write(self, state='running'):
        try:
            atomic_json(self.path, dict(state=state, stage=self.stage, pid=os.getpid(), updated=timestamp()))
        except OSError:
            pass  # A display failure must never change the science run.

    def start(self):
        self.write()
        def loop():
            while not self.stop.wait(2):
                self.write()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def close(self, state='finished'):
        self.stop.set()
        if self.thread:
            self.thread.join()
        self.write(state)


def display_array(array, max_side=160):
    import numpy as np
    a = np.asarray(array, float)
    stride = max(1, int(np.ceil(max(a.shape[-2:])/max_side)))
    a = a[..., ::stride, ::stride]
    rounded = np.round(a, 5)
    return np.where(np.isfinite(rounded), rounded, None).tolist()


def render_inputs(data, run):
    """Return PNG bytes/metadata for the verified stack; source pixels unchanged."""
    import io
    import numpy as np
    from matplotlib.image import imsave
    stack = data['stack']; finite = stack[np.isfinite(stack)]
    low, high = np.percentile(finite, [1, 99]) if finite.size else (0., 1.)
    high = max(high, low+1e-6)
    frames, images = [], {}
    for i, (frame, az, el) in enumerate(zip(stack, data['azimuths'], data['elevations'])):
        normalized = np.clip((np.where(np.isfinite(frame), frame, low)-low)/(high-low), 0, 1)
        rgba = np.stack([normalized]*3+[np.isfinite(frame).astype(float)], axis=-1)
        buffer = io.BytesIO(); imsave(buffer, rgba, format='png')
        name = f'frame_{i:02d}.png'; images[name] = buffer.getvalue()
        frames.append(dict(index=i, pid=run['frames'][i]['pid'], azimuth_deg=float(az), elevation_deg=float(el), image=name))
    metadata = dict(frames=frames, image_shape=list(stack.shape[1:]), pixel_m=run['image_posting_m'],
                    display_range=[float(low), float(high)], demo=bool(run.get('demo', False)),
                    test_pixel=[run['counterfactual']['row_px'], run['counterfactual']['col_px']],
                    meaning='Aligned input intensities. Shared 1st-99th percentile display stretch; transparent pixels have no data.')
    return metadata, images


class LiveFeedback:
    def __init__(self, campaign, stage, *, interval=3.):
        self.folder = Path(campaign)/'live'; self.stage = stage
        self.interval = interval; self.last = -float('inf'); self.last_region = -float('inf')
        self.state = dict(stage=stage)
        self.warned = False

    def failure(self, exc):
        if not self.warned:
            print(f'Live display unavailable: {exc}. Scientific processing continues.', flush=True)
            self.warned = True

    def inputs(self, data, run):
        try:
            # The campaign freezes the source ZIP. Reuse its display images in subsequent workers.
            if (self.folder/'inputs.json').exists():
                return
            metadata, images = render_inputs(data, run)
            self.folder.mkdir(parents=True, exist_ok=True)
            for name, payload in images.items():
                path = self.folder/name; temporary = path.with_suffix('.png.part')
                temporary.write_bytes(payload); temporary.replace(path)
            atomic_json(self.folder/'inputs.json', metadata)
        except Exception as exc:
            self.failure(exc)

    def update(self, *, force=False, **fields):
        try:
            self.state.update(fields)
            now = time.monotonic()
            if not force and now-self.last < self.interval:
                return
            self.last = now
            self.state['updated'] = timestamp()
            atomic_json(self.folder/(self.stage+'.json'), self.state)
        except Exception as exc:
            self.failure(exc)

    def regional(self, info, name):
        try:
            now = time.monotonic()
            if now-self.last_region < self.interval and info['cells_visited'] < info['cells_total']:
                return
            self.last_region = now
            import numpy as np
            sample = info.get('last_fit')
            fit = None
            if sample is not None:
                fit = {k: sample[k] for k in ('row_px', 'col_px', 'root_row_px', 'root_col_px', 'score', 'index',
                                              'height_m', 'width_m', 'contrast', 'identifiability', 'common_fraction',
                                              'endpoint_censored', 'score_scale')}
                fit.update(frames=sample['frames'].tolist(), slope_rc=list(sample['slope_rc']))
                null = sample['residual_null']; projected = sample['projected_template']
                after = null+sample['contrast']*projected
                fit['frame_delta_chi2'] = np.sum(null**2-after**2, axis=1).tolist()
                fit['null_energy'] = float(np.sum(null**2)); fit['fitted_energy'] = float(np.sum(after**2))
                common = sample['common']
                residual = np.full(sample['patch'].shape, np.nan); residual[:, common] = after
                fit.update(observed=display_array(sample['patch']), template=display_array(sample['template']),
                           residual=display_array(residual), common_mask=common.tolist())
                fit['meaning'] = 'Latest assessed cell from the regional scan; winning grid parameters, not measured object dimensions. Residuals are whitened after nuisance projection.'
            self.update(force=True, kind='regional', subrun=name, message='Scanning the regional shadow bank',
                        regional_updated=timestamp(),
                        cells_visited=info['cells_visited'], cells_assessed=info['cells_assessed'], cells_total=info['cells_total'],
                        image_shape=list(info['score'].shape), score=display_array(info['score']),
                        assessment=display_array(info['status']), common_fraction=display_array(info['common_fraction']), fit=fit)
        except Exception as exc:
            self.failure(exc)

    def terrain(self, result, cfg, native_pixel_m, project, test_pixel):
        try:
            import numpy as np
            row, col = (int(np.floor(v)) for v in test_pixel)
            primary = result['primary']
            layers = {k: project(primary[k]) for k in
                      ('slope_deg', 'rms_height_m', 'positive_relief_m', 'negative_relief_m')}
            limits = dict(slope_deg=cfg.slope_limit_deg, rms_height_m=cfg.rms_limit_m,
                          positive_relief_m=cfg.relief_limit_m, negative_relief_m=cfg.relief_limit_m)
            def finite(value):
                return float(value) if np.isfinite(value) else None
            sample = {k: dict(value=finite(a[row, col]), limit=limits[k],
                              ratio=finite(a[row, col]/limits[k])) for k, a in layers.items()}
            self.update(force=True, terrain=dict(updated=timestamp(), row_px=row, col_px=col,
                native_pixel_m=native_pixel_m, diameter_m=result['effective_diameter_m'],
                footprint_resolved=bool(result['footprint_resolved']), configuration_label=cfg.label,
                sample=sample, index=finite(project(result['score'])[row, col]),
                slope=display_array(layers['slope_deg']), rms=display_array(layers['rms_height_m'])))
        except Exception as exc:
            self.failure(exc)

    def adaptive(self, info):
        """A pass snapshot, including the actual larger extraction patch."""
        try:
            import numpy as np
            result = info['pass_result']; sample = info['sample']; fit = None
            if sample is not None:
                best = result['best']; row, col = info['centre']
                null = sample['residual_null']; after = null+best['contrast']*sample['projected_template']
                residual = np.full(sample['patch'].shape, np.nan); residual[:, sample['common']] = after
                fit = dict(row_px=row, col_px=col, root_row_px=row+best['root_offset'][0],
                    root_col_px=col+best['root_offset'][1], score=best['score'], index=None,
                    score_scale=None, height_m=best['height_m'], width_m=best['width_m'],
                    contrast=best['contrast'], identifiability=best['identifiability'],
                    common_fraction=result['common_fraction'], endpoint_censored=result['endpoint_censored'],
                    frames=result['frames'], slope_rc=list(sample['slope_rc']),
                    frame_delta_chi2=best['frame_delta_chi2'], null_energy=result['null_energy'],
                    fitted_energy=result['fitted_energy'], patch_size_px=sample['patch'].shape[1],
                    observed=display_array(sample['patch']), template=display_array(sample['template']),
                    residual=display_array(residual), common_mask=sample['common'].tolist(),
                    meaning='Experimental adaptive pass. Equivalent shadow dimensions and uncalibrated score; no hazard-probability or measured dimension-accuracy claim.')
            self.update(force=True, kind='adaptive', subrun='adaptive context', fit=fit,
                regional_updated=timestamp(), message=f'Adaptive pass {result["scale"]}x at {info["centre"]}',
                adaptive=dict(centre=info['centre'], scale=result['scale'], radius_px=result['radius_px'],
                    support_px=result['support_px'], status=result['status'],
                    height_range_m=result.get('height_range_m'), width_range_m=result.get('width_range_m'),
                    endpoint_reasons=[v['reason'] for v in result.get('endpoints', [])],
                    spatial_degree=result.get('spatial_degree')))
        except Exception as exc:
            self.failure(exc)

    def adaptive_progress(self, info):
        try:
            last = info['last']; fields = {}
            if self.state.get('adaptive', {}).get('centre') != last['centre']:
                fields['fit'] = None  # a cached cell has no current fit snapshot
            self.update(kind='adaptive', cells_visited=info['processed'], cells_total=info['requested'],
                cells_assessed=info['processed'], score=display_array(info['score']),
                assessment=display_array(info['status']),
                message=f'Adaptive context: {info["processed"]}/{info["requested"]}; {last["status"]}', **fields)
        except Exception as exc:
            self.failure(exc)

    def field(self, title, array, **metrics):
        try:
            self.update(force=True, kind='field', message=title, field=display_array(array), field_title=title,
                        metrics=metrics, fit=None)
        except Exception as exc:
            self.failure(exc)
