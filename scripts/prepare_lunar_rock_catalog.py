"""Download two NASA Apollo meshes and retain small, attributed shape proxies.

Run explicitly, outside the offline science campaign. Source archive hashes,
metadata and reduction assumptions are recorded. Existing catalogs are never
overwritten. The sample identities are fixed before any detector evaluation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.hati_core.rock_scenes import NASA_CREDIT, read_obj, convex_proxy


def fetch(url):
    with urlopen(url, timeout=60) as response:
        return response.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=ROOT/'data/rock_shapes/apollo_proxy_v1')
    args = ap.parse_args(); out = args.output
    if out.exists() and any(out.iterdir()):
        ap.error('output must be new or empty; existing catalogs are frozen')
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    root = 'https://ares-a3d.s3.us-gov-east-1.amazonaws.com/samples/'
    for sample, split in [('10017-15', 'development'), ('10021-79', 'evaluation')]:
        metadata_url = root+sample+'/'+sample+'_A3D_EXPLORER_metadata.json'
        metadata_bytes = fetch(metadata_url); metadata = json.loads(metadata_bytes)
        filename = metadata['mesh_filename']
        url = root+sample+'/'+metadata['mesh_foldername']+'/'+filename.split('.')[0]+'.zip'
        print('Downloading NASA mesh '+sample, flush=True)
        archive = fetch(url)
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            names = [n for n in z.namelist() if Path(n).name == filename]
            if len(names) != 1:
                raise ValueError('expected one declared OBJ payload')
            payload = z.read(names[0])  # no archive paths are extracted
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source.obj'; source.write_bytes(payload)
            vertices, faces = read_obj(source)
        v, f = convex_proxy(vertices)
        proxy = '# NASA Apollo '+sample+'; 96-direction convex shape proxy; original units\n'
        proxy += ''.join('v '+' '.join(f'{x:.12g}' for x in row)+'\n' for row in v)
        proxy += ''.join('f '+' '.join(str(int(x)+1) for x in row)+'\n' for row in f)
        path = out/(sample+'_proxy.obj'); path.write_text(proxy, encoding='utf-8', newline='\n')
        (out/(sample+'_metadata.json')).write_bytes(metadata_bytes)
        rows.append(dict(id=sample+'_convex_proxy', parent_rock=sample.split('-')[0], split=split,
                         path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                         source_url='https://ares.jsc.nasa.gov/astromaterials3d/sample-details.htm?sample='+sample,
                         archive_url=url, archive_sha256=hashlib.sha256(archive).hexdigest(),
                         source_obj_sha256=hashlib.sha256(payload).hexdigest(),
                         metadata_url=metadata_url, metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
                         source_vertices=len(vertices), source_triangles=len(faces),
                         proxy_vertices=len(v), proxy_triangles=len(f), credit=NASA_CREDIT,
                         geometry_assumptions='96-direction convex proxy discards concavities and fine detail. '
                         'Returned, potentially cut or broken laboratory subsample; scene dimensions, burial and orientation '
                         'are imposed assumptions. Not a representative polar boulder population.'))
    manifest = dict(schema_version=1, created=datetime.now(timezone.utc).isoformat(), meshes=rows,
                    source_policy='https://ares.jsc.nasa.gov/astromaterials3d/faqs.htm',
                    purpose='Independent synthetic shape-transfer diagnostics; no measured boulder size prior.')
    (out/'catalog.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print('Catalog: '+str(out/'catalog.json'), flush=True)


if __name__ == '__main__':
    main()
