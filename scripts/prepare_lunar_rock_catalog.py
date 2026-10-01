"""Download NASA Apollo meshes and retain small, attributed shape proxies.

Run explicitly, outside the offline science campaign. Source archive hashes,
metadata and reduction assumptions are recorded. Existing catalogs are never
overwritten. The sample identities and their splits are fixed before any
detector evaluation.

--set v1 builds the original two-rock catalog. --set v2 (the default) holds
all 22 Apollo rocks with exterior models in NASA Astromaterials 3D: the two v1
rocks keep their splits, and the 20 others are split by parent rock with a
fixed seed, about a third to evaluation. --reuse names an existing catalog
folder whose rows are copied byte for byte instead of downloaded again.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.request import urlopen
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.hati_core.rock_scenes import NASA_CREDIT, read_obj, convex_proxy

V1 = [('10017-15', 'development'), ('10021-79', 'evaluation')]
V2_NEW = ['10022-53', '12013-11', '12019-0', '12038-7', '14305-18', '14316-0', '14321-1404', '15016-0', '15556-0',
          '60019-4', '60025-241', '60639-0', '65035-0', '67016-2', '70017-8', '70175-0', '70295-0', '76535-0',
          '78236-0', '79115-0']
V2_SEED, V2_EVALUATION = 20261001, 6
# Rock types from the Astromaterials 3D sample catalog pages (read 1 October 2026); NASA's metadata files omit them.
CLASSIFICATION = {
    '10017-15': 'High-Ti, high-K ilmenite basalt', '10021-79': 'Regolith breccia', '10022-53': 'High-Ti, high-K ilmenite basalt',
    '12013-11': 'Granitic breccia', '12019-0': 'Low-Ti pigeonite basalt', '12038-7': 'Feldspathic basalt',
    '14305-18': 'Crystalline matrix breccia', '14316-0': 'Regolith breccia', '14321-1404': 'Regolith breccia',
    '15016-0': 'Low-Ti olivine basalt', '15556-0': 'Low-Ti basalt', '60019-4': 'Ancient regolith breccia',
    '60025-241': 'Ferroan anorthosite', '60639-0': 'Regolith breccia', '65035-0': 'Glass-coated dimict breccia',
    '67016-2': 'Fragmental breccia', '70017-8': 'High-Ti basalt', '70175-0': 'Pyroclastic-glass breccia',
    '70295-0': 'Regolith breccia', '76535-0': 'Troctolite', '78236-0': 'Shocked norite', '79115-0': 'Regolith breccia'}
ARCHIVE_ROOT = 'https://ares-a3d.s3.us-gov-east-1.amazonaws.com/samples/'


def samples(name):
    if name == 'v1':
        return list(V1)
    order = np.random.default_rng(V2_SEED).permutation(len(V2_NEW))
    evaluation = {V2_NEW[i] for i in order[:V2_EVALUATION]}
    return list(V1)+[(s, 'evaluation' if s in evaluation else 'development') for s in V2_NEW]


def fetch(url):
    with urlopen(url, timeout=120) as response:
        return response.read()


def build(sample, split, out):
    metadata_url = ARCHIVE_ROOT+sample+'/'+sample+'_A3D_EXPLORER_metadata.json'
    metadata_bytes = fetch(metadata_url); metadata = json.loads(metadata_bytes)
    filename = metadata['mesh_filename']
    url = ARCHIVE_ROOT+sample+'/'+metadata['mesh_foldername']+'/'+filename.split('.')[0]+'.zip'
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
    return dict(id=sample+'_convex_proxy', parent_rock=sample.split('-')[0], split=split,
                classification=CLASSIFICATION.get(sample),
                path=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                source_url='https://ares.jsc.nasa.gov/astromaterials3d/sample-details.htm?sample='+sample,
                archive_url=url, archive_sha256=hashlib.sha256(archive).hexdigest(),
                source_obj_sha256=hashlib.sha256(payload).hexdigest(),
                metadata_url=metadata_url, metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
                source_vertices=len(vertices), source_triangles=len(faces),
                proxy_vertices=len(v), proxy_triangles=len(f), credit=NASA_CREDIT,
                geometry_assumptions='96-direction convex proxy discards concavities and fine detail. '
                'Returned, potentially cut or broken laboratory subsample; scene dimensions, burial and orientation '
                'are imposed assumptions. Not a representative polar boulder population.')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--set', choices=['v1', 'v2'], default='v2')
    ap.add_argument('--output', type=Path)
    ap.add_argument('--reuse', type=Path, help='existing catalog folder; its rows are copied instead of downloaded')
    args = ap.parse_args()
    out = args.output or ROOT/f'data/rock_shapes/apollo_proxy_{args.set}'
    if out.exists() and any(out.iterdir()):
        ap.error('output must be new or empty; existing catalogs are frozen')
    reused = {}
    if args.reuse:
        old = json.loads((args.reuse/'catalog.json').read_text(encoding='utf-8'))
        reused = {row['id'].replace('_convex_proxy', ''): row for row in old['meshes']}
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for sample, split in samples(args.set):
        if sample in reused:
            row = dict(reused[sample], split=split, classification=CLASSIFICATION.get(sample))
            for name in (row['path'], sample+'_metadata.json'):
                shutil.copyfile(args.reuse/name, out/name)
            if hashlib.sha256((out/row['path']).read_bytes()).hexdigest() != row['sha256']:
                raise ValueError('reused proxy does not match its recorded checksum: '+sample)
            print('Reusing '+sample, flush=True)
        else:
            row = build(sample, split, out)
        rows.append(row)
    manifest = dict(schema_version=1, created=datetime.now(timezone.utc).isoformat(), meshes=rows,
                    source_policy='https://ares.jsc.nasa.gov/astromaterials3d/faqs.htm',
                    split_rule=('v1 rocks keep their splits; the others are split by parent rock, '
                                f'{V2_EVALUATION} to evaluation by numpy default_rng({V2_SEED}).permutation')
                    if args.set == 'v2' else 'fixed by hand',
                    purpose='Independent synthetic shape-transfer diagnostics; no measured boulder size prior.')
    (out/'catalog.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print('Catalog: '+str(out/'catalog.json'), flush=True)


if __name__ == '__main__':
    main()
