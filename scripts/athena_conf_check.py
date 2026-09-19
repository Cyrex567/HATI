"""Georeferenced categorical stereo-confidence report; no ordinal percentiles."""
import json
from pathlib import Path
import sys
import rasterio
from pyproj import CRS,Transformer

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
import athena_counterfactual as ac
from src.hati_core.stereo_confidence import SOURCE,category_counts,sample_categories


def main():
    path=ROOT/'data/athena/NAC_DTM_NOBILE03_CONF.IMG'
    with rasterio.open(path) as src:
        conf=src.read(1,masked=True).astype(float).filled(float('nan'))
        # Use the product CRS and actual touchdown, not historical pixel constants.
        projected=CRS.from_user_input(src.crs)
        x,y=Transformer.from_crs(projected.geodetic_crs,projected,always_xy=True).transform(ac.TD_LON,ac.TD_LAT)
        row,col=src.index(x,y)
        report=dict(source=str(path),lookup_source=SOURCE,
                    touchdown=sample_categories(conf,row,col),scene_categories=category_counts(conf),
                    interpretation='Codes are documented categories. Successful correlation is not a local height-accuracy or landing-safety guarantee.')
    out=ROOT/'output/athena';out.mkdir(parents=True,exist_ok=True)
    (out/'athena_conf_categories.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':main()
