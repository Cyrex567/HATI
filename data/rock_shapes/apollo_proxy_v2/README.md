# Apollo shape proxies, 22 rocks

These OBJ files are small derived geometry assets from NASA Astromaterials 3D: every Apollo rock there with an exterior (photogrammetry) model on 1 October 2026. `catalog.json` records, for each rock, its source page, archive URL and hash, the original OBJ hash, the metadata hash, its rock type, attribution and the imposed geometry assumptions.

- **Development (15):** 10017,15; 12013,11; 12019,0; 12038,7; 14305,18; 14321,1404; 15016,0; 15556,0; 60019,4; 60639,0; 65035,0; 67016,2; 70017,8; 76535,0; 78236,0. Planted calibration rocks (T14, T18) draw their bodies from these.
- **Evaluation (7):** 10021,79; 10022,53; 14316,0; 60025,241; 70175,0; 70295,0; 79115,0. Only evaluation scenes use these (T10's Apollo-derived controls).

The two rocks of `apollo_proxy_v1` keep their splits and their files, copied byte for byte. The other 20 were split by parent rock before any detector evaluation, 6 to evaluation, by `numpy.random.default_rng(20261001).permutation`; `scripts/prepare_lunar_rock_catalog.py` holds the rule.

The conversion selects extreme source vertices in 96 fixed directions and builds their convex hull. Concavities and fine details are removed. Source coordinates keep their original units; the planted-rock sampler rescales each body to drawn dimensions and varies yaw and burial. The scenes test assumed shapes, not measured boulder heights or the polar boulder population. NASA's samples are returned laboratory pieces; samples other than the ",0" main masses can include cut or broken surfaces.

Rebuild into a new directory with `python scripts/prepare_lunar_rock_catalog.py --set v2 --reuse data/rock_shapes/apollo_proxy_v1 --output /path/to/new/folder`. The science runner never downloads source meshes. Git preserves OBJ and metadata bytes to keep their checksums stable across Windows and WSL.

NASA credit: These 3D reconstructed image data were produced at the Lunar Sample Laboratory Facility for Astromaterials 3D in NASA's Acquisition & Curation Office and were funded by NASA Planetary Data Archiving, Restoration, and Tools Program, Proposal No.: 15-PDART15_2-0041.

Source and reuse information: [NASA Astromaterials 3D FAQ](https://ares.jsc.nasa.gov/astromaterials3d/faqs.htm).
