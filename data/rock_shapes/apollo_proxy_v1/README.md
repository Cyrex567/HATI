# Apollo shape proxies for synthetic experiments

These two OBJ files are small derived geometry assets from NASA Astromaterials 3D. `catalog.json` records the exact source URLs, archive and original OBJ hashes, source metadata hashes, attribution and the imposed geometry assumptions.

- **10017,15** (parent 10017): development split.
- **10021,79** (parent 10021): evaluation split. The campaign's Apollo-derived controls use only this split.

The conversion selects extreme source vertices in 96 fixed directions and builds their convex hull. Concavities and fine details are removed. Source coordinates keep their original units; scene placement deliberately rescales each axis to declared synthetic dimensions and varies yaw and burial. The resulting scenes are tests of assumed shapes, not measurements of boulders' height or of their lunar population distribution. NASA's samples are returned laboratory subsamples and can include broken or cut surfaces. One evaluation parent cannot support a claim of morphology generalization.

Rebuild into a **new directory** with `python scripts/prepare_lunar_rock_catalog.py --output /path/to/new/catalog-folder`. The science runner never downloads source meshes. Git preserves OBJ and metadata bytes to keep their checksums stable across Windows and WSL.

NASA credit: These 3D reconstructed image data were produced at the Lunar Sample Laboratory Facility for Astromaterials 3D in NASA's Acquisition & Curation Office and were funded by NASA Planetary Data Archiving, Restoration, and Tools Program, Proposal No.: 15-PDART15_2-0041.

Source and reuse information: [NASA Astromaterials 3D FAQ](https://ares.jsc.nasa.gov/astromaterials3d/faqs.htm).
