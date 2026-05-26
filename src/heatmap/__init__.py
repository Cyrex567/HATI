"""HATI v2.0 sub-resolution hazard heatmap.

Modules
-------
dem_features : Tier-1 DEM-derived channels (MDS, RMS slope, IQR slope,
               IQR curvature, RMS planar deviation, TPI, TRI).
fusion       : literature-anchored fusion of channels into a [0, 1] heatmap.

See heatmap_explained.md (companion document) for the method walkthrough.
"""

from . import dem_features  # noqa: F401
from . import fusion  # noqa: F401
