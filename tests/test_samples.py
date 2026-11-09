"""Checks against OME-TIFFs from elsewhere: Bio-Formats exports, pyvips, microscopes.

Skipped unless ``OME_TIFF_SAMPLES`` names a directory of them:

    OME_TIFF_SAMPLES=~/ome-tiff-samples uv run pytest

Every ``.tif``/``.tiff`` under it is read as the action would read it, header and
pairing first, then a corner of every image's pixels. A file the converter refuses must
be refused with `Unreadable`, a reason it can give, rather than with anything else.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

import app as M
from ome_tiff import Unreadable, match_series, read_ome

ROOT = os.environ.get("OME_TIFF_SAMPLES")
SAMPLES = (
    sorted(p for p in Path(ROOT).expanduser().rglob("*") if p.name.lower().endswith(M.READABLE_SUFFIXES))
    if ROOT
    else []
)

pytestmark = [
    pytest.mark.samples,
    pytest.mark.skipif(not SAMPLES, reason="set OME_TIFF_SAMPLES to a directory of OME-TIFFs"),
]


@pytest.mark.parametrize("path", SAMPLES, ids=[p.name for p in SAMPLES])
def test_a_sample_is_read_or_refused_with_a_reason(path):
    with tifffile.TiffFile(path) as tif:
        try:
            ome = read_ome(tif, path.name)
            images = match_series(tif, ome, path.name)
        except Unreadable:
            return
        for item in images:
            if item.series is None:
                assert item.problem
                continue
            plan = M.plan_image(item)
            for rgb_channel in plan.rgb_channels:
                array, _, _ = M.field_array(
                    M.series_array(item.series, item.split), rgb_channel, plan.positions.descending
                )
                corner = array.isel(y=slice(0, 16), x=slice(0, 16))
                read = dict(zip(corner.dims, np.asarray(corner).shape))
                assert (read["y"], read["x"]) == (
                    min(16, array.sizes["y"]),
                    min(16, array.sizes["x"]),
                )
