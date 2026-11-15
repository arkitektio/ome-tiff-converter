"""Convert an OME-TIFF into one ``ArrayDataset`` per OME Image.

A TIFF with no OME-XML is converted too, one dataset per tifffile series, with only what
its header says: its shape, its pixel type, and an ImageJ calibration where there is one
(`ome_tiff.plain_images`). It is not placed, and its channels are not named.

The same contract as the ND2, LIF and OME converters, read natively: no Bio-Formats and
no JVM. tifffile reads each plane (or tile) out of the file on demand, so the pixels are
never decoded into RAM or copied to a scratch store before upload, and ome-types reads
the OME-XML the file carries — the same document Bio-Formats would have produced, here
read as written rather than re-derived. See `ome_tiff` for how series and Images are
paired, and what is refused.

Everything the file says about an image is pinned to a *coordinate*: facts true of the
whole image (its OME record, the microscope, the objective, the stage, the environment,
when it was acquired) hang off the anchor that pins no axis; facts true of one channel
(its histogram, name, fluor, detector, light path) hang off that channel's anchor.
Physical scale is an edge, not a property. An image shares the file's micrometer world
only where its Planes say where it sits along EVERY spatial axis it has: x and y from the
first plane's position, and z for a stack from every plane's (`world_offsets`).
Otherwise it is registered, calibrated, into a micrometer space of its own, and staged
there alone.

A file of several Images (the fields of a multipoint run, the wells of a plate, a slide
and its label) uploads each as a dataset of its own. Images whose fields touch share one
mosaic scene; every other image gets a scene of its own — see `scene_groups`.

Nothing about a scene is bootstrapped: every lens and layer is authored explicitly,
**one layer per channel**. The only exception is a packed 8-bit RGB pixel — see
`packs_one_pixel`.

Absent facts stay absent. An image whose pixels the file does not calibrate is **not**
registered into the world: ``scale={x: 1.0}`` is not a neutral default but the claim that
one pixel is one micrometer. Such a dataset is staged in its own pixel grid instead.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import struct
import tempfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np
import xarray as xr
from arkitekt import App, DescriptorOperator, Requires, Task, run
from mikro import Mikro, build_pyramid, canonical, mikro_service, scales_from, space_3d
from mikro.api.schema import (
    AcquisitionWindowInput,
    ArrayDataset,
    AxisAnchorInput,
    AxisType,
    Blending,
    ColorMap,
    CoordinateAnchorInput,
    File,
    LabelInput,
    OmeMetadataInput,
    OptikitStateInput,
    PlacementValidity,
    PreferredView,
    ProjectionMode,
    Scene,
    SourceFileInput,
    StageStateInput,
    ValueHistogramInput,
    ValueRelation,
)
from mikro.arkitekt.specs import EXTENSION
from mikro.inputs.render import (
    RGB_PRIMARIES,
    color_for_wavelength,
    composite_graph,
    fallback_color,
    rgba,
)
from mikro.pyramid import axes_for

from converter_metadata import device, devices, setting, text
from logos import declare_logos, logo_of
from ome_metadata import (
    COLOR_BY_EMISSION,
    FAR_RED,
    channel_lightpath,
    channel_optikit,
    find_instrument,
    find_objective,
    ome_rgba,
    qty,
    to_micrometer,
    um,
    unit_symbol,
)
from ome_tiff import (
    POSITION_CONVENTION,
    Calibration,
    OmeImage,
    PlanePositions,
    Unreadable,
    Window,
    acquisition_windows,
    image_footprint,
    plane_positions,
    read_calibration,
    read_images,
)

logger = logging.getLogger(__name__)

# The version of the app. A release is the commit tagged with it (v1.2.3).
__version__ = "0.1.1"

app = App(
    "ome-tiff-converter",
    __version__,
    description=(
        "Converts OME-TIFF files into mikro datasets natively, one per OME Image, with "
        "their OME metadata: channels, instrument, light path, stage positions and "
        "plane times. Plain TIFFs without OME-XML convert too, one dataset per series."
    ),
    author="jhnnsrs",
)
app.service(mikro_service)
declare_logos(app)



# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The axes the world below is built over. Every scale and offset is derived by
# intersecting an image's own dims with this set: `register` raises if a scale or an
# offset names an axis the two spaces do not share.
WORLD_AXES = frozenset({"z", "y", "x"})

# The axis type for each name this converter can emit. Stated rather than inferred:
# the library's fallback calls every unrecognised name SPACE, which is right for
# z/y/x and silently wrong for anything else — and it matters twice over, because
# the pyramid coarsens exactly the SPACE axes.
AXIS_TYPES: dict[str, AxisType] = {
    "t": AxisType.TIME,
    "c": AxisType.CHANNEL,
    "z": AxisType.SPACE,
    "y": AxisType.SPACE,
    "x": AxisType.SPACE,
}

AXIS_LONG_NAMES = {
    "t": "time",
    "c": "channel",
    "z": "depth",
    "y": "height",
    "x": "width",
}

# `ValueHistogramInput.from_array` calls `np.asarray`, so it is the one place a lazy
# image is forced into memory. Above this many bytes per channel slice the histogram
# is computed from the finest pyramid level that fits instead.
HISTOGRAM_BUDGET_BYTES = 256 * 1024 * 1024

# Past this many layers a scene stops being useful: a 96-well run of four channels is
# 384. The world still holds every registration and outlives the run, so a subset can be
# staged later.
MAX_SCENE_LAYERS = 48

# How far apart two fields may be and still count as touching, as a fraction of the
# field: a grid acquired with a little overlap or a little gap is one mosaic, the wells
# of a plate are not.
MOSAIC_GAP_FRACTION = 0.1

# An automatic pyramid halves until its coarsest level fits a tile of this many pixels
# on its longer side: enough to open a 12k stitched image in one fetch, and no more
# levels than a 512-pixel field needs.
PYRAMID_TOP_PIXELS = 512

# The longer side of a scene snapshot, in pixels. A mosaic's tiles share it: each is
# rendered at its share of the side, so 48 tiles cost what one field does.
SNAPSHOT_PIXELS = 1024


# The suffixes a TIFF goes by. Which of them is an OME-TIFF is decided by its content:
# a ``.tif`` written by Bio-Formats or tifffile is one as much as a ``.ome.tif`` is.
READABLE_SUFFIXES = (".tif", ".tiff", ".tf2", ".tf8", ".btf")

# A tiled image is stored in tiles of 256 or 512 pixels, and tifffile's store hands out
# one chunk per tile: 40 000 of them for a whole slide, each its own task. The upload
# reads them in blocks of at least this many pixels a side instead.
MIN_CHUNK_PIXELS = 2048


# ---------------------------------------------------------------------------
# The array
# ---------------------------------------------------------------------------


def series_array(
    series: Any, split: Sequence[tuple[str, int]] | None = None
) -> xr.DataArray:
    """The base level of one TIFF series, lazily: read where it lies when a chunk is
    computed.

    ``aszarr(level=0)`` and not ``aszarr()``: a pyramidal OME-TIFF's store is otherwise a
    multiscale group rather than an array. Its stored levels are not reused — the
    converter builds its own, to the depth and method it states (`image_pyramid`).

    Dims are tifffile's, in stored order (``T C Z Y X``, ``Y X S`` …), with singleton
    axes already dropped. A generic series' page axis ``I`` is unfolded by `split`
    (`ome_tiff.page_split`) into the T, C and Z the OME record says it holds. The
    TiffFile must stay open while the array is read.
    """
    import dask.array as da
    import zarr

    array = da.from_zarr(zarr.open(series.aszarr(level=0), mode="r"))
    dims = [str(axis) for axis in series.axes]
    if split and "I" in dims:
        at = dims.index("I")
        array = array.reshape(
            (*array.shape[:at], *(size for _, size in split), *array.shape[at + 1 :])
        )
        dims = [*dims[:at], *(axis for axis, _ in split), *dims[at + 1 :]]
    merged = {
        dims.index(axis): max(array.chunks[dims.index(axis)][0], MIN_CHUNK_PIXELS)
        for axis in ("Y", "X")
        if axis in dims
    }
    if merged:
        array = array.rechunk(merged)
    return xr.DataArray(array, dims=dims)


def field_array(
    whole: xr.DataArray,
    rgb_channel: int | None,
    descending_z: bool,
) -> tuple[xr.DataArray, dict[str, str], bool]:
    """One image (and, for several colour pictures, one of them) as a named array, its
    extra axis types, and whether ``c`` is a packed sample axis.

    The same reduction the ND2 converter makes: dims lower-cased, ``c`` kept even at
    extent 1 (the per-channel anchors key off it), any other singleton dropped, and
    anything left that is not t/c/z/y/x declared INDEX so the pyramid does not halve it —
    a FLIM file's lifetime bins, say, which tifffile reads out of an OME modulo
    annotation as an axis of their own.

    An RGB pixel is three samples — tifffile's ``S``. With several channels of them
    (``C`` × ``S``) each channel is a picture of its own, and is converted as one
    (`rgb_channel`). A stack recorded top-down (`descending_z`) is turned over, so that z
    grows with its index, as the scale registering it claims.
    """
    array = whole

    packed_samples = False
    if "S" in array.dims:
        if "C" in array.dims:
            if rgb_channel is None:
                raise ValueError(
                    f"{array.sizes['C']} channels of {array.sizes['S']} samples each: "
                    "convert them one channel at a time."
                )
            array = array.isel(C=rgb_channel)
        packed_samples = array.sizes["S"] in (3, 4)
        array = array.rename({"S": "c"})

    array = array.rename({d: str(d).lower() for d in array.dims if str(d) in "TCZYX"})

    if descending_z and "z" in array.dims:
        array = array.isel(z=slice(None, None, -1))

    if "c" not in array.dims:
        array = array.expand_dims("c")

    dropped = [
        d for d in array.dims if str(d) not in ("c", "y", "x") and array.sizes[d] == 1
    ]
    if dropped:
        array = array.squeeze(dropped, drop=True)

    extra_types = {str(d): "INDEX" for d in array.dims if str(d) not in AXIS_TYPES}

    return (
        xr.DataArray(array.data, dims=[str(d) for d in array.dims]),
        extra_types,
        packed_samples,
    )


def auto_levels(sizes: Mapping[str, int]) -> int:
    """How many pyramid levels an image needs: halvings until it fits `PYRAMID_TOP_PIXELS`.

    Including the base, so a 1024-pixel field is two levels and an 11k stitched image is
    six. A fixed count is wrong in both directions — levels below one tile are wasted
    uploads, and a stitched image stopped at a fixed depth still opens at megapixels.

    Never fewer than two: a one-level pyramid uploads with an empty ``scales`` list, which
    no other converter here sends — one coarser level costs a quarter of a small field.
    """
    side = max(int(sizes.get("y", 1)), int(sizes.get("x", 1)))
    levels = 1
    while side > PYRAMID_TOP_PIXELS or levels < 2:
        side = -(-side // 2)
        levels += 1
    return levels


def image_pyramid(
    data: xr.DataArray,
    levels: int,
    method: str,
    types: Mapping[str, str],
    scratch: str,
) -> list[xr.DataArray]:
    """The pyramid of a memory-mapped base: level 0 left mapped, level 1 persisted.

    mikro uploads each level as a separate computation, and a lazy level is chained all
    the way back to level 0 — so a pyramid left wholly lazy reads the base once per
    level, six passes over a 40 GB stitched image. mikro's own remedy (``materialize``)
    copies the whole base to a scratch Zarr first, in ``TMPDIR`` whatever
    ``MIKRO_SCRATCH_DIR`` says, which for a base that is already a random-access store
    is a full copy for nothing.

    The middle path: level 1 — a quarter of the base, an eighth for a z-stack — is
    written once into `scratch` (a directory the action owns, beside the download, and
    removes once this dataset is uploaded), and every coarser level is built from it. Two passes over the base in
    all: one for level 1, one for the upload of level 0 itself.
    """
    import dask.array as da

    pyramid = build_pyramid(
        data, levels=levels, method=method, types=types, materialize=False
    )
    if len(pyramid) <= 2:
        return pyramid

    path = tempfile.mkdtemp(prefix="level1-", dir=scratch)
    pyramid[1].data.to_zarr(path, overwrite=True)
    level1 = xr.DataArray(da.from_zarr(path), dims=pyramid[1].dims)
    return [
        pyramid[0],
        *build_pyramid(
            level1, levels=len(pyramid) - 1, method=method, types=types, materialize=False
        ),
    ]

# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def voxel_scale(
    calibration: Calibration, dims: Sequence[str]
) -> tuple[dict[str, float] | None, str | None]:
    """The per-axis factor from this dataset's indices into micrometers.

    Returns ``(None, reason)`` when the file does not calibrate the pixels in a length.
    Registering anyway would positively claim one pixel is one micrometer, at the wrong
    size next to genuinely calibrated images.

    A missing *axial* spacing is different: omitting ``z`` and passing ``z=1.0`` are the
    same claim, so there is no way to abstain. The assumption is recorded instead.
    """
    shared = [d for d in dims if d in WORLD_AXES]
    if calibration.x_um is None or calibration.y_um is None:
        return (
            None,
            "the pixels are not calibrated — not registered into a physical space",
        )
    note = None
    if "z" in shared and calibration.z_um is None:
        note = "z spacing unknown, registered as 1 um per plane"
    sizes = {"x": calibration.x_um, "y": calibration.y_um, "z": calibration.z_um}
    return {axis: sizes[axis] or 1.0 for axis in shared}, note


def world_offsets(
    positions: PlanePositions,
    scale: Mapping[str, float],
    data: xr.DataArray,
) -> dict[str, float] | None:
    """Where pixel (0, 0, 0) sits in the shared world, or None when the file does not say.

    All or nothing: every spatial axis the image has must be placed — x and y from the
    first plane's position (read as that of the first pixel: `ome_tiff`), z from the
    stack's lowest plane. A space shared with other images is the claim that positions in
    it compare; an axis left at 0 would put the image where the instrument never said it
    was. A flat image is placed without a z.
    """
    offsets: dict[str, float] = {}
    for axis, value in (("x", positions.x_um), ("y", positions.y_um)):
        if axis in data.dims and axis in scale and value is not None:
            offsets[axis] = value
    if "z" in data.dims and positions.bottom_um is not None:
        offsets["z"] = positions.bottom_um
    spatial = {axis for axis in ("z", "y", "x") if axis in data.dims}
    return offsets if spatial <= offsets.keys() else None


# ---------------------------------------------------------------------------
# Anchors
# ---------------------------------------------------------------------------


def acquisition_window(window: Window | None) -> AcquisitionWindowInput | None:
    if window is None:
        return None
    return AcquisitionWindowInput(start=window.start, end=window.end)


def image_record(
    file: File, item: OmeImage, instrument: Any | None, rgb_channel: int | None
) -> str:
    """The OME record of this one image, as the JSON string the anchor takes.

    Reduced to this Image and its instrument rather than the whole file's document: a
    96-well plate would otherwise carry 96 copies of it, and the reduced document is the
    one that is actually true of this dataset. Wrapped with where it came from, so the
    blob is a complete record even when read on its own.
    """
    from ome_types.model import OME

    payload: dict[str, Any] = {
        "format": "ome-tiff",
        "source_file": file.name,
        "source_file_id": str(file.id),
        "image_index": item.index,
        "image_id": item.id,
    }
    if rgb_channel is not None:
        payload["channel"] = rgb_channel
    try:
        reduced = OME(
            images=[item.image],
            instruments=[instrument] if instrument is not None else [],
        )
        payload["ome"] = json.loads(reduced.model_dump_json(exclude_unset=True))
    except Exception:
        # A reduced document can fail to revalidate when the Image references something
        # left behind (an experimenter, a plate). Not worth failing an upload over.
        logger.warning("Could not rebuild a reduced OME document for %s", item.id)
        payload["ome"] = json.loads(item.image.model_dump_json(exclude_unset=True))
    return json.dumps(payload)


def series_optikit(
    file: File,
    item: OmeImage,
    instrument: Any | None,
    positions: PlanePositions,
    data: xr.DataArray,
    scale_note: str | None,
    time_note: str | None,
) -> OptikitStateInput | None:
    """The hardware state of the whole image: what the OME record says of the
    microscope, the objective, the environment and the stage, and of the run itself."""
    image = item.image
    pixels = image.pixels
    planes = list(pixels.planes)
    first = planes[0] if planes else None
    env = getattr(image, "imaging_environment", None)
    microscope = getattr(instrument, "microscope", None) if instrument else None
    objective_settings = getattr(image, "objective_settings", None)
    objective = find_objective(instrument, image)

    stage = None
    x = f"{positions.x_um:g} um" if positions.x_um is not None else None
    y = f"{positions.y_um:g} um" if positions.y_um is not None else None
    z = f"{positions.focus_um:g} um" if positions.focus_um is not None else None
    if any((x, y, z)):
        # State what is known, and leave the rest unstated rather than at 0.
        stage = StageStateInput(x=x, y=y, z=z)

    temperature = None
    if env is not None:
        temperature = qty(env.temperature, env.temperature_unit, "degC")

    def verbatim(axis: str) -> str | None:
        """A plane position that is not a length — OME's default unit, ``reference
        frame``, is not — kept as the microscope wrote it, but never placed."""
        if first is None:
            return None
        value = getattr(first, f"position_{axis}")
        unit = getattr(first, f"position_{axis}_unit")
        if value is None or to_micrometer(value, unit) is not None:
            return None
        return f"{value:g} {unit_symbol(unit)}"

    state_devices = devices(
        device(
            "microscope",
            "microscope",
            setting("instrument-id", string=text(getattr(instrument, "id", None))),
            setting("manufacturer", string=text(getattr(microscope, "manufacturer", None))),
            setting("model", string=text(getattr(microscope, "model", None))),
            setting("serial-number", string=text(getattr(microscope, "serial_number", None))),
            setting("type", string=text(getattr(microscope, "type", None))),
        ),
        # The run itself. The only home the source file's identity has, so `source-file`
        # and `source-file-id` are load-bearing rather than decorative.
        device(
            "acquisition",
            "software",
            setting("source-file", string=file.name),
            setting("source-file-id", string=str(file.id)),
            setting("image-index", number=float(item.index)),
            setting("image-id", string=item.id),
            setting("image-name", string=item.name),
            setting("image-description", string=text(image.description)),
            setting(
                "acquisition-date",
                string=image.acquisition_date.isoformat() if image.acquisition_date else None,
            ),
            setting("time-note", string=time_note),
            setting("dimension-order", string=text(pixels.dimension_order)),
            setting("pixel-type", string=str(data.dtype)),
            setting("significant-bits", number=pixels.significant_bits),
            *(setting(f"size-{axis}", number=float(size)) for axis, size in data.sizes.items()),
            setting(
                "physical-size-x",
                quantity=um(pixels.physical_size_x, pixels.physical_size_x_unit),
            ),
            setting(
                "physical-size-y",
                quantity=um(pixels.physical_size_y, pixels.physical_size_y_unit),
            ),
            setting(
                "physical-size-z",
                quantity=um(pixels.physical_size_z, pixels.physical_size_z_unit)
                if "z" in data.dims
                else None,
            ),
            setting(
                "time-increment",
                quantity=qty(pixels.time_increment, pixels.time_increment_unit, "s"),
            ),
            setting("z-order", string="reversed from top-down" if positions.descending else None),
            setting(
                "stack-bottom",
                quantity=f"{positions.bottom_um:g} um"
                if positions.bottom_um is not None and "z" in data.dims
                else None,
            ),
            # Says out loud what the registration had to assume. There is no way to
            # express "unknown" through a BY_DIMENSION edge.
            setting("scale-note", string=scale_note),
        ),
        device(
            "objective",
            "objective",
            setting("model", string=text(getattr(objective, "model", None))),
            setting("manufacturer", string=text(getattr(objective, "manufacturer", None))),
            setting(
                "magnification",
                number=getattr(objective, "nominal_magnification", None)
                or getattr(objective, "calibrated_magnification", None),
            ),
            setting("numerical-aperture", number=getattr(objective, "lens_na", None)),
            setting(
                "working-distance",
                quantity=um(
                    getattr(objective, "working_distance", None),
                    getattr(objective, "working_distance_unit", None),
                ),
            ),
            setting("immersion", string=text(getattr(objective, "immersion", None))),
            setting("correction", string=text(getattr(objective, "correction", None))),
            setting(
                "correction-collar",
                number=getattr(objective_settings, "correction_collar", None),
            ),
            setting("medium", string=text(getattr(objective_settings, "medium", None))),
            setting(
                "refractive-index",
                number=getattr(objective_settings, "refractive_index", None),
            ),
        ),
        device(
            "environment",
            "environment",
            setting(
                "air-pressure",
                quantity=qty(
                    getattr(env, "air_pressure", None),
                    getattr(env, "air_pressure_unit", None),
                    "hPa",
                ),
            ),
            setting("humidity", number=getattr(env, "humidity", None)),
            setting("co2-percent", number=getattr(env, "co2_percent", None)),
        ),
        device(
            "stage",
            "stage",
            # Only where there is a position to read it by.
            setting(
                "position-convention",
                string=POSITION_CONVENTION
                if positions.x_um is not None or positions.y_um is not None
                else None,
            ),
            setting("position-x", string=verbatim("x")),
            setting("position-y", string=verbatim("y")),
            setting("position-z", string=verbatim("z")),
        ),
        device(
            "timing",
            "acquisition",
            setting("plane-count", number=float(len(planes)) if planes else None),
            setting(
                "exposure-time",
                quantity=qty(first.exposure_time, first.exposure_time_unit, "s")
                if first is not None
                else None,
            ),
        ),
    )

    if stage is None and temperature is None and not state_devices:
        return None
    return OptikitStateInput(stage=stage, temperature=temperature, devices=state_devices)


def plain_optikit(
    file: File,
    item: OmeImage,
    calibration: Calibration,
    data: xr.DataArray,
    scale_note: str | None,
) -> OptikitStateInput | None:
    """The record of a series of a TIFF without OME-XML: where it came from and what its
    header says, and nothing about an instrument it does not name."""
    series = item.series
    state_devices = devices(
        device(
            "acquisition",
            "software",
            setting("source-file", string=file.name),
            setting("source-file-id", string=str(file.id)),
            setting("image-index", number=float(item.index)),
            setting("image-id", string=item.id),
            setting("image-name", string=item.name),
            setting("tiff-series-kind", string=text(getattr(series, "kind", None))),
            setting("tiff-axes", string=text(getattr(series, "axes", None))),
            setting("pixel-type", string=str(data.dtype)),
            setting(
                "stored-pixel-type",
                string=str(series.dtype) if series.dtype != data.dtype else None,
            ),
            *(setting(f"size-{axis}", number=float(size)) for axis, size in data.sizes.items()),
            setting(
                "physical-size-x",
                quantity=f"{calibration.x_um:g} um" if calibration.x_um is not None else None,
            ),
            setting(
                "physical-size-y",
                quantity=f"{calibration.y_um:g} um" if calibration.y_um is not None else None,
            ),
            setting(
                "physical-size-z",
                quantity=f"{calibration.z_um:g} um"
                if calibration.z_um is not None and "z" in data.dims
                else None,
            ),
            setting("scale-note", string=scale_note),
        ),
    )
    if not state_devices:
        return None
    return OptikitStateInput(devices=state_devices)


def channel_label(channel: Any | None, index: int, implicit_rgb: bool) -> str:
    if implicit_rgb:
        return (
            ("Red", "Green", "Blue", "Alpha")[index] if index < 4 else f"Sample {index}"
        )
    if channel is not None:
        for candidate in (channel.name, channel.fluor, channel.id):
            rendered = text(candidate)
            if rendered:
                return rendered
    return f"Channel {index}"


def channel_color(
    channel: Any | None, index: int, implicit_rgb: bool, total: int
) -> tuple[int, int, int, int]:
    """What to draw this channel in.

    OME's own colour first — what the acquisition software displayed, and the one of
    these that is a recorded fact rather than a guess (white is OME's default, and so no
    choice: `ome_rgba`). Then its emission, which at least lands in the right part of the
    spectrum. Then a palette, so that two channels never silently collide.
    """
    if implicit_rgb:
        # Index 3 of an RGBA pixel is alpha — a mask, not a colour. White labels it
        # without painting it the same red as channel 0.
        if index >= 3:
            return rgba((255, 255, 255))
        return rgba(RGB_PRIMARIES[index])

    if channel is not None:
        from_file = ome_rgba(channel)
        if from_file is not None:
            return from_file
        emission = to_micrometer(channel.emission_wavelength, channel.emission_wavelength_unit)
        if emission is not None:
            return color_for_wavelength(emission * 1000, COLOR_BY_EMISSION, FAR_RED)

    return fallback_color(index, total)


@dataclass(frozen=True)
class ChannelSpec:
    """One channel, as both the anchors and the layers need it.

    Computed once because the histogram is the expensive part — it is the render
    metadata the anchor carries, and computing it twice would double the only forced
    materialisation in the converter.
    """

    index: int
    label: str
    color: tuple[int, int, int, int]
    histogram: ValueHistogramInput
    clim: tuple[float, float]
    ome: Any | None


def display_window(shown: np.ndarray) -> tuple[float, float]:
    """The contrast window a channel's layer opens with: p1/p99 of what it *shows*.

    `shown` is the picture, which is not the voxels wherever the layer projects — see
    `channel_specs`. p1/p99 rather than min/max, because a single hot pixel should not
    decide the display range of the whole channel — except where the percentiles coincide
    (a channel that is over 99 % one value), which would be a window of no width.
    """
    shown = shown[np.isfinite(shown)]
    low, high = (float(v) for v in np.percentile(shown, [1, 99]))
    if high <= low:
        low, high = float(shown.min()), float(shown.max())
    if high <= low:
        high = low + 1.0
    return low, high


def ome_channels(
    image: Any, channels_in_array: int, packed_samples: bool, rgb_channel: int | None
) -> list[Any]:
    """The OME Channels behind the array's ``c`` axis, index for index, or none.

    A colour picture is one OME Channel of three samples, and converted one at a time
    (`rgb_channel`): its samples are named as such, not after it. Otherwise the OME
    Channels describe the array's channels only where there is one per channel. A modulo
    annotation along C, for one, splits OME's channels into two axes, and indexing one
    by the other would hang a channel's detector and light path on the wrong signal.
    """
    channels = list(image.pixels.channels)
    if rgb_channel is not None:
        return [channels[rgb_channel]] if rgb_channel < len(channels) else []
    if packed_samples:
        return channels[:1]
    if len(channels) == channels_in_array:
        return channels
    if channels:
        logger.info(
            "%s: %s OME channel(s) for %s array channel(s); channels are not described",
            text(image.id),
            len(channels),
            channels_in_array,
        )
    return []


def channel_specs(
    data: xr.DataArray,
    pyramid: Sequence[xr.DataArray],
    channels: Sequence[Any],
    packed_samples: bool,
    is_volume: bool,
) -> list[ChannelSpec]:
    """One spec per channel of the array.

    A contrast window is per channel by nature, which is why each histogram is computed
    from that channel's slice alone rather than from the stack. OME keeps no display
    window this converter reads, so the window is derived from the picture.

    Named and coloured as samples wherever tifffile packed them, even when the image is
    not collapsed into one RGB layer: a 16-bit colour camera's three samples are still
    red, green and blue.
    """
    specs: list[ChannelSpec] = []
    total = data.sizes["c"]

    for index in range(total):
        channel = channels[index] if index < len(channels) and not packed_samples else None

        base = data.isel(c=index)
        slice_ = base
        if base.nbytes > HISTOGRAM_BUDGET_BYTES:
            # `from_array` materialises whatever it is handed, so a multi-gigabyte
            # channel is summarised from the finest pyramid level that fits. Max
            # downsampling keeps only block maxima, so every level shifts the
            # distribution up — hence the finest, not the coarsest.
            slice_ = next(
                (
                    level.isel(c=index)
                    for level in pyramid[1:]
                    if level.isel(c=index).nbytes <= HISTOGRAM_BUDGET_BYTES
                ),
                pyramid[-1].isel(c=index),
            )

        values = np.asarray(slice_)

        # What the layer draws. A z-stack is drawn as a maximum projection, and the
        # maxima along z are a different population from the voxels: in sparse
        # fluorescence the stack's p99 sits barely above background while nearly every
        # projected pixel holds something brighter.
        shown = values
        if is_volume:
            projection = base.max("z")
            if slice_ is base:
                shown = values.max(axis=slice_.dims.index("z"))
            elif projection.nbytes <= HISTOGRAM_BUDGET_BYTES:
                shown = np.asarray(projection)
            else:
                shown = values.max(axis=slice_.dims.index("z"))

        specs.append(
            ChannelSpec(
                index=index,
                label=channel_label(channel, index, packed_samples),
                color=channel_color(channel, index, packed_samples, total),
                histogram=ValueHistogramInput.from_array(values),
                clim=display_window(shown),
                ome=channel,
            )
        )

    return specs


def channel_anchors(
    specs: Sequence[ChannelSpec], image: Any, instrument: Any | None
) -> list[CoordinateAnchorInput]:
    """One anchor per channel: its contrast window, its identity, its optics."""
    return [
        CoordinateAnchorInput(
            axis_anchors=(AxisAnchorInput(axis="c", value=spec.index),),
            value_histogram=spec.histogram,
            label=LabelInput(label=spec.label),
            microscope=(
                channel_optikit(spec.ome, spec.index, spec.label, instrument)
                if spec.ome is not None
                else None
            ),
            light_graph=(
                channel_lightpath(spec.ome, image, instrument)
                if spec.ome is not None
                else None
            ),
        )
        for spec in specs
    ]


def channel_lens(dataset: Any, index: int) -> Any:
    """The lens that pins channel `index`: what a layer over one channel draws.

    A layer draws its lens and nothing narrower, so a channel is a lens of its own.
    A dataset with no channel axis is one signal already, and its whole lens is it.
    """
    if "c" not in dataset.axis_names:
        return dataset.lens()
    return dataset.lens(c=index)


def author_channel_layers(
    mikro: Mikro,
    dataset: ArrayDataset,
    scene: Any,
    specs: Sequence[ChannelSpec],
    is_volume: bool,
    implicit_rgb: bool,
    order_offset: int = 0,
) -> int:
    """One layer per channel of `dataset`, composited into `scene`.

    Per channel, because one layer over a multi-channel dataset means one recipe standing
    in for every channel: no per-channel colour, no per-channel contrast, nothing to
    toggle. It is also the only place OME's own channel colour can land.

    The one exception is a packed RGB(A) series, one layer over its samples: those
    "channels" are one photograph, and separating them gives toggles that are only ever
    right in one combination. `implicit_rgb` is trusted only as far as it can be checked —
    this function refuses it for any channel count that is not 3 or 4. That refusal is the
    backstop for the whole rule, kept here rather than left to the server because it fires
    first and demotes instead of failing the upload.

    One lens per channel, beside the dataset's whole one: a layer draws its lens and
    nothing narrower, so the channel a layer shows is the one its lens pins.

    **Which kind of layer** — each recipe has a server kind that states it as fields:

    * one channel, flat → ``create_intensity_layer``
    * one channel, through a z-stack → ``create_volume_layer``
    * a packed RGB(A) pixel, flat → ``create_rgb_layer``

    ``create_layer`` and a render graph are left for the one shape none of them can say,
    a packed RGB(A) **z-stack**: an RGB layer carries no projection mode.

    Returns the number of layers created — not ``len(specs)`` in the RGB case.
    """
    lens = dataset.lens()

    if implicit_rgb and len(specs) not in (3, 4):
        logger.warning(
            "Refusing to composite %s channel(s) into one RGB layer; a render graph "
            "covers exactly one channel unless the channels are the 3 or 4 samples of "
            "one pixel. Authoring one layer per channel instead.",
            len(specs),
        )
        implicit_rgb = False

    if implicit_rgb:
        if is_volume:
            # The one shape the flat kinds cannot say. Three primaries even for RGBA:
            # the fourth sample is alpha, a mask rather than a colour. `rgb_graph` says
            # nearly this, but its bare triples are what the server rejects.
            mikro.create_layer(
                lens=lens.id,
                scene=scene.id,
                render_graph=composite_graph(
                    [
                        {
                            "lens": channel_lens(dataset, index).id,
                            "color": rgba(primary),
                            "colormap": ColorMap.INTENSITY,
                            # Already display-referred; stretching it would change the
                            # colours the image was authored with.
                            "clim_min": 0.0,
                            "clim_max": 255.0,
                            "mode": ProjectionMode.MIP,
                        }
                        for index, primary in enumerate(RGB_PRIMARIES)
                    ]
                ),
                blending=Blending.ADDITIVE,
                visible=True,
                order=order_offset,
            )
            return 1

        # Flat: the photograph is exactly what an RGB layer is for, and it says so as
        # three component lenses, one per sample, rather than as a tree a reader has to
        # recognise. One contrast window across all three — they are components of one
        # picture, not three signals with their own scaling — left at the full 8-bit
        # range for the reason above.
        #
        # No `blending` argument here, and none accepted: the server forces NORMAL,
        # because a photograph sits *over* what is beneath it rather than summing with
        # it. That is a change from the ADDITIVE this path used to pass.
        mikro.create_rgb_layer(
            lens=lens.id,
            scene=scene.id,
            red_lens=channel_lens(dataset, 0).id,
            green_lens=channel_lens(dataset, 1).id,
            blue_lens=channel_lens(dataset, 2).id,
            clim_min=0.0,
            clim_max=255.0,
            visible=True,
            order=order_offset,
        )
        return 1

    for spec in specs:
        # `color`, not a colormap: what this converter knows about a channel is an RGBA
        # quad — OME's channel colour, or the one its emission implies —
        # and no named map means it. That fact used to force the whole layer onto a
        # render graph; an intensity layer carries it as a field now.
        settings = dict(
            lens=channel_lens(dataset, spec.index).id,
            scene=scene.id,
            color=spec.color,
            # Of what the layer shows, not of the voxels — see `channel_specs`.
            clim_min=spec.clim[0],
            clim_max=spec.clim[1],
            blending=Blending.ADDITIVE,
            visible=True,
            order=order_offset + spec.index,
        )

        if is_volume:
            # A z-stack has no single plane to show, so each channel projects through the
            # volume rather than picking one slice. Same kind of layer, one field more.
            mikro.create_volume_layer(mode=ProjectionMode.MIP, **settings)
        else:
            mikro.create_intensity_layer(**settings)

    return len(specs)


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Preview:
    """One series drawn the way its layers draw it, small: a maximum projection.

    Rendered in `convert_image`, while the pyramid it is read from still exists, and only
    blended into a scene snapshot once the scene's other series have landed.
    `origin_um` and `pixel_um` are ``(y, x)`` in the world, or None for a series that is
    not registered into it — such a series is the whole of its own scene.
    """

    rgb: np.ndarray
    origin_um: tuple[float, float] | None
    pixel_um: tuple[float, float] | None


def preview_level(pyramid: Sequence[xr.DataArray], side: int) -> int:
    """The finest pyramid level whose longer side fits `side`, else the coarsest."""
    for index, level in enumerate(pyramid):
        if max(level.sizes.get("y", 1), level.sizes.get("x", 1)) <= side:
            return index
    return len(pyramid) - 1


def render_preview(
    level: xr.DataArray, specs: Sequence[ChannelSpec], implicit_rgb: bool, side: int
) -> np.ndarray:
    """What the layers over `level` show, as 8-bit RGB: each channel's maximum through z,
    in its own colour and contrast window, summed — the scene blends ADDITIVE.

    The first time point and the first index of any other axis: a snapshot is one
    picture. An RGBA series' alpha is a mask, and is not drawn, as its layer does not.
    """
    level = level.isel({d: 0 for d in level.dims if d not in ("c", "z", "y", "x")})
    if "z" in level.dims:
        level = level.max("z")
    step = -(-max(level.sizes["y"], level.sizes["x"]) // side)
    level = level.isel(y=slice(None, None, step), x=slice(None, None, step))

    rgb = np.zeros((level.sizes["y"], level.sizes["x"], 3), dtype=np.float32)
    for spec in specs:
        if implicit_rgb and spec.index >= 3:
            continue
        low, high = spec.clim
        values = np.asarray(level.isel(c=spec.index), dtype=np.float32)
        shown = np.clip((values - low) / (high - low), 0.0, 1.0)
        rgb += shown[..., None] * np.asarray(spec.color[:3], dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def mosaic(previews: Sequence[Preview]) -> np.ndarray:
    """The previews of one scene, each at its place in the world.

    One preview is its own picture. Several are laid out at their stage positions on one
    canvas at the finest pixel size among them (coarser where that would exceed
    `SNAPSHOT_PIXELS`), each resampled to it nearest-neighbour;
    where tiles overlap the brighter wins, rather than the overlap summing to a seam.
    """
    if len(previews) == 1 or any(p.origin_um is None for p in previews):
        return previews[0].rgb

    pixel = min(min(p.pixel_um) for p in previews)
    top = min(p.origin_um[0] for p in previews)
    left = min(p.origin_um[1] for p in previews)
    bottom = max(p.origin_um[0] + p.rgb.shape[0] * p.pixel_um[0] for p in previews)
    right = max(p.origin_um[1] + p.rgb.shape[1] * p.pixel_um[1] for p in previews)
    # Coarser where the tiles span more than the snapshot's side: a strip of tiles, or
    # two fields far apart, would otherwise grow the canvas with the distance between.
    pixel = max(pixel, max(bottom - top, right - left) / SNAPSHOT_PIXELS)

    shape = (max(1, round((bottom - top) / pixel)), max(1, round((right - left) / pixel)))
    canvas = np.zeros((*shape, 3), dtype=np.uint8)
    for preview in previews:
        rows = max(1, round(preview.rgb.shape[0] * preview.pixel_um[0] / pixel))
        cols = max(1, round(preview.rgb.shape[1] * preview.pixel_um[1] / pixel))
        y0 = min(round((preview.origin_um[0] - top) / pixel), shape[0] - 1)
        x0 = min(round((preview.origin_um[1] - left) / pixel), shape[1] - 1)
        rows, cols = min(rows, shape[0] - y0), min(cols, shape[1] - x0)
        picked = preview.rgb[
            (np.arange(rows) * preview.rgb.shape[0] // rows)[:, None],
            np.arange(cols) * preview.rgb.shape[1] // cols,
        ]
        target = canvas[y0 : y0 + rows, x0 : x0 + cols]
        np.maximum(target, picked, out=target)
    return canvas


def encode_png(rgb: np.ndarray) -> bytes:
    """`rgb` (8-bit, height × width × 3) as a PNG, with nothing but the standard library."""

    def chunk(tag: bytes, body: bytes) -> bytes:
        return (
            struct.pack(">I", len(body))
            + tag
            + body
            + struct.pack(">I", zlib.crc32(tag + body))
        )

    height, width, _ = rgb.shape
    # Every scanline is prefixed with its filter type; 0 is none.
    rows = np.concatenate(
        [np.zeros((height, 1), dtype=np.uint8), rgb.reshape(height, width * 3)], axis=1
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(np.ascontiguousarray(rows).tobytes(), 6))
        + chunk(b"IEND", b"")
    )


def snapshot_scene(
    mikro: Mikro, scene: Scene, entries: Sequence[Converted], scratch: str
) -> None:
    """Attach a maximum projection of `scene` as its snapshot: a preview a client can
    show without compositing the layers itself."""
    previews = [entry.preview for entry in entries if entry.preview is not None]
    if not previews:
        return
    path = os.path.join(scratch, f"snapshot-{scene.id}.png")
    with open(path, "wb") as out:
        out.write(encode_png(mosaic(previews)))
    with open(path, "rb") as picture:
        mikro.create_scene_snapshot(
            file=picture, scene=scene.id, name=f"{scene.name} (maximum projection)"
        )


def compose_scene(
    mikro: Mikro, name: str, coordinate_system: Any, entries: Sequence[Converted]
) -> Scene:
    """A scene over `coordinate_system`, with every layer authored explicitly.

    ``create_scene`` rather than ``stage`` / ``create_scene_from_coordinate_system``:
    the bootstrap decides on your behalf how many registered sources become layers,
    which recipe each one gets, and — for anything multi-channel — collapses a whole
    dataset into a single layer. This converter knows exactly what it uploaded and what
    each channel is, so none of those are decisions worth delegating. It adopts the
    world system, then authors one lens per dataset and one layer per channel, each
    with its own colour, contrast window and projection.

    `entries` is one series, or the tiles of one mosaic — never the whole file; which
    series share a scene is the action's decision (see `share_a_scene`), not this
    function's.

    The scene carries only what is genuinely scene-wide. ``blending`` is ADDITIVE
    because that is how fluorescence channels combine — and how three RGB primaries
    recompose a photograph. ``preferred_view`` is a viewer hint and nothing more: the
    projection that actually renders a volume lives on each layer's channel node, so
    this only says which way a viewer should open, and only claims a preference when
    something in the scene has depth to look into.
    """
    scene = mikro.create_scene(
        name=name,
        coordinate_system=coordinate_system.id,
        blending=Blending.ADDITIVE,
        preferred_view=(
            PreferredView.THREE_D
            if any(entry.is_volume for entry in entries)
            else PreferredView.TWO_D
        ),
    )

    # Ordered so a series' channels stay together in the layer stack rather than
    # interleaving with the next series'.
    order = 0
    for entry in entries:
        order += author_channel_layers(
            mikro,
            entry.dataset,
            scene,
            entry.specs,
            entry.is_volume,
            entry.implicit_rgb,
            order,
        )

    # Nominated through the whole lens: on an unsliced lens that is the dataset's own
    # default scene, and so where its thumbnail (`latestSnapshot`) comes from. A
    # mosaic's tiles all nominate the mosaic.
    for entry in entries:
        mikro.set_lens_default_scene(lens=entry.dataset.lens().id, scene=scene.id)

    logger.info(
        "Scene %s over %s: %s layer(s) from %s dataset(s)",
        scene.id,
        scene.world_coordinate_system.name,
        order,
        len(entries),
    )
    return scene


def stage_group(
    mikro: Mikro,
    title: str,
    world: Any,
    entries: Sequence[Converted],
    scratch: str | None = None,
) -> None:
    """Compose the scene(s) for one group out of `scene_groups`.

    The series placed in the world share one scene over it. Any other series is staged
    alone: over its own micrometer space when it is calibrated but its position is not
    fully known (`world_offsets`), over its own pixel grid when it is not calibrated
    either. Both are the honest frame for it, and one it shares with nothing.

    With a `scratch` directory, every scene also gets its snapshot (`snapshot_scene`).
    """

    def compose(name: str, system: Any, members: Sequence[Converted]) -> None:
        scene = compose_scene(mikro, name, system, members)
        if scratch is None:
            return
        try:
            snapshot_scene(mikro, scene, members, scratch)
        except Exception:  # the scene is composed; a picture of it is not worth it
            logger.exception("Could not snapshot %s", name)

    registered = [entry for entry in entries if entry.registered]
    if registered:
        layer_count = sum(entry.layer_count for entry in registered)
        if layer_count > MAX_SCENE_LAYERS:
            logger.warning(
                "%s would compose %s layers from %s series; not building a scene over "
                "more than %s. The world holds every registration and outlives this run, "
                "so a subset can be composed later.",
                title,
                layer_count,
                len(registered),
                MAX_SCENE_LAYERS,
            )
        else:
            compose(title, world, registered)

    for entry in entries:
        if entry.registered:
            continue
        if entry.own_space is not None:
            compose(entry.name, entry.own_space, [entry])
            continue
        if entry.dataset.intrinsic_system is None:
            logger.warning(
                "Series %s has no intrinsic coordinate system to compose over; the "
                "dataset is uploaded but has no scene.",
                entry.series,
            )
            continue
        compose(entry.name, entry.dataset.intrinsic_system, [entry])


# ---------------------------------------------------------------------------
# Which images share a scene
# ---------------------------------------------------------------------------


def scene_groups(
    footprints: Sequence[tuple[float, float, float, float] | None],
    layers: Sequence[int],
) -> list[list[int]]:
    """The images each scene composes, one list per scene, in order of first image.

    Images whose fields touch make one picture — the fields of a grid acquired but not
    stitched — and share one mosaic scene. Touching means connected: every field reaches
    the others through fields that overlap it or sit within `MOSAIC_GAP_FRACTION` of it.
    Everything else gets a scene of its own: the wells of a plate, points picked across
    a slide, a slide's label beside the slide, and any image not placed in x and y. A
    scene over those would be mostly empty space.

    A mosaic over more than `MAX_SCENE_LAYERS` layers is not composed: `stage_group`
    would refuse it and leave a large grid without any scene. Its images get a scene
    each instead.

    Decided from the header (`plan_image`), before any pixel is read, so that each
    snapshot is rendered at the size its scene will have.
    """
    seen: set[int] = set()
    groups: list[list[int]] = []
    for start in range(len(footprints)):
        if start in seen:
            continue
        seen.add(start)
        if footprints[start] is None:
            groups.append([start])
            continue
        component, frontier = [start], [start]
        while frontier:
            here = frontier.pop()
            for other, box in enumerate(footprints):
                if other not in seen and box is not None and _touch(footprints[here], box):
                    seen.add(other)
                    component.append(other)
                    frontier.append(other)
        component.sort()
        if len(component) > 1 and sum(layers[i] for i in component) > MAX_SCENE_LAYERS:
            groups.extend([i] for i in component)
        else:
            groups.append(component)
    return sorted(groups, key=lambda group: group[0])


def _touch(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    """Whether two ``(y0, x0, y1, x1)`` boxes overlap or nearly abut."""
    gap_y = MOSAIC_GAP_FRACTION * min(a[2] - a[0], b[2] - b[0])
    gap_x = MOSAIC_GAP_FRACTION * min(a[3] - a[1], b[3] - b[1])
    return (
        a[0] <= b[2] + gap_y
        and b[0] <= a[2] + gap_y
        and a[1] <= b[3] + gap_x
        and b[1] <= a[3] + gap_x
    )


# ---------------------------------------------------------------------------
# What landed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Converted:
    """What the action needs to know about a series after it has landed."""

    series: int
    name: str
    dataset: ArrayDataset
    specs: list[ChannelSpec]
    #: Placed in the file's shared world (`world_offsets`).
    registered: bool
    is_volume: bool
    implicit_rgb: bool
    preview: Preview | None = None
    #: The micrometer space of its own a calibrated series that is not placed in the
    #: world is registered into; None for the placed and the uncalibrated.
    own_space: Any | None = None
    #: Its field in the world, ``(y0, x0, y1, x1)`` in micrometers, when it is placed.
    footprint: tuple[float, float, float, float] | None = None
    #: The name of the image it was made of: its scene's, unless it shares a mosaic.
    point_name: str = ""

    @property
    def layer_count(self) -> int:
        """How many layers this series composes into a scene.

        Not ``len(specs)``: a packed RGB(A) series is one layer over its 3 or 4
        samples. Kept here rather than recomputed at each call site so the budget check
        in `stage_group` and what `author_channel_layers` actually creates cannot drift.

        For every other series this is ``len(specs)`` — one layer per channel — and that
        is the invariant, not an implementation detail.
        """
        return 1 if self.implicit_rgb else len(self.specs)


def packs_one_pixel(packed_samples: bool, channels: int, dtype: Any) -> bool:
    """Whether this image's channels are the samples of one pixel — the ONLY exception.

    ONE LAYER PER CHANNEL. Exactly one thing may say yes: tifffile's sample axis
    (``packed_samples`` out of `field_array`), which it exposes only for a pixel the
    TIFF stores as several samples. Nothing that *counts* channels may: collapsing three dyes destroys
    per-channel colour, contrast and toggling.

    And the samples have to be 8-bit. An RGB layer is drawn unstretched over 0..255; a
    deeper sample axis gets a layer and a window per sample instead.
    """
    return packed_samples and channels in (3, 4) and np.dtype(dtype) == np.uint8


# ---------------------------------------------------------------------------
# The plan: what the header says, before any pixel is read
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Planned:
    """One readable image, as far as its OME record and TIFF header describe it."""

    item: OmeImage
    calibration: Calibration
    positions: PlanePositions
    #: Its field in the world, ``(y0, x0, y1, x1)`` in micrometers, where it is placed
    #: in x and y; what decides which images share a scene (`scene_groups`).
    footprint: tuple[float, float, float, float] | None
    #: How many layers it will compose into (`layers_of`).
    layers: int
    #: The colour pictures it is converted as, one at a time; ``[None]`` for one image.
    rgb_channels: list[int | None]


def layers_of(sizes: Mapping[str, int], dtype: Any) -> int:
    """How many layers an image will compose into: one per channel, and one per colour
    picture where its samples are an 8-bit RGB(A) pixel (`packs_one_pixel`)."""
    channels = sizes.get("C", 1)
    samples = sizes.get("S", 1)
    if samples > 1 and packs_one_pixel(True, samples, dtype):
        return channels
    return channels * samples


def plan_image(item: OmeImage) -> Planned:
    sizes = item.sizes
    calibration = item.header_calibration or read_calibration(item.image)
    positions = plane_positions(item.image)
    rgb_channels: list[int | None] = [None]
    if "S" in sizes and "C" in sizes:
        rgb_channels = list(range(sizes["C"]))
    return Planned(
        item=item,
        calibration=calibration,
        positions=positions,
        footprint=image_footprint(
            positions, calibration, sizes.get("Y", 1), sizes.get("X", 1)
        ),
        layers=layers_of(sizes, item.series.dtype),
        rgb_channels=rgb_channels,
    )


# ---------------------------------------------------------------------------
# One image
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FileContext:
    """What every image of one file shares: read once, before the first upload."""

    file: File
    #: None for a TIFF without OME-XML.
    ome: Any | None
    world: Any
    levels: int
    method: str
    #: Whether the file holds several images, and so names each after its own.
    several: bool


def convert_image(
    mikro: Mikro,
    context: FileContext,
    planned: Planned,
    rgb_channel: int | None,
    scratch: str,
    preview_side: int | None = None,
) -> Converted:
    """Upload one image (one colour picture of it, for several), and place it if the
    file says where. Never stages it: that depends on the images around it — see
    `scene_groups`.

    Placed means in the shared world, and only when the file says where along every
    spatial axis (`world_offsets`). A calibrated image it does not place is registered
    into a micrometer space of its own, at its origin: its scale holds, its position is
    claimed against nothing.
    """
    file, item = context.file, planned.item
    image = item.image
    instrument = find_instrument(context.ome, image)
    positions = planned.positions

    array, extra_types, packed_samples = field_array(
        series_array(item.series, item.split), rgb_channel, positions.descending
    )

    # Canonical order — time, then channel, then space — is load-bearing: the render
    # axes are derived from the *position* of the spatial axes.
    data = canonical(array, types=extra_types)
    if data.dtype == np.bool_:
        # A bilevel TIFF (a mask, a scan) reads as bool, which no histogram, contrast
        # window or pyramid takes. 0 and 1 in eight bits are the same values.
        data = data.astype(np.uint8)
    dims = [str(d) for d in data.dims]

    implicit_rgb = packs_one_pixel(packed_samples, data.sizes["c"], data.dtype)
    if packed_samples and not implicit_rgb:
        logger.warning(
            "%s has a packed sample axis but %s %s sample(s) are no 8-bit RGB(A) "
            "picture; treating them as channels, one layer each.",
            item.id,
            data.sizes["c"],
            data.dtype,
        )
    is_volume = data.sizes.get("z", 1) > 1

    pyramid = image_pyramid(
        data, context.levels or auto_levels(data.sizes), context.method, extra_types, scratch
    )
    scales = scales_from(pyramid, context.method)

    scale, scale_note = voxel_scale(planned.calibration, dims)
    whole_window, timepoint_windows, time_note = (
        acquisition_windows(image, data.sizes.get("t", 1))
        if image is not None
        else (None, [], None)
    )

    channels = (
        ome_channels(image, data.sizes["c"], packed_samples, rgb_channel)
        if image is not None
        else []
    )
    specs = channel_specs(data, pyramid, channels, packed_samples, is_volume)

    name = file.name
    if context.several:
        name += f" - {item.name or f'image {item.index + 1}'}"
    point_name = name
    if rgb_channel is not None:
        label = text(channels[0].name) if channels else None
        name += f" - {label or f'channel {rgb_channel}'}"

    series_identifier = item.id
    if rgb_channel is not None:
        series_identifier += f":C{rgb_channel}"

    if image is None:
        # A plain TIFF: there is no OME record to carry, only what its header says.
        image_anchor = CoordinateAnchorInput(
            axis_anchors=(),
            microscope=plain_optikit(file, item, planned.calibration, data, scale_note),
        )
    else:
        image_anchor = CoordinateAnchorInput(
            axis_anchors=(),
            acquisition_window=acquisition_window(whole_window),
            ome_metadata=OmeMetadataInput(
                metadata_string=image_record(file, item, instrument, rgb_channel)
            ),
            microscope=series_optikit(
                file, item, instrument, positions, data, scale_note, time_note
            ),
        )

    dataset = mikro.create_array_dataset(
        data=pyramid[0],
        scales=scales,
        name=name,
        axes=axes_for(
            data, types=dict.fromkeys(extra_types, "INDEX"), long_names=AXIS_LONG_NAMES
        ),
        anchors=[
            # No label on the image anchor: a label names a channel, and the image's name
            # is the dataset's.
            image_anchor,
            *channel_anchors(specs, image, instrument),
            # When each timepoint was taken: intervals a nominal increment cannot state.
            *(
                CoordinateAnchorInput(
                    axis_anchors=(AxisAnchorInput(axis="t", value=index),),
                    acquisition_window=acquisition_window(window),
                )
                for index, window in timepoint_windows
            ),
        ],
        # The link back to the bytes these arrays were read out of: OME's own Image id
        # (and the colour picture of it), which survives a reader that lists them in
        # another order.
        source_files=[
            SourceFileInput(
                file=file.id,
                series_identifier=series_identifier,
                # tifffile hands back the stored values, and this converter only renames
                # and reorders axes (and turns a top-down stack over): the histograms are
                # true of the file too. A series missing planes, which tifffile would
                # fill with zeros, is refused before it gets here (`ome_tiff`).
                value_relation=ValueRelation.IDENTICAL,
            )
        ],
    )

    offsets = world_offsets(positions, scale, data) if scale is not None else None
    footprint = None
    if scale is not None and offsets is not None:
        footprint = (
            offsets["y"],
            offsets["x"],
            offsets["y"] + data.sizes["y"] * scale["y"],
            offsets["x"] + data.sizes["x"] * scale["x"],
        )

    preview = None
    if preview_side is not None:
        try:
            level = pyramid[preview_level(pyramid, preview_side)]
            rgb = render_preview(level, specs, implicit_rgb, preview_side)
            placed = footprint is not None
            preview = Preview(
                rgb,
                (offsets["y"], offsets["x"]) if placed else None,
                (
                    scale["y"] * data.sizes["y"] / rgb.shape[0],
                    scale["x"] * data.sizes["x"] / rgb.shape[1],
                )
                if placed
                else None,
            )
        except Exception:  # only costs the snapshot
            logger.exception("Could not render a preview of %s", name)

    if scale is None:
        logger.info("%s: %s", name, scale_note)
        return Converted(
            item.index,
            name,
            dataset,
            specs,
            False,
            is_volume,
            implicit_rgb,
            preview,
            point_name=point_name,
        )

    if offsets is None:
        logger.info(
            "%s: the file does not say where it sits along every axis; registered into a "
            "space of its own rather than the shared world",
            name,
        )
        own = space_3d(mikro, f"{name} (micrometers)", unit="micrometer")
        own.register(dataset, validity=PlacementValidity.INFERRED, scale=scale)
        return Converted(
            item.index,
            name,
            dataset,
            specs,
            False,
            is_volume,
            implicit_rgb,
            preview,
            own,
            point_name=point_name,
        )

    context.world.register(
        dataset, validity=PlacementValidity.INFERRED, scale=scale, **offsets
    )
    return Converted(
        item.index,
        name,
        dataset,
        specs,
        True,
        is_volume,
        implicit_rgb,
        preview,
        footprint=footprint,
        point_name=point_name,
    )


# ---------------------------------------------------------------------------
# The action
# ---------------------------------------------------------------------------


def preview_sides(groups: Sequence[Sequence[int]]) -> dict[int, int]:
    """The side each image's preview is rendered at: a mosaic's images share the
    snapshot's side, an image staged alone has all of it."""
    sides: dict[int, int] = {}
    for group in groups:
        side = (
            SNAPSHOT_PIXELS
            if len(group) == 1
            else max(64, int(SNAPSHOT_PIXELS / np.sqrt(len(group))))
        )
        sides.update(dict.fromkeys(group, side))
    return sides


@app.action(
    name="Convert OME-TIFF file to Datasets",
    description=(
        "Read an OME-TIFF natively (no Bio-Formats) and upload every OME Image as a "
        "pyramided dataset with its OME metadata: channels, instrument, light path, "
        "objective and plane times, placed in a shared micrometer space at its stage "
        "position where the file records it along every axis (otherwise in a calibrated "
        "space of its own), and staged in a scene — one mosaic for images whose fields "
        "touch — with a maximum projection as its snapshot. A plain TIFF without "
        "OME-XML converts as one dataset per series, uncalibrated unless ImageJ "
        "calibrated it."
    ),
    collections=["conversion"],
    logo=logo_of("convert_ome_tiff_file_to_datasets"),
)
def convert_ome_tiff_file_to_datasets(
    file: Annotated[File, Requires(key=EXTENSION, operator=DescriptorOperator.IN, value=["tif", "tiff", "ome.tif", "ome.tiff", "btf", "ome.btf", "tf2", "tf8"])],
    mikro: Mikro,
    task: Task,
    levels: int = 0,
    method: str = "max",
    stage_scene: bool = True,
    world_name: str = "",
    snapshot: bool = True,
) -> list[ArrayDataset]:
    """Convert every OME Image of an OME-TIFF into a dataset, or every series of a
    plain TIFF without OME-XML.

    `levels` of 0 picks the pyramid depth per image (`auto_levels`). Returns a list
    rather than yielding: a mosaic scene is composed after its last image, and an
    abandoned generator would leave it unstaged.
    """
    assert file.store, "No file provided"
    if not file.name.lower().endswith(READABLE_SUFFIXES):
        raise ValueError(f"{file.name} is not a TIFF file.")

    task.progress(0, "Downloading file")
    # Under its own name, into a scratch dir of its own — never the working directory,
    # which the `finally` below would then delete from. MIKRO_SCRATCH_DIR when set,
    # TMPDIR otherwise; a tmpfs /tmp puts the download in RAM.
    workdir = tempfile.mkdtemp(
        prefix="ome-tiff-converter-", dir=os.environ.get("MIKRO_SCRATCH_DIR") or None
    )

    try:
        path = file.store.download(os.path.join(workdir, os.path.basename(file.name)))
        task.progress(5, "Reading the header")

        import tifffile

        with tifffile.TiffFile(path) as tif:
            ome, images = read_images(tif, file.name)
            failures: list[tuple[str, Exception]] = [
                (item.id, Unreadable(f"{item.id}: {item.problem}"))
                for item in images
                if item.problem
            ]
            for where, exc in failures:
                logger.warning("%s of %s is not converted: %s", where, file.name, exc)
            readable = [item for item in images if item.series is not None]
            if not readable:
                raise Unreadable(
                    f"None of the {len(images)} image(s) of {file.name} can be read; "
                    f"first: {failures[0][1]}"
                )

            planned = [plan_image(item) for item in readable]
            groups = scene_groups(
                [plan.footprint for plan in planned], [plan.layers for plan in planned]
            )
            sides = preview_sides(groups) if snapshot and stage_scene else {}

            # One per invocation: a physical space belongs to nobody and outlives every
            # dataset put into it. Two invocations mint two disjoint worlds.
            world = space_3d(
                mikro, world_name or f"{file.name} (micrometers)", unit="micrometer"
            )
            context = FileContext(
                file=file,
                ome=ome,
                world=world,
                levels=levels,
                method=method,
                several=len(images) > 1,
            )

            units = [
                (k, plan, rgb_channel)
                for k, plan in enumerate(planned)
                for rgb_channel in plan.rgb_channels
            ]
            converted: dict[int, list[Converted]] = {}
            for done, (k, plan, rgb_channel) in enumerate(units):
                where = (
                    f"image {k + 1}/{len(planned)} ({plan.item.id})"
                    + (f", channel {rgb_channel}" if rgb_channel is not None else "")
                )
                task.progress(10 + int(85 * done / len(units)), f"Converting {where}")
                # Per image, and gone once it is uploaded: a plate's persisted levels
                # would otherwise all wait for the end of the file.
                scratch = tempfile.mkdtemp(prefix="image-", dir=workdir)
                try:
                    converted.setdefault(k, []).append(
                        convert_image(
                            mikro, context, plan, rgb_channel, scratch, sides.get(k)
                        )
                    )
                except Exception as exc:  # one bad image must not cost the others
                    logger.exception("%s of %s failed", where, file.name)
                    failures.append((where, exc))
                finally:
                    shutil.rmtree(scratch, ignore_errors=True)

            entries = [entry for k in sorted(converted) for entry in converted[k]]
            datasets = [entry.dataset for entry in entries]

            if stage_scene:
                for group in groups:
                    members = [entry for k in group for entry in converted.get(k, [])]
                    if not members:
                        continue
                    title = (
                        members[0].point_name
                        if len(group) == 1
                        else f"{file.name} ({len(group)} images)"
                    )
                    try:
                        stage_group(
                            mikro, title, world, members, workdir if snapshot else None
                        )
                    except Exception:  # the datasets are uploaded; a scene is not worth them
                        logger.exception("Could not stage %s", title)

        if failures and not datasets:
            raise RuntimeError(
                f"All {len(failures)} image(s) failed; first: {failures[0][1]}"
            ) from failures[0][1]
        if failures:
            logger.warning(
                "%s image(s) failed: %s", len(failures), [where for where, _ in failures]
            )

        task.progress(100, f"Converted {len(datasets)} dataset(s)")
        return datasets

    finally:
        # The whole scratch directory, and only ever that directory, which this
        # converter made and owns.
        shutil.rmtree(workdir, ignore_errors=True)
        if os.path.exists(workdir):
            # Cleanup must never mask whatever actually went wrong above.
            logger.warning("Could not remove the scratch directory %s", workdir)


if __name__ == "__main__":
    run(app)
