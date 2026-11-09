"""Reading an OME-TIFF: which TIFF series is which OME Image, and what the Planes say.

tifffile reads the pixels and ome-types reads the OME-XML. Each does its job well. What
neither does is say which of tifffile's series is which OME ``Image``, and the rest of
the converter hangs every fact about an image off that pairing. `match_series` makes it,
and refuses what it cannot make:

* tifffile **skips** an Image it finds no data for, or one stored discontiguously, so
  its series are not always the document's Images in order. Matched by position only
  when the counts agree (nothing was skipped). Otherwise each series is matched by the
  IFD its first plane is stored in, which the OME document states.
* tifffile **zero-fills** planes it cannot find, with a warning in its log. A dataset
  uploaded from that would be zeros labelled ``IDENTICAL`` to the file. A series with
  a missing plane is refused (`missing_planes`).
* A **multi-file** OME-TIFF stores some planes in other files, named by UUID. The
  converter downloads one file, so those planes are missing, and the Image is refused
  (`foreign_files`). A ``BinaryOnly`` file keeps its metadata in a companion
  ``.companion.ome`` and has no Images of its own; it is refused outright.
* A writer may state no ``TiffData`` at all (pyvips writes ``MetadataOnly``). tifffile
  then makes one generic series of the file. That is paired with the document's Image
  only where there is one of each and their extents agree (`describes`).

The rest reads where and when each Image was taken from its ``Plane`` records:
`plane_positions` for the stage, `acquisition_windows` for the clock. OME does not say
whether a plane position is the centre of the field or its first pixel. This converter
reads it as the first pixel, as `ome-converter` does, so the same file lands in the same
place through either converter.
"""

from __future__ import annotations

import datetime
import logging
import os
from dataclasses import dataclass
from typing import Any

from ome_metadata import to_micrometer, unit_symbol

logger = logging.getLogger(__name__)

# How the converter reads a plane's PositionX/Y. Recorded on every image, because OME
# leaves it open and a reader of the dataset cannot tell from the numbers alone.
POSITION_CONVENTION = "first pixel"

# OME's time vocabulary, in seconds per unit. Anything absent is not read as a time.
SECONDS_PER_TIME_UNIT: dict[str, float] = {
    "d": 86400.0,
    "h": 3600.0,
    "min": 60.0,
    "s": 1.0,
    "ds": 1e-1,
    "cs": 1e-2,
    "ms": 1e-3,
    "µs": 1e-6,
    "us": 1e-6,
    "ns": 1e-9,
    "ps": 1e-12,
}


class Unreadable(ValueError):
    """This file, or this image of it, cannot be converted faithfully."""


# ---------------------------------------------------------------------------
# The document
# ---------------------------------------------------------------------------


def read_ome(tif: Any, file_name: str) -> Any:
    """The OME document of an open `tif`, or `Unreadable` when it has none to convert.

    Validated first. A file that does not validate is still read, without validation,
    because writers in the wild bend the schema in small ways (an unknown enum member, an
    attribute in the wrong namespace) that do not touch the pixels.
    """
    from ome_types import from_xml

    if not tif.is_ome or not tif.ome_metadata:
        raise Unreadable(f"{file_name} is a TIFF without OME-XML, not an OME-TIFF.")
    try:
        ome = from_xml(tif.ome_metadata)
    except Exception as exc:
        logger.warning("%s: the OME-XML does not validate (%s); reading it anyway", file_name, exc)
        ome = from_xml(tif.ome_metadata, validate=False)

    if getattr(ome, "binary_only", None) is not None:
        companion = text_of(getattr(ome.binary_only, "metadata_file", None))
        raise Unreadable(
            f"{file_name} is the binary part of a multi-file OME-TIFF: its metadata is in "
            f"{companion or 'a companion .ome file'}. Convert the file that holds it."
        )
    if not ome.images:
        raise Unreadable(f"{file_name} declares no OME Image.")
    return ome


def text_of(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(getattr(value, "value", value)).strip()
    return rendered or None


# ---------------------------------------------------------------------------
# Series ↔ Image
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OmeImage:
    """One OME Image, with the TIFF series that holds its pixels.

    `series` is None when the Image cannot be read from this file, and `problem` says
    why. The Image is then not converted, but it is reported.
    """

    #: Position in the OME document: what the dataset's source link names.
    index: int
    image: Any
    series: Any | None
    problem: str | None = None
    #: For a generic series (`describes`): how its page axis ``I`` unfolds into OME's
    #: T, C and Z, slowest first. None for an OME series, which tifffile already shaped.
    split: tuple[tuple[str, int], ...] | None = None

    @property
    def sizes(self) -> dict[str, int]:
        """The series' extent along each axis, with the page axis unfolded."""
        sizes = series_sizes(self.series)
        if not self.split:
            return sizes
        unfolded: dict[str, int] = {}
        for axis, size in sizes.items():
            if axis == "I":
                unfolded.update(self.split)
            else:
                unfolded[axis] = size
        return unfolded

    @property
    def name(self) -> str | None:
        return text_of(self.image.name)

    @property
    def id(self) -> str:
        return text_of(self.image.id) or f"Image:{self.index}"


def first_plane_ifd(image: Any) -> int | None:
    """The IFD the Image's first plane (C = Z = T = 0) is stored in, as the OME document
    states it, or None when it states none."""
    for block in image.pixels.tiff_data_blocks:
        if (block.first_c or 0) == 0 and (block.first_z or 0) == 0 and (block.first_t or 0) == 0:
            return int(block.ifd or 0)
    return None


def foreign_files(image: Any, root_uuid: str | None, file_name: str) -> list[str]:
    """The other files this Image keeps planes in, by the name its TiffData gives them.

    A UUID is this file's when it is the document's own UUID, or when it names this file.
    The converter is handed one file and cannot read planes kept in another.
    """
    own_name = os.path.basename(file_name).lower()
    root = (root_uuid or "").strip().lower()
    elsewhere: list[str] = []
    for block in image.pixels.tiff_data_blocks:
        uuid = block.uuid
        if uuid is None:
            continue
        value = (getattr(uuid, "value", None) or "").strip().lower()
        name = (getattr(uuid, "file_name", None) or "").strip()
        if (root and value == root) or (name and name.lower() == own_name):
            continue
        elsewhere.append(name or value or "an unnamed file")
    return sorted(set(elsewhere))


def missing_planes(series: Any) -> int:
    """How many of the series' planes tifffile did not find, and would read as zeros."""
    return sum(1 for page in series.pages if page is None)


def describes(image: Any, series: Any) -> bool:
    """Whether the OME Image has exactly this series' extent: the same plane, and as many
    values per pixel position (channels × samples × planes) as it."""
    pixels = image.pixels
    sizes = series_sizes(series)
    if not set(sizes) <= {"I", "Y", "X", "S"}:
        return False
    if (sizes.get("Y"), sizes.get("X")) != (int(pixels.size_y), int(pixels.size_x)):
        return False
    rest = 1
    for axis, size in sizes.items():
        if axis not in ("Y", "X"):
            rest *= size
    return rest == int(pixels.size_c) * int(pixels.size_z) * int(pixels.size_t)


def page_split(image: Any, series: Any) -> tuple[tuple[str, int], ...] | None:
    """How a generic series' page axis ``I`` unfolds into OME's T, C and Z, slowest
    first, or None when it has none.

    A generic series is the file's pages in order, and ``DimensionOrder`` says in which:
    ``XYZCT`` pages run through Z fastest, then C, then T. OME counts an RGB pixel's
    samples as channels (``SizeC`` 3 for one RGB channel), and tifffile has them on ``S``
    already, so the pages hold ``SizeC / samples`` channels.
    """
    sizes = series_sizes(series)
    if "I" not in sizes:
        return None
    pixels = image.pixels
    extent = {
        "Z": int(pixels.size_z),
        "C": int(pixels.size_c) // sizes.get("S", 1),
        "T": int(pixels.size_t),
    }
    order = (text_of(pixels.dimension_order) or "XYZCT")[2:]
    split = tuple((axis, extent[axis]) for axis in reversed(order))
    product = 1
    for _, size in split:
        product *= size
    if product != sizes["I"]:
        raise Unreadable(
            f"{text_of(image.id)}: {sizes['I']} pages for {product} planes; the pages "
            "cannot be read as its planes"
        )
    return split


def match_series(tif: Any, ome: Any, file_name: str) -> list[OmeImage]:
    """Every OME Image of the file, paired with the TIFF series that holds its pixels.

    By position when tifffile made exactly one series per Image. It walks the Images in
    order and only ever skips one, so equal counts mean nothing was skipped. Otherwise
    by the IFD each Image says its first plane is in. With no OME series at all, by
    `describes`. An Image matched to nothing, kept
    partly in another file, or missing planes, carries its `problem` and no series.
    """
    series = [s for s in tif.series if getattr(s, "kind", None) == "ome"]
    images = list(ome.images)
    if (
        not series
        and len(images) == 1
        and len(tif.series) == 1
        and describes(images[0], tif.series[0])
    ):
        # A writer that states no TiffData — pyvips writes ``MetadataOnly`` — leaves
        # tifffile no OME series to make, only the file's one generic series. One Image
        # of exactly that series' extent can only be it.
        logger.info(
            "%s: the OME-XML names no TiffData; its one Image is the file's one series",
            file_name,
        )
        series = list(tif.series)
        split = page_split(images[0], series[0])
        missing = missing_planes(series[0])
        if missing:
            return [OmeImage(0, images[0], None, f"{missing} of its planes are not in the file")]
        return [OmeImage(0, images[0], series[0], split=split)]
    if not series:
        raise Unreadable(
            f"tifffile could not read the OME series of {file_name}; it holds OME-XML "
            "that does not describe its own pages."
        )

    if len(series) == len(images):
        paired: list[Any | None] = list(series)
    else:
        by_ifd = {}
        for s in series:
            first = s.pages[0] if len(s.pages) else None
            if first is not None:
                by_ifd[first.index] = s
        paired = [by_ifd.get(first_plane_ifd(image)) for image in images]
        logger.info(
            "%s: tifffile read %s series for %s OME Image(s); matched by first IFD",
            file_name,
            len(series),
            len(images),
        )

    out: list[OmeImage] = []
    for index, (image, s) in enumerate(zip(images, paired)):
        elsewhere = foreign_files(image, ome.uuid, file_name)
        problem = None
        if elsewhere:
            problem = (
                "keeps planes in other files of a multi-file OME-TIFF "
                f"({', '.join(elsewhere)}), which were not uploaded with it"
            )
        elif s is None:
            problem = "tifffile found no pixels for it in this file"
        elif missing := missing_planes(s):
            problem = f"{missing} of its planes are not in the file"
        out.append(OmeImage(index, image, None if problem else s, problem))
    return out


# ---------------------------------------------------------------------------
# Calibration and placement
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Calibration:
    """Micrometers per pixel along each axis, or None where the file does not say in a
    unit that is a length."""

    x_um: float | None
    y_um: float | None
    z_um: float | None


def read_calibration(image: Any) -> Calibration:
    pixels = image.pixels
    return Calibration(
        to_micrometer(pixels.physical_size_x, pixels.physical_size_x_unit),
        to_micrometer(pixels.physical_size_y, pixels.physical_size_y_unit),
        to_micrometer(pixels.physical_size_z, pixels.physical_size_z_unit),
    )


@dataclass(frozen=True)
class PlanePositions:
    """Where the stage was, in micrometers, as the Image's Planes record it.

    `x_um`/`y_um` are those of its first plane. `z_um` holds one position per z index,
    at the first timepoint and channel, and is None unless every plane of the stack has
    one, and they differ: a stack placed by half its planes, or by one z for all of
    them, is not placed.
    """

    x_um: float | None
    y_um: float | None
    z_um: tuple[float, ...] | None
    #: The first plane's PositionZ, also for an image that is not a stack.
    focus_um: float | None

    @property
    def descending(self) -> bool:
        """Whether the stack was recorded top-down: z position falling with its index."""
        return self.z_um is not None and len(self.z_um) > 1 and self.z_um[-1] < self.z_um[0]

    @property
    def bottom_um(self) -> float | None:
        return min(self.z_um) if self.z_um else None


def plane_positions(image: Any) -> PlanePositions:
    """The stage positions the Image's Planes record, read as lengths.

    A position in OME's default unit, ``reference frame``, is not a length, and reads as
    absent. That is the common case, and it leaves the image out of the shared world.
    """
    planes = list(image.pixels.planes)
    if not planes:
        return PlanePositions(None, None, None, None)

    def position(plane: Any, axis: str) -> float | None:
        return to_micrometer(
            getattr(plane, f"position_{axis}"), getattr(plane, f"position_{axis}_unit")
        )

    first = min(planes, key=lambda p: (p.the_t or 0, p.the_z or 0, p.the_c or 0))
    first_c = first.the_c or 0
    size_z = int(image.pixels.size_z)

    z: dict[int, float] = {}
    for plane in planes:
        if (plane.the_t or 0) != 0 or (plane.the_c or 0) != first_c:
            continue
        value = position(plane, "z")
        if value is not None:
            z.setdefault(plane.the_z or 0, value)
    stack = tuple(z[i] for i in range(size_z)) if size_z > 1 and len(z) == size_z else None
    if stack is not None and len(set(stack)) == 1:
        # The same z for every plane is where the focus or the stage's home was, not
        # where each plane was: the stack moved on a drive that was not read out.
        stack = None

    return PlanePositions(position(first, "x"), position(first, "y"), stack, position(first, "z"))


def unit_seconds(value: Any, unit: Any) -> float | None:
    """An OME time in seconds; OME's default unit for a time is the second."""
    if value is None:
        return None
    symbol = unit_symbol(unit)
    if symbol is None:
        return float(value)
    factor = SECONDS_PER_TIME_UNIT.get(symbol)
    return None if factor is None else float(value) * factor


@dataclass(frozen=True)
class Window:
    start: datetime.datetime
    end: datetime.datetime


def acquisition_windows(
    image: Any, timepoints: int
) -> tuple[Window | None, list[tuple[int, Window]], str | None]:
    """When this Image was acquired: as a whole, per timepoint, and why not if not.

    From its ``AcquisitionDate`` plus each plane's ``DeltaT``. The date has to say its
    time zone: an xsd:dateTime may leave it out, and stamping UTC on a local time would
    be off by hours. A date without a zone is still recorded verbatim, on the image
    anchor, but states no window. With no DeltaT the window is the date alone.
    """
    date = getattr(image, "acquisition_date", None)
    if date is None:
        return None, [], None
    if date.tzinfo is None or date.utcoffset() is None:
        return None, [], "AcquisitionDate states no time zone; no acquisition window is stated"

    deltas: dict[int, list[float]] = {}
    for plane in image.pixels.planes:
        seconds = unit_seconds(plane.delta_t, plane.delta_t_unit)
        if seconds is not None:
            deltas.setdefault(plane.the_t or 0, []).append(seconds)

    def at(seconds: float) -> datetime.datetime:
        return (date + datetime.timedelta(seconds=seconds)).astimezone(datetime.UTC)

    every = [s for values in deltas.values() for s in values]
    whole = Window(at(min(every)), at(max(every))) if every else Window(at(0.0), at(0.0))
    per_t = (
        [(t, Window(at(min(deltas[t])), at(max(deltas[t])))) for t in sorted(deltas) if t < timepoints]
        if timepoints > 1
        else []
    )
    return whole, per_t, None


def image_footprint(
    positions: PlanePositions, calibration: Calibration, size_y: int, size_x: int
) -> tuple[float, float, float, float] | None:
    """The image's field in the world, ``(y0, x0, y1, x1)`` in micrometers, from the
    header alone, or None when it is not placed in x and y."""
    if None in (positions.x_um, positions.y_um, calibration.x_um, calibration.y_um):
        return None
    return (
        positions.y_um,
        positions.x_um,
        positions.y_um + size_y * calibration.y_um,
        positions.x_um + size_x * calibration.x_um,
    )


def series_sizes(series: Any) -> dict[str, int]:
    """The series' extent along each of tifffile's axes."""
    return {str(axis): int(size) for axis, size in zip(series.axes, series.shape)}
