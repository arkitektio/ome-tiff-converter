"""Checks against real OME-TIFFs, written here by tifffile: what is read, what is refused.

Every file is made in the test's own directory, so these run offline like the rest. The
last checks run the whole action over a file, against a Mikro client that records what
it is asked to do instead of reaching a server.
"""

import datetime
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

import app as M
from ome_tiff import (
    Unreadable,
    acquisition_windows,
    match_series,
    plane_positions,
    read_calibration,
    read_ome,
)

UM = {"PositionXUnit": "µm", "PositionYUnit": "µm", "PositionZUnit": "µm"}


def positions(n, x, y, z=None):
    """Plane metadata for `n` planes at stage (x, y), and per plane z where given."""
    plane = {
        "PositionX": [float(x)] * n,
        "PositionY": [float(y)] * n,
        **{key: [unit] * n for key, unit in UM.items()},
    }
    plane["PositionZ"] = [float(v) for v in z] if z is not None else [0.0] * n
    return plane


def write(path, *images, **options):
    """An OME-TIFF of one Image per ``(array, metadata)``."""
    with tifffile.TiffWriter(path, ome=True) as writer:
        for data, metadata in images:
            writer.write(data, metadata=metadata, **options)
    return str(path)


def opened(path):
    tif = tifffile.TiffFile(path)
    return tif, read_ome(tif, Path(path).name)


def rewrite_xml(path, change):
    """The file at `path`, with `change` applied to its OME-XML."""
    with tifffile.TiffFile(path) as tif:
        xml = tif.ome_metadata
    tifffile.tiffcomment(path, change(xml).encode())


@pytest.fixture
def stack_and_photo(tmp_path):
    rng = np.random.default_rng(0)
    stack = rng.integers(0, 4000, (3, 2, 32, 48), dtype=np.uint16)  # Z C Y X
    photo = rng.integers(0, 255, (16, 16, 3), dtype=np.uint8)
    path = write(
        tmp_path / "pair.ome.tif",
        (
            stack,
            {
                "axes": "ZCYX",
                "Name": "stack",
                "PhysicalSizeX": 0.5,
                "PhysicalSizeY": 0.5,
                "PhysicalSizeZ": 2.0,
                "Channel": {"Name": ["DAPI", "GFP"]},
                # Recorded top-down: z falls as the index grows.
                "Plane": positions(6, 100, 200, [9, 9, 7, 7, 5, 5]),
            },
        ),
        (photo, {"axes": "YXS", "Name": "photo"}),
    )
    return path, stack, photo


def test_each_image_is_paired_with_its_series(stack_and_photo):
    path, stack, photo = stack_and_photo
    tif, ome = opened(path)
    with tif:
        images = match_series(tif, ome, "pair.ome.tif")
        assert [(i.name, i.problem) for i in images] == [("stack", None), ("photo", None)]
        np.testing.assert_array_equal(np.asarray(M.series_array(images[0].series)), stack)
        np.testing.assert_array_equal(np.asarray(M.series_array(images[1].series)), photo)
        assert M.series_array(images[1].series).dims == ("Y", "X", "S")


def test_the_planes_place_the_stack(stack_and_photo):
    path, _, _ = stack_and_photo
    tif, ome = opened(path)
    with tif:
        stack, photo = ome.images
        at = plane_positions(stack)
        assert (at.x_um, at.y_um, at.z_um) == (100.0, 200.0, (9.0, 7.0, 5.0))
        assert at.descending and at.bottom_um == 5.0
        assert read_calibration(stack) == M.Calibration(0.5, 0.5, 2.0)
        # tifffile writes no plane positions for an image it is told none of.
        assert plane_positions(photo).x_um is None
        assert read_calibration(photo).x_um is None


def test_a_position_in_reference_frame_units_is_not_a_place(tmp_path):
    path = write(
        tmp_path / "frame.ome.tif",
        (
            np.zeros((8, 8), np.uint8),
            {"axes": "YX", "PhysicalSizeX": 1.0, "PhysicalSizeY": 1.0,
             "Plane": {"PositionX": [5.0], "PositionY": [6.0]}},
        ),
    )
    rewrite_xml(path, lambda xml: xml.replace('PositionXUnit="µm"', "").replace('PositionYUnit="µm"', ""))
    tif, ome = opened(path)
    with tif:
        # OME's default unit for a plane position is "reference frame", not a length.
        assert plane_positions(ome.images[0]).x_um is None


def test_a_pyramid_is_read_at_its_base(tmp_path):
    data = np.random.default_rng(1).integers(0, 255, (2, 256, 256), dtype=np.uint8)
    path = tmp_path / "pyramid.ome.tif"
    with tifffile.TiffWriter(path, ome=True) as writer:
        writer.write(data, subifds=2, metadata={"axes": "CYX"}, tile=(64, 64))
        writer.write(data[:, ::2, ::2], subfiletype=1, tile=(64, 64))
        writer.write(data[:, ::4, ::4], subfiletype=1, tile=(64, 64))
    tif, ome = opened(str(path))
    with tif:
        [image] = match_series(tif, ome, path.name)
        array = M.series_array(image.series)
        assert array.shape == (2, 256, 256)
        # Tiles are read in blocks, not one task each.
        assert array.data.chunks[-1] == (256,)
        np.testing.assert_array_equal(np.asarray(array), data)


def test_a_compressed_file_decodes(tmp_path):
    data = np.arange(5 * 64 * 64, dtype=np.uint16).reshape(5, 64, 64)
    path = str(tmp_path / "lzw.ome.tif")
    tifffile.imwrite(path, data, ome=True, compression="lzw", metadata={"axes": "TYX"})
    tif, ome = opened(path)
    with tif:
        [image] = match_series(tif, ome, "lzw.ome.tif")
        np.testing.assert_array_equal(np.asarray(M.series_array(image.series)), data)


def test_a_plain_tiff_is_refused(tmp_path):
    path = str(tmp_path / "plain.tif")
    tifffile.imwrite(path, np.zeros((8, 8), np.uint8))
    with tifffile.TiffFile(path) as tif, pytest.raises(Unreadable, match="without OME-XML"):
        read_ome(tif, "plain.tif")


def test_a_binary_only_file_is_refused(tmp_path):
    path = write(tmp_path / "part.ome.tif", (np.zeros((8, 8), np.uint8), {"axes": "YX"}))
    head = '<?xml version="1.0" encoding="UTF-8"?>'

    def binary_only(xml):
        uuid = xml.split('UUID="')[1].split('"')[0]
        return (
            head
            + '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06" '
            + f'UUID="{uuid}"><BinaryOnly MetadataFile="set.companion.ome" '
            + f'UUID="{uuid}"/></OME>'
        )

    rewrite_xml(path, binary_only)
    with tifffile.TiffFile(path) as tif, pytest.raises(Unreadable, match="set.companion.ome"):
        read_ome(tif, "part.ome.tif")


def test_an_image_kept_in_another_file_is_refused(tmp_path):
    path = write(
        tmp_path / "set_0.ome.tif",
        (np.ones((2, 8, 8), np.uint8), {"axes": "CYX", "Name": "here"}),
        (np.ones((2, 8, 8), np.uint8), {"axes": "CYX", "Name": "elsewhere"}),
    )

    def second_plane_elsewhere(xml):
        # The second Image keeps its second plane in another file of the set.
        tail = xml.rsplit('<TiffData IFD="2" PlaneCount="2"/>', 1)
        assert len(tail) == 2, xml
        return tail[0] + (
            '<TiffData IFD="2" PlaneCount="1"/>'
            '<TiffData FirstC="1" IFD="0" PlaneCount="1">'
            '<UUID FileName="set_1.ome.tif">urn:uuid:00000000-0000-0000-0000-000000000001</UUID>'
            "</TiffData>"
        ) + tail[1]

    rewrite_xml(path, second_plane_elsewhere)
    tif, ome = opened(path)
    with tif:
        here, elsewhere = match_series(tif, ome, "set_0.ome.tif")
        assert here.series is not None and here.problem is None
        assert elsewhere.series is None and "set_1.ome.tif" in elsewhere.problem


def test_a_skipped_image_does_not_shift_the_others(tmp_path):
    path = write(
        tmp_path / "three.ome.tif",
        (np.full((8, 8), 1, np.uint8), {"axes": "YX", "Name": "first"}),
        (np.full((8, 8), 2, np.uint8), {"axes": "YX", "Name": "second"}),
    )

    def phantom_between(xml):
        # An Image whose plane is not in the file: tifffile skips it, so its series are
        # no longer the Images in order.
        first_end = xml.index("</Image>") + len("</Image>")
        phantom = (
            '<Image ID="Image:9" Name="phantom"><Pixels ID="Pixels:9" '
            'DimensionOrder="XYCZT" Type="uint8" SizeX="8" SizeY="8" SizeC="1" SizeZ="1" '
            'SizeT="1"><Channel ID="Channel:9:0" SamplesPerPixel="1"/>'
            '<TiffData IFD="50" PlaneCount="1"/></Pixels></Image>'
        )
        return xml[:first_end] + phantom + xml[first_end:]

    rewrite_xml(path, phantom_between)
    tif, ome = opened(path)
    with tif:
        first, phantom, second = match_series(tif, ome, "three.ome.tif")
        assert int(np.asarray(M.series_array(first.series)).max()) == 1
        assert phantom.series is None and phantom.problem
        assert int(np.asarray(M.series_array(second.series)).max()) == 2


def test_a_zoned_acquisition_date_states_windows(tmp_path):
    path = write(
        tmp_path / "timed.ome.tif",
        (
            np.zeros((3, 8, 8), np.uint8),
            {"axes": "TYX", "Plane": {"DeltaT": [0.0, 1.5, 4.0], "DeltaTUnit": ["s"] * 3}},
        ),
    )
    rewrite_xml(
        path,
        lambda xml: xml.replace(
            '<Pixels', '<AcquisitionDate>2026-03-01T10:00:00+01:00</AcquisitionDate><Pixels', 1
        ),
    )
    tif, ome = opened(path)
    with tif:
        whole, per_t, note = acquisition_windows(ome.images[0], 3)
    start = datetime.datetime(2026, 3, 1, 9, 0, tzinfo=datetime.UTC)
    assert note is None
    assert (whole.start, whole.end) == (start, start + datetime.timedelta(seconds=4))
    assert [(t, w.start) for t, w in per_t] == [
        (0, start),
        (1, start + datetime.timedelta(seconds=1.5)),
        (2, start + datetime.timedelta(seconds=4)),
    ]


def test_a_local_acquisition_date_states_no_window(tmp_path):
    path = write(tmp_path / "local.ome.tif", (np.zeros((8, 8), np.uint8), {"axes": "YX"}))
    rewrite_xml(
        path,
        lambda xml: xml.replace(
            '<Pixels', '<AcquisitionDate>2026-03-01T10:00:00</AcquisitionDate><Pixels', 1
        ),
    )
    tif, ome = opened(path)
    with tif:
        whole, per_t, note = acquisition_windows(ome.images[0], 1)
    assert whole is None and per_t == [] and "time zone" in note


# --- the whole action -----------------------------------------------------------------


class Space:
    def __init__(self, name, log):
        self.name, self.id, self.log = name, name, log

    def register(self, dataset, validity, scale, **offsets):
        self.log.append((self.name, dataset.name, scale, offsets))


class Dataset:
    def __init__(self, name, data, axis_names):
        self.name, self.id, self.data = name, name, data
        self.axis_names = axis_names
        self.intrinsic_system = SimpleNamespace(id=f"{name} pixels", name=f"{name} pixels")

    def lens(self, **pins):
        return SimpleNamespace(id=f"{self.name}|{pins}")


class Recorder:
    """The Mikro client, as far as the action uses it: every call recorded."""

    def __init__(self):
        self.calls = []
        self.datasets = []

    def create_array_dataset(self, *, data, scales, name, axes, anchors, source_files):
        # Read every level, as an upload would: a lazy read that fails, fails here.
        values = np.asarray(data)
        for level in scales:
            np.asarray(level.data if hasattr(level, "data") else level)
        dataset = Dataset(name, values, tuple(str(d) for d in data.dims))
        dataset.anchors, dataset.source_files = anchors, source_files
        self.datasets.append(dataset)
        return dataset

    def create_scene(self, **kw):
        self.calls.append(("create_scene", kw))
        return M.Scene.model_construct(
            id=kw["name"], name=kw["name"], world_coordinate_system=SimpleNamespace(name="w")
        )

    def __getattr__(self, name):
        return lambda **kw: self.calls.append((name, kw))


def run_action(tmp_path, monkeypatch, source, **options):
    registered = []
    monkeypatch.setattr(M, "space_3d", lambda mikro, name, unit: Space(name, registered))
    upload = Path(source)

    def download(target):
        shutil.copy(upload, target)
        return target

    file = SimpleNamespace(name=upload.name, id="file-1", store=SimpleNamespace(download=download))
    mikro = Recorder()
    monkeypatch.setenv("MIKRO_SCRATCH_DIR", str(tmp_path))
    returned = M.convert_ome_tiff_file_to_datasets(
        file=file, mikro=mikro, task=SimpleNamespace(progress=lambda *a, **k: None), **options
    )
    return returned, mikro, registered


def test_a_grid_and_a_far_field_convert_and_stage(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    tile = {"axes": "CYX", "PhysicalSizeX": 1.0, "PhysicalSizeY": 1.0,
            "Channel": {"Name": ["DAPI", "GFP"]}}
    tiles = [rng.integers(0, 4000, (2, 32, 32), dtype=np.uint16) for _ in range(3)]
    source = write(
        tmp_path / "grid.ome.tif",
        (tiles[0], {**tile, "Name": "A", "Plane": positions(2, 0, 0)}),
        (tiles[1], {**tile, "Name": "B", "Plane": positions(2, 32, 0)}),
        (tiles[2], {**tile, "Name": "far", "Plane": positions(2, 5000, 5000)}),
    )
    (tmp_path / "scratch").mkdir()
    returned, mikro, registered = run_action(tmp_path / "scratch", monkeypatch, source)

    assert [d.name for d in returned] == [
        "grid.ome.tif - A", "grid.ome.tif - B", "grid.ome.tif - far"
    ]
    for dataset, data in zip(returned, tiles):
        np.testing.assert_array_equal(dataset.data, data)
        [source_file] = dataset.source_files
        assert source_file.series_identifier.startswith("Image:")
    # Each at its first pixel, in the one world.
    assert [(name, offsets) for _, name, _, offsets in registered] == [
        ("grid.ome.tif - A", {"x": 0.0, "y": 0.0}),
        ("grid.ome.tif - B", {"x": 32.0, "y": 0.0}),
        ("grid.ome.tif - far", {"x": 5000.0, "y": 5000.0}),
    ]
    # The two touching tiles are one mosaic; the far field is a scene of its own.
    scenes = [kw["name"] for name, kw in mikro.calls if name == "create_scene"]
    assert scenes == ["grid.ome.tif (2 images)", "grid.ome.tif - far"]
    layers = [kw for name, kw in mikro.calls if name == "create_intensity_layer"]
    assert len(layers) == 6
    assert sum(1 for name, _ in mikro.calls if name == "create_scene_snapshot") == 2
    # The channels carry their OME names.
    labels = [a.label.label for a in returned[0].anchors if a.label is not None]
    assert labels == ["DAPI", "GFP"]


def test_an_rgb_photo_and_a_stack_convert(tmp_path, monkeypatch, stack_and_photo):
    path, stack, _ = stack_and_photo
    (tmp_path / "scratch").mkdir()
    returned, mikro, registered = run_action(tmp_path / "scratch", monkeypatch, path)

    stack_dataset, photo_dataset = returned
    # Turned over, so that z grows with the index: canonical order is c, z, y, x.
    np.testing.assert_array_equal(
        stack_dataset.data, np.moveaxis(stack, 1, 0)[:, ::-1]
    )
    [(_, name, scale, offsets)] = registered
    assert name == stack_dataset.name
    assert scale == {"z": 2.0, "y": 0.5, "x": 0.5}
    assert offsets == {"x": 100.0, "y": 200.0, "z": 5.0}
    # The photograph is not calibrated: one RGB layer, in its own pixel grid.
    assert photo_dataset.data.shape == (3, 16, 16)
    rgb = [kw for name, kw in mikro.calls if name == "create_rgb_layer"]
    assert len(rgb) == 1
    volume = [kw for name, kw in mikro.calls if name == "create_volume_layer"]
    assert len(volume) == 2


def metadata_only(xml):
    """The OME-XML a writer that names no TiffData writes: pyvips' ``MetadataOnly``."""
    import re

    return re.sub(r"<TiffData[^>]*/>", "<MetadataOnly/>", xml)


def test_one_image_without_tiff_data_is_the_files_one_series(tmp_path):
    photo = np.random.default_rng(3).integers(0, 255, (24, 16, 3), dtype=np.uint8)
    path = write(tmp_path / "vips.ome.tif", (photo, {"axes": "YXS"}))
    rewrite_xml(path, metadata_only)
    tif, ome = opened(path)
    with tif:
        assert tif.series[0].kind != "ome"
        [image] = match_series(tif, ome, "vips.ome.tif")
        np.testing.assert_array_equal(np.asarray(M.series_array(image.series)), photo)


def test_an_image_without_tiff_data_of_another_extent_is_refused(tmp_path):
    path = write(tmp_path / "vips.ome.tif", (np.zeros((24, 16), np.uint8), {"axes": "YX"}))
    rewrite_xml(path, lambda xml: metadata_only(xml).replace('SizeX="16"', 'SizeX="17"'))
    tif, ome = opened(path)
    with tif, pytest.raises(Unreadable):
        match_series(tif, ome, "vips.ome.tif")


def test_pages_without_tiff_data_unfold_into_channels_and_planes(tmp_path, monkeypatch):
    # Pages stored the way XYZCT orders them: z fastest, then c.
    data = np.random.default_rng(4).integers(0, 4000, (3, 2, 8, 12), dtype=np.uint16)  # C Z Y X
    path = write(
        tmp_path / "vips.ome.tif",
        (data, {"axes": "CZYX", "PhysicalSizeX": 1.0, "PhysicalSizeY": 1.0,
                "Channel": {"Name": ["a", "b", "c"]}}),
    )
    rewrite_xml(path, metadata_only)
    tif, ome = opened(path)
    with tif:
        assert tif.series[0].axes == "IYX"
        [image] = match_series(tif, ome, "vips.ome.tif")
        assert image.split == (("T", 1), ("C", 3), ("Z", 2))
        assert image.sizes == {"T": 1, "C": 3, "Z": 2, "Y": 8, "X": 12}

    (tmp_path / "scratch").mkdir()
    [dataset], mikro, _ = run_action(tmp_path / "scratch", monkeypatch, path)
    # One layer per channel, through z: not one layer over a page index.
    assert dataset.axis_names == ("c", "z", "y", "x")
    np.testing.assert_array_equal(dataset.data, data)
    assert len([kw for name, kw in mikro.calls if name == "create_volume_layer"]) == 3
    assert [a.label.label for a in dataset.anchors if a.label is not None] == ["a", "b", "c"]


def test_one_z_for_every_plane_does_not_place_the_stack(tmp_path):
    path = write(
        tmp_path / "home.ome.tif",
        (np.zeros((3, 8, 8), np.uint8),
         {"axes": "ZYX", "PhysicalSizeX": 1.0, "PhysicalSizeY": 1.0, "PhysicalSizeZ": 1.0,
          "Plane": positions(3, 10, 20, [42, 42, 42])}),
    )
    tif, ome = opened(path)
    with tif:
        at = plane_positions(ome.images[0])
    assert (at.x_um, at.y_um, at.z_um, at.focus_um) == (10.0, 20.0, None, 42.0)


def test_a_time_series_states_its_windows_on_the_dataset(tmp_path, monkeypatch):
    path = write(
        tmp_path / "timed.ome.tif",
        (
            np.zeros((3, 8, 8), np.uint8),
            {"axes": "TYX", "Plane": {"DeltaT": [0.0, 1.5, 4.0], "DeltaTUnit": ["s"] * 3}},
        ),
    )
    rewrite_xml(
        path,
        lambda xml: xml.replace(
            '<Pixels', '<AcquisitionDate>2026-03-01T10:00:00+01:00</AcquisitionDate><Pixels', 1
        ),
    )
    (tmp_path / "scratch").mkdir()
    [dataset], _, _ = run_action(tmp_path / "scratch", monkeypatch, path)
    image = dataset.anchors[0]
    start = datetime.datetime(2026, 3, 1, 9, 0, tzinfo=datetime.UTC)
    assert image.acquisition_window.start == start
    assert image.acquisition_window.end == start + datetime.timedelta(seconds=4)
    timepoints = [a for a in dataset.anchors if a.axis_anchors and a.axis_anchors[0].axis == "t"]
    assert [a.axis_anchors[0].value for a in timepoints] == [0, 1, 2]


def test_a_big_endian_file_uploads_its_values(tmp_path, monkeypatch):
    data = np.arange(2 * 16 * 16, dtype=np.uint16).reshape(2, 16, 16) * 7
    path = str(tmp_path / "motorola.ome.tif")
    tifffile.imwrite(path, data, ome=True, byteorder=">", metadata={"axes": "CYX"})
    (tmp_path / "scratch").mkdir()
    [dataset], _, _ = run_action(tmp_path / "scratch", monkeypatch, path)
    np.testing.assert_array_equal(dataset.data, data)
    assert dataset.data.dtype.isnative
