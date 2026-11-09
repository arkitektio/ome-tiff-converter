"""Offline checks of the converter: what it reads, where it places it, what it authors.

No backend: the Mikro client is a recorder, so the assertions land on the calls that
would reach the server rather than on a property that predicts them. The checks that
read real OME-TIFFs (written here by tifffile) are in `test_ome_tiff.py`.

The layer, pyramid, preview and staging checks are the ND2 converter's, unchanged: the
converters share that code, and the rule it serves.

    ONE LAYER PER CHANNEL. The only exception is a packed RGB(A) pixel.
"""

import numpy as np
import pytest
import xarray as xr

import app as M
from ome_tiff import Calibration, PlanePositions


class _RecordingMikro:
    """Stands in for the Mikro client: records every mutation the layer author makes."""

    def __init__(self, calls: list) -> None:
        self._calls = calls

    def __getattr__(self, name: str):
        return lambda **kw: self._calls.append((name, kw))


@pytest.mark.parametrize(
    ("sizes", "levels"),
    [
        ({"y": 512, "x": 512}, 2),
        ({"y": 1024, "x": 1024}, 2),
        ({"y": 11003, "x": 11022}, 6),
        ({"y": 64, "x": 64}, 2),
        ({"y": 3741, "x": 3680}, 4),
    ],
)
def test_auto_levels(sizes, levels):
    assert M.auto_levels(sizes) == levels

@pytest.mark.parametrize(
    ("packed", "channels", "dtype", "expected"),
    [
        (True, 3, np.uint8, True),
        (True, 4, np.uint8, True),
        (False, 3, np.uint8, False),
        (True, 6, np.uint8, False),
        # An RGB layer is drawn unstretched over 0..255: a deeper sample axis is a colour
        # camera's acquisition and gets a layer and a window per sample instead.
        (True, 3, np.uint16, False),
        (True, 3, np.float32, False),
    ],
)
def test_packs_one_pixel(packed, channels, dtype, expected):
    assert M.packs_one_pixel(packed, channels, dtype) is expected

def test_display_window():
    assert M.display_window(np.arange(101.0).reshape(1, 101)) == (1.0, 99.0)
    # Over 99 % one value: the percentiles coincide, and a window of no width draws
    # nothing.
    dots = np.zeros((64, 64), dtype=np.uint16)
    dots[0, :8] = 500
    assert M.display_window(dots) == (0.0, 500.0)
    assert M.display_window(np.full((4, 4), 7.0)) == (7.0, 8.0)



class FakeId:
    id = "fake"
    name = "fake"


class FakeLens:
    def __init__(self, id):
        self.id = id


class FakeDataset:
    """A dataset as far as the layer author reads one: its axes, and its lenses.

    A lens' id spells out what it pins — ``"lens"`` for the whole lens, ``"lens:c=2"``
    for channel 2 — so a recorded call can be asked which channel it draws.
    """

    axis_names = ("c", "z", "y", "x")

    def lens(self, **selections):
        for axis in selections:
            if axis not in self.axis_names:
                raise ValueError(f"Invalid axis {axis!r} for axes {list(self.axis_names)}")
        if not selections:
            return FakeLens("lens")
        return FakeLens("lens:" + ",".join(f"{a}={v}" for a, v in selections.items()))


def pinned_channel(lens_id):
    """The channel a fake lens id pins, or None for the whole lens and for no lens."""
    if not isinstance(lens_id, str) or not lens_id.startswith("lens:"):
        return None
    pins = dict(pin.split("=") for pin in lens_id[len("lens:") :].split(","))
    return int(pins["c"]) if "c" in pins else None


def fake_specs(n):
    return [
        M.ChannelSpec(
            index=i,
            label=f"ch{i}",
            color=(i * 60 % 256, 0, 0, 255),
            histogram=M.ValueHistogramInput(
                histogram=[1.0], bins=[0.0, 1.0], min=0.0, max=1.0, p1=0.0, p99=1.0
            ),
            # Deliberately not the histogram's p1/p99, so a check can tell which of the
            # two a layer was given.
            clim=(10.0 + i, 20.0 + i),
            ome=None,
        )
        for i in range(n)
    ]


def layers_for(specs, implicit_rgb, is_volume=True):
    calls = []
    made = M.author_channel_layers(
        _RecordingMikro(calls), FakeDataset(), FakeId(), specs, is_volume, implicit_rgb
    )
    assert made == len(calls), "layer count must match the calls made"
    return calls


def selected_channels(call):
    """Every channel index this layer covers, whichever kind of layer it is.

    An unrecognised call is an error rather than an empty list: otherwise the converter
    moving to a mutation this walker does not know would make every check below pass
    vacuously against nothing at all.
    """
    name, kw = call
    if name in ("create_intensity_layer", "create_volume_layer"):
        index = pinned_channel(kw.get("lens"))
        assert index is not None, f"{name} call's lens pins no channel: {kw!r}"
        return [index]
    if name == "create_rgb_layer":
        found = [pinned_channel(kw.get(k)) for k in ("red_lens", "green_lens", "blue_lens")]
        assert None not in found, f"{name} call is missing a component lens: {kw!r}"
        return found
    if name == "create_layer":
        found = []

        def walk(node):
            if node is None:
                return
            if node.get("kind") == "channel":
                index = pinned_channel(node.get("lens"))
                assert index is not None, f"channel node's lens pins no channel: {node!r}"
                found.append(index)
            for child in node.get("children") or ():
                walk(child)

        graph = kw.get("render_graph")
        assert graph is not None, f"create_layer call carries no render_graph: {kw!r}"
        walk(graph.model_dump()["root"])
        assert found, f"create_layer graph has no channel node: {kw!r}"
        return found
    raise AssertionError(f"unrecognised layer mutation {name!r}; teach the walker about it")


def test_a_z_stack_is_one_volume_layer_per_channel():
    four = layers_for(fake_specs(4), False)
    assert len(four) == 4
    assert {n for n, _ in four} == {"create_volume_layer"}
    assert sorted(i for c in four for i in selected_channels(c)) == [0, 1, 2, 3]
    assert sorted(tuple(kw["color"]) for _, kw in four) == sorted(
        tuple(s.color) for s in fake_specs(4)
    )
    assert sorted((kw["clim_min"], kw["clim_max"]) for _, kw in four) == [
        (10.0 + i, 20.0 + i) for i in range(4)
    ]


def test_a_packed_rgb_z_stack_is_one_render_graph_layer():
    rgb = layers_for(fake_specs(3), True)
    assert len(rgb) == 1
    assert rgb[0][0] == "create_layer"
    assert sorted(selected_channels(rgb[0])) == [0, 1, 2]
    assert len(layers_for(fake_specs(4), True)) == 1


@pytest.mark.parametrize("n", [2, 5])
def test_implicit_rgb_over_a_non_pixel_channel_count_is_refused(n):
    assert len(layers_for(fake_specs(n), True)) == n


def test_flat_channels_are_intensity_layers():
    flat = layers_for(fake_specs(4), False, is_volume=False)
    assert {n for n, _ in flat} == {"create_intensity_layer"}
    assert sorted(i for c in flat for i in selected_channels(c)) == [0, 1, 2, 3]
    assert [kw.get("mode") for _, kw in flat] == [None] * 4


def test_a_flat_packed_rgb_pixel_is_one_rgb_layer():
    flat_rgb = layers_for(fake_specs(3), True, is_volume=False)
    assert [n for n, _ in flat_rgb] == ["create_rgb_layer"]
    assert selected_channels(flat_rgb[0]) == [0, 1, 2]
    kw = flat_rgb[0][1]
    assert (kw["clim_min"], kw["clim_max"]) == (0.0, 255.0)
    assert "blending" not in kw


def test_rgba_labels_and_colours():
    assert [M.channel_label(None, i, True) for i in range(4)] == [
        "Red",
        "Green",
        "Blue",
        "Alpha",
    ]
    assert [M.channel_color(None, i, True, 4) for i in range(4)] == [
        (255, 0, 0, 255),
        (0, 255, 0, 255),
        (0, 0, 255, 255),
        (255, 255, 255, 255),
    ]


def test_a_persisted_level_one_changes_no_value(tmp_path):
    import dask.array as da

    base = xr.DataArray(
        da.from_array(np.random.default_rng(0).integers(0, 255, (2, 4, 64, 64), dtype=np.uint8), chunks=(1, 1, 64, 64)),
        dims=("c", "z", "y", "x"),
    )
    lazy = M.build_pyramid(base, levels=4, method="max", materialize=False)
    persisted = M.image_pyramid(base, 4, "max", {}, str(tmp_path))
    assert len(persisted) == 4
    for a, b in zip(lazy, persisted):
        assert a.dims == b.dims and a.dtype == b.dtype
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # Level 1 is read back from the scratch directory, not re-derived from the base.
    assert any(tmp_path.iterdir())


# --- the scene snapshot ---------------------------------------------------------------


def test_a_preview_is_the_maximum_projection_in_each_channels_window():
    data = np.zeros((2, 3, 4, 4), dtype=np.uint16)
    data[0, 1, 0, 0] = 20  # channel 0, bright at one plane only
    data[1, 2, 3, 3] = 15  # channel 1, halfway into its window
    level = xr.DataArray(data, dims=("c", "z", "y", "x"))
    specs = [
        M.ChannelSpec(0, "a", (255, 0, 0, 255), None, (0.0, 20.0), None),
        M.ChannelSpec(1, "b", (0, 255, 0, 255), None, (10.0, 20.0), None),
    ]

    rgb = M.render_preview(level, specs, implicit_rgb=False, side=4)

    assert rgb.shape == (4, 4, 3) and rgb.dtype == np.uint8
    assert tuple(rgb[0, 0]) == (255, 0, 0)
    assert tuple(rgb[3, 3]) == (0, 127, 0)
    assert rgb[1:3, 1:3].max() == 0


def test_a_preview_fits_its_side():
    level = xr.DataArray(np.zeros((1, 100, 50), dtype=np.uint8), dims=("c", "y", "x"))
    specs = [M.ChannelSpec(0, "a", (255, 255, 255, 255), None, (0.0, 1.0), None)]
    assert M.render_preview(level, specs, False, side=32).shape == (25, 13, 3)


def test_an_rgba_previews_alpha_is_not_drawn():
    level = xr.DataArray(np.full((4, 2, 2), 255, dtype=np.uint8), dims=("c", "y", "x"))
    specs = [
        M.ChannelSpec(i, "", M.channel_color(None, i, True, 4), None, (0.0, 255.0), None)
        for i in range(4)
    ]
    assert tuple(M.render_preview(level, specs, True, side=2)[0, 0]) == (255, 255, 255)


def test_tiles_are_laid_out_at_their_stage_positions():
    red = np.zeros((2, 2, 3), dtype=np.uint8)
    red[..., 0] = 200
    green = np.zeros((2, 2, 3), dtype=np.uint8)
    green[..., 1] = 100
    canvas = M.mosaic(
        [
            M.Preview(red, (0.0, 0.0), (1.0, 1.0)),
            M.Preview(green, (0.0, 2.0), (1.0, 1.0)),
        ]
    )
    assert canvas.shape == (2, 4, 3)
    assert tuple(canvas[0, 0]) == (200, 0, 0)
    assert tuple(canvas[1, 3]) == (0, 100, 0)


def test_the_snapshot_is_a_png():
    import zlib

    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    png = M.encode_png(rgb)
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    start = png.index(b"IDAT") + 4
    length = int.from_bytes(png[start - 8 : start - 4], "big")
    rows = np.frombuffer(zlib.decompress(png[start : start + length]), np.uint8)
    rows = rows.reshape(2, 1 + 3 * 3)
    assert (rows[:, 0] == 0).all()
    assert (rows[:, 1:].reshape(2, 3, 3) == rgb).all()


def test_a_scene_gets_its_snapshot(tmp_path):
    calls = []
    scene = M.Scene.model_construct(id="s1", name="scene")
    entry = M.Converted(
        0, "x", None, [], True, False, False,
        M.Preview(np.zeros((2, 2, 3), np.uint8), None, None),
    )
    M.snapshot_scene(_RecordingMikro(calls), scene, [entry], str(tmp_path))
    [(name, kw)] = calls
    assert name == "create_scene_snapshot"
    assert kw["scene"] == "s1"
    assert kw["file"].name.endswith(".png")


def test_a_mosaic_never_exceeds_the_snapshot_side():
    tile = np.full((4, 4, 3), 255, dtype=np.uint8)
    far = 10 * M.SNAPSHOT_PIXELS
    canvas = M.mosaic(
        [M.Preview(tile, (0.0, 0.0), (1.0, 1.0)), M.Preview(tile, (0.0, far), (1.0, 1.0))]
    )
    assert max(canvas.shape[:2]) <= M.SNAPSHOT_PIXELS
    assert canvas[0, 0].any() and canvas[0, -1].any()


def test_an_unplaced_series_is_staged_alone_in_its_own_space(monkeypatch):
    composed = []
    monkeypatch.setattr(
        M, "compose_scene", lambda mikro, name, system, members: composed.append((name, system, [m.name for m in members]))
    )
    world, own = object(), object()
    entries = [
        M.Converted(0, "a", FakeDataset(), fake_specs(1), True, False, False, None),
        M.Converted(1, "b", FakeDataset(), fake_specs(1), True, False, False, None),
        M.Converted(2, "c", FakeDataset(), fake_specs(1), False, False, False, None, own),
    ]
    M.stage_group(None, "file", world, entries)
    assert composed == [("file", world, ["a", "b"]), ("c", own, ["c"])]


def test_a_composed_scene_is_each_datasets_default():
    calls = []

    class Mikro(_RecordingMikro):
        def create_scene(self, **kw):
            calls.append(("create_scene", kw))
            return M.Scene.model_construct(
                id="s1", name=kw["name"], world_coordinate_system=FakeId()
            )

    entries = [
        M.Converted(tile, f"t{tile}", FakeDataset(), fake_specs(1), True, False, False, None)
        for tile in range(2)
    ]
    M.compose_scene(Mikro(calls), "mosaic", FakeId(), entries)
    nominations = [kw for name, kw in calls if name == "set_lens_default_scene"]
    assert nominations == [{"lens": "lens", "scene": "s1"}] * 2




# --- tifffile's array, one image at a time ------------------------------------------


def whole(dims, shape, fill=None):
    data = np.zeros(shape, dtype=np.uint8) if fill is None else fill
    return xr.DataArray(data, dims=dims)


@pytest.mark.parametrize(
    ("dims", "shape", "expected"),
    [
        # An RGB TIFF: tifffile's sample axis.
        (("Y", "X", "S"), (8, 8, 3), (3, True)),
        # Dyes, stored per channel.
        (("C", "Y", "X"), (3, 8, 8), (3, False)),
        (("Z", "C", "Y", "X"), (5, 4, 8, 8), (4, False)),
        # One channel, dropped by tifffile: c comes back at extent 1.
        (("Y", "X"), (8, 8), (1, False)),
    ],
)
def test_the_sample_axis_decides_packing(dims, shape, expected):
    array, _, packed = M.field_array(whole(dims, shape), None, False)
    assert (int(array.sizes["c"]), packed) == expected


def test_several_colour_channels_are_converted_one_at_a_time():
    data = np.arange(2, dtype=np.uint8)[:, None, None, None] * np.ones((2, 4, 4, 3), np.uint8)
    both = whole(("C", "Y", "X", "S"), data.shape, data)
    with pytest.raises(ValueError):
        M.field_array(both, None, False)
    for c in range(2):
        array, _, packed = M.field_array(both, c, False)
        assert packed and array.sizes["c"] == 3 and int(array.max()) == c


def test_a_top_down_stack_is_turned_over():
    data = np.arange(4, dtype=np.uint8)[:, None, None] * np.ones((4, 2, 2), np.uint8)
    stack = whole(("Z", "Y", "X"), data.shape, data)
    up, _, _ = M.field_array(stack, None, True)
    assert [int(up.isel(c=0, z=i).max()) for i in range(4)] == [3, 2, 1, 0]
    kept, _, _ = M.field_array(stack, None, False)
    assert [int(kept.isel(c=0, z=i).max()) for i in range(4)] == [0, 1, 2, 3]


def test_a_modulo_axis_is_declared_index_not_space():
    # tifffile reads a FLIM file's lifetime bins out of its modulo annotation as "H".
    _, extra, _ = M.field_array(whole(("H", "C", "Y", "X"), (5, 2, 4, 4)), None, False)
    assert extra == {"H": "INDEX"}


@pytest.mark.parametrize(
    ("sizes", "dtype", "layers"),
    [
        ({"C": 3, "Y": 8, "X": 8}, np.uint16, 3),
        ({"Y": 8, "X": 8, "S": 3}, np.uint8, 1),
        ({"C": 2, "Y": 8, "X": 8, "S": 3}, np.uint8, 2),
        # A 16-bit colour camera's samples are a layer each.
        ({"Y": 8, "X": 8, "S": 3}, np.uint16, 3),
        ({"Y": 8, "X": 8}, np.uint8, 1),
    ],
)
def test_the_layers_an_image_will_compose(sizes, dtype, layers):
    assert M.layers_of(sizes, dtype) == layers


# --- placement -----------------------------------------------------------------------


def test_voxel_scale():
    scale, note = M.voxel_scale(Calibration(0.65, 0.65, 0.3), ["c", "z", "y", "x"])
    assert scale == {"z": 0.3, "y": 0.65, "x": 0.65} and note is None
    scale, note = M.voxel_scale(Calibration(0.65, 0.65, None), ["c", "z", "y", "x"])
    assert scale["z"] == 1.0 and "z spacing unknown" in note


def test_uncalibrated_pixels_are_not_registered():
    scale, note = M.voxel_scale(Calibration(None, None, None), ["c", "y", "x"])
    assert scale is None and "not calibrated" in note


def test_only_a_fully_placed_image_shares_the_world():
    flat = xr.DataArray(np.zeros((1, 4, 6)), dims=("c", "y", "x"))
    stack = xr.DataArray(np.zeros((1, 3, 4, 6)), dims=("c", "z", "y", "x"))
    scale = {"z": 2.0, "y": 0.5, "x": 0.5}

    # The first plane's position is that of the first pixel.
    at = PlanePositions(100.0, 200.0, None, 7.0)
    assert M.world_offsets(at, scale, flat) == {"x": 100.0, "y": 200.0}
    # A stack needs every plane's z; the first plane's focus alone does not place it.
    assert M.world_offsets(at, scale, stack) is None
    stacked = PlanePositions(100.0, 200.0, (9.0, 7.0, 5.0), 9.0)
    assert M.world_offsets(stacked, scale, stack) == {"x": 100.0, "y": 200.0, "z": 5.0}
    # No stage, no place.
    assert M.world_offsets(PlanePositions(None, None, None, None), scale, flat) is None


def test_a_top_down_stack_is_read_as_such():
    assert PlanePositions(0.0, 0.0, (9.0, 7.0, 5.0), 9.0).descending
    assert not PlanePositions(0.0, 0.0, (5.0, 7.0, 9.0), 5.0).descending
    assert not PlanePositions(0.0, 0.0, None, 5.0).descending


# --- scenes --------------------------------------------------------------------------


def box(y, x, side=10.0):
    return (y, x, y + side, x + side)


def test_a_grid_of_touching_fields_is_one_mosaic():
    grid = [box(0, 0), box(0, 10), box(10, 0), box(10, 10)]
    assert M.scene_groups(grid, [2] * 4) == [[0, 1, 2, 3]]


def test_wells_far_apart_are_a_scene_each():
    wells = [box(0, 0), box(0, 9000), box(9000, 0)]
    assert M.scene_groups(wells, [2] * 3) == [[0], [1], [2]]


def test_two_grids_are_two_mosaics():
    footprints = [box(0, 0), box(5000, 5000), box(0, 10), box(5000, 5010)]
    assert M.scene_groups(footprints, [1] * 4) == [[0, 2], [1, 3]]


def test_an_unplaced_image_shares_no_mosaic():
    assert M.scene_groups([box(0, 0), None, box(0, 10)], [1] * 3) == [[0, 2], [1]]


def test_a_touching_grid_over_the_layer_budget_gets_a_scene_each():
    grid = [box(0, 10 * i) for i in range(13)]
    assert M.scene_groups(grid, [4] * 13) == [[i] for i in range(13)]


def test_a_mosaic_shares_the_snapshot_side():
    sides = M.preview_sides([[0, 1, 2, 3], [4]])
    assert sides[0] == sides[3] == M.SNAPSHOT_PIXELS // 2
    assert sides[4] == M.SNAPSHOT_PIXELS


# --- colours and names ---------------------------------------------------------------


class Colour:
    def __init__(self, rgba):
        self.rgba = rgba

    def as_rgb_tuple(self, alpha=True):
        return self.rgba if alpha else self.rgba[:3]


def ome_channel(color=None, emission=None, name=None, fluor=None):
    from types import SimpleNamespace

    return SimpleNamespace(
        color=Colour(color) if color else None,
        emission_wavelength=emission,
        emission_wavelength_unit=None,
        name=name,
        fluor=fluor,
        id="Channel:0:0",
    )


def test_colours_follow_ome_then_the_emission_then_the_palette():
    assert M.channel_color(ome_channel(color=(0, 255, 0, 1.0)), 0, False, 2) == (0, 255, 0, 255)
    # White is OME's default: no choice was made, so the emission decides.
    # An emission without a unit is in micrometers, OME's default for a length.
    red = M.channel_color(ome_channel(color=(255, 255, 255, 1.0), emission=0.7), 0, False, 2)
    assert red == M.color_for_wavelength(700.0, M.COLOR_BY_EMISSION, M.FAR_RED)
    assert M.channel_color(ome_channel(), 1, False, 2) == M.fallback_color(1, 2)


def test_a_channel_label_is_its_name_then_its_fluor():
    assert M.channel_label(ome_channel(name="DAPI"), 0, False) == "DAPI"
    assert M.channel_label(ome_channel(fluor="GFP"), 0, False) == "GFP"
    assert M.channel_label(None, 2, False) == "Channel 2"
