# ome-tiff-converter

<!-- arkitekt:badge -->
<!-- The release workflow puts the 'Open in Arkitekt' badge here on its first run. -->
<!-- /arkitekt:badge -->

Converts OME-TIFF files into mikro datasets **natively**: no Bio-Formats, no JVM. One
pyramided `ArrayDataset` per OME Image. Plain TIFFs without OME-XML convert too (see
[Plain TIFFs](#plain-tiffs)).

It is the sibling of `nd2-converter`, `lif-converter` and `ome-converter`, with the same
output contract. It reads the pixels with [tifffile](https://pypi.org/project/tifffile/)
and the OME-XML with [ome-types](https://pypi.org/project/ome-types/):

- **Lazy reads.** Each plane, or each block of tiles, is read out of the file when a chunk
  of it is uploaded. Nothing is decoded up front, and the file is never copied to a
  scratch Zarr. Only pyramid level 1 (¼ of the base, ⅛ for a z-stack) is persisted, next
  to the download, so that the coarser levels do not each re-read the base. A pyramidal
  OME-TIFF is read at its base, and the converter builds its own levels.
- **The OME record.** Each image's own OME document (its Image and its Instrument) goes
  on the image anchor as `{"format": "ome-tiff", …}` in the anchor's metadata blob.
  - The image anchor pins no axis, so it holds over the whole image. It carries the
    microscope, the objective and its settings, the environment, the stage, and the
    acquisition window.
  - Each channel anchor gets its name, fluor, colour, wavelengths, detector and light
    source settings, and a light path built from the Instrument
    (`ome_metadata.py`, shared with `ome-converter`).
  - A time series also states each timepoint's own window on its `{t: i}` anchor, from
    `AcquisitionDate` plus each plane's `DeltaT`. This is done only when
    `AcquisitionDate` names its time zone. A local time is recorded verbatim, and no
    window is claimed for it.
- **OME-derived display.** Each channel is drawn in its OME colour. OME's default white
  is not a choice, so a white channel falls back to the colour its emission implies, and
  failing that to a palette. It opens on p1/p99 of what its layer shows.
- **Tuned pyramids.** `levels=0` (the default) halves until the coarsest level is
  ≤ 512 px.

## Plain TIFFs

A TIFF with no OME-XML at all is converted too, as one dataset per tifffile series: a
plain image, a multi-page file, an ImageJ hyperstack, or a slide scanner's pyramid (read at
its base) and its label. Only what the header says is recorded, and nothing is invented:

- **Axes** are tifffile's. A run of pages it cannot name (`I` in a multi-page file, `Q` in
  a shaped one) is kept as an `INDEX` axis. It is not guessed to be z, t or c.
- **Calibration** is read only from ImageJ, and only when its `unit` is a length: the
  resolution tags give x/y, and `spacing` gives z. A baseline `XResolution` is a print
  resolution and is not read, because 72 dpi would claim 353 µm pixels. An uncalibrated
  series is staged in its own pixel grid. A calibrated one goes in a micrometer space of
  its own.
- **Placement**: never. A plain TIFF records no stage position.
- **The record**: the image anchor carries no OME metadata. Its `acquisition` device names
  the source file, the series (`Series:<n>`, also the dataset's source link), tifffile's
  series kind and axes, the pixel type and the sizes.
- **Channels** are named `Channel <n>` (`Red`/`Green`/`Blue` for an 8-bit RGB pixel) and
  are coloured from the palette.

## What is refused

The converter is handed one file. It refuses what it cannot read faithfully from that file
rather than upload something else:

| The file | What happens |
|---|---|
| A TIFF with OME-XML that cannot be read faithfully | Refused as an OME-TIFF (below). It never falls back to being read as a plain TIFF. |
| `BinaryOnly`: the pixels of a multi-file set, with the metadata in a `.companion.ome` | Refused, naming the companion. |
| An Image that keeps planes in another file of a multi-file set (TiffData `UUID FileName=…`) | That Image is skipped and reported; the others convert. |
| An Image some of whose planes are not in the file | Skipped. tifffile would fill them with zeros, which would then be uploaded as `IDENTICAL` to the file. |
| An Image tifffile finds no pixels for | Skipped and reported. |
| OME-XML that names no TiffData (pyvips' `MetadataOnly`) | Read when it is one Image over the file's one series, of the same extent. Otherwise refused. |

tifffile skips an Image it finds no data for, so its series are not always the
document's Images in order. Series are matched by position only when the counts agree.
Otherwise each series is matched by the IFD its Image says its first plane is in
(`ome_tiff.match_series`).

Where an OME modulo annotation splits OME's channels in two (FLIM lifetime bins along C,
say), tifffile reads the split axis as one of its own. It is kept as an `INDEX` axis,
and the channels are not described, rather than indexing one axis by the other.

## Placement

Every calibrated image is registered into one micrometer world. An axis whose
`PhysicalSize` is not a length (`pixel`, `reference frame`) is not given a scale, and an
uncalibrated image is staged in its own pixel grid.

- **x/y**: from the first plane's `PositionX`/`PositionY`. OME does not say whether that
  is the centre of the field or its first pixel. The converter reads it as the first
  pixel, as `ome-converter` does, so the same file lands in the same place through
  either converter. The convention is recorded on the image anchor's `stage` device.
  A position in OME's default unit, `reference frame`, is not a length, and does not
  place the image. That is the common case.
- **z**: a stack is placed at its lowest plane, and only when every plane of it records a
  `PositionZ`. A stack recorded top-down is turned over, so that z grows with the index.
- **All or nothing**: an image shares the world only where every spatial axis it has is
  placed. Otherwise it is registered, calibrated, into a micrometer space of its own.

## Several images and scenes

Every OME Image is uploaded and registered. Images whose fields touch share one mosaic
scene, for example the fields of a grid acquired but not stitched. Images that do not
touch get a scene each, as do images that are not placed: the wells of a plate, or a
slide and its label. A mosaic is capped at 48 layers, so a larger touching grid gets a
scene per image instead. Which images share a scene is decided from the header, before
any pixel is read, so each snapshot is rendered at the size of its scene.

An RGB image is one RGB layer when its samples are 8-bit. Several RGB pictures in one
Image (`C` × `S`) become one dataset each, and they share their Image's scene.

## The action

**Convert OME-TIFF file to Datasets** (`convert_ome_tiff_file_to_datasets`) takes a mikro
`File` (`.tif`, `.tiff`, `.tf2`, `.tf8` or `.btf`; whether it is an OME-TIFF or a plain TIFF
is decided by its content), plus `levels` (0 = auto), `method`, `stage_scene`, `world_name` and
`snapshot`. `snapshot` is on by default: each staged scene gets a maximum projection of
its layers as its snapshot, and each staged scene is set as the default scene of every
dataset in it. The action returns the datasets it made. If one image fails, it is logged
and the rest still convert.

## Developing

```sh
uv run pytest                                      # offline: no server; OME-TIFFs written by tifffile
OME_TIFF_SAMPLES=~/ome-tiff-samples uv run pytest  # also read every TIFF (OME or plain) under that directory
uv run arkitekt check
uv run arkitekt plugin build -p linux/amd64
```
