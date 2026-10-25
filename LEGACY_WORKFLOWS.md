# Historical bin-filter datasets and CLI workflows

These examples document earlier TIFF, planar and height-field experiments.
For new collection and deployment, use the [Bin Setup operator guide](STANDALONE_UI.md).
Local data and outputs are not included in Git; old preview clouds may have been
removed because they can be regenerated from saved source data.

## The October 16 dataset: curved bowl workflow

The capture data and bulk offline outputs below are local collection artifacts
and are not included in Git. These links and replay commands assume that dataset
is present. The historical rollback bundle is retained locally under
[`../ebots_system_artifacts/artifacts_cell_4/calibration/deployment/bin/fremont-2026-10-16/`](../ebots_system_artifacts/artifacts_cell_4/calibration/deployment/bin/fremont-2026-10-16/).
The [runtime validation report](outputs/2026-10-16/runtime_validation/REPORT.md)
and its small result files remain available locally. Regenerable old cloud
previews may have been removed during cleanup; the source captures and model
remain available for replay.

The organized dataset is [data/dataset_2026-10-16](data/dataset_2026-10-16).
Its [manifest](data/dataset_2026-10-16/dataset.yaml) registers all 30 completed
captures, and [capture_index.csv](data/dataset_2026-10-16/capture_index.csv) maps
IDs to original timestamp folders and annotations. The original TIFFs in
`data/empty_captures` and `data/filled_captures` remain unchanged. Registration
references those files; `captures/train` and `captures/test` inside the dataset
are symlink views, not extra copies.

The captures were collected on October 16, 2026, approximately 11:07–11:26 PDT.
Capture metadata records 8.0 ms for both spotlight and structured-light
exposures. There are 60 original depth TIFFs and 60 original spotlight TIFFs.
Calibration copies are frozen under the dataset's `calibration` directory. The
source calibration file modification times precede acquisition, but the captures
did not embed calibration snapshots; those times alone cannot prove which
calibration files the running camera service used.

The split is frozen before fitting:

| Capture group | Train | Test |
| --- | --- | --- |
| Empty | `empty_01`–`empty_07` (7) | `empty_08`–`empty_10` (3) |
| Filled | All other filled IDs (12) | `filled_03`, `filled_06`, `filled_09`, `filled_12`, `filled_16`, `filled_17`, `filled_18`, `filled_20` (8) |
| **Total** | **19** | **11** |

Only the seven both-empty training captures fit the surface. Filled captures
01–03 have parts in A only, 04–06 in B only, and 07–20 in both bins. Spotlight
review supplied per-bin labels and placement notes. `filled_16` is held out with
`filled_17` because their A-bin placement may be shared. Consequently this actual
split differs from the 20/10 template for a new collection below.

The reviewed [bin_config.yaml](data/dataset_2026-10-16/bin_config.yaml) selects
`kind: height_field` for both bins. `calibrate` dispatches from those per-bin
settings; no top-level `model_type` setting is required. The exported model type
is `offline_height_field_bins`. Run from this directory:

```bash
python3 bin_filter.py calibrate --dataset data/dataset_2026-10-16 \
  --output outputs/2026-10-16/model_v1 --stride 1 --output-format yaml

python3 bin_filter.py validate --dataset data/dataset_2026-10-16 \
  --model outputs/2026-10-16/model_v1/model.yaml \
  --output outputs/2026-10-16/validation_train --split train --stride 1

python3 bin_filter.py validate --dataset data/dataset_2026-10-16 \
  --model outputs/2026-10-16/model_v1/model.yaml \
  --output outputs/2026-10-16/validation_test --split test --stride 1

python3 bin_filter.py filter --dataset data/dataset_2026-10-16 \
  --model outputs/2026-10-16/model_v1/model.yaml \
  --capture filled_01 --bin bin_a \
  --output outputs/2026-10-16/filtered_filled_01_bin_a --stride 1
```

These commands produce the versioned model and validation reports at the named
locations: [model.yaml](outputs/2026-10-16/model_v1/model.yaml),
[training report](outputs/2026-10-16/validation_train/validation.json), and
[test report](outputs/2026-10-16/validation_test/validation.json). This run has
completed for all 30 captures; see the [results and review images](outputs/2026-10-16/REPORT.md).
File existence and low empty-bin residue alone are not an acceptance result.
The subsequently supplied STL was also compared with the empty training surfaces;
the [CAD review](outputs/2026-10-16/cad_review/README.md) found a close geometric match
under the 1-unit = 1-mm assumption. That separate review did not change the fitted
model or its held-out results.
Each output directory must be new. Use another version name if it already exists. Keep
`--stride 1` for this fit so the cell support reflects the original depth
density. Height-field fitting uses all points in its fitting region and ignores
the planar-only `--max-fit-points` sampling limit.

The height field is fixed in the B2 robot frame, in millimetres. Its configuration
sets `xy_bounds_mm: [[xmin, ymin], [xmax, ymax]]` for each bin, a background-only
`z_bounds_mm: [zmin, zmax]` search band, and a `seed_xy_mm: [x, y]` point selecting
the connected observed bowl surface. These are reviewed regions,
not CAD dimensions or automatic bin detection. The current fit uses 1 mm cells,
at least three points per capture/cell, and support from at least five empty
training captures. It first computes a median Z per capture/cell, then a median
across captures. A reference-height cutoff separates the low bowl surface from
the rim; disconnected background is excluded.

Filtering interpolates only where all four surrounding reference cells are
observed. For each interpolation square, the noise tolerance uses the largest
per-capture 99th-percentile absolute Z residual plus a 0.5 mm margin, with a
1 mm minimum. Squares exceeding the 4 mm maximum, lacking repeated support, or
disconnected from the seed become **unknown**; they are not filled in or assigned
a clipped tolerance. The saved model includes the reference heights, support
counts, noise tolerances, and coverage diagnostics.

At a supported XY location, points within the tolerance of the learned surface
are removed. Points more than the tolerance above it, up to the configured
150 mm height limit, are retained. Lower points and points above that limit are
outside; unknown regions are excluded. The training rim cutoff does not impose
a ceiling on live parts above the rim. This model represents one height per XY
location, so it does not model overhangs or unseen vertical surfaces.

In the classified previews, green means retained, blue means learned surface,
purple means unknown, and grey means outside. Inspect unknown coverage alongside
empty-bin leakage: a small retained count can also result from missing reference
coverage. Compare loaded results with their raw clouds and spotlight images to
check which parts remain. Held-out results are evaluation evidence, not fitting
input. Moving a camera changes the reference scene and requires reassessment
and a matching dataset/model. The [runtime integration](RUNTIME.md) uses the
same classification in C++ and selects a bin through the GetPose `bin_id` port.
Its optional ICP handles small observable bin movements; larger or poorly
observed changes require a new initial reference and validation.

## Archived single-exposure TIFF collection plan

For new factory data, use the [current EXR collection workflow](MODEL_STORAGE.md).
The material below preserves the earlier manually organized TIFF process.

The following 30-set template covers one exposure. For multiple settings,
follow the anchor and held-out intermediate plan in
[EXPOSURE_PROFILES.md](EXPOSURE_PROFILES.md#collect-anchors-and-independent-intermediate-checks).
Each anchor needs its own independent empty training and test captures.

One **capture set** means one stationary scene acquisition containing both bins,
with a depth TIFF and a spotlight TIFF from each camera. The initial target is
30 sets: 10 with both bins empty and 20 with parts. This gives **60 depth images
and 60 spotlight images**, plus metadata. Keep the camera rig and both bins fixed
throughout the dataset. Park the robot and suction cup clear of the bins before
each capture.

Choose the held-out sets before fitting. This is a template for a new collection,
not the exact split of the organized October 16 dataset:

| Scene category | Total sets | Train | Test |
| --- | ---: | ---: | ---: |
| Both bins empty | 10 | 7 | 3 |
| A: one thin part at center/wall/corner; B: empty | 3 | 2 | 1 |
| A: empty; B: one thin part at center/wall/corner | 3 | 2 | 1 |
| A: a few parts; B: normal fill | 3 | 2 | 1 |
| A: normal fill; B: a few parts | 3 | 2 | 1 |
| Both normal: three layouts, two captures per layout | 6 | 4 | 2 |
| Both densely filled | 2 | 1 | 1 |
| **Total** | **30** | **20** | **10** |

Vary the part placements between loaded sets and cover both bins. For the normal
fill group, hold out one entire layout and both of its repeat captures. Keep all
other captures of an unchanged placement in the same split too. Label each bin
separately: use `empty` or `loaded` only, and describe thin/few/normal/dense and
placement details in the capture note. When one bin is empty and the other
contains parts, record that distinction.
Only training sets labelled empty for **both** bins fit the bin surfaces. The
planar calibrator requires at least three such independent sets; the height-field
support threshold defaults to five. This plan supplies seven. Loaded data and
the held-out empty sets reveal unwanted bin points and parts removed by the filter. They do not
provide ground-truth part labels automatically.

Keep one CAD model if the two bins are identical, or one model per bin if their
shapes differ. Keep its units and dimensions with the model. A shared CAD shape
still needs a separate placement for each bin in the robot frame. The scripts do
not automatically align CAD to captures; use measured dimensions and cloud
inspection to review the initial bin geometry.

For every set retain:

- Both original depth TIFFs and both original spotlight TIFFs, unchanged.
- Capture timestamp and completion status; bin A/B content labels and split.
- The actual spotlight and structured-light exposure used, in milliseconds.
- The camera calibration and B2 robot calibration used for scene 0, including
  their units and coordinate conventions.

For a single-exposure model, use one fixed setting matching the intended runtime
setting. A profile dataset can contain multiple depth exposures and separate
spotlight exposures. The examples below use **8.0 ms only as an editable
example**. Record each acquisition's value rather than relying on a default or
directory name. A manifest exposure value cannot change images already taken.
The acquisition metadata records the exposure supplied to the camera request;
independent sensor-register readback is not currently available.

## Collect new captures

Use the [Bin Setup UI](STANDALONE_UI.md) on the local PC. It guides scene
preparation and captures each exposure through the camera and Eyes services.
Each acquisition saves both Float32 depth EXRs, both spotlight EXRs, requested
settings, frozen calibration and a completion manifest. The UI waits for an
operator confirmation before each scene and can train and validate the completed
session. The fixed camera calibration is sufficient; no current arm pose or
Brain sequence is required.

## Manual registration of legacy captures

Current complete EXR sessions use `import-session` or `build-session`; no
manual exposure/bin/split entry is needed. The commands below remain available
for older TIFF data and individual imports.

Choose a new dataset directory on the healthy disk. Its parent may exist, but
the dataset directory itself must not exist before `init`:

```bash
python3 bin_filter.py init --dataset /healthy/path/bin_dataset
```

Initialization creates `dataset.yaml` and `bin_config.yaml` and snapshots this
workcell's `calibration/deployment/camera/scene0_denali0_params.yaml` and
`calibration/deployment/robot/B2_params.yaml` for scene 0 from `artifacts_cell_4`. Verify these match the files used
when collecting the images. Keep the snapshots fixed for the dataset; start a
new dataset after changing camera or robot calibration.

`add` registers the source images; **it does not copy them**. It stores relative
paths for files inside the dataset directory and absolute paths for external
files. Keep those files available and unchanged, or copy them to their permanent
healthy-disk location before registering them. SHA-256 hashes recorded at import
are checked when loading, including the calibration snapshots. Identical input
captures cannot be registered again under a different ID or split.

Register each completed timestamp directory with a unique ID, the split, and the
content of each bin:

```bash
python3 bin_filter.py add --dataset /healthy/path/bin_dataset \
  --capture-dir /healthy/path/captures/2026-10-16-10_00_00_000000 \
  --id empty_01 --split train --bin-a empty --bin-b empty --exposure-ms 8.0

python3 bin_filter.py add --dataset /healthy/path/bin_dataset \
  --capture-dir /healthy/path/captures/2026-10-16-10_05_00_000000 \
  --id loaded_01 --split train --bin-a loaded --bin-b loaded --exposure-ms 8.0 \
  --note 'Both normal fill; layout 1, repeat 1'
```

Replace the example timestamp paths with your captures. Use `--split test`
for the three held-out empty sets and seven held-out loaded sets. IDs can encode
the placement category, such as `loaded_wall_01`.

`--exposure-ms` records the structured-light depth exposure. Supply
`--spotlight-exposure-ms` when recording a separate spotlight setting. Use
`--group-id` to identify repeated captures of one layout or acquisition session,
including across exposures, and keep that group in one split. Available saved
acquisition metadata is checked against supplied settings. See the
[profile registration examples](EXPOSURE_PROFILES.md#register-and-fit-a-new-model).

The importer also supports contour records containing both depth and spotlight
pairs. Use `--record /absolute/path/to/robot_pose/contour_<id>.yaml` to select the
intended contour acquisition explicitly when a transfer folder contains multiple
records; a path relative to `--capture-dir` also works. The later inspection
capture and the earlier bin contour capture can
show different scene states; keep each pair associated with its own acquisition.

For existing unfiltered point clouds, an explicit alternative is:

```bash
python3 bin_filter.py add --dataset /healthy/path/bin_dataset \
  --cloud /healthy/path/raw_scene.xyz --cloud-frame vision --unfiltered \
  --id raw_empty_01 --split train --bin-a empty --bin-b empty --exposure-ms 8.0
```

Use `--cloud-frame robot` only for a cloud already expressed in the dataset's
selected reference frame (B2 in the historical example).
Both formats use millimetres. `--unfiltered` confirms that the cloud still
contains the original bin surfaces; it does not undo earlier processing.
Processed `contour_*.ply` files are rejected. Prefer original paired TIFFs for
new collections; a cloud-only entry has no camera images for visual comparison.
XYZ text loading works with the base requirements; PLY and other Open3D formats
require the optional `open3d` package.

## Inspect the reconstructed scene

First reconstruct an empty scene to check the calibration and bin placement:

```bash
python3 bin_filter.py inspect --dataset /healthy/path/bin_dataset \
  --capture empty_01 --output /tmp/bin_preview
```

This exports `raw_robot_mm.ply` and `inspection.json`. Add `--show` to open the
optional Open3D viewer, or `--pick` to select points with Shift+left click and print
their robot XYZ coordinates after closing the window. Picking visible floor
points helps determine the floor seed. Review both camera reconstructions for
alignment. Do not infer correct alignment from a successful command alone.

Every `--output` directory must be new. Use another path, such as `model_v2`, for
subsequent runs; existing results are never overwritten.

## Optional planar-bin workflow

Use this model for bins whose observed floor and walls are adequately planar.
The `init` command creates a planar template for a new dataset. The October 16
dataset instead uses the height-field configuration described above. Both bins
must use the same model type; planar entries may omit `kind` or set `kind: planar`.

Review `bin_config.yaml` in the dataset before fitting. Its `bins.bin_a` and
`bins.bin_b` entries need the actual inner geometry and an approximate placement
in the robot frame:

- `robot_from_bin_seed` places the local bin axes in robot coordinates. The
  initial X/Y positions, `(523, -419)` and `(523, -587)` mm, come from the existing
  bin crop centers. Identity rotation is only an initial assumption; check the
  real bin orientation and floor height in the reconstructed cloud. Its Z
  translation starts as `null`: **replace it with the approximate observed floor
  height before fitting**. Floor height is not automatically initialized.
- `interior_polygon_mm` is the measured inner footprint, with vertices ordered
  counterclockwise in the bin's local XY plane. Replace the initial `null` value
  from real measurements or CAD. The sequence's **190 × 190 mm crop is not the
  physical inner bin size** and must not be substituted for it.
- `floor_search_band_mm` and `wall_search_band_mm` bound surface fitting around
  the initial geometry. Their 8 mm defaults are starting values to review.
  The floor seed must put the real floor within the configured search band.
- `wall_height_range_mm: [8, 60]` selects the height band above the floor used to
  fit walls. `max_height_mm: 180` limits the retained interior height. These are
  illustrative values to adapt to the real bins and visibility.

Each empty training capture must contain enough observed depth to support the
floor and every configured wall. A hidden wall cannot be recovered reliably from
these observations: fitting fails if a required surface has insufficient support.
Review the geometry and image coverage, or recollect the scene, before trying
again. There is no automatic CAD fit for an unobserved wall.

For example, an inner rectangle 100 mm wide and 80 mm long, centered at the
local origin, would use this polygon **only if those are its measured dimensions**:

```yaml
interior_polygon_mm:
  - [-50, -40]
  - [50, -40]
  - [50, 40]
  - [-50, 40]
```

The example is counterclockwise when viewed from above the local +Z axis.
`robot_from_bin_seed` maps these local coordinates to the robot frame. For the
identity rotation in the initial file, the last column contains the bin's robot
X, Y and approximate floor Z; fit errors tell you when the measured surfaces
are outside the search bands. Use `inspect --pick` to print selected robot XYZ
coordinates (Shift+left click, then Q), or measure them in your own PLY viewer.

The wall fitting height range is measured from the **seed frame's local Z=0**.
The fitted model's `max_height_mm` is measured along the fitted floor normal.
Planar walls extend to that maximum height for bin membership; the model does
not represent a rim, overhang, rib, or a change in wall slope.

Optional per-bin `fit` settings include `ransac_distance_mm` (0.5),
`residual_quantile` (0.99), `alignment_margin_mm` (0.2),
`min_tolerance_mm` (0.2), and `max_tolerance_mm` (2.0). These are starting
defaults, not measurements of your bin. For each surface, the fitter takes the
largest per-capture absolute-residual quantile plus the alignment margin, with
the minimum tolerance as a floor. It rejects a result above the maximum tolerance
rather than clipping it. Surface support and spatial coverage must also pass.
Detailed fitted values and per-capture residuals are saved in `model.yaml`.

For an optional CAD overlay, export a triangle mesh in STL, OBJ, or PLY and record
its coordinate units; those are supported by
[Open3D mesh file I/O](https://www.open3d.org/docs/release/tutorial/geometry/file_io.html#mesh).
This tool has no STEP reader. Supply both the mesh and a YAML transform:

```bash
python3 bin_filter.py inspect --dataset /healthy/path/bin_dataset \
  --capture empty_01 --output /tmp/bin_preview_cad \
  --cad /healthy/path/bin.stl --cad-scale-to-mm 1 \
  --cad-transform /healthy/path/bin_a_cad_pose.yaml --show
```

The transform YAML contains `robot_from_cad`, a 4×4 rigid matrix mapping scaled
CAD coordinates into the robot frame. For example, the following is an identity
template; replace its rotation and translation with the actual bin placement:

```yaml
robot_from_cad:
  - [1, 0, 0, 0]
  - [0, 1, 0, 0]
  - [0, 0, 1, 0]
  - [0, 0, 0, 1]
```

Use scale `1` for CAD coordinates already in millimetres or `1000` for metres;
the transform translation is always in millimetres. Use a separate transform for
each bin even if they share a mesh. The overlay helps review dimensions and
manual placement. Calibration fits observed planes rather than aligning CAD.

Fit a versioned model, then assess the held-out sets. `calibrate` reads the
dataset's `bin_config.yaml` by default; `--config /path/to/settings.yaml` selects
another reviewed configuration:

```bash
python3 bin_filter.py calibrate --dataset /healthy/path/bin_dataset \
  --output /healthy/path/bin_dataset/model_v1

python3 bin_filter.py validate --dataset /healthy/path/bin_dataset \
  --model /healthy/path/bin_dataset/model_v1/model.json \
  --output /healthy/path/bin_dataset/validation_v1 --split test
```

## Outputs and review for either model

The model directory contains `model.json`, numerical `arrays/`, `settings.yaml`,
and calibration snapshots. `--output-format yaml` explicitly exports legacy YAML. `validate` writes `validation.json` plus a folder for each capture and
bin. Each folder contains:

| File | Use |
| --- | --- |
| `foreground_robot_mm.ply` | All retained points for this bin, in robot millimetres. |
| `classified_preview_robot_mm.ply` | Sampled visual review: green retained, blue floor/surface, orange planar walls, purple height-field unknown, grey outside. |
| `report.json` | Point counts, unknown count, empty-bin leakage fraction, and review context. |

Review the validation report and point-cloud outputs. Empty scenes should leave
few interior points. In loaded scenes, inspect parts near the floor, walls, and
corners for unwanted removal as well as remaining bin points. Compare the same
held-out sets between model versions and retain the reports. The empty leakage
fraction uses only classified surface/interior points as its denominator;
unknown height-field regions are excluded. Review their coverage separately.

These reports contain geometry diagnostics, not automatic grasp-success or
part-retention accuracy. Compare loaded results with the raw cloud and spotlight
images. A single-exposure model rejects captures at other settings. A profile
bundle selects an exact anchor or an interpolated reference between anchors;
exposures outside its range are rejected. Reports identify that selection and
provide per-exposure and per-camera diagnostics for evaluating the new range.
Read [the profile acceptance guidance](EXPOSURE_PROFILES.md#review-acceptance-by-exposure-and-by-camera)
before treating interpolation as physically validated.

Calibration and model loading also reject overlapping usable bin volumes.
Height-field models conservatively require disjoint XY region interiors.
Validation reports `kept_by_multiple_bins` for each capture as an additional check.
These checks concern model membership, not robot collision clearance.

To export the filtered result for one bin and capture:

```bash
python3 bin_filter.py filter --dataset /healthy/path/bin_dataset \
  --model /healthy/path/bin_dataset/model_v1/model.json \
  --capture loaded_01 --bin bin_a \
  --output /healthy/path/bin_dataset/filtered_v1
```

Choose `bin_b` for the other bin. These commands operate on saved files and do
not install a model into the online C++ contour pipeline. Runtime loading and
the sequence's `bin_id` selection are described in [RUNTIME.md](RUNTIME.md).

`model.json` and its arrays are model data, **not** ROS parameter files. Both model
types retain their legacy provenance fields and use the selected robot frame
in millimetres (B2 in the historical example). Planar
models store `robot_from_bin`, inward floor/wall normals, plane offsets, and
tolerances. Height-field models store the robot XY grid, nullable reference
heights and tolerances, repeated-capture support, and fit diagnostics. Exposure
bundles store these fields separately for each anchor and specify interpolation
between them. The C++ integration applies the selected reference before summit
detection. No summit detector, tool-clearance planner, or live ROS client runs in
these scripts. Spotlight images are retained as review evidence; both filters
use depth geometry.

## Coordinate and depth conventions

- Raw depth is **camera optical Z in millimetres**, not range from the optical
  center and not robot Z. Preserve float32 TIFF values; do not rescale depth into
  an 8-bit display image. Invalid depth pixels must not become valid 3D points.
- Each camera's `cam_extrinsic` transforms camera coordinates into the shared
  calibrated vision frame. The selected scene's hand-eye transform maps vision
  to its chosen robot reference frame. The model and reconstructed point clouds
  use that frame in millimetres.
- In the hand-eye calibration's six-value vector, the first three values are
  translation in millimetres. The last three are an **axis-angle rotation vector
  in degrees**, as used by `PoseVecToMat` in `math_helper.cuh`. They are not Euler
  roll/pitch/yaw, despite some historical `xyzrpy` names in the codebase.
- A contour record's selected point and processed PLY are in the robot frame;
  `returned_pose` is in the vision frame. The processed contour cloud has already
  been filtered. Reconstruct the original depth maps for this offline workflow.
