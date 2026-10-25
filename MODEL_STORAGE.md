# Collection inputs and model outputs

The [standalone Bin Setup UI](STANDALONE_UI.md) prepares readable sessions and
captures directly through the native camera client. Each exposure acquires every
selected bin in one scene through both stereo cameras. No collection sequence
or Brain action is required.

```text
bin-setup-2026-10-23-03/
  session.json                    # Scene, bin IDs, stages, deployment calibration references
  recipe.yaml                     # Frozen acquisition settings
  bin_config.yaml                 # Frozen geometry/uncertainty fitting settings
  configs/depth_*ms.yaml           # Unique exposure-specific camera recipes
  stage_confirmations/*.json      # Explicit operator Ready confirmations
  completed_stages/*.json          # Verified stage checkpoints
  captures/<capture_id>/
    depth/Frame{0,1}.exr           # Original Float32 optical-Z depth, mm
    spotlight/Frame{0,1}.exr
    requested_settings.yaml
    calibration/{camera,robot}.yaml
    capture_manifest.yaml          # Written last, only on successful completion
  previews/                       # Fresh previews; excluded from the dataset
  previous_attempts/              # Preserved interrupted stages
  logs/                           # Native capture diagnostics
  selected_candidate.json          # Latest completed model build for UI review
  candidate/                      # Later builds use candidate-02, candidate-03, ...
    build.json
    dataset/                      # Imported labels and references to original EXRs
    model/
      model.json
      arrays/bins/<bin>/...        # Typed numerical payloads
      calibration/{camera,robot}.yaml
      settings.yaml
      cad_reference.yaml          # Optional commissioned plan snapshot
      cad/<bin-id>/<mesh>.stl      # Optional packaged CAD source
    validation/                   # Python geometry diagnostics and PLY previews
    production_validation/
      acceptance.json             # Held-out production detector results
      pipeline_config.yaml        # Exact replay configuration
      <capture>/<bin>/result.json
      <capture>/<bin>/replay.log
```

Keep the session with its candidate: the dataset references the original EXRs.
New sessions reference existing camera files under
`artifacts/calibration/deployment/camera/` and reference-frame files under
`artifacts/calibration/deployment/robot/`. Eyes uses the shared
`artifacts/workflow_configs/bin_calibration.yaml`. No new calibration setup files
or per-session Eyes workflows are generated. Capture and model calibration
snapshots retain the actual inputs as evidence. Older sessions with a top-level
`calibration/` directory or `eyes_capture.yaml` remain readable.
The candidate model directory is self-contained when copied with its arrays and
calibration. For a CAD-guided candidate, keep its packaged reference meshes as well.
Deployment exports a smaller runtime bundle, described below. A CAD plan
is a build input; choosing it does not edit the session's frozen recipe, bin
configuration, labels or captures. Later builds preserve earlier candidates.
The UI waits before each scene, records confirmation, and sweeps exposures without
changing the arrangement. Interrupted stages are archived and retaken, never
combined with a newly arranged partial sweep.
New guided sessions make each loaded arrangement a separate stage and capture
group. The default `guided.loaded_arrangements: 3` collects three arrangements
per resistor location and fill condition. With
`guided.resistor_capture_mode: simultaneous`, each resistor capture records
exactly one resistor in every selected bin at the same named location.
`guided.wall_sides` selects
`[left, right, bottom, top]`, each with its own directional stages;
center and the bottom-left diagonal are separate locations. Historical diagonal
stages keep their `corner` IDs. Saved `target_region_fraction` values preserve
the original schematic guidance; they are not measured resistor annotations.
Production acceptance reports `operator_confirmed_directional` evidence and
checks broad directions inside the reliable measured bowl footprint.
Each arrangement has one exposure sweep and one Ready confirmation.
Empty training and validation repetitions are
unchanged. Existing saved recipes and plans retain their recorded grouping;
older repeated captures do not become independent arrangements by relabeling.
Saved recipes without `guided.resistor_capture_mode` retain the sequential plan
with one loaded bin and the others empty. Existing labels and captures are not
converted into evidence of simultaneous resistor placement.
The original left-wall labels may omit `wall_side`; that omission means left
for display. Added right, bottom and top wall labels record their `wall_side`.

New sessions use each sweep value for both spotlight and structured light on
both cameras. The capture plan records both exposure fields explicitly. Older
saved sessions retain their original independent settings when verified or imported.

## Runtime deployment

**Use model** installs only the files needed to load and run an accepted model:

```text
artifacts/calibration/deployment/bin/<model-name>/
  model.json
  arrays/bins/<bin-id>/...
```

The exporter keeps measured geometry, registration and camera-profile arrays,
and the metadata needed to check them. The small `calibration_reference` field
records the expected reconstruction values and selected hand-eye transform;
the runtime directory contains no copied calibration files. It also omits CAD
meshes and CAD diagnostic arrays, fitting settings, plots, acceptance reports
and configuration backups.
Full evidence remains in the local session and its candidate directories.
Deployment state, rollback history, locks and failed-install diagnostics are
stored separately in `data/bin_model/deployment/`. Keep that directory if you
need the UI's rollback action. Existing rollback models remain runtime bundles.

The pipeline's `bin_filter_model_path` points to the runtime `model.json`.
Camera and robot configuration continue to come from the station's existing
`calibration/deployment/camera/` and `calibration/deployment/robot/` directories;
Perception compares their reconstruction values with `calibration_reference`
before using the model. Offline capture/candidate calibration snapshots remain
with the local dataset for reproducibility.

## Command-line build

The UI runs these operations. They also work independently from this directory:

```bash
python3 bin_filter.py build-session \
  --session /path/to/completed/session \
  --output /path/to/new/candidate \
  --pipeline-config /path/to/artifacts/config/pose_estimation/contour_summit_pose_params.yaml \
  --replay-executable /path/to/install/perception_node/lib/perception_node/summit_replay
```

To add commissioned CAD guidance, append this argument to the build command:

```text
--cad-reference /path/to/bin_filter_offline/configs/cad_references/bin-reference.yaml
```

Omit this option for the unchanged measured-only build. The plan must match the
session's scene/reference frame and include every selected bin. Meshes are STL
files with explicit scale and per-bin placement. See the
[commissioning example](STATION_SETUP.md#optional-cad-reference). CAD placement
and optional bounded refinement use empty training measurements only; held-out
captures remain evaluation evidence. Selecting CAD does not automatically activate
the resulting candidate or replace a prior candidate's acceptance report.

The modern recipe selects `calibrated_bin_background`. Fitting learns geometry
from empty training captures, plus separate camera corrections at each exposure.
It does not pool camera errors into one shared uncertainty threshold. Python
validation exports diagnostics; native replay evaluates the actual detector,
including registration and production postfilters. Empty scenes must produce no
pick target with successful registration; resistor scenes must produce a
target near the center or in the instructed broad direction, inside each bin's
reliable measured bowl footprint. These checks use operator-confirmed contents;
exact resistor footprints were not independently annotated. All selected bins
are evaluated separately even when they share the same capture. Normal/full scenes must
produce targets. Every prescribed stage must cover every measured exposure.

A successful build is not automatically accepted. Failed production checks are
saved for review, and activation refuses failed, incomplete or synthetic evidence.
The production detector may reject a thin resistor even when the background
filter retains some points; this is reported rather than silently weakening its
settings. Physical capture tests do not establish grasp success.

For separate diagnostics:

```bash
python3 bin_filter.py import-session --session /path/to/session --dataset /path/to/new/dataset
python3 bin_filter.py calibrate --dataset /path/to/dataset --output /path/to/new/model
python3 bin_filter.py validate --dataset /path/to/dataset \
  --model /path/to/model/model.json --output /path/to/new/geometry_review
python3 bin_filter.py validate-production --dataset /path/to/dataset \
  --model /path/to/model/model.json --pipeline-config /path/to/pipeline.yaml \
  --replay-executable /path/to/summit_replay --output /path/to/new/production_review
```

Import checks complete manifests, paired Float32 EXRs, dimensions, file sizes,
recorded settings, stage labels and calibration contents. Exposure metadata is
explicitly `requested_not_sensor_verified`. New capture and model files add no
hash checks or format-version dispatch. Historical TIFF import remains supported.

## Binary arrays

`model.json` holds model identity, local placement, exposure metadata, fitting
diagnostics and array descriptors, for example:

```json
{"file": "arrays/bins/bin_a/geometry/surface_height_mm.bin", "dtype": "float64_le", "shape": [160, 150]}
```

Arrays are row-major little-endian Float64/Int64 payloads. They store geometry
heights/tolerances/support, measured rim/boundary samples, and camera bias,
uncertainty, validity and support maps. Unknown heights/tolerances use NaN in
binary; other values follow the constraints of the corresponding model field.
Loaders check relative paths, data types, dimensions and exact byte lengths.
There are no checksums in the array descriptors.

CAD-guided bundles additionally retain each bin's `cad_reference` metadata and
binary `nominal_height_mm`, `apparent_deviation_mm` and `association_mask` arrays.
Metadata records `seed_bin_from_cad`, the final `bin_from_cad`, setup parameters
and diagnostics. The source STL is packaged at `cad/<bin-id>/<mesh>.stl`; the
model root also contains a `cad_reference.yaml` plan snapshot. These arrays use
the same typed binary representation. Actual background heights, rim/boundary
samples and camera profiles remain measured; unsupported
measurements are not replaced by CAD. Apparent deviation is measured minus
nominal and can include unresolved sensor error, so it is not a manufacturing
tolerance measurement. Runtime filtering uses the measured arrays without mesh
processing. Keep the complete candidate model directory to preserve its reference
evidence; the deployed runtime export intentionally omits CAD source and diagnostics.

Read [MODERN_MODEL.md](MODERN_MODEL.md) for the measured model contract. Small
calibration and tuning YAML snapshots remain human-readable; the large grids
are binary. Legacy YAML models can still be read or explicitly exported:

```bash
python3 bin_filter.py convert-model --model /path/to/old/model.yaml --output /path/to/new/model
python3 bin_filter.py convert-model --model /path/to/model/model.json \
  --output /path/to/new/export --output-format yaml
```

Conversion changes storage, not measured geometry or validation evidence.
