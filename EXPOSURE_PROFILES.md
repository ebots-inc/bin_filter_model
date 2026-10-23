# Exposure support

New factory sessions use the camera-aware model in [MODERN_MODEL.md](MODERN_MODEL.md).
Collect its measured exposure anchors using the [standalone UI](STANDALONE_UI.md).
The new runtime corrects each camera before merging and rejects unmeasured
settings unless an interval has explicit held-out validation. The default
4/8/12/16/20 ms recipe therefore establishes five settings, not all possible
exposures or lighting. Spotlight exposure is separate from depth exposure.
The guide applies to one or more configured bins in any selected scene. It
uses that scene's existing deployment camera calibration and the selected
robot frame. Two image pairs refer to the Denali sensors, not two physical bins.

## Archived pooled and experimental height-field workflows

The sections below describe the older pooled and experimental model types.
They do not describe the new `calibrated_bin_background` model or its runtime.

The older height-field runtime uses one shared height field. Fit that field
with `--pool-exposures` to include empty-bin captures from multiple exposure
settings. The same surface and learned residual tolerances apply to every
request, without exposure metadata, profile selection, or interpolation at
runtime. The separate `--exposure-profiles` mode remains an offline experiment
and produces a different format that the current runtime does not load.

The active artifact configuration still points to the frozen October 16 model
at **8.0 ms**. That bundle and its calibration snapshots are unchanged. Its
recorded validation covers 8 ms only. The existing dataset has no measured
4–20 ms sweep, so a new pooled model needs new captures and held-out evaluation
before replacing the configured bundle. Repository settings do not establish
which libraries or parameters a running process has loaded.

## Fit one shared model for production

Register independent training captures with every configured bin empty at each
intended setting, and hold out separate empty and loaded recordings at those settings. Keep the
true exposure in the offline dataset so reports can reveal settings that fail;
this does not add metadata to ROS requests. Use `kind: height_field` for every
configured bin in the dataset's `bin_config.yaml`.

```bash
python3 bin_filter.py calibrate --dataset /healthy/path/bin_exposure_dataset \
  --pool-exposures --stride 1 --output /healthy/path/bin_pooled_v1

python3 bin_filter.py validate --dataset /healthy/path/bin_exposure_dataset \
  --model /healthy/path/bin_pooled_v1/model.json \
  --split test --stride 1 --output /healthy/path/bin_pooled_validation_v1
```

The fitter pools all training captures with every configured bin empty into one
model per bin. It keeps the existing per-capture surface estimates, support checks, and residual
tolerance limits. It requires at least two exposure settings and at least three
independent captures in total; the configured per-cell support requirement may
require more. Collect repeated captures at every setting and check held-out
results separately for each setting and camera.

The export retains
`model_type: offline_height_field_bins`, with direct `bins` and sorted
`exposure_settings_ms` provenance. It has no scalar `exposure_ms` requirement or
profile interpolation policy. `--pool-exposures`, `--exposure-ms`, and
`--exposure-profiles` are mutually exclusive.

Offline filtering and validation use this same grid for any recorded exposure;
the provenance list is not an enforced range or evidence that untested settings
work. Review the actual residuals, unknown coverage, and retention of the
smallest parts across the production settings. Pooling can widen tolerances or
make inconsistent cells unknown. Existing models trained only at 8 ms remain
unchanged and do not become multi-exposure models by removing runtime checks.

## What changes with exposure

The filter uses depth geometry, not spotlight brightness. Changing acquisition
exposure can change depth bias, noise, valid-pixel coverage, and the relative
contribution of the two cameras. Brightness normalization of the spotlight
images cannot repair those depth measurements. Missing depths cannot be
recovered by widening a foreground threshold.

The older pooled model merges both camera clouds before fitting. Per-camera
diagnostics therefore matter: different errors or dropout in the two views can
alter the combined surface. Inspect each camera's signed surface residual and
coverage as well as the merged result. Camera-dependent error may eventually
justify separate camera reference or bias models; exposure profiles alone do
not correct every such error.

## Collect anchors and independent intermediate checks

For a proposed **4–20 ms** operating range, start with anchors at **4, 8, 12, 16,
and 20 ms**. These are collection targets, not validated operating settings.
At every anchor collect at least seven independent training captures and three
independent test captures with every configured bin empty. The default fitter requires
support from at least five training captures at each reference cell and
interpolation square; seven acquisitions do not guarantee sufficient local
coverage.

Keep cameras, bins, calibration, reconstruction settings, and relevant lighting
conditions fixed. Capture independent sessions or repeats, rather than copying
one image or replaying it under different exposure labels. Preserve actual
acquisition metadata and both original depth/spotlight pairs. Record gain,
projector, or ambient-light changes in the notes if they occur; exposure alone
does not describe the entire acquisition condition.

Reserve **6, 10, 14, and 18 ms**, plus at least one non-round setting such as
**9.3 ms**, for independent tests. Test empty scenes and loaded scenes at the
anchors and intermediate settings. Include the smallest required parts at the
floor, walls, corners, and rim, plus sparse and dense loading. Inspect each bin
separately. Captures with one loaded bin and all others empty are optional extra
mixed-occupancy checks; the default guide collects the selected bins together.

Keep an unchanged loaded layout and its repeated captures across exposures in
one split. Use `--group-id` for each shared layout/acquisition group; registration
and test validation use it to detect train/test leakage. Use different groups
for independent empty acquisition sessions or repeats. Group names do not make
repeated observations independent.

Use the [Bin Setup UI](STANDALONE_UI.md) on the local PC. It performs fresh
acquisitions at each planned exposure after the operator confirms the scene.
New guided sessions use three loaded arrangements at the center, beside each of
the four walls and at the indicated bottom-left corner, placing exactly one
resistor in every selected bin at the same named location simultaneously.
The UI also collects three arrangements per fill condition. Each arrangement
stays unchanged through its exposure sweep; the UI
then waits for the operator to rearrange the parts and click Ready again.
Reposition every bin's resistor between resistor arrangements, keeping one per
bin. Validation evaluates each bin separately within the shared capture.
Empty stages retain their automatic repetitions.
Each selected exposure is applied to both structured light and
spotlight on both cameras. It saves both original Float32 EXR pairs,
completion manifests, requested settings, and per-capture calibration evidence.
Session settings reference existing deployment calibration files; no new
calibration configuration is created. No TCP or detected-part pose is needed. Use **Build and validate** after collection, or
`import-session` before the manual fitting commands below. Exposure values are
requested settings, not independent sensor-register readback.

## Optional offline profile experiment

Run from `bin_filter_offline/`. Create a new dataset using the calibration that
was active during collection, then review its `bin_config.yaml` for the actual
height-field geometry. Every configured bin must use `kind: height_field` for
exposure profiles. The default `init` template is planar; do not fit that template as a
curved-bowl profile configuration. The command examples below use the historical
`bin_a`/`bin_b` layout. For another layout, use `init --bins <names...>` and
repeat `add --bin-label NAME=empty` or `NAME=loaded` for every configured bin.

```bash
python3 bin_filter.py init --dataset /healthy/path/bin_exposure_dataset

python3 bin_filter.py add --dataset /healthy/path/bin_exposure_dataset \
  --capture-dir /healthy/path/captures/empty_s01_e4 \
  --id empty_s01_e4 --split train --bin-a empty --bin-b empty \
  --exposure-ms 4.0 --spotlight-exposure-ms 4.0 --group-id empty_session_01

python3 bin_filter.py add --dataset /healthy/path/bin_exposure_dataset \
  --capture-dir /healthy/path/captures/thin_wall_test_e10 \
  --id thin_wall_test_e10 --split test --bin-a loaded --bin-b empty \
  --exposure-ms 10.0 --spotlight-exposure-ms 10.0 --group-id thin_wall_test_layout \
  --note 'Held-out layout; thinnest required part against A wall'
```

Replace the paths and settings with the captured values. Here `--exposure-ms`
means the **structured-light depth exposure**. `--spotlight-exposure-ms` records
the separate review-image setting when supplied. Otherwise its own saved
metadata is used, or the spotlight exposure remains unknown (`null`); it is
never assumed equal to depth exposure. The importer checks available acquisition
metadata; do not relabel a capture to bypass those checks.

These manual import examples use matching depth/spotlight values only as an
example. Use the values actually recorded in each acquisition. New EXR session
imports read the independently configured values from the collection plan and
requested-settings files automatically.

After registering all planned captures and reviewing the height-field settings:

```bash
python3 bin_filter.py calibrate --dataset /healthy/path/bin_exposure_dataset \
  --exposure-profiles --interpolation-margin-mm 0.5 --stride 1 \
  --output /healthy/path/bin_exposure_profiles_v1

python3 bin_filter.py validate --dataset /healthy/path/bin_exposure_dataset \
  --model /healthy/path/bin_exposure_profiles_v1/model.json \
  --split test --stride 1 --output /healthy/path/bin_exposure_validation_v1

python3 bin_filter.py filter --dataset /healthy/path/bin_exposure_dataset \
  --model /healthy/path/bin_exposure_profiles_v1/model.json \
  --capture thin_wall_test_e10 --bin bin_a --stride 1 \
  --output /healthy/path/thin_wall_test_e10_filtered_v1
```

`--exposure-profiles` groups **training** captures with every configured bin empty by their
recorded depth exposure. It requires at least two anchor settings, fits each
anchor independently, and applies the existing per-bin support and tolerance
checks to every fit. Loaded captures and test captures never fit a profile.
Use a new output directory for each run. The `0.5` mm interpolation margin is a
software default to evaluate, not a measured error bound for the proposed range.

To fit a legacy single-exposure model from a mixed dataset instead:

```bash
python3 bin_filter.py calibrate --dataset /healthy/path/bin_exposure_dataset \
  --exposure-ms 8.0 --stride 1 --output /healthy/path/bin_8ms_v2
```

This selects the 8 ms training group. It does not overwrite or widen the frozen
8 ms model. Without pooling, profile mode, or an explicit single-exposure selection,
mixed training exposures are rejected.

## Model format and selection

Offline single-exposure replay retains its exact exposure check. The current
Perception runtime ignores exposure provenance on shared models. An offline profile
bundle uses
`model_type: offline_height_field_exposure_profiles`, with a shared configured robot frame,
millimetre units, scene, calibration, and settings. Its `profiles`
list is sorted by `exposure_ms`; every entry contains its `training_captures`
and independent height fields under `bins.<configured-bin-name>`. The interpolation
policy is stored as:

```yaml
interpolation:
  method: linear_height_max_tolerance
  margin_mm: 0.5
```

Calibration snapshots remain alongside the bundle.

For a requested exposure `e` strictly between consecutive anchors `e0` and
`e1`, the `linear_height_max_tolerance` policy computes:

```text
alpha = (e - e0) / (e1 - e0)
height(e, x, y) = (1 - alpha) * height(e0, x, y) + alpha * height(e1, x, y)
tolerance(e, square) = max(tolerance(e0, square), tolerance(e1, square))
                       + interpolation_margin_mm
```

Both endpoints must support the reference vertices and interpolation square.
Missing support at either endpoint remains unknown. A derived tolerance above
`max_tolerance_mm` makes the square unknown; it is never clipped to the cap.
Only the reliable component connected to the configured bin seed remains
usable. If the seed cannot support the selection, selection fails instead of
manufacturing a reference. Interpolation may therefore have less usable area
than either anchor. This check occurs when an intermediate exposure is selected;
failure of one interval's common reliable seed does not invalidate the exact
endpoint profiles.

Exact anchors use their original heights and tolerances without adding the
interpolation margin. A small `1e-5` ms match tolerance handles floating-point
representation at anchors. Other values outside the anchor range are rejected;
there is no nearest-profile fallback or extrapolation. Exposure must be finite
and positive. Subject to the support checks, any finite positive interior value
can be selected; values are not rounded to an integer or to a predefined list.
Offline selection does not change the saved model.

## Review acceptance by exposure and by camera

Each per-bin `report.json` contains `structured_light_exposure_ms`,
`spotlight_exposure_ms`, `group_id`, and an `exposure_selection` object with
`requested_exposure_ms`, `lower_exposure_ms`, `upper_exposure_ms`, `weight`, and
`mode` (`single`, `exact`, or `interpolated`) for profile experiments. Use the
bracket to distinguish direct anchor evidence from an interpolated result.
Pooled models instead report `mode: pooled` and `exposure_settings_ms`, with no
requested-exposure bracket or interpolation weight.

For height fields, `camera_diagnostics` describes each camera's input:

| Field | Interpretation |
| --- | --- |
| `camera_index` | Source camera; unavailable for a merged cloud-only import. |
| `sampled_depth_pixels`, `valid_depth_pixels`, `valid_depth_fraction` | Positive finite depth coverage over the sampled whole image, not only the bin ROI. |
| `roi_point_count`, `known_reference_point_count` | Reconstructed points within the bin ROI and points with a supported selected reference. |
| `unknown_point_count`, `unknown_fraction_of_roi` | ROI points excluded because their selected reference is unknown. |
| `supported_squares_total`, `supported_squares_observed`, `observed_support_fraction` | Coverage of supported reference squares by the current camera's observations. |
| `signed_residual_quantiles_mm` | `p01`, `p50`, and `p99` of robot Z minus selected surface height at supported points. |
| `residual_interpretation` | `empty_background` for an empty-labelled bin, or `includes_foreground` for loaded data. |

Loaded-scene residuals include parts and cannot be interpreted directly as
background sensor noise. Cloud-only imports have no original per-camera pixel
validity information; the camera identity and depth-validity fields are `null`.

`validation.json` adds `exposure_summary`, grouped by depth exposure, bin, and
empty/loaded label. Each group includes `capture_count`, classification `counts`,
`classified_bin_point_count`, `selection_modes`, and `empty_leakage_fraction`.
The leakage denominator excludes unknown/outside points, so inspect the coverage
and counts alongside it. The model and reports retain manual-review status;
successful selection or a report being written is not an automatic deployment
approval or proof of physical interpolation accuracy.

Evaluate these together for every held-out exposure:

- Empty-bin retained points and downstream false summit candidates after the
  production voxel/radius filters.
- Valid depth coverage, reference coverage, unknown and outside counts, and
  signed residual distributions for each camera. A low leakage fraction can
  result from missing or excluded points rather than a useful reference.
- Retention of the smallest required parts and features, including at boundaries
  and sparsely observed locations. Compare with the original images and clouds;
  point counts alone are not ground-truth segmentation accuracy.
- With ICP enabled, registration acceptance, residuals, observability, and
  displacement at each exposure, including deliberately moved physical bins.

Set a minimum required part/feature height and retained area for the task before
accepting a bundle. A point is kept only when its measured height above the
selected surface exceeds the local tolerance. Tolerances near the current
4 mm cap can remove thin features; interpolation adds margin and may remove
more. Raising the cap is not a substitute for demonstrating part retention.
The downstream radius and summit stages impose additional requirements, so a
geometric height threshold alone cannot guarantee a pickable part.

Passing synthetic interpolation tests proves implementation behavior. It does
not prove exposure invariance, reconstruct missing sensor data, or establish
robot pick success. If intermediate tests fail, collect closer anchors, improve
acquisition quality, or restrict the operating settings based on evidence.

## Use a reviewed pooled model in the current runtime

Install the pooled model with its calibration snapshots, then point
`bin_filter_model_path` in the active `contour_summit_pose_params.yaml` to its
versioned `model.yaml`. Rebuild and restart Perception after code changes, or
restart after changing its model path. The current runtime loads the direct
`bins` grid and ignores exposure provenance; profile bundles are not
supported. It does not require new exposure or camera-ID fields from Eyes.

ICP and classification use the same shared reference. ICP estimates physical
bin displacement, not exposure correction. No capture-specific exposure model
is selected. Calibration matching, bin geometry validation, and ICP quality
checks still apply. [RUNTIME.md](RUNTIME.md) preserves the old service's
deployment notes; its old exposure interface and launch commands do not
describe the current `perception_node` port.

The October 16 reports remain historical 8 ms evidence. New 4–20 ms operation
requires its own recorded captures, intermediate tests, and physical validation.

## Software verification on October 21, 2026

- All **71 Python tests passed**, including exposure selection, interpolation,
  malformed inputs, acquisition metadata, group split checks, diagnostics, and
  the CLI workflows.
- C++ host unit checks passed. Python/C++ classifications agreed for the frozen
  model's **74,027 probes per bin** and the synthetic/interpolation parity cases
  across seven tested exposure selections.
- Replaying the existing 8 ms data through the updated offline tools preserved
  raw foreground counts: **28** for `empty_08/bin_a` and **18,506** for
  `filled_03/bin_a`.
- The perception service, node, and direct-service replay tool linked in an
  isolated build, and the height-field CTest passed. With ICP enabled, the
  recorded `empty_08/bin_a` replay correctly returned no pose after radius
  filtering; `filled_09/bin_a` returned a candidate.
- Synthetic profile fixtures exercised the full service with the unchanged
  recorded **8 ms** TIFFs: exact anchor selection preserved the legacy loaded
  result, and selection between artificial 4/20 ms anchors at weight 0.25
  passed classification, ICP, and visualization checks. The interpolated
  visualization contained **26,474 reference vertices**, matching the Python
  selected surface. These fixtures test software integration; they are not
  captures or evidence of sensor performance at 4 or 20 ms.
- The frozen model SHA-256 remains
  `6a92cd64de306e4151a52996ecec57cd2f3b3ded77d69c2c3bbeb43ded3e0e7a`.

Installed runtime libraries and running services were not changed by these
checks. These are software and recorded-data checks. The new physical exposure sweep
and its held-out acceptance measurements remain to be collected.
