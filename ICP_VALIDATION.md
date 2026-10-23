# Per-capture bin registration validation

This optional feature aligns **each selected bin independently** against its own
selected reliable surface. A legacy model uses its frozen single-exposure
reference; an exposure-profile bundle selects or interpolates the reference
before registration. A new camera-aware model uses its measured top rim and
boundary. Registration estimates a local rigid bin displacement in the model's
configured robot frame, in millimetres. It does not change camera or robot
calibration.

Current collection and model validation support every named bin in the selected
scene. Use the [standalone UI](STANDALONE_UI.md) for new sessions and production
replay. The commands and measured results below describe the historical Cell 4
scene-0/B2 two-bin setup; they do not restrict new scenes or bin layouts.

## Historical Cell 4 startup

1. Rebuild Perception with the changes; no common-interface, Brain, Eyes, or XML
   update is needed for this addition.
2. Check `deep_pose_config/ros_config/pth_contour_pose_config.yaml` and restart
   the contour node with that YAML. Its checked-in `bin_icp.enabled` is currently
   `true`; a repository setting alone does not establish the deployed value.
3. Existing `bin_id="bin_a"` and `bin_id="bin_b"` requests select independent
   references. Inspect the `Bin ICP` log and saved `processing.bin_registration`
   metadata. `applied: false` means the nominal pose passed the quality checks
   within the measurement-noise tolerance.

The historical development build was isolated under
`/tmp/bin_filter_perception_build`; it did not replace the live install.
See [RUNTIME.md](RUNTIME.md#optional-per-capture-bin-icp) for parameters, transform
direction, rejection behavior, and deployment details.

For exposure profiles, ICP and classification must use the same selected
reference at the reported acquisition exposure. Inspect selection metadata as
well as registration diagnostics. ICP does not establish exposure invariance
or correct missing depth. Follow [EXPOSURE_PROFILES.md](EXPOSURE_PROFILES.md) to
collect and validate a new range; the configured October 16 bundle and the
results below remain **8 ms only**.

## Repeatable offline checks

Build these targets in a configured Perception build directory:

```bash
cmake --build /tmp/bin_filter_perception_build --target \
  pth_contour_pose_node replay_bin_height_field_service \
  replay_bin_registration test_bin_surface_registration -j2
ctest --test-dir /tmp/bin_filter_perception_build --output-on-failure \
  -R '^test_(bin_surface_registration|bin_height_field_filter|bin_filter_calibration|contour_peak_detector|contour_capture_writer)$'
```

For the existing direct-service replay command, add `--icp`. It reconstructs the
saved depths and exercises the production service, including bin selection,
classification, radius filtering, and summit detection. It never acquires new
camera data or requests robot motion. Use `--expect-empty` for an empty selected
bin; a correctly rejected empty scene is a passing test.

For controlled pose tests, first export reconstructed **unfiltered robot-frame XYZ**
using the existing offline dataset loader. This example uses the historical B2
dataset. From `bin_filter_offline/`:

```bash
python3 - <<'PY'
from pathlib import Path
import numpy as np
import dataset

root = Path("data/dataset_2026-10-16")
manifest, calibration = dataset.load_dataset(root)
out = Path("outputs/2026-10-16/icp_validation")
out.mkdir(parents=True, exist_ok=True)
for name in ("empty_08", "filled_20"):
    capture = dataset.get_capture(manifest, name)
    points = dataset.load_capture_points(root, capture, calibration)
    np.asarray(points, dtype=np.float32).tofile(out / f"{name}_robot_xyz.bin")
PY
```

Inside the Perception container, after sourcing ROS and the workspace:

```bash
model=/root/artifacts/deep_pose_config/bin_filter_models/2026-10-16/model_v1/model.yaml
captures=/root/artifacts/bin_filter_offline/outputs/2026-10-16/icp_validation
OMP_NUM_THREADS=4 /tmp/bin_filter_perception_build/replay_bin_registration \
  "$model" "$captures/empty_08_robot_xyz.bin"
OMP_NUM_THREADS=4 /tmp/bin_filter_perception_build/replay_bin_registration \
  "$model" "$captures/filled_20_robot_xyz.bin" --loaded
```

The packed input format is little-endian float32 XYZ, with no header, in the
model's robot frame and millimetres (B2 for this historical dataset). This test
input differs from the saved text point-cloud format.
The historical runner tests its two named bins with identity, translations `(2,-2,2)` and
`(-2,2,-2)` mm, and translation `(1,-1,1)` mm plus 0.02 rad rotation about axis
`(1,2,3)`, centered on each reference's centroid. The entire recorded cloud is
transformed for each case; the synthetic unit test separately exercises one bin
moving independently with the other bin included as unrelated clutter.

Empty tests require accepted registration, less than 0.75 mm maximum transform
error over reference vertices, and zero points after 0.5 mm voxelization and the
45-neighbor / 4 mm radius filter. Loaded tests use the same transform-error limit
and require nonempty retained data and at least 98% precision and recall against
the unchanged-scene classifier's retained point indices. These latter metrics
measure regression agreement, **not ground-truth segmentation accuracy**.

## Results on 2026-10-16

Model SHA-256:
`6a92cd64de306e4151a52996ecec57cd2f3b3ded77d69c2c3bbeb43ded3e0e7a`.
The model and calibration bundle were not refitted for ICP.
These historical results precede exposure-profile interpolation and do not
validate a multi-exposure bundle or the currently deployed process.

| Check | Result |
| --- | --- |
| Five targeted CTests | Passed |
| Synthetic independent-bin transforms, part occlusion, and rejection checks | Passed |
| Eight direct-service cases: empty_08, filled_03, filled_06, filled_20, each selecting A and B | Passed; original raw classification counts preserved |
| Eight digitally shifted empty-cloud cases, including identity | Passed; maximum reference-vertex error 0.225 mm; zero points after radius filtering |
| Eight digitally shifted loaded-cloud cases, including identity | Passed; maximum reference-vertex error 0.189 mm |
| Loaded retained-point agreement with unchanged-scene baseline | Precision at least 99.71%; recall at least 99.97% |
| Perception pre-commit checks | Passed |

The selected unchanged-scene replays all validated the nominal pose without
applying a correction. Four loaded selections returned candidates and four
empty selections returned no pose. Measured nominal plane residuals ranged from
0.101 to 0.292 mm. Full service replay time was approximately 0.35–0.50 s,
excluding acquisition and startup; this is not an ICP-only latency measurement.

Detailed local outputs are in `outputs/2026-10-16/icp_validation/`: `summary.json`,
the eight per-capture JSON/log pairs, `shifted_empty.log`, and
`shifted_loaded.log`. That outputs directory remains ignored by Git. Earlier
experimental outputs are archived in subdirectories and are not final results.
The small CPU shift runner and production GPU service use different voxel/radius
implementations, so their post-radius counts need not match exactly.

## Limitations

Digitally transformed recorded data preserves the original sensor noise and
visibility. Physical bin motion changes shadows, occlusions, and reconstruction
noise; these tests do not substitute for shifted empty and loaded captures from
the real cameras. No robot pick-success percentage is established.

Repeat empty and loaded physical-bin checks at each new exposure anchor and at
held-out intermediate settings. Exposure-dependent noise or dropout can affect
registration support and observability as well as foreground classification.
Passing a digitally shifted 8 ms replay does not establish those properties at
4–20 ms. The current artifact's ICP limits are documented in
[RUNTIME.md](RUNTIME.md#optional-per-capture-bin-icp); historical test results
should not be interpreted as validation of every later configuration change.

A broad CTest attempt also selected two unrelated checks: `test_ros2_libs` was
not built in the isolated build, and `test_pth_seg_pose_node` exceeded its 30 s
startup timeout while creating a TensorRT engine from the installed segmentation
model. Those checks are not counted as passing contour/ICP validation.
