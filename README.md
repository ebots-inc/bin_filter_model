# Bin Setup and offline bin filtering

**Bin Setup** collects bin images, fits the measured background model, validates
it on held-out scenes, and deploys a passing model. It runs as a local browser
application. Collection connects to camera and Eyes services and never moves
the robot; no collection sequence or HMI modification is required.

**Start with the [illustrated operator guide](STANDALONE_UI.md).** It explains
each click, what to put in every bin, when to move parts, CAD upload, validation,
and using the model from the real HMI workflow.

## Start

Complete the one-time [station installation](STATION_SETUP.md#station-installation),
then run from the workspace root:

```bash
bin_filter_offline/.venv/bin/python bin_filter_offline/bin_setup.py
```

Open **http://localhost:8765** and leave the terminal running. The defaults use
`ebots_system_artifacts/artifacts_cell_4` and `data/bin_model` in this workspace.
Use `--artifacts-root` and `--data-root` for another station. Container names are
unrestricted: automatic discovery uses the artifacts mount and installed tools.
Use `--camera-container` or `--replay-container` when several containers match.

The host server needs no ROS environment. Collection needs the camera and Eyes
services, without a running Perception node. Build/validation uses the standalone
Summit replay executable in its build environment. Activation additionally needs
an updated, idle Perception service with the matching workflow loaded.

## Workflow

1. Select the commissioned scene, calibration and frame. Choose a saved layout,
   or click **Create bin layout**: capture empty bins, draw a box around each,
   review the detected bowl and rim, and save. No coordinate measurements are
   required. Select the bins and click **New setup**, or **Open session** to
   continue a saved collection.
2. Check the camera connection and preview the bins.
3. Follow the scene instructions. **Scene ready — capture exposures** captures
   the prepared arrangement at every exposure, then waits before the next stage.
4. For loaded stages, reposition or rotate parts **between arrangements**, keeping
   them still **within each exposure sweep**.
5. Optionally upload an STL, review its placement against empty training data,
   and confirm a CAD reference for the build.
6. Click **Build and validate**, review the result, then **Use this model** after
   physical validation passes and Perception is ready.

![The app pauses between arrangements, keeping parts still during each exposure sweep.](docs/capture-cycle.svg)

The default collection uses **4, 8, 12, 16 and 20 ms** for both spotlight and
structured light on both cameras. All selected bins are captured together.
The current guide totals **170 captures / 26 Ready confirmations**: empty
training, fresh empty validation, one resistor in six broad areas, normal fill
and full fill. Every loaded condition has three different arrangements.
Normal/full amounts follow the station's usual production fill, without an
exact part count. Older sessions retain their original instructions.

Any configured scene and one, two or more bins are supported. Eyes reads
`artifacts/workflow_configs/bin_calibration.yaml`, referencing existing camera
calibration in `artifacts/calibration/deployment/camera/`. The reference frame
comes from existing robot calibration. The UI derives bin coordinates and rim
height from a calibrated depth preview after the operator marks each bin. Saved
layouts stay under `data/bin_model/layouts/`. The fixed camera transform is needed; the
current robot arm pose is not.

## Model and files

The model contains measured bin-local interior geometry, the flat top rim and
its observed boundary, bin placement, and per-camera/per-exposure correction and
uncertainty. Rim registration checks visible rim and boundary coverage; hidden
side walls are not the denominator of that coverage check.

Optional CAD guides association with the real measured surface. Printing or
machining deviations stay in the measured model. CAD does not fill unseen areas,
provide calibration, or increase detection tolerances. The runtime uses the
fitted grids and measured rim points; it does not align an STL per pose request.

Large numerical arrays are stored as typed binary files beside a small
`model.json`. The deployed folder contains only `model.json` and its numerical
`arrays/`:

```text
ebots_system_artifacts/artifacts_cell_4/calibration/deployment/bin/<model-name>/
```

Production reads the shared camera and robot calibration under
`artifacts/calibration/deployment/camera/` and `robot/`. Small expected numeric
values in the model's `calibration_reference` metadata let the loader detect a
calibration mismatch. Frozen calibration files remain with the offline captures
and candidates for replay.

Captures, CAD uploads, settings, candidate models and validation reports stay
under `data/bin_model/<session-name>/`. Deployment state and rollback history
stay under `data/bin_model/deployment/`. Building a candidate does not activate it.

Passing validation qualifies the tested conditions. It does not establish
arbitrary lighting, intermediate untested exposures, exact resistor localization,
grasp success or a production latency target. See the operator guide for how
results are checked and the station guide for saved-data timing measurements.

## Documentation

| Need | Read |
| --- | --- |
| Collect data and use the UI | [Operator guide](STANDALONE_UI.md) |
| Install camera/replay helpers, configure scenes, troubleshoot, benchmark | [Station setup](STATION_SETUP.md) |
| Input/output layout and saved evidence | [Model storage](MODEL_STORAGE.md) |
| Measured model and fitting rules | [Model details](MODERN_MODEL.md) |
| Perception integration | [Runtime integration](RUNTIME.md) |
| Older October 16 TIFF, planar and height-field commands | [Historical workflows](LEGACY_WORKFLOWS.md) |

## Development checks

From this directory, with the dependencies installed:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python bin_filter.py demo --output /tmp/bin_filter_demo
```

The demo uses generated data and requires no cameras or robot connection. Tests
exercise input handling, geometry, capture workflow and deployment behavior;
synthetic tests do not establish real-bin accuracy.
