# Bin model collection recipe

The standalone [Bin Setup application](../README.md)
([step-by-step operator guide](../STANDALONE_UI.md)) guides the operator through acquisition, model building and validation. It connects to the camera nodes directly through `bin_capture`; it needs no Brain sequence or HMI changes. This directory supplies its collection recipe and fitting layout; [prepare_session.py](../prepare_session.py) prepares and verifies sessions. Preparation creates files only and never starts an acquisition.

The default [recipe](recipe.yaml) uses **4, 8, 12, 16 and 20 ms**. Each capture applies the same exposure to **spotlight and structured light on both sensors**: the 4 ms step requests `[4, 4]` for both lighting modes, then the 8 ms step requests `[8, 8]`, and so on. Gain and black level stay fixed. Each acquisition covers every selected physical bin in one scene and saves the two depth and two spotlight images. The two Denali sensors do not limit how many bins can be collected. The capture client currently supports Denali 0 within the selected scene.

Select the scene, existing camera calibration, robot frame/calibration, and matching measured bin layout in the app. Scenes 0–4 and any other configured scene ID from 0–255 are supported. The supplied [bin_config.yaml](bin_config.yaml) is the existing scene 0 / B2 example; its coordinates must not be reused for a different scene or frame. Each layout declares `scene_id`, `frame_id`, and one or more named entries under `bins`. Bin names may contain letters, digits, `.`, `_`, or `-`, starting with a letter or digit. The recipe may select a subset with `bin_ids`; omitting it selects every bin in the layout.

Camera calibration is read directly from **`artifacts/calibration/deployment/camera/`**. Robot calibration comes from `artifacts/calibration/deployment/robot/` and must include the selected scene's hand-eye transform. Preparation never generates or duplicates deployment calibration files.

## Guided setup

Prepare or open a session in the standalone application. Arrange the bins as shown and press **Scene ready** before each stage. An acknowledgement applies only to that stage. The exposure sweep runs without another interruption while the arrangement stays unchanged. Each loaded condition has three separate arrangement stages: rotate or reposition parts before confirming the next arrangement.

Names are readable and unique, such as `bin-setup-2026-10-23-01`. An optional operator name is accepted only if it is unused. With the default five exposures, four wall sides and three loaded arrangements, the guide has **26 stages and 170 captures**, regardless of the number of selected bins. All selected bins are captured together. Every stage needs its own Ready confirmation.

| Stage | Arrangement | Captures with default exposures |
| --- | --- | --- |
| Empty training | Every selected bin empty | 35 |
| Fresh empty validation | Every selected bin empty | 15 |
| Resistor at center, toward each of four sides, then toward the bottom-left diagonal | Exactly one resistor in every selected bin at the same named location. Rotate or reposition each resistor within its indicated direction between arrangements, keeping one per bin. | 3 arrangements × 5 exposures × 6 locations = 90 |
| Normal fill | Representative everyday load in every selected bin. Rearrange between sweeps, keeping roughly the same fill. | 3 arrangements × 5 exposures |
| Full fill | Representative full load in every selected bin. Rearrange between sweeps, keeping roughly the same fill. | 3 arrangements × 5 exposures |

Use the thinnest production resistor. Wall stages cover the broad left (−X), right (+X), bottom (−Y), and top (+Y) directions in bin-local coordinates. The historical corner stage means the bottom-left (−X, −Y) diagonal; it does not require a rectangular corner. The UI shows a curved bowl with the selected direction highlighted. Keep the resistor on the lower bowl, off the rim and steep wall. The resistor icon is an example position, not a live detection; the diagram does not imply the same orientation as the camera image. Validation checks measured bin support and the operator-confirmed direction. It does not independently verify the exact resistor footprint or distance to a wall. Normal/full stages use representative fill instructions without prescribing an invented part count. Resistor thickness and production acceptance limits remain unset until measured or commissioned. Only the initial empty stage supplies training data; loaded scenes are validation only.

The recipe sets `guided.loaded_arrangements: 3`, `guided.wall_sides: [left, right, bottom, top]`, and `guided.resistor_capture_mode: simultaneous`. Each loaded arrangement appears separately in the UI as 0/5 captures, with its own capture group and confirmation. Do not move parts between exposures in one sweep: the fixed arrangement allows comparison across exposures. Empty training and fresh empty validation retain their automatic repetitions. Fresh empty validation checks every selected bin while all are empty; resistor validation evaluates each bin's retention and expected target region separately within the shared captures.

Existing saved sessions keep their recorded stage grouping. Recipes without `guided.resistor_capture_mode` keep the sequential instructions with one loaded bin and the others empty. An old loaded stage with 15 captures still runs three sweeps after one confirmation. Existing images are not relabeled as simultaneous captures or different arrangements. Older wall labels without `wall_side` mean left (−X); newly added wall sides record that field explicitly.

The application records `stage_confirmations/<stage>.json` before that stage's first capture. A browser refresh can reopen the session. Stopping lets the current capture finish and prevents another exposure from starting. Retrying an interrupted stage preserves its old files under `previous_attempts/` and collects the entire unchanged arrangement again after a new confirmation.

**Quick Bin Check** uses the installed model without training or replacing it. Empty every selected bin and confirm Scene ready; it collects **15 fresh test captures**, three rounds per exposure. Default names are `bin-check-YYYY-MM-DD-NN`. Empty checks cover leakage and registration; resistor retention needs the guided loaded stages.

## Session preparation API

The standalone application calls:

```python
prepare_session(artifacts, output_root, runtime_root, recipe,
                mode="guided", name=None,
                runtime_artifacts_root="/root/artifacts")
```

It snapshots the recipe and selected fitting settings, creates the labeled capture plan, and installs unique per-exposure part configurations. The session keeps an identical copy of each configuration in `configs/`. Eyes resolves depth settings by the unique part name in the shared artifact tree, so **the app and Eyes must use the same artifacts directory**. Calibration paths in those configurations refer directly to `runtime_artifacts_root/calibration/deployment/...`. The host data directory and Eyes' data mount may differ.

`session.json` records the host `artifacts_root`, selected `bin_ids`, and calibration entries such as `{"artifact_path": "calibration/deployment/camera/scene0_denali0_params.yaml"}`. `resolve_calibration(folder, session, name, artifacts=None)` resolves these references, with an optional artifact root override for moved datasets. Old sessions with `{"path": "calibration/camera.yaml"}` snapshots remain readable.

Preparation validates the existing shared Eyes configuration at **`artifacts/workflow_configs/bin_calibration.yaml`**. That file must contain the selected scene, logical Denali 0 camera, and the same selected deployment camera file. The session records its path without copying or rewriting it. No per-session `eyes_capture.yaml` is generated, and no Brain sequence is modified. Guided setup is the default; `mode="check"` makes a quick-check plan. Existing empty-only and loaded-arrangement sessions remain readable and verifiable.

For developer preparation without opening the app:

```bash
python3 bin_filter_offline/prepare_session.py prepare \
  --mode guided \
  --output-root "$PWD/data/bin_model" \
  --runtime-root /tmp/bin-model-capture
```

Here `--output-root` stores the host session, while `--runtime-root` records the native session root. The standalone UI uses temporary camera-container storage for acquisition and copies completed images directly into the local session. Artifacts retain only the exposure part configurations that Eyes needs to serve capture requests. `--runtime-artifacts-root` defaults to `/root/artifacts`; set it when the camera environment mounts artifacts elsewhere. Eyes resolves its `artifacts/...` camera path beneath `EBOTS_ROOT`, so the runtime artifact tree must be `EBOTS_ROOT/artifacts`. For a native capture process running directly on the host, use the host session root for both session roots and pass the host artifact root as `--runtime-artifacts-root`. Add `--name resistor-baseline` for a custom name. Use `--exposures 4 6 8 10 12 14 16 18 20` for every 2 ms; any sorted, unique exposure list within 4–20 ms is accepted and applies to both lighting modes. This CLI prepares files only; the standalone app handles the ready confirmations and acquisition.

For station commissioning, if Eyes is not already serving the matching camera calibration, start the single Eyes process in its usual configured environment with the shared configuration:

```bash
ros2 run ebots_eyes ebots_eyes \
  -cfg /root/artifacts/workflow_configs/bin_calibration.yaml \
  --ros-args -p mode:=hardware
```

The executable is `ebots_eyes` (the node is named `eyes_node`). This command is an explicit camera-service startup step, not something the preparation helper runs. The camera node must already be available; use one Eyes service instance for the scene. The shared Eyes workflow must use `artifacts/calibration/deployment/camera/...`, which Eyes resolves beneath `EBOTS_ROOT`. Do not use an absolute `/root/artifacts/...` camera parameter: the current Eyes parser prefixes the root itself. The native capture configuration uses the absolute runtime artifact path. Both point to the same existing deployment camera file. Use the same deployment calibration throughout a collection; after changing calibration, start a new session and restart Eyes with that calibration. The camera and native capture process may use different data mounts, but their shared deployment files must agree.

The shared workflow can list several configured scenes; each scene has its own
existing deployment camera file. For example, a commissioned scene 1 entry is:

```yaml
scene_id: 1
name: bin_model_capture
ros_action: pose
camera:
  - cam_id: 0
    parameters:
      - artifacts/calibration/deployment/camera/scene1_denali0_params.yaml
```

Place entries under the workflow's `scenes` list. Add only scenes with an existing
matching calibration and measured bin layout; preparation reports a missing or
mismatched entry before creating session files. `cam_id: 0` is the logical stereo
pair within that scene, independent of the hardware PCIe card number. Existing
saved sessions keep their old metadata and remain readable. A previously saved
session with a fixed spotlight exposure retains that recorded setting during
verification; its captured data is never relabeled. Every new preparation uses
matched exposures, including when the input recipe still contains an old fixed
`spotlight_exposure_ms` field. That field is omitted from the new frozen recipe.

## Session contents and verification

```text
<session>/
  session.json                 # Plan, stage instructions, labels, exposure/group IDs
  recipe.yaml                  # Frozen acquisition recipe from bin_filter_offline/configs
  bin_config.yaml              # Frozen selected-bin fitting settings
  configs/depth_*ms.yaml       # Exact per-exposure part configurations
  stage_confirmations/*.json   # Standalone UI Ready acknowledgements
  captures/<capture_id>/
    depth/Frame{0,1}.exr
    spotlight/Frame{0,1}.exr
    requested_settings.yaml
    calibration/{camera,robot}.yaml  # Acquisition evidence, not service configuration
    capture_manifest.yaml      # Published only when acquisition finishes
```

Verify a completed session with:

```bash
python3 bin_filter_offline/prepare_session.py verify /absolute/path/to/session
```

`verify_capture(folder, session, capture)` performs the same per-capture checks during collection without requiring future stages. Whole-session verification also checks the complete plan and stage metadata against the frozen recipe. Historical display wording is preserved and may differ from current guidance; capture settings, labels, placements, arrangement groups and confirmations must still match. Checks cover completion, file sizes, EXR dimensions/encoding, camera/robot calibration contents, prescribed settings and training/validation labels. Guided/check captures require a matching operator confirmation. Capture calibration evidence is compared with the referenced deployment files; those per-capture files preserve the calibration used during acquisition and are never new deployment configuration. There are no hash checks. Exposure metadata records requested settings, not sensor readback; the fitter also compares image dimensions with capture calibration evidence.

The standalone app builds a candidate and reports validation results before **Use model** can install it. Capture completion itself does not establish model acceptance. Raw EXRs remain available; model arrays are saved as binary payloads with a small manifest. Accepted runtime models are exported to **`artifacts/calibration/deployment/bin/<model-name>/`**, containing only `model.json` and the required binary arrays. The model metadata records expected calibration values; Perception reads the existing shared deployment camera and robot files. No calibration files are copied into the bin-model directory. CAD sources, fitting settings, previews and validation reports stay with the local session under `data/bin_model/`; deployment state and rollback history stay in `data/bin_model/deployment/`. See the [offline tool documentation](../README.md) for build, review, activation and rollback details. Collecting a range of exposures does not establish reliability at untested settings.

## Offline development checks

Install dependencies and validate the recipe without writing:

```bash
python3 -m pip install -r bin_filter_offline/requirements.txt
python3 -B bin_filter_offline/prepare_session.py check
PYTHONPATH=bin_filter_offline python3 -B -m unittest discover -s bin_filter_offline/tests -p 'test_prepare_session.py' -v
```

Tests use temporary artifact trees and synthetic EXRs, with no hardware. For a manual dry preparation, copy the source calibration/settings and `workflow_configs/bin_calibration.yaml` into a temporary artifact tree and pass `--artifacts-root /tmp/copied-artifacts`; generated part settings then stay in that tree.
