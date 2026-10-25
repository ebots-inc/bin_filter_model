# Bin Setup station installation and troubleshooting

This page is for station integrators. Operators should start with the
[illustrated step-by-step guide](STANDALONE_UI.md).

## Station installation

From the repository root:

```bash
python3 -m venv bin_filter_offline/.venv
bin_filter_offline/.venv/bin/python -m pip install -r bin_filter_offline/requirements.txt
docker ps --format '{{.Names}}'
read -r -p 'Camera/Eyes container name: ' camera_container
bash bin_filter_offline/native/build.sh "$camera_container"
```

At the prompt, enter the actual running camera/Eyes container name from
`docker ps`. The name can differ from station to station.
The build installs the native helper at
`/root/bin_filter_offline/native/bin_capture` inside that container. It builds
the camera transport and Float32 EXR writer without building or starting
Perception. Source code, collection recipes and fitting configuration live in
`bin_filter_offline/`; no Python tools are installed into artifacts.

Camera buffers use CUDA shared memory, but the capture helper copies those
buffers directly to CPU memory for EXR saving. The Eyes container's Open3D library
can be CPU-only; its CUDA module is not required for collection. The camera/Eyes
environment still needs its normal GPU access.

Start the server as your normal user:

```bash
bin_filter_offline/.venv/bin/python bin_filter_offline/bin_setup.py
```

The default `--camera-container auto` discovers a running container by its
artifacts bind mount and installed Eyes tools. **Container names are unrestricted.**
If multiple containers match, use `--camera-container YOUR_CAMERA_CONTAINER`.
`--capture-executable` overrides the helper's absolute path inside that container.

The server binds to loopback and defaults to this checkout's
`ebots_system_artifacts/artifacts_cell_4` and `data/bin_model`. Override these with
`--artifacts-root` and `--data-root`. `--recipe /path/to/recipe.yaml` selects other
acquisition defaults; the normal recipe is `bin_filter_offline/configs/recipe.yaml`. Host Python needs neither ROS nor CUDA. It invokes the
native client through Docker and maps paths using the actual bind mounts.
Capture output is transferred to the host session with the host user's ownership;
Eyes' `/root/data` mount can differ from the host data directory.


## Scene, bins and existing calibration

Select **scene 0, 1, 2, 3 or 4** in the UI; additional configured scene IDs also
appear. A session collects one scene and any nonempty selection of its bins.
The two Denali sensors form a stereo pair; they do not limit the physical bin
count. One, two, three or more bins are supported.

The shared Eyes workflow is:

```text
artifacts/workflow_configs/bin_calibration.yaml
```

Its scene entries reference the existing camera calibration files under:

```text
artifacts/calibration/deployment/camera/
```

For example, this is a scene-0 entry in the workflow:

```yaml
ros_vis: false
scenes:
  - scene_id: 0
    name: bin_model_capture
    ros_action: pose
    camera:
      - cam_id: 0
        parameters:
          - artifacts/calibration/deployment/camera/scene0_denali0_params.yaml
```

Add the station's actual scene entries and their existing calibration paths to
this shared workflow when commissioning another scene. Eyes expects these paths
relative to `EBOTS_ROOT`; do not prepend `/root/` to its `parameters` entries.
Creating a session reads this workflow and checks the selected scene/calibration.
It does not generate another Eyes workflow or calibration setup file. Older
sessions and their recorded files remain readable.

The reference-frame selector reads existing files from
`artifacts/calibration/deployment/robot/` with a hand-eye entry for the selected
scene. The fixed camera-to-frame transform is needed; the current arm pose is not.

The normal layout workflow is **Create bin layout** in the local UI. After
selecting the commissioned scene, camera calibration and reference frame, the
operator empties the bins, captures a top-down depth view, draws a box around
each complete bin and reviews the detected bowl and flat rim. No manual
`xy_bounds_mm`, seed coordinates, Z range or rim-height input is needed.
The app derives these values from calibrated depth and stores the confirmed
layout at `data/bin_model/layouts/<readable-name>.yaml`. The preview and review
evidence stay with local data, outside artifacts. See the
[operator walkthrough](STANDALONE_UI.md#first-layout-or-bins-moved-mark-them-on-a-picture).

For an expert-maintained preset, `bin_filter_offline/configs/layouts/` remains
supported. A layout contains `scene_id`, `frame_id`, an optional readable `name`
and a `bins` mapping. The default fitting settings are shown in
`bin_filter_offline/configs/bin_config.yaml`; do not copy its measured coordinates
to another station. Names such as `resistors`, `left` and `right` are supported.
The UI lists matching presets and locally saved layouts, and lets the operator
select all or a subset of their bins.

The checked-in default is the measured scene-0/B2 layout. Other scenes need their
commissioned camera calibration, hand-eye transform and Eyes workflow entry;
the layout can then be created in the UI. Changing the scene selector does not
calibrate the station. The sidebar selections apply to the next new setup;
an opened session retains its recorded scene, bins and capture plan.

Collection preparation code is `bin_filter_offline/prepare_session.py`. The
recipe and default fitting settings are in `bin_filter_offline/configs/`.
Artifacts keep the station workflow, existing calibration and runtime models.
Eyes still reads generated per-exposure part configurations from its artifacts
part-config lookup directory, but temporary camera images are transferred from
the container to the local session rather than stored under artifacts.


## Camera connection

Keep production stopped while collecting so another workflow does not change
settings or compete for captures. Park the arm clear of the camera view.

If needed, start the existing camera node in the Eyes environment using the
station's workcell manifest:

```bash
ros2 run ebots_cameras ebots_cameras --ros-args -p mode:=hardware
```

Start Eyes with the shared workflow, in another terminal in that environment:

```bash
ros2 run ebots_eyes ebots_eyes \
  -cfg /root/artifacts/workflow_configs/bin_calibration.yaml \
  --ros-args -p mode:=hardware
```

An existing Eyes process configured for the same scenes and calibration can be
used. Do not start a second Eyes server for the same scene. If calibration or
workflow entries change, reload Eyes with the matching shared workflow before
collection. The app does not start or restart station services.

The UI checks readiness at startup and after opening/preparing a session.
**Check connection** refreshes the result. It checks the session's scene, or the
selected scene when no session is open. Missing camera status, stereo sensors,
acquisition service and `/eyes_depthN` for scene N are listed separately.
Preview and Scene ready stay disabled until readiness passes, and check again
before acquisition. A failed check preserves captures and scene confirmations.
Service-name discovery alone is not sufficient: a server must be available.

**Preview cameras** captures a fresh pair outside the training/validation plan.
Raw camera images alone are insufficient; Eyes supplies reconstructed depth.
Logs are saved with failed requests. Requested settings and calibration copies
are evidence, not independent sensor-register or live Eyes calibration readback.
Keep deployment calibration unchanged throughout a session. Capture evidence is
checked against it during verification. Saved captures and full offline
candidates retain frozen calibration files for reproducible replay. Deployed
bin packages contain only model metadata and arrays. Live operation reads the
existing files under `artifacts/calibration/deployment/camera/` and `robot/`;
expected numeric values in `model.json`'s `calibration_reference` metadata
detect mismatches between the fitted model and the loaded calibration.


## Optional CAD reference

Use CAD alongside empty training captures when the nominal bowl outline or flat
rim helps select the observed bin surfaces. Measured heights and deviations stay
in the model, including differences caused by 3D printing or machining. There
is no manufacturing-process preset that widens the foreground threshold. The
recorded measured-minus-CAD deviation also contains unresolved camera error;
it is not a dimensional inspection of the manufactured bin.

The normal operator path is [upload and preview CAD in the UI](STANDALONE_UI.md#7-add-cad-if-you-have-it):
choose an STL, confirm its units and orientation for each bin, preview its
translation placement against empty training data, and confirm the reference.
Uploads and previews remain in the session, outside artifacts. Uploaded STL
files are limited to 16 MiB and 100,000 triangles. This path needs no Open3D
Python package.

For an existing commissioned reference, the UI also accepts a matching named
CAD plan under `bin_filter_offline/configs/cad_references/`. Keep its STL files
relative to that plan, inside the local configuration tree. The session's scene, reference frame and every selected bin must
match. A mesh can be shared by multiple physical bins with separate transforms.
Existing artifact-based CAD plans remain readable for compatibility; keep new
uploaded references with the session.

Before qualifying either path:

1. Verify the CAD's units, upward direction and orientation. `scale_to_mm` is
   explicit because STL does not record units. `bin_from_cad` maps scaled CAD
   coordinates into that bin's local model frame; `robot_from_bin` separately
   places the bin in the calibrated reference frame.
2. Check placement against **empty training captures only**. Keep validation
   captures held out. The upload preview keeps scale and chosen orientation
   fixed while fitting translation. It does not infer CAD units or a symmetric
   bin's intended orientation.
3. Review the top view, cross-sections, retained geometry/rim fractions and
   held-out native replay. Compare with a measured-only candidate on empty
   leakage, resistor retention, registration and runtime. The association limits
   permit some exclusion; they do not guarantee that every measured region
   survives.

This example illustrates the plan structure for **one bin**. Its identity
transform is valid only if the STL is already expressed in that bin's local
frame; it is **not** the transform for the supplied wok-style CAD. Commission
the actual values and add entries for all bins to be built:

```yaml
name: Commissioned curved-bin reference
scene_id: 0
frame_id: B2
bins:
  bin_a:
    mesh_path: curved-bin.stl
    scale_to_mm: 1.0
    bin_from_cad:
      - [1, 0, 0, 0]
      - [0, 1, 0, 0]
      - [0, 0, 1, 0]
      - [0, 0, 0, 1]
    association_margin_mm: 3.0
    min_measured_retained_fraction: 0.9
    max_reference_residual_mm: 10.0
    min_overlap_fraction: 0.5
    alignment:
      enabled: false
```

These are setup compatibility and association settings, separate from runtime
part-detection tolerance. `max_reference_residual_mm` limits the 95th-percentile
absolute measured-minus-nominal residual. A rejected reference needs review of units, placement,
shape and measured coverage. Do not increase detection tolerance to hide a CAD
mismatch. CAD selects from supported measurements; it never fills an unseen
region, supplies independent capture support, or substitutes synthetic rim points.
Runtime uses the resulting measured grids and rim points; it does not load or
align a mesh for each pose request.

STL reading and a commissioned transform require no Open3D Python dependency.
For optional bounded refinement, install Open3D in the host fitting environment:

```bash
bin_filter_offline/.venv/bin/python -m pip install open3d
```

Then replace `alignment` with explicit bounds appropriate to the setup, for
example:

```yaml
alignment:
  enabled: true
  max_translation_mm: 5.0
  max_rotation_deg: 5.0
  max_correspondence_distance_mm: 5.0
  max_iterations: 50
  min_overlap_fraction: 0.5
```

Alignment keeps scale fixed, uses only empty training measurements, and rejects
corrections outside the configured bounds. These example limits require station
review. CAD selection alone does not establish better accuracy or lower latency.
The same physical validation gate applies before **Use this model** is enabled.


## Production replay installation

Collection needs no Perception container. **Build and validate** additionally
needs the matching standalone `summit_replay` executable. Build it in the normal
Perception build environment:

```bash
cd /root/perception/workspace
source /opt/ros/humble/setup.bash
colcon build --packages-up-to perception_node
source install/setup.bash
```

The host discovers the matching container when replay is requested. Override with
`--replay-container YOUR_REPLAY_CONTAINER` or `--replay-executable` if needed.
The default executable is
`/root/perception/workspace/install/perception_node/lib/perception_node/summit_replay`.
It processes saved files without starting a Perception node. Model, capture,
calibration and output paths must be visible through that container's bind
mounts. Keep the executable and model-loading libraries from the same build.


## GPU processing and replay timing

After rebuilding Perception, the calibrated bin pipeline defaults to GPU camera
correction/background classification and a reduced input for measured-rim
registration. These optional settings belong at the top level of
`config/pose_estimation/contour_summit_pose_params.yaml`:

```yaml
use_gpu_bin_filter: true
use_reduced_registration_input: true
```

Camera profile arrays remain on the GPU between requests. Correction and
classification preserve the CPU model's exposure, gain, black-level, uncertainty,
and unknown-cell checks. These bin-filter steps avoid downloading the full
cloud; registration transfers a selected subset to the CPU. Its selection retains
a conservative neighborhood of the measured lip and boundary, including the
alignment search margin. Full-resolution foreground classification
still uses the corrected scene cloud. Registration changes model lookup
coordinates, not the returned scene points. CPU input and legacy models retain
their existing CPU filtering path. Setting either option to `false` provides
the previous path for comparison. Reload the workflow or restart Perception after
changing pipeline settings.

Use the native `summit_replay` executable **inside the Perception container** to
measure saved-depth processing. No running camera, Eyes, or Perception node is
needed. First set `model_bundle` and `capture_dir` to the actual completed model
and saved capture directories visible in that container. The following example
uses a scene-0 `bin_a` capture at 8 ms; adjust the scene, bin, exposure, gain, and
black level to the capture's metadata:

```bash
source /opt/ros/humble/setup.bash
source /root/perception/workspace/install/setup.bash

model_bundle=/absolute/path/to/candidate/model
capture_dir=/absolute/path/to/completed/capture
replay=/root/perception/workspace/install/perception_node/lib/perception_node/summit_replay

replay_inputs=(
  --model "$model_bundle"
  --camera-calibration "$capture_dir/calibration/camera.yaml"
  --robot-calibration "$capture_dir/calibration/robot.yaml"
  --config /root/artifacts/config/pose_estimation/contour_summit_pose_params.yaml
  --depth0 "$capture_dir/depth/Frame0.exr"
  --depth1 "$capture_dir/depth/Frame1.exr"
  --scene-id 0 --bin bin_a
  --exposure-ms 8,8 --gain 0,0 --black-level 240,240
  --device CUDA:0 --warmup 2 --repeat 20
)

"$replay" "${replay_inputs[@]}" --gpu-filter on --registration-input reduced \
  --output /tmp/bin-replay-optimized.json
"$replay" "${replay_inputs[@]}" --gpu-filter off --registration-input full \
  --output /tmp/bin-replay-baseline.json
```

Both runs use the same saved capture and selected GPU. The baseline keeps the
existing CPU camera correction/classification and full registration input while
the rest of the pipeline runs on the same device. A pure CPU run uses
`--device CPU:0`. The host `host_replay.py` validation adapter does not currently
forward these benchmark options; run the native executable directly as above.
Capture calibration snapshots reproduce the saved measurement; live operation
continues to use the deployment calibration under the artifacts folder.

Each benchmark records one cold call separately, performs the requested extra
warmups, and reports median, p95, and minimum latency across the measured repeats.
The output JSON includes `benchmark.cold_ms`, `benchmark.warm_ms`,
`benchmark.stages_ms`, and `benchmark.registration_input_points`.
`depth_to_cloud` includes camera back-projection, transforms, correction, and
fusion. The other stages cover registration, background classification, and
foreground processing/summit extraction. Inspect the result's success/error and
registration acceptance together with timings; a fast rejected request is not a
successful pose result. Repeat comparisons on resistor, normal-fill, and full-fill
captures from every selected bin.

**These measurements start from depth images already loaded on the selected
device.** They exclude hardware capture/exposure, structured-light depth
reconstruction, model loading, file I/O, initial input upload, and ROS transport.
They also exclude the live request's final cache cleanup. RViz is disabled during
replay. Per-stage GPU synchronization makes stage timings
comparable but adds measurement overhead. These timings are not the full HMI
`GetPose` latency and do not demonstrate a 100 ms end-to-end processing budget;
structured-light reconstruction and service overhead require separate live timing.


## Activate and run the HMI contour test

For the cell-4 scene-0/B2 layout with `bin_a` and `bin_b`, use the existing
**Celestica Contour Map Test** in the HMI's test workflows. Its files are:

| Purpose | Path under the artifacts folder |
| --- | --- |
| HMI sequence | `sequence/celestica/test/contour_map_test.xml` |
| Workflow and calibration selection | `workflow_configs/contour_map_test.yaml` |
| Requested camera settings and pose task | `config/parts/bin_picking/highest_part.yaml` |
| Active bin model and filtering settings | `config/pose_estimation/contour_summit_pose_params.yaml` |

`highest_part.yaml` uses the collected 8 ms setting explicitly for both sensors:

```yaml
exposure_time_spot_light: [8.0, 8.0]
exposure_time_structure_light: [8.0, 8.0]
gain: [0.0, 0.0]
black_level: [240, 240]
```

The calibrated model requires both sensors' structured-light exposures, gains
and black levels; a singleton exposure or omitted gain/black level is rejected.
To use another exposure qualified by the candidate's validation report, change
all four spotlight/structured-light values together. The default collection
tests 4, 8, 12, 16 and 20 ms. Keep gain and black level equal to the collected
values. Reload cached part settings before running after an edit.
The model reload service does not reload part settings; restart
Brain/Eyes/Perception to clear those caches.

1. Wait for **Build and validate** to finish and review a passing physical
   production-validation result. Building alone does not change the active
   model. Finish collection before switching the station to its HMI workflow.
2. Run the station's camera, Eyes, updated Perception, Brain and HMI services in
   hardware mode. If Brain/Eyes/Perception were already running with older
   binaries or cached part settings, restart those services through the normal
   station procedure after validation completes.
3. In HMI, select **Celestica Contour Map Test** and click **Launch** once.
   Wait for the nodes to be ready and leave the button at **Run**. Launch loads
   the scene-0 workflow; Run executes its sequence. Perception needs the workflow
   loaded and no active pose requests for model activation.
4. In Bin Setup, click **Use this model** and wait for successful activation.
   It installs `model.json` and its arrays under
   `calibration/deployment/bin/<model-name>/`, updates
   `bin_filter_model_path` in `contour_summit_pose_params.yaml`, sets
   `bin_icp.reference_mode: measured_rim`, and verifies the model loaded by
   Perception. Activation restores the previous configuration if loading fails.
   The old model path remains selected until this step succeeds; do not point
   the pipeline at a build's temporary directory.
5. Place parts in both bins and click **Run** in HMI. The sequence requests a
   pose from `bin_a`, then `bin_b`, and stops after one pass. Both must return
   poses for overall success. An empty bin returns no pose and makes the test
   fail; inspect Perception logs to distinguish an empty result from a capture
   or registration error. The workflow enables Summit visualization; use the
   `perception_config/ros_config/rviz/contour_summit.rviz` display configuration.

This sequence acquires camera data and returns poses without moving the robot.
The Celestica `sequence/celestica/bin_picking/bin_a.xml` and `bin_b.xml` files are
placeholders; a robot picking cycle still needs its grasp, motion and placement
sequence. The other HMI entry, **Pose Estimation Test (Perception Contour
Summit)**, is a separate multi-scene test and is not configured for this bin
model. For another scene or bin layout, provide a matching HMI sequence,
workflow, part settings and deployment target before using its model.


Manual `bin_filter.py activate` and `rollback` commands require both
`--models-root ebots_system_artifacts/artifacts_cell_4/calibration/deployment/bin`
and `--deployment-root data/bin_model/deployment`, together with the matching
pipeline configuration. Runtime packages contain `model.json` and `arrays/`
under the model root. Camera and robot calibration stays in its shared deployment
directories; state, locks and history live outside artifacts under the deployment root. The UI
chooses these default paths automatically.

## Stopping, recovery and daily checks

**Stop after current capture** finishes the in-flight request and starts no new
capture. **Open session** resumes at the first unfinished stage. After an
interruption, Ready preserves that stage's previous attempt and retakes the
whole stage. Earlier completed stages remain intact.

A depth-request timeout does not mean Eyes has finished using the capture
buffers. The updated native client revokes a request that is still queued;
if Eyes is already using its buffers, cleanup waits until Eyes releases them
or its process stops. The UI stays busy during cleanup and starts no further
capture. After three minutes it reports that it is waiting for camera/Eyes to
release buffers; this is a notification, not a forced process termination.
Leave the server running until cleanup finishes. The failed attempt is retained,
and Ready then retakes the unfinished arrangement with a new attempt directory.
If Eyes cannot finish, stop that Eyes process before retrying.

This timeout protection needs both the updated native helper (rebuilt with
`bin_filter_offline/native/build.sh YOUR_CAMERA_CONTAINER`) and the corrected
Eyes build. The Eyes import-close fix remains necessary; caller-side waiting
does not replace it. Read-only connection checks still have a bounded timeout.

If a capture reports `cudaErrorMemoryAllocation` at `cudaMalloc(&addr, shm->size)`,
the camera transport has run out of GPU memory. This is unrelated to whether
the failed exposure is 12 ms or another value. A CPU-only Open3D build still
uses CUDA for camera buffers and shared-memory transfers. Older Eyes depth
requests left an imported CUDA buffer open after each capture, allowing GPU
memory to accumulate. The corrected Eyes code closes that import when each
request finishes, including on errors.

Restart Eyes after deploying the corrected build to release buffers retained
by the old process. After **Check connection** passes, open the same session,
recreate the unfinished arrangement in all selected bins, and click **Scene
ready — capture exposures**. This preserves the partial attempt and retakes
its whole exposure sweep (five captures with the defaults). Earlier completed
stages are preserved; do not reset the dataset.

Closing or refreshing the browser does not stop the server. It finishes the
current stage and waits for Ready. Ctrl+C requests a stop between captures and
waits for the active operation to finish. One server may own a data root at a time.

For sessions from an older root-run server, stop it before opening from the host.
If necessary, repair ownership only for that session and its shared session/part
configuration directories, preserving captures and previous attempts. New host
sessions avoid this ownership mismatch.

**New empty check** takes 15 fresh empty test captures at the default exposures
and checks registration and false foreground against the active model without
refitting. Full setup is required to qualify resistor retention, new lighting or
camera settings.

See [saved inputs and outputs](MODEL_STORAGE.md) and the
[measured model contract](MODERN_MODEL.md).
