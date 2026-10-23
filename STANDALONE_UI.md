# Standalone Bin Setup

Run `bin_setup.py` on the local PC and open **http://localhost:8765**. The UI
collects data through the existing camera and Eyes services, fits a candidate,
and validates saved captures. Collection does not need a running Perception
node, Brain, HMI or robot sequence. The application never moves the robot.

## Station installation

From the repository root:

```bash
python3 -m venv bin_filter_offline/.venv
bin_filter_offline/.venv/bin/python -m pip install -r bin_filter_offline/requirements.txt
bin_filter_offline/.venv/bin/python bin_filter_offline/install_tools.py \
  --artifacts-root ebots_system_artifacts/<YOUR_ARTIFACTS_CELL: e.g.artifacts_cell_4>
bash bin_filter_offline/native/build.sh <YOUR_CAMERA_CONTAINER: e.g. eyes_container>
```

Replace `YOUR_CAMERA_CONTAINER` with the actual running Eyes container name.
The build installs the native helper at
`/root/artifacts/data_saving/bin_model/tools/native/bin_capture` inside that
container. It builds the camera transport and Float32 EXR writer without building
or starting Perception. Refreshing staged Python tools preserves `tools/native`
and keeps the previous Python copy in `tools.previous`.

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
acquisition defaults. Host Python needs neither ROS nor CUDA. It invokes the
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

An integrator defines each measured bin layout under
`artifacts/data_saving/bin_model/layouts/`. A layout contains `scene_id`,
`frame_id`, an optional readable `name`, and a `bins` mapping with one entry per
bin. Use `data_saving/bin_model/bin_config.yaml` as the fitting-settings example.
Set the actual bounds, interior-floor seed, height range and top-rim band in the
chosen frame. Names such as `resistors`, `left` and `right` are supported.
Do not copy another scene's coordinates without measuring them. The UI offers
matching layouts and lets the operator select all or a subset of their bins.

The checked-in default is the measured scene-0/B2 layout. Other scenes need their
commissioned camera calibration, hand-eye transform, Eyes workflow entry and
measured layout in the artifacts tree. Changing the scene selector does not
create these measurements. The sidebar selections apply to the next new setup;
an opened session retains its recorded scene, bins and capture plan.

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
checked against it during verification. Capture and model calibration snapshots
are retained for reproducibility, not additional setup files to maintain.

## Operator workflow

1. Select the scene, camera calibration, reference frame, measured layout and bins.
   Enter an optional readable session name and sorted exposures from 4–20 ms.
   Defaults are **4, 8, 12, 16, 20 ms**; additional measured anchors can be entered.
   Each value sets both spotlight and structured-light exposures on both cameras:
   a 4 ms step uses 4 ms for both, an 8 ms step uses 8 ms for both, and so on.
   Click **New setup**. Existing names are refused.
2. Follow the stage instructions and diagram. **Scene ready — capture exposures**
   records your confirmation and captures that arrangement at every selected
   exposure. Leave every selected bin unchanged during the sweep, so exposure
   comparisons use the same part positions. The app stops before the next
   arrangement and waits for another Ready click.
3. Empty training and fresh empty validation come first. Then place **exactly
   one resistor in each selected bin**, collecting all bins together at the same
   named location: center, beside **each of the four walls**, then the indicated
   bottom-left corner.
   Each location has **three arrangements**: after each sweep, rotate or
   reposition every bin's resistor within its indicated region before clicking
   Ready again. Keep exactly one resistor in each bin. Finally collect three normal-fill
   and three full-fill arrangements in all selected bins. Rearrange the parts
   between sweeps while keeping roughly the same fill amount; use your usual
   production amounts, with no exact count needed. Empty stages retain their
   automatic repetitions and do not ask for rearrangement.
4. **Build and validate** fits empty training data, then evaluates held-out empty,
   resistor and filled captures through geometry checks and production replay.
   Review failures by capture, bin and exposure. Rebuilding preserves earlier
   candidates and writes `candidate-02`, `candidate-03`, and so on.
5. **Use this model** requires passed physical evidence and an available
   Perception Summit reload service. It installs the bundle, verifies the loaded
   identity and restores the previous configuration if activation fails. The
   configured pipeline must match the model's scene and reference frame. Use
   `--pipeline-config` and `--models-root` for another deployment target.

With the default five exposures, the guide has **170 captures and 26 Ready
confirmations** for any number of selected bins. All selected bins share each
capture. Each loaded
arrangement shows **0/5** in the sidebar and needs a separate Ready click.
The three arrangements still total 15 captures per resistor location or fill
condition. Only empty training captures fit the model. Fresh empty validation
checks each selected bin while all are empty; loaded validation evaluates every
bin's resistor and expected location separately within the shared captures.
Validation does not establish grasp success, unmeasured exposure settings or
arbitrary ambient lighting.

These counts apply to newly prepared sessions with the default three loaded
arrangements, four walls, and simultaneous resistor placement in all selected
bins. The diagram labels **Left (−X), Right (+X),
Bottom (−Y), and Top (+Y)** in the bin's local coordinate system. The selected
wall is gold and the dashed amber region marks the interior floor placement
area. Keep the resistor off the rim and side wall. The resistor icon is an example
position, not a live detection; camera images may use a different orientation.
The corner stages still use only the indicated bottom-left (−X, −Y) corner.

Existing saved sessions retain their recorded plan. Older sequential sessions
still ask for one loaded bin and the other bins empty; follow their current
instructions and diagrams. An older loaded stage showing **0/15** runs all three
exposure sweeps after one Ready click. An older wall stage without a side label
uses the left (−X) wall. Its recorded images are not relabeled as simultaneous
captures, different arrangements, or other wall sides.

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
   `perception_config/bin_filter_models/installed/<model-name>/`, updates
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
