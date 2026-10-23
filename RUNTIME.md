# Runtime integration

The current `perception_node` loads the new camera-aware binary bundle and the
legacy models. New models use `model_type: calibrated_bin_background`, measured
top-rim/boundary registration (`reference_mode: measured_rim`), and per-camera
exposure/gain/black-level correction before merging foreground. See
[model details](MODERN_MODEL.md) and the [standalone collection UI](STANDALONE_UI.md).

The active station config is
`artifacts_cell_4/config/pose_estimation/contour_summit_pose_params.yaml`.
It still points to the October 16 legacy model until a qualified replacement is
explicitly activated. The standalone UI's Use model operation installs a
candidate, calls `bin_model/reload`, checks its reported path/name, and restores
the previous configuration on failure. `bin_model/status` reports the loaded
identity without capturing. Reload refuses concurrent pose work. Collection
itself uses `bin_capture` and requires no Perception workflow or Brain sequence.
New sessions select a configured scene and robot frame, and collect any number
of named bins defined by that scene's measured layout. Camera calibration comes
directly from `artifacts/calibration/deployment/camera/`; robot calibration comes
from the matching deployment robot file. Each named bin gets its own geometry,
registration reference, and validation results.

RViz's `contour_summit.rviz` preset shows the observed input, reference,
foreground and targets. New measured models additionally publish the top lip
on `/summit/bin_rim_cloud` (cyan) and its boundary on
`/summit/bin_boundary_cloud` (orange). The reference is green; observed input
retains its source coloring. The retained publishers allow RViz to connect after
a request. No target is published when registration/detection fails.

## Historical pth_contour_pose integration

The rest of this document preserves the previous service and its historical
paths/commands. These are not instructions for launching the new standalone UI.

The historical Cell 4 setup selected the measured height-field filter in
`deep_pose_config/ros_config/pth_contour_pose_config.yaml`. The XML sequences
request a named bin. The contour service removes that bowl's measured background
and rejects the other bin before finding a summit.

The runtime model bundle is
`deep_pose_config/bin_filter_models/2026-10-16/model_v1/`. It is an unchanged copy
of the trained model and calibration snapshots; the original offline results
remain under `bin_filter_offline/outputs/2026-10-16/`.

The runtime also supports exposure-profile bundles with exact anchor selection
and interpolation between anchors. See [EXPOSURE_PROFILES.md](EXPOSURE_PROFILES.md)
for collection, formulas, and acceptance checks. This capability does not expand
the measured range of the configured 8 ms model; its recorded results remain
8 ms evidence. Repository configuration does not confirm the state of a running
process.

The Cell 4 ROS YAML and this complete versioned model bundle are tracked in Git,
even though other contents of `deep_pose_config/` remain ignored. Pull these
artifact changes along with the code and mount `artifacts_cell_4` at
`/root/artifacts` in the containers. Keep the calibration snapshots with the model.
The frozen model's historical `online_integration` field describes its original
offline export. The ROS YAML selects the runtime implementation.

## Sequence syntax

```xml
<Action ID="GetPose"
  part_name="capacitor_submit"
  arm_name="B2"
  pose_request="6DOF"
  scene_id="0"
  blocking_type="0"
  bin_id="bin_a"
  pose_prior="capacitor_submit:0,0,0,0,0,0"
  pose_prior_options="capacitor_submit,robot,180,10000"
  rectify_pose_options="capacitor_submit,0;1;2,2;2;0,0/0/1;0/0/1;0/-1/0,0;0;0,robot;robot;robot"
  output_dir="{data_capture_dir}"
  output_poses="{capacitor_submit_poses}" />
```

Use `bin_id="bin_b"` for B. Both picking XMLs already contain these changes.
The four-field `pose_prior_options` preserves the pose frame and prior limits;
there are no manual crop dimensions or center coordinates. Keep it when using
`pose_prior`, because Brain requires matching prior options.

`bin_id` applies to one requested part per GetPose action. It is sent with the
individual request, including asynchronous task/result selection. The gripper
measurement GetPose actions are unchanged and omit `bin_id`.

## Runtime configuration

The historical Cell 4 ROS YAML contained:

```yaml
background_filter_mode: "height_field"
bin_filter_model_path: "/root/artifacts/deep_pose_config/bin_filter_models/2026-10-16/model_v1/model.yaml"
```

Keep the sibling `calibration/camera.yaml` and `calibration/robot.yaml` files
with the model. Startup verifies their stored hashes and compares the active
camera reconstruction fields and B2 scene-0 hand-eye calibration with them.
Unrelated TCP/tool fields may change without invalidating the bin model. A changed
camera calibration or camera placement requires reassessment and a matching
model/calibration. A single-exposure model requires its recorded exposure; an
exposure-profile bundle supports selection only within its anchor range and
requires physical validation of that range. With ICP disabled, bin placement
must also remain fixed. ICP below handles only small, observable bin movements.

The measured model requires **8.0 ms**. Start Eyes with the existing workcell
exposure file:

```bash
ros2 run ebots_eyes ebots_eyes --ros-args \
  --params-file /root/artifacts/config/eyes_camera_data.yaml
```

For an already running, rebuilt Eyes process, the equivalent parameter command is:

```bash
ros2 param set /eyes_node camera_data_cuda_shm.scene0.exposure_time_ms 8.0
```

The camera service reports the exposure supplied to the acquisition request;
there is currently no independent sensor-register readback. The contour service
checks that reported value against the selected model. A legacy model rejects
a mismatch; a profile bundle selects its anchor or interpolation bracket and
rejects nonpositive, nonfinite, and out-of-range settings. Editing the YAML alone
does not change an already running Eyes process. The plain `run-eyes` alias may
not load the file.
For recorded `camera_source: sim` inputs, `sim_exposure_time_ms` must state the
recorded exposure and be supported by the model; it cannot change the saved TIFFs.

To use a reviewed profile bundle, retain its sibling calibration snapshots and
change `bin_filter_model_path` to the new versioned `model.yaml`. Rebuild and
restart Perception with profile support and configure Eyes for the intended
supported exposure. Do not replace the frozen 8 ms bundle with a hand-edited
exposure range. The selected reference is shared by ICP, raw-point
classification, and reference visualization for that capture.

## Rebuild and restart the matching components

The ROS interfaces changed in `ebots_ros2_common`: `EyesRequest` and
`PthContourMapSummitPose` gained `bin_id`, and `CameraDataCudaShm` now reports
`exposure_time_ms`. Rebuild all consuming workspaces against the same interface
versions, then restart their processes together. A container relaunch by itself
does not compile these source changes. Do this between robot runs.

In each relevant container, use the normal full-workspace build:

```bash
bash /root/colcon_build.sh
```

Then source the rebuilt install in the terminal used to restart that component:

| Container | Setup file |
| --- | --- |
| Brain | `/root/brain/workspace/install/setup.bash` |
| Eyes | `/root/workspace/install/setup.bash` |
| Perception | `/root/perception/workspace/install/setup.bash` |

Rebuild/restart HMI against its updated common interfaces too if it is running as
a consumer. Preserve the normal Brain/hardware launcher for this cell; this
change does not start robot motion. Start Eyes with the exposure file shown above.
Restart the contour node using `run-contour-map`, or explicitly:

```bash
ros2 launch ebots_perception_ros2 pth_contour_pose_launch.py \
  params_file:=/root/artifacts/deep_pose_config/ros_config/pth_contour_pose_config.yaml
```

Use a fresh shell or source the rebuilt setup files; already loaded libraries and
ROS type support do not update in running processes. The development checks use
isolated build directories and do not replace the live install or restart nodes.

## Processing and diagnostics

### Optional per-capture bin ICP

The checked-in Cell 4 contour ROS YAML currently has `bin_icp.enabled: true`.
Restart the rebuilt contour node with that YAML to use the setting; the source
file alone does not establish whether a deployed process has ICP enabled.
For this historical example, the `bin_id="bin_a"` / `bin_id="bin_b"` XML
requests use separate reference surfaces and independent corrections. New
models use whichever bin names are declared in their selected scene layout.
No new common interfaces or Brain/Eyes changes are required for this addition.
See [ICP_VALIDATION.md](ICP_VALIDATION.md) for activation steps, replay commands,
measured results, and the remaining physical-bin validation.

ICP starts from the nominal reference position every time; it does not carry a
previous correction forward or modify the saved model. The current Cell 4
settings select `bin_icp.reference_mode: upper_band` with
`bin_icp.min_reference_height_mm: -100.0`. This selects reliable reference points
whose nominal B2 Z is at least -100 mm for both fitting and quality evaluation.
The 40% coverage threshold therefore counts only that fixed upper band; hidden
floor and lower-wall points do not contribute to its denominator. The complete
height field remains available for background classification after alignment.

For the frozen October 16 model, this leaves 4,493 reference points in A and
4,108 in B; 40% requires at least 1,798 and 1,644 matches respectively, alongside
the other quality gates. This is the curved upper inner-wall region up to the
model's -82 mm reference ceiling. A literal flat horizontal rim would need a
different measured reference and additional geometry to constrain every pose
axis. Normals are estimated from the complete model before the band is selected,
so the artificial lower edge does not change the wall normals.

Registration uses per-XY-cell median samples of scene points near that band and
trimmed, Huber-weighted point-to-plane matches against its measured surface. Median sampling avoids the surface bias seen with ordinary 3D voxel averages
on these captures. Elevated part points are downweighted or excluded. This is a local alignment method, not a global
bin search or a semantic bin/part segmentation method.

The following values are the current Cell 4 artifact settings, not a statement
of the deployed node's values or the library defaults:

| Parameter | Cell 4 setting | Meaning |
| --- | ---: | --- |
| `bin_icp.reference_mode` | `upper_band` | Fixed upper reference region for alignment and coverage |
| `bin_icp.min_reference_height_mm` | -100.0 | Minimum nominal B2 reference Z, in mm |
| `bin_icp.voxel_size_mm` | 1.0 | Registration-only XY cell size for median sampling; classification still uses raw points |
| `bin_icp.max_correspondence_mm` | 6.0 | Maximum distance of an ICP correspondence |
| `bin_icp.max_translation_mm` | 2.0 | Maximum displacement of the reference bin's centroid |
| `bin_icp.max_rotation_deg` | 1.0 | Maximum total rotation from the reference pose |
| `bin_icp.min_reference_coverage` | 0.4 | Minimum fraction of the selected upper-band reference supported by the current scene |
| `bin_icp.max_plane_rmse_mm` | 0.8 | Maximum point-to-plane RMS residual of final supported reference points |
| `bin_icp.nominal_plane_rmse_mm` | 0.3 | Retain the nominal pose when it already passes quality checks below this residual |
| `bin_icp.min_information_ratio` | 0.0001 | Minimum normalized pose-information eigenvalue ratio |
| `bin_icp.max_iterations` | 50 | Maximum local optimization iterations |

Before optimizing, ICP checks the nominal pose. If it passes coverage and
observability checks with an RMS residual at or below 0.3 mm, the identity
transform is accepted and `applied: false` is recorded. This noise tolerance
avoids unnecessary corrections to already aligned bins; it is not a guarantee
that movement below a particular distance will be detected. The selected
threshold covers nominal residuals observed in the recorded Cell 4 replays.

An optimized correction additionally requires convergence, at least 200 supported
reference points, support in all four XY quadrants, and at least 20% residual
improvement when the nominal pose passes the general quality checks. Quality
checks use the complete fixed upper-band reference before fitting trims any
matches. Each XY quadrant still needs at least 20 supported reference points.
A flat or mostly hidden upper region
can leave some pose axes unobservable; such requests fail with `Bin ICP rejected`
and a reason instead of silently falling back to the unaligned model.

The [recorded native replay comparison](../ebots_ros2_perception/workspace/perception_core/tools/bin_registration_replay/README.md#local-smoke-verification-2026-10-22)
accepted all eight upper-band cases from four saved empty/loaded captures, with
100% band coverage and 0.0884–0.1777 mm residuals. These runs checked nominal
alignment and retained full-model classification. The incident's failed frame
was not saved, and these results do not establish recovery from physical bin
motion or severe upper-edge occlusion.

The saved transform maps reference robot-frame coordinates to current
robot-frame coordinates (B2 in this historical example):
`p_current = T_reference_to_scene * p_reference`. Classification applies its
inverse to live points, uses the selected exposure's height field, and retains the
corresponding **original** live points. Therefore summit coordinates and robot
pick poses retain their existing frame conventions. RViz shows the corrected
reference, and successful capture metadata records the transform, centroid
displacement, rotation, coverage, residual, information ratio, and iteration count.
Rejected registrations are reported in the contour log; no pose is returned.

This corrects small **bin** displacement with fixed cameras and valid hand-eye
calibration. It does not recalibrate the camera or robot, select a new acquisition
exposure, widen the selected foreground tolerance, or move the robot. Exposure
profile selection and its interpolation margin happen before ICP. Thin parts close to the
surface can still bias alignment; the quality gates cannot prove every match is
bin material. Validate deliberately shifted empty and loaded physical bins before
using ICP for production picking. Large bin movement needs a new initial pose
or model, rather than arbitrarily increasing these local limits.

The order is exposure/reference selection → stereo depth reconstruction → configured robot frame → named-bin
height-field classification → optional extra crop → voxel downsample → radius
outlier removal → existing summit detector. Classification precedes voxel means
so foreground and bowl points cannot be averaged together before the learned
threshold is applied. The saved model is cached at startup, not refitted per pick.
When enabled, per-bin ICP runs between the robot-frame transform and height-field
classification; its downsampled fitting copy does not replace the raw input.

Only points above the selected bin's reliable reference surface survive.
Unobserved/high-uncertainty cells, other bins, background, and points behind
the surface are excluded. The old empty XYZ template is not loaded in this mode.
Missing or unknown bin names fail before acquisition; an empty selected bin
returns failure instead of falling back to a different bin. Small remaining noise
still passes through the existing radius filter before summit detection.

The response pose stays in the **vision frame**; its robot-frame summit is still
recorded. Capture metadata now includes the selected bin, model path/hash,
filter order, reported acquisition exposure, selected anchor or interpolation
bracket in `processing.bin_filter_exposure`, and raw classification counts. Its
fields are `requested_exposure_ms`, `lower_exposure_ms`, `upper_exposure_ms`,
`weight`, and `mode` (`single`, `exact`, or `interpolated`). RViz's
cropping-cloud topic shows the selected bin's supported learned surface.

Other workcells keep `background_filter_mode: legacy_template` by default. That
mode uses the existing empty-cloud subtraction and rejects named-bin requests
rather than silently ignoring them. To deliberately restore the old Cell 4
workflow, both the YAML mode and sequence requests/crop boxes must be changed
consistently.

The model's measured thresholds are approximately 1–4 mm. Very thin features and
parts in excluded rim/unknown areas can still be lost. The offline tests and
recorded-data replays do not establish robot pick success.

## Validation results

The historical 8 ms recorded replay passed eight selected-bin cases: A-only, B-only, both-loaded,
and both-empty scenes. Loaded selections produced summit candidates; empty
selections returned no pose, including when the other bin was loaded. Unit tests,
calibration checks, and Python/C++ classifier parity also passed.

Those results do not validate interpolation or a 4–20 ms operating range.
Evaluate a new profile bundle on independent empty and loaded captures at every
anchor and held-out intermediate setting, then validate the selected reference
with the production radius/summit stages and physical-bin ICP checks.

See the [validation report](outputs/2026-10-16/runtime_validation/REPORT.md) for
counts, logs, timing scope, and remaining hardware validation. In a known-empty
bin, the existing radius filter can report that it removed every point; the
replays confirmed that this rejects the small residual noise without producing
a false summit.
