# Runtime integration

The production `summit_pose` step uses a deployed bin model to align the measured
rim, correct each camera's depth, remove the measured bin surface, and detect a
part summit. Collection and training use the separate [Bin Setup UI](STANDALONE_UI.md).

## Files used by Perception

Deployed models live in the station artifacts:

```text
artifacts/calibration/deployment/bin/<model-name>/
  model.json
  arrays/bins/<bin-id>/...
```

`model.json` selects the geometry, measured rim/boundary reference and camera
profiles stored in `arrays/`. Its small `calibration_reference` metadata records
the expected camera reconstruction values and selected hand-eye transform.
Perception reads the shared deployment camera and robot files below and checks
those values before using the model. The bin directory has no calibration copies.

Training captures, CAD meshes, fitting settings, previews and validation results
stay in the local session under `data/bin_model/<session>/`. Deployment state,
rollback history and failed-install diagnostics stay in
`data/bin_model/deployment/`. None of those results is needed in the deployed
model directory. A CAD-guided candidate also exports only its measured runtime
arrays: Perception does not open or align the STL during a pose request.

The Cell 4 pipeline configuration is
`artifacts_cell_4/config/pose_estimation/contour_summit_pose_params.yaml`:

```yaml
background_filter_mode: height_field
bin_filter_model_path: /root/artifacts/calibration/deployment/bin/fremont-bin-calibration-1023-02/model.json
calibration_yaml_path: /root/artifacts/calibration/deployment/camera/scene0_denali0_params.yaml
robot_calibration_yaml_path: /root/artifacts/calibration/deployment/robot/B2_params.yaml
use_gpu_bin_filter: true
use_reduced_registration_input: true
bin_icp:
  enabled: true
  reference_mode: measured_rim
```

These paths assume the station artifacts are mounted at `/root/artifacts`.
Other scenes, robot frames and bin layouts need their own matching model and
calibration. The loader compares camera reconstruction fields and the selected
scene's hand-eye transform with the expected model values; changing unrelated tool
parameters does not change the bin geometry. Moving a camera or changing its
calibration requires reassessing the model.

## Install and verify a model

1. Finish the guided collection in Bin Setup and run **Build and validate**.
2. Review the production validation result. Installation requires complete,
   passing physical-capture evidence.
3. While the robot is idle, use **Use model**. The UI exports the runtime files,
   updates the pipeline path, calls `bin_model/reload`, and checks the returned
   model path and name. A failed reload restores the previous configuration.
4. Check the loaded identity with `bin_model/status` before running the normal
   sequence. Status and reload do not acquire camera images.

Restart Perception after updating native binaries; editing files does not replace
already loaded code or models. The UI's deployment state records whether the
runtime acknowledged the model. A configured path alone is not confirmation.
The reload service refuses concurrent pose work. **Roll back** uses the previous
runtime model and saved configuration; its local deployment history must remain
available.

The current `fremont-bin-calibration-1023-02` candidate passed 270 held-out bin
checks at 4, 8, 12, 16 and 20 ms. Those results establish these exposure anchors;
they do not establish every intermediate exposure or grasp success. Intermediate
exposures require separately validated intervals. Gain and black level must match
the model's camera profiles. Capture metadata records requested exposure rather
than independent sensor readback.

## Run the HMI sequence

The Cell 4 `contour_map_test.xml` sequence requests one named bin per `GetPose`:

```xml
<Action ID="GetPose"
  part_name="bin_picking/highest_part"
  pose_request="6DOF"
  scene_id="0"
  bin_id="{contour_bin_id}"
  blocking_type="0"
  output_poses="{contour_bin_a_poses}" />
```

The requested bin ID must exist in the model. The sequence and part task select
the capture and summit step; the shared pipeline YAML selects the deployed
model. The current part configuration requests 8 ms for both structured light
and spotlight on both sensors. The sequence's GetPose actions do not move the
robot. Use the normal station HMI startup and wait for camera/Perception readiness
before running it.

## Registration and filtering

The modern model uses measured flat-rim samples to constrain tilt and height,
and measured boundary samples to constrain in-plane motion. The 40% minimum
reference coverage applies to the rim registration reference. Hidden bowl floor
and lower sidewall points do not enlarge that denominator. The complete measured
interior is still used to remove the bin background after alignment.

Each request starts from the model's nominal pose. The local registration has
bounded translation and rotation, residual, coverage and observability checks;
it is not a global search for a moved bin. Rejection returns no pick pose.
CPU/GPU correction uses separate profiles for each camera and exposure; reduced
registration input limits the points used for fitting while foreground filtering
retains the full supported measurements. Actual post-capture latency must be
measured on the deployed machine; passing model validation is not a latency test.

## RViz

Load `perception_node/config/rviz/contour_summit.rviz`. The reference cloud is
green and observed input retains its source coloring. The measured flat rim is
cyan on `/summit/bin_rim_cloud`; the boundary is orange on
`/summit/bin_boundary_cloud`. Foreground and summit displays show the retained
part data and returned targets. Publishers retain their latest debug result so
RViz can connect after a request. No target is published for rejected registration
or detection.

## Saved-data replay and legacy models

`summit_replay` runs the production detector against saved depth and calibration
without commanding camera hardware. The standalone build uses it for held-out
acceptance. Keep its exact pipeline configuration and results in the candidate's
`production_validation/` directory outside artifacts.

Legacy YAML models remain readable. They retain their original single-exposure
or profile behavior and may use `full_surface` or `upper_band` registration.
Do not transfer their settings to a modern measured-rim model. The older
[ICP validation notes](ICP_VALIDATION.md) and [exposure-profile notes](EXPOSURE_PROFILES.md)
remain historical evidence; current setup is documented above and in the
[operator guide](STANDALONE_UI.md).
