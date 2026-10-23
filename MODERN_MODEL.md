# Measured bin model

`model_type: calibrated_bin_background` selects the camera-aware fitter. Its
`model.json` and binary `arrays/` directory contain three distinct measurements
for each bin:

- `geometry`: the connected interior height field, in local bin millimetres.
- `registration`: measured flat-lip surface samples and separate physical
  boundary samples next to an observed lower surface.
- `camera_profiles`: per-camera, per-exposure bias, empirical uncertainty,
  validity and independent capture support, including the lip footprint.

`robot_from_bin` places each local shape in the calibrated robot frame. The
fitter keeps independently measured bins separate. `extract_shape` exports local
geometry and registration without placement or camera error profiles; a new site
still needs its own placement, camera profiles and acceptance data. Explicit
rotated placements require local fitting bounds (`coordinate_frame: bin`). The
station's existing robot-coordinate bounds get a translation-only local origin;
no orientation is inferred from a guessed CAD model.

The geometric reference is an empirical consensus of empty captures. Camera
bias is relative to that reference, not an absolute metrology claim. Each camera
is corrected in local Z before merging. Unsupported camera cells are excluded.
The foreground threshold uses the larger of the camera's empirical clearance
uncertainty and the measured geometry residual after camera correction. Geometry
residuals account for interpolation and discretization; replacing them with a
constant floor can produce false foreground near curved surfaces. Corrected
measurements supply output/pick coordinates. Source camera indices and original
coordinates remain available as provenance; bin-registration transforms remain
lookup corrections rather than movements of the observed scene.

Camera uncertainty is the worst per-capture residual quantile plus a configured
margin. It is a clearance envelope, not a Gaussian standard deviation. Gain and
black level are recorded alongside each camera profile and must match at use.
Historical inputs with unknown acquisition settings can produce development
candidates, but cannot establish production acceptance.

Measured exposure anchors are usable individually. Intermediate values require
an explicit `validated_intervals_ms` interval supported by held-out production
replay. Bias interpolates linearly; uncertainty uses the larger endpoint envelope
plus an interpolation margin, and both endpoint validity masks must be valid.
Extrapolation is rejected. `candidate_interval=True` is an offline-only preview
for testing a proposed interval; it does not enable that interval in the model.

The interior's `max_reference_height_mm` does not crop registration data. Lip
fitting independently uses `registration.height_band_mm` and low measured slope.
Boundary samples are the nearest stable measured lip row to observed lower
geometry. Missing depth, the image border and the ROI border never manufacture
an edge. Too few samples, inadequate independent capture support or missing
quadrant coverage reject fitting. Counts, measured extents and support are saved
with the model for review.

The station recipe trains at 4, 8, 12, 16 and 20 ms. That establishes those five
anchors only, after the operator's fresh empty, resistor, normal and full scenes
pass the production pipeline. It does not establish every value between them.
Factory acquisition settings are explicitly *requested*, not sensor-verified.

The binary representation uses little-endian float64/int64 arrays with relative
file paths and explicit dimensions. No camera images, operator labels or large
height grids are embedded in YAML. Calibration and fitting settings remain small
readable snapshots beside the model. Legacy YAML models can still be read and
explicitly exported, without changing their original behavior.
