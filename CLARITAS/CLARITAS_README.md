# CLARITAS

CLARITAS is a single-file CUDA Monte Carlo model of light transport through
loess and kaolin suspensions in a cylindrical acrylic sample cell. It supports
material, concentration and wavelength sweeps, prescribed free surfaces,
PSD-resolved stationary sediment fields, detector response and transport
diagnostics. Production optical calculations run on the GPU; host code manages
configuration, reference calculations, reporting and plotting.

## Setup and execution

The supplied dependency bounds are in `requirements.txt`: NumPy, Matplotlib and
CuPy for CUDA 12. Production requires a compatible NVIDIA GPU, driver and CUDA
runtime. Host checks were tested with Python 3.12.14. Importing `CLARITAS` does
not start a simulation or require CuPy; production execution does.

```bash
python -m venv .venv
# Activate .venv using your operating system's usual command.
python -m pip install -r requirements.txt
python -m pip check
python validate_claritas.py
python test_diagnostics.py
python CLARITAS.py
```

For CPU-only checks, install `numpy>=1.26,<3` and `matplotlib>=3.8,<4` instead of
the complete requirements. Run `python validate_claritas.py --cuda-full` on a
CUDA-capable system for compilation, host/device comparisons, diagnostic
deposition, stationary-solver and small-ray checks. A skipped check is not a pass.

## Configuration

Edit the `USER CONFIGURATION` section near the top of `CLARITAS.py`; there is no
command-line configuration parser. `RUN_SWEEP=False` uses `MATERIAL`,
`CONCENTRATION_G_L` and `WAVELENGTH_NM`. With `RUN_SWEEP=True`, the Cartesian
product of `MATERIALS`, `CONCENTRATIONS_G_L` and `WAVELENGTHS_NM` is used.
Concentration is bulk sediment loading integrated over the actual water volume,
not the local concentration at a sensor or boundary. Geometry uses metres and
heights above the internal bottom; CUDA uses the sensor plane as internal z=0.

| Control | Supplied default |
| --- | --- |
| Sweep | Loess and kaolin; 0.5, 2 and 4 g/L; 622 nm |
| Rays per case | 100,000,000 |
| Internal bore diameter | 0.098 m |
| Sediment model | `schiller_naumann_meridional_advection_diffusion` |
| Vortex | `localized_scully_vortex`; wall-centre height difference 0.050 m |
| Meridional circulation ratio | 0.5 |
| Particle event law | `hybrid_mie_exact_rate` |
| Particle imaginary refractive index | 0.0 |
| Diagnostic grid | 256 × 256 × 384; all launched rays eligible |
| Failure policy | `report`; failed-ray stop fraction 0.001 |

These defaults are substantial GPU workloads. For a small execution check,
explicitly reduce `N_RAYS` and, if necessary, the diagnostic grid/ray limit in
your working copy. `LIGHT_TRANSPORT_DIAGNOSTIC_MAX_RAYS` is set from `N_RAYS`
when the module is loaded. Display controls affect rendering, not raw arrays.

`SEDIMENT_DISTRIBUTION` selects uniform or stationary fields. Stationary transport
supports Stokes slip, finite-Re Schiller–Naumann drift/diffusion, and meridional
advection/diffusion. Both localized Scully and parabolic vortices are supported;
the vortex is prescribed rather than formed through a time-dependent flow model.
The constructor's model-code dictionaries list additional comparison and
diagnostic options. `legacy_*` output keys and `legacy_polar_pdf` are active
comparison interfaces: the latter treats the Beta law as a polar probability
density rather than 3-D angular radiance.

`MODEL_DEFAULT_KWARGS` is a constructor template; `_build_model` applies current
configuration values. `COMPARISON_GEOMETRY` and `COMPARISON_SEDIMENT` describe the
supplied comparison setting (93 mm bore, finite-Re drift/diffusion, zero
meridional circulation). They are not the current defaults. Match flags compare
settings only and do not establish scientific validation.

## Outputs and failure accounting

Results are written beneath `OUTPUT_DIR` (`claritas_output`), with timestamped
sessions by default and subdirectories for material, wavelength and concentration.
Configured export switches determine which CSVs, arrays and plots are produced.

- `run_config.json`, `run_summary.json`, `simulation_summary.csv`: resolved
  settings, executing source hash, numerical summaries and integrity assessments.
- `detector_response.csv`: absolute detector scores, normalized angular response,
  native/mirror counts and uncertainty estimates.
- `particle_optical_properties.csv`, `phase_function.csv`: per-bin optical
  properties and the reporting phase-function average.
- Sediment, local PSD, sensor-ring and vortex CSV/PNG products: spatial and
  size-resolved diagnostics of the production field.
- `light_transport_volume.npz`, section CSVs/PNGs and representative trajectories:
  observed production histories. Track-length totals are clipped to the diagnostic
  grid; they are not a measurement of full physical path length.
- Integrity reports, `water_boundary_report.json`, `sweep_progress.json` and
  `case_failure_report.json`: failure categories, limited outputs and checkpoints.
  Unsafe diagnostic bundles may export `light_transport_safe_raw_fields.npz`.

Failed rays remain in launched denominators; they are not removed by successful-ray
renormalization. Valid prefixes can remain in history diagnostics. `report` can
limit diagnostic outputs without stopping valid production cases; `strict` requests
a stop after reports and safe outputs are saved. Configuration errors, malformed
arrays and CUDA/programming errors remain fatal. Policy thresholds are operational
budgets, not established physical-accuracy criteria.

## Scientific scope and validation limits

Geometric particle interfaces, coherent Mie scattering and incoherent morphology
events remain separate channels. Production sampling uses the selected bin's phase
table; the exported reporting average must not replace those per-bin distributions.
Stored size-dependent optical profiles are exploratory hypotheses, not measured
sample constants. The configured imaginary index overrides the stored k(d) values.

Changing wavelength recomputes explicitly wavelength-dependent quantities, but
water/acrylic indices and particle n(d) have no automatic spectral dispersion.
Optical overrides need an external physical basis. Stationary sediment solutions
balance transport at ∂C/∂t=0; they do not predict transient settling or vortex
formation. Optional aggregation and fragmentation pathways remain available but
are disabled in the configured model template.

Normalized detector agreement concerns angular shape. Absolute amplitude requires
separate experimental validation. Configuration-match flags and
`parabolic_transport_validation_pass` are limited checks, not evidence of amplitude
accuracy. No production sweep or CUDA execution was performed during publication
cleanup; see `CLEANUP_REPORT.md` for the checks actually completed and the small
metadata/interface migration.
