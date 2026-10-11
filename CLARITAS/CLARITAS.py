#!/usr/bin/env python3
"""CLARITAS: CUDA Monte Carlo light transport in sediment suspensions.

Configure single cases or material/concentration/wavelength sweeps below.
Production optical physics and stationary sediment fields are computed in CUDA;
host helpers provide reference calculations, diagnostics and CSV/JSON reporting.
Sediment transport is stationary, not a transient settling simulation.

Wavelength-dependent Mie efficiencies, per-bin phase functions, morphology
coherence and internal absorption are recomputed. Refractive-index dispersion
tables are not supplied: water, acrylic and particle n(d) remain fixed unless
explicitly overridden. The configured particle imaginary index uses the stored exploratory k(d) arrays.
These arrays are not measured spectral optical constants.
Supplied comparison settings at 622 nm are distinct from the current defaults.

The executing source SHA-256 is recorded in run_config.json and run_summary.json.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
from pathlib import Path
from datetime import datetime
import csv
import hashlib
import itertools
import json
import math
import time
import numpy as np

# ============================================================================
# USER CONFIGURATION -- normally edit only this section
# ============================================================================


# Operational stop limits, not physics validation criteria. None disables a limit.
DIAGNOSTIC_POLICY = "report"  # report or strict
MAX_FAILED_RAY_FRACTION = 0.001  # abort after exporting case results if exceeded
MAX_UNEXPLAINED_EVENT_FRACTION = 0.001
MAX_DIAGNOSTIC_CLOSURE_ERROR = 0.001
ABORT_ON_SYSTEMIC_DIAGNOSTICS = False  # diagnostic-only faults normally limit outputs
STRICT_DIAGNOSTIC_CLOSURE_ERROR = 2e-6

# ---------------------------------------------------------------------------
# SIMULATION
# ---------------------------------------------------------------------------
RUN_SWEEP = False
MATERIAL = "loess"  # "loess" or "kaolin"
CONCENTRATION_G_L = 0.5
WAVELENGTH_NM = 622.0  # supplied comparison wavelength = 622 nm
N_RAYS = 10_000_000
RANDOM_SEED = 24620001
GPU_DEVICE = 0

# ---------------------------------------------------------------------------
# SWEEP CONFIGURATION (used only when RUN_SWEEP=True)
# ---------------------------------------------------------------------------
MATERIALS = ["loess", "kaolin"]
CONCENTRATIONS_G_L = [0.5, 2.0, 4.0]
WAVELENGTHS_NM = [622.0]
USE_SAME_SEED_FOR_SWEEP = False  # False => deterministic seed offset per case

# ---------------------------------------------------------------------------
# SEDIMENT SPATIAL DISTRIBUTION
# ---------------------------------------------------------------------------
# Sediment is stationary: use either a
# uniform suspension or a PSD-resolved STEADY-STATE transport field in which
# settling, turbulent/eddy diffusion, vortex-driven radial drift and (for the
# optional model below) steady meridional advection are balanced.
#
# "steady_state" means partial C / partial t = 0; it does NOT mean that the
# settling/diffusion transport processes are absent.
SEDIMENT_DISTRIBUTION = "steady_state"  # "uniform" or "steady_state"

# Native steady-state transport law used when SEDIMENT_DISTRIBUTION is
# "steady_state":
#   "stokes_drift_diffusion"
#   "schiller_naumann_drift_diffusion"                 (supplied comparison)
#   "schiller_naumann_meridional_advection_diffusion"  (requires vortex)
SEDIMENT_TRANSPORT_MODEL = "schiller_naumann_meridional_advection_diffusion"
SEDIMENT_EDDY_DIFFUSIVITY_M2_S = 1.0e-3
SEDIMENT_PARTICLE_DENSITY_KG_M3 = 2600.0
WATER_DENSITY_KG_M3 = 997.0
WATER_DYNAMIC_VISCOSITY_PA_S = 8.9e-4
GRAVITY_M_S2 = 9.81

# Optional steady bulk meridional circulation. The supplied comparison uses zero.
# Both supported vortices use the same circulation closure.
MERIDIONAL_CIRCULATION_RATIO = 0.5

# Stationary sediment quadrature, grid and solver controls.
SEDIMENT_TRANSPORT_QUADRATURE_POINTS = 4096
SEDIMENT_RADIAL_LOOKUP_POINTS = 2049
MERIDIONAL_GRID_NR = 48
MERIDIONAL_GRID_NZ = 96
MERIDIONAL_SOLVER_MAX_ITERATIONS = 8000
MERIDIONAL_SOLVER_CHECK_INTERVAL = 250
MERIDIONAL_SOLVER_TOLERANCE = 1.0e-6
MERIDIONAL_SOLVER_SOR = 1.75

# ---------------------------------------------------------------------------
# VORTEX / FREE-SURFACE MODEL
# ---------------------------------------------------------------------------
# ENABLE_VORTEX means include the existing prescribed steady vortex/free-surface
# field.  It does NOT simulate vortex formation through time.  The selected
# model shapes the water/air boundary and supplies the matching centrifugal
# drift and optional meridional flow field.  Both supported models are fully
# coupled to uniform and steady-state sediment configurations.
# A uniform sediment distribution may still be combined with vortex optical
# geometry.
ENABLE_VORTEX = True
VORTEX_MODEL = "localized_scully_vortex"  # or "parabolic_vortex" (rigid-body rotation)
VORTEX_WALL_CENTER_HEIGHT_DIFFERENCE_M = 0.050
VORTEX_CORE_RADIUS_M = 0.020

# ---------------------------------------------------------------------------
# SAMPLE CELL GEOMETRY
# ---------------------------------------------------------------------------
# Heights are measured upward from the INTERNAL BOTTOM/WATER-BASE PLANE.
# SAMPLE_CELL_DIAMETER_M is the internal water-bore diameter (2*tube_inner_radius).
# The supplied comparison apparatus uses a 93 mm bore; current defaults use 98 mm.
SAMPLE_CELL_DIAMETER_M = 0.098
SAMPLE_CELL_HEIGHT_M = 0.491
WATER_LEVEL_M = 0.142
SENSOR_RING_HEIGHT_M = 0.093

# Wall and ring dimensions propagate consistently when the bore diameter changes.
ACRYLIC_WALL_THICKNESS_M = 0.0035
SENSOR_RING_INNER_CLEARANCE_M = 0.0005
SENSOR_RING_RADIAL_WIDTH_M = 0.0150
COUNTERBORE_DEPTH_M = 0.008
THROUGH_BORE_DIAMETER_M = 0.004
COUNTERBORE_DIAMETER_M = 0.0087
BOTTOM_DISC_THICKNESS_M = 0.003

# Source launch radius follows the sensor-ring outer edge unless overridden.
SOURCE_LAUNCH_RADIUS_M = None

# ---------------------------------------------------------------------------
# OPTICAL PROPERTIES / MULTI-WAVELENGTH ASSUMPTIONS
# ---------------------------------------------------------------------------
WATER_REFRACTIVE_INDEX = 1.33
ACRYLIC_REFRACTIVE_INDEX = 1.4906
AIR_REFRACTIVE_INDEX = 1.0

# The configured optical profile uses size-dependent n(d) and stored k(d).
# Stored k(d) profiles are exploratory, not measured wavelength-dependent data.
PARTICLE_N_REAL_OVERRIDE = None
# None: select the stored k(d) array separately for each material (works in a sweep).
# 0.0: disable particle absorption by setting every bin's k to zero.
# Positive scalar: set the same k for every bin and every material.
# Per-bin array: match PSD order and length (38 loess / 37 kaolin); run one material.
# All modes use the same selected k values across wavelength sweeps. Use separate
# runs with justified optical constants to represent wavelength-dependent k.
PARTICLE_K_IMAG_OVERRIDE = None

# Particle surface parameters.
PARTICLE_SURFACE_RMS_SLOPE_DEG = 25.0
WAVE_SURFACE_RMS_HEIGHT_NM = 100.0

# ---------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------
OUTPUT_DIR = "claritas_output"
CREATE_TIMESTAMPED_SESSION = True
SAVE_CSV = True
SAVE_PLOTS = True
SHOW_PLOTS = False
SAVE_PHASE_FUNCTION_CSV = True
EXPORT_SEDIMENT_FIELDS = True
EXPORT_LOCAL_PSD_PROFILES = True
SAVE_SEDIMENT_DIAGNOSTIC_PLOTS = True
SAVE_PSD_DIAGNOSTIC_PLOTS = True
DETAILED_DIAGNOSTICS_PER_SWEEP_CASE = False
SAVE_VORTEX_PROFILE = True
SUMMARY_ANGLES_DEG = [0, 20, 90, 120, 140, 170]

# CUDA-resident light-history diagnostics.  These diagnostics observe the
# production histories without consuming RNG or changing any transport choice.
# Track length is deposited exactly by voxel-boundary traversal; it is not a
# count of sampled points.  The bounded ray subset keeps memory/runtime explicit.
SAVE_LIGHT_TRANSPORT_VISUALISATIONS = True
SAVE_LIGHT_TRANSPORT_DATA = True
LIGHT_TRANSPORT_DIAGNOSTIC_MAX_RAYS = N_RAYS
# Finer diagnostic cells: twice the resolution in each axis (8x grid memory).
LIGHT_TRANSPORT_GRID_NX = 256
LIGHT_TRANSPORT_GRID_NY = 256
LIGHT_TRANSPORT_GRID_NZ = 384
LIGHT_TRANSPORT_REPRESENTATIVE_RAYS = 256
LIGHT_TRANSPORT_MAX_SEGMENTS_PER_RAY = 512
LIGHT_TRANSPORT_VERTICAL_SLICE_THICKNESS_M = 0.001
LIGHT_TRANSPORT_SENSOR_SLICE_THICKNESS_M = 0.001
LIGHT_TRANSPORT_LOG_FLOOR = 1.0e-12
# Display-only normalization and rendering. These controls never modify exported arrays.
LIGHT_TRANSPORT_PLOT_DPI = 400
# Marker sizes are areas in points squared; small dots avoid overlapping blobs.
LIGHT_TRANSPORT_DISPLAY_LOW_PERCENTILE = 2.0
LIGHT_TRANSPORT_DISPLAY_HIGH_PERCENTILE = 99.5
LIGHT_TRANSPORT_3D_MIN_DISPLAY_PERCENTILE = 5.0
LIGHT_TRANSPORT_3D_MAX_DISPLAY_VOXELS = 2000000
LIGHT_TRANSPORT_3D_ALPHA_MIN = 0.06
LIGHT_TRANSPORT_3D_ALPHA_MAX = 0.50
LIGHT_TRANSPORT_3D_MARKER_SIZE_MIN = 0.15
LIGHT_TRANSPORT_3D_MARKER_SIZE_MAX = 0.80
LIGHT_TRANSPORT_3D_SURFACE_MARGIN_M = 0.005

# Sediment reporting controls only; these do NOT change transport physics.
SEDIMENT_PROFILE_RADII_FRACTION = [0.0, 0.5, 0.9]
SEDIMENT_PROFILE_HEIGHT_POINTS = 200
PSD_PROFILE_HEIGHT_FRACTIONS = [0.05, 0.25, 0.50, 0.75, 0.95]
SEDIMENT_OUTPUT_NR = 64
SEDIMENT_OUTPUT_NZ = 96
VORTEX_OUTPUT_NR = 256

# ---------------------------------------------------------------------------
# RUNTIME CONTROLS (defaults)
# ---------------------------------------------------------------------------
CHUNK_SIZE = 250_000
STATISTICS_BATCH_RAYS = 250_000

# ============================================================================
# END USER CONFIGURATION
# ============================================================================


def _current_source_sha256():
    """Return the SHA-256 of the executing source for reproducibility."""
    return hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest()


# ============================================================================
# PRODUCTION PHYSICS
# ============================================================================
LOESS_DIAMETER_M = np.array(
    [
        1.729e-6,
        1.981e-6,
        2.269e-6,
        2.599e-6,
        2.976e-6,
        3.409e-6,
        3.905e-6,
        4.472e-6,
        5.122e-6,
        5.867e-6,
        6.72e-6,
        7.697e-6,
        8.816e-6,
        10.097e-6,
        11.565e-6,
        13.246e-6,
        15.172e-6,
        17.377e-6,
        19.904e-6,
        22.797e-6,
        26.111e-6,
        29.907e-6,
        34.255e-6,
        39.234e-6,
        44.938e-6,
        51.471e-6,
        58.953e-6,
        67.523e-6,
        77.34e-6,
        88.583e-6,
        101.46e-6,
        116.21e-6,
        133.103e-6,
        152.453e-6,
        174.616e-6,
        200.000e-6,
        229.075e-6,
        262.376e-6,
    ],
    dtype=np.float64,
)
LOESS_WEIGHTS = np.array(
    [
        157,
        227,
        294,
        354,
        414,
        487,
        592,
        747,
        975,
        1291,
        1704,
        2197,
        2736,
        3288,
        3822,
        4196,
        4372,
        4391,
        4352,
        4362,
        4508,
        4826,
        5279,
        5758,
        6080,
        6106,
        5786,
        5149,
        4342,
        3404,
        2456,
        1662,
        1175,
        858,
        631,
        463,
        333,
        230,
    ],
    dtype=np.float64,
)
KAOLIN_DIAMETER_M = np.array(
    [
        0.172e-6,
        0.197e-6,
        0.226e-6,
        0.259e-6,
        0.296e-6,
        0.339e-6,
        0.389e-6,
        0.445e-6,
        0.51e-6,
        0.584e-6,
        0.669e-6,
        0.766e-6,
        0.877e-6,
        1.005e-6,
        1.151e-6,
        1.318e-6,
        1.51e-6,
        1.729e-6,
        1.981e-6,
        2.269e-6,
        2.599e-6,
        2.976e-6,
        3.409e-6,
        3.905e-6,
        4.472e-6,
        5.122e-6,
        5.867e-6,
        6.72e-6,
        7.697e-6,
        8.816e-6,
        10.097e-6,
        11.565e-6,
        13.246e-6,
        15.172e-6,
        17.377e-6,
        19.904e-6,
        22.797e-6,
    ],
    dtype=np.float64,
)
KAOLIN_WEIGHTS = np.array(
    [
        217,
        547,
        1112,
        2032,
        2985,
        3492,
        3308,
        2644,
        1893,
        1300,
        916,
        700,
        601,
        584,
        637,
        757,
        948,
        1208,
        1530,
        1899,
        2309,
        2770,
        3312,
        3973,
        4772,
        5681,
        6583,
        7267,
        7478,
        7042,
        6113,
        5057,
        3680,
        2330,
        1287,
        631,
        284,
    ],
    dtype=np.float64,
)

DETECTOR_ANGLES_DEG = np.arange(0, 180, 10, dtype=np.float64)

# Execution-integrity codes identify failure stages without changing physics.
# The high-level status array remains available for output compatibility.
PRODUCTION_FAILURE_NAMES = {
    0: 'PROD_OK',
    1: 'PARTICLE_MICROSURFACE_NUMERICAL',
    2: 'PARTICLE_MICROSURFACE_WATCHDOG',
    3: 'PARTICLE_MACRO_GEOMETRY_FAILURE',
    4: 'PARTICLE_MACRO_BOUNCE_LIMIT',
    5: 'ACRYLIC_ANNULUS_NO_INTERSECTION',
    6: 'ACRYLIC_ANNULUS_BOUNCE_LIMIT',
    7: 'BOTTOM_DISC_GEOMETRY_FAILURE',
    8: 'BOTTOM_DISC_BOUNCE_LIMIT',
    9: 'WATER_BOUNDARY_NO_INTERSECTION',
    10: 'TRACE_ITERATION_LIMIT',
    11: 'NONFINITE_POSITION',
    12: 'NONFINITE_DIRECTION',
    13: 'INVALID_MEDIUM_STATE',
    14: 'INVALID_EVENT_STATE',
    15: 'OTHER_EXPLICIT_FAILURE',
    16: 'SPHEROID_GEOMETRY_FAILURE',
    17: 'HEADSPACE_BOUNDARY_NO_INTERSECTION',
    18: 'SPATIAL_MAJORANT_VIOLATION',
}
PRODUCTION_MEDIUM_NAMES = {
    0: 'unknown/pre-water',
    1: 'water',
    2: 'acrylic_annulus',
    3: 'bottom_acrylic',
    4: 'particle',
    5: 'headspace',
}
# legacy_* detector fields describe the angular-window comparison estimator.
# legacy_polar_pdf treats the Beta law as a polar PDF rather than radiance.
# These public keys/options remain available; they are active comparison paths.
PRODUCTION_EVENT_CHANNEL_NAMES = {
    0: 'none',
    1: 'geometric_particle',
    2: 'morphology_wave_particle',
}
PRODUCTION_WATCHDOG_CODES = frozenset((2, 4, 6, 8, 10))
PRODUCTION_GEOMETRY_FAILURE_CODES = frozenset((3, 5, 7, 9, 13, 16, 17))
PRODUCTION_NONFINITE_CODES = frozenset((11, 12))
PRODUCTION_FAILURE_SUBCODE_NAMES = {
    0: 'NONE',
    10: 'VNDF_PROJECTED_AREA_INVALID',
    11: 'VNDF_CDF_INVALID',
    12: 'VNDF_INVERSE_CDF_INVALID',
    13: 'VNDF_VISIBILITY_FAILURE',
    14: 'VNDF_NORMALIZATION_FAILURE',
    20: 'HEIGHT_CDF_INVALID',
    21: 'HEIGHT_SAMPLE_INVALID',
    22: 'FREE_PATH_INVALID',
    23: 'HEIGHT_LAMBDA_INVALID',
    24: 'HEIGHT_C0_INVALID',
    25: 'HEIGHT_LOG_C0_INVALID',
    26: 'HEIGHT_G1_INVALID',
    27: 'HEIGHT_ESCAPE_PROB_INVALID',
    28: 'HEIGHT_RANDOM_INVALID',
    29: 'HEIGHT_LOG_CNEXT_INVALID',
    30: 'MICROFACET_NORMAL_INVALID',
    31: 'REFLECTION_VECTOR_INVALID',
    32: 'REFRACTION_VECTOR_INVALID',
    33: 'SIDE_STATE_INVALID',
    34: 'NONFINITE_DIRECTION',
    35: 'FRESNEL_INVALID',
    36: 'HEIGHT_CNEXT_RANGE_INVALID',
    37: 'HEIGHT_INVERSE_CDF_INVALID',
    38: 'HEIGHT_OUTPUT_INVALID',
    39: 'HEIGHT_FREE_PATH_INVALID',
    40: 'MICRO_WATCHDOG',
    41: 'MACRO_WATCHDOG',
    1001: 'SPHERE_ENTRY_SIDE',
    1002: 'SPHERE_CHORD',
    1003: 'SPHERE_EXIT_SIDE',
    1004: 'SPHERE_INTERNAL_SIDE',
    1005: 'SPHERE_BOUNCE_LIMIT',
    1101: 'SPHEROID_ASPECT',
    1102: 'SPHEROID_AXES',
    1103: 'SPHEROID_ENTRY',
    1104: 'SPHEROID_CHORD',
    1105: 'SPHEROID_EXIT_SIDE',
    1106: 'SPHEROID_INTERNAL_SIDE',
    1107: 'SPHEROID_BOUNCE_LIMIT',
    2001: 'ACRYLIC_NO_INTERSECTION',
    2002: 'ACRYLIC_BOUNCE_LIMIT',
    2101: 'BOTTOM_NONPOSITIVE_T',
    2102: 'BOTTOM_TANGENTIAL',
    2103: 'BOTTOM_BOUNCE_LIMIT',
    2104: 'BOTTOM_TOPOLOGY_INVALID',
    3001: 'WATER_NO_BOUNDARY',
    3002: 'TRACE_ITERATION_LIMIT',
    3003: 'NONFINITE_POSITION',
    3004: 'NONFINITE_DIRECTION',
    3005: 'WATER_OUTSIDE_INNER_RADIUS',
    3006: 'HEADSPACE_NO_BOUNDARY',
    4001: 'SEDIMENT_MAJORANT_VIOLATION',
}

# ============================ COMPLEX PARTICLE OPTICS ============================
# Explicit per-bin complex refractive-index profiles at 622 nm.
# These are exploratory hypotheses, NOT measured optical constants for the user's samples.
# They exist to test whether size-dependent n(d)+i*k(d) can materially change the detector response.
#
# Loess profile: smooth effective-mineral mixture that is clay/iron richer in the fine fraction
# and quartz/feldspar richer in the coarse fraction. These stored profiles are exploratory, not sample measurements.
# Every one of the 38 loess PSD bins receives its own n and k value.
DEFAULT_LOESS_N_REAL = np.array(
    [
        1.607451331,
        1.605504695,
        1.603586747,
        1.601692366,
        1.599827047,
        1.597981118,
        1.596160167,
        1.594367874,
        1.592599298,
        1.590855346,
        1.589138183,
        1.587447487,
        1.585783658,
        1.584147841,
        1.582538755,
        1.580958153,
        1.579405657,
        1.577883002,
        1.576389253,
        1.57492664,
        1.573494964,
        1.572095154,
        1.570728147,
        1.569395256,
        1.568097028,
        1.56683506,
        1.565610853,
        1.564425894,
        1.563282045,
        1.562181851,
        1.561127985,
        1.560123766,
        1.559173695,
        1.558283428,
        1.557461192,
        1.556719679,
        1.556082305,
        1.555628472,
    ],
    dtype=np.float64,
)
DEFAULT_LOESS_K_IMAG = np.array(
    [
        0.001036314391,
        0.00100409545,
        0.0009725096047,
        0.0009414702151,
        0.0009110650451,
        0.0008811348103,
        0.0008517691111,
        0.0008230252467,
        0.000794822161,
        0.0007671732463,
        0.0007401114665,
        0.0007136302637,
        0.0006877346223,
        0.0006624410284,
        0.000637728537,
        0.000613623132,
        0.000590118135,
        0.0005672389351,
        0.0005449708121,
        0.000523346451,
        0.0005023623575,
        0.0004820319492,
        0.0004623687456,
        0.0004433917156,
        0.0004251090282,
        0.0004075438773,
        0.0003907181311,
        0.0003746536642,
        0.0003593777604,
        0.0003449270403,
        0.0003313401273,
        0.0003186645197,
        0.0003069633941,
        0.0002963152118,
        0.0002868310859,
        0.0002786768356,
        0.0002721434123,
        0.0002680974742,
    ],
    dtype=np.float64,
)

# Kaolin gets an explicit 37-bin conservative clay-dominant profile so the code remains usable
# for both materials; configured k is supplied separately.
DEFAULT_KAOLIN_N_REAL = np.array(
    [
        1.585,
        1.584305748,
        1.583603199,
        1.582905961,
        1.58222285,
        1.58152895,
        1.580825132,
        1.580137092,
        1.579439632,
        1.5787465,
        1.57805136,
        1.5773587,
        1.576666415,
        1.575969469,
        1.575275553,
        1.574582453,
        1.573886743,
        1.573193902,
        1.572497863,
        1.571803467,
        1.571108815,
        1.570415873,
        1.569720958,
        1.569026043,
        1.568332463,
        1.567638212,
        1.566943504,
        1.566249071,
        1.565554649,
        1.564860252,
        1.564166201,
        1.563471767,
        1.562777499,
        1.562083009,
        1.561388825,
        1.560694246,
        1.56,
    ],
    dtype=np.float64,
)
DEFAULT_KAOLIN_K_IMAG = np.array(
    [
        0.0008,
        0.0007805609537,
        0.0007608895687,
        0.0007413668971,
        0.0007222398082,
        0.000702810588,
        0.0006831036844,
        0.0006638385767,
        0.0006443096912,
        0.0006249020138,
        0.0006054380686,
        0.0005860435904,
        0.0005666596099,
        0.0005471451365,
        0.0005277154959,
        0.0005083086708,
        0.0004888288063,
        0.0004694292508,
        0.0004499401693,
        0.0004304970855,
        0.0004110468237,
        0.0003916444313,
        0.0003721868175,
        0.000352729216,
        0.0003333089771,
        0.0003138699307,
        0.0002944181087,
        0.0002749739813,
        0.0002555301742,
        0.0002360870658,
        0.000216653628,
        0.0001972094688,
        0.0001777699824,
        0.0001583242393,
        0.0001388871099,
        0.0001194389002,
        0.0001,
    ],
    dtype=np.float64,
)

DEFAULT_PARTICLE_WAVELENGTH_M = 622e-9


def vortex_surface_geometry(
    model: str,
    zmean: float,
    radius: float,
    delta_h: float,
    core_radius: float,
    gravity_m_s2: float = 9.81,
):
    """Return deterministic geometry diagnostics for the selected free surface.

    ``delta_h`` is wall height minus centre height.  For the localized Scully
    surface, ``core_radius`` is the velocity-core radius ``a``.  The returned
    centre/wall positions use the same z coordinate as ``zmean``.
    """
    model = str(model).strip().lower()
    zmean = float(zmean)
    R = float(radius)
    D = float(delta_h)
    a = float(core_radius)
    g = float(gravity_m_s2)
    if not (R > 0.0 and np.isfinite(R)):
        raise ValueError('vessel radius must be finite and positive')
    if not (D >= 0.0 and np.isfinite(D)):
        raise ValueError(
            'vortex wall-centre height difference must be finite and nonnegative'
        )
    if not (g > 0.0 and np.isfinite(g)):
        raise ValueError('gravity must be finite and positive')
    R2 = R * R
    if not (R2 > 0.0 and np.isfinite(R2)):
        raise ValueError('vessel radius squared is outside the finite numerical range')
    if model == 'flat' or D == 0.0:
        return dict(
            center_z=zmean,
            wall_z=zmean,
            h_inf=0.0,
            h_bar=0.0,
            angular_velocity_rad_s=0.0,
            vtheta_peak_m_s=0.0,
            equivalent_rigid_body_rpm=0.0,
            equivalent_core_rpm=0.0,
        )
    if model == 'parabolic_vortex':
        omega = math.sqrt(2.0 * g * D / R2)
        rpm = omega * 60.0 / (2.0 * math.pi)
        if not (np.isfinite(omega) and np.isfinite(rpm)):
            raise ValueError('parabolic vortex derived angular velocity is non-finite')
        return dict(
            center_z=zmean - 0.5 * D,
            wall_z=zmean + 0.5 * D,
            h_inf=D,
            h_bar=0.5 * D,
            angular_velocity_rad_s=omega,
            vtheta_peak_m_s=omega * R,
            equivalent_rigid_body_rpm=rpm,
            equivalent_core_rpm=float('nan'),
        )
    if model != 'localized_scully_vortex':
        raise ValueError(f'unknown free-surface model {model!r}')
    if not (a > 0.0 and R > 0.0):
        raise ValueError(
            'localized Scully vortex requires positive core and vessel radii'
        )
    a2 = a * a
    if not (a2 > 0.0 and np.isfinite(a2)):
        raise ValueError(
            'localized Scully core radius squared is outside the finite numerical range'
        )
    h_inf = D * (a2 + R2) / R2
    h_bar = h_inf * (1.0 - (a2 / R2) * math.log1p(R2 / a2))
    omega_c = math.sqrt(2.0 * g * h_inf / a2)
    if not all(np.isfinite(q) for q in (h_inf, h_bar, omega_c)):
        raise ValueError(
            'localized Scully derived geometry or angular velocity is non-finite'
        )
    return dict(
        center_z=zmean - h_bar,
        wall_z=zmean - h_bar + D,
        h_inf=h_inf,
        h_bar=h_bar,
        angular_velocity_rad_s=omega_c,
        vtheta_peak_m_s=0.5 * omega_c * a,
        equivalent_rigid_body_rpm=float('nan'),
        equivalent_core_rpm=omega_c * 60.0 / (2.0 * math.pi),
    )


def meridional_velocity_host(
    r_m,
    z_m,
    zmean,
    zmin,
    radius,
    delta_h,
    core_radius,
    circulation_ratio,
    gravity_m_s2=9.81,
    free_surface_model='localized_scully_vortex',
):
    """Axisymmetric, incompressible secondary-circulation reference field.

    The streamfunction is

        psi = -A r^2 (1-r^2/R^2)^2 eta(1-eta),
        eta = (z-z_bottom)/(z_surface(r)-z_bottom),

    with ``A = 2 U_m``.  ``U_m`` is tied to the selected vortex rather than an
    arbitrary absolute speed: ``U_m = chi * max(v_theta)``.  Positive ``chi`` gives outward flow near
    the bottom, upward return in the outer annulus, inward flow near the free
    surface and downward axial return.  Because psi is constant on r=0, r=R,
    z=z_bottom and z=z_surface(r), the field is incompressible and tangent to
    all vessel boundaries in the continuous model.
    """
    r = np.asarray(r_m, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    R = float(radius)
    a = float(core_radius)
    chi = float(circulation_ratio)
    g = float(gravity_m_s2)
    model = str(free_surface_model).strip().lower()
    D = float(delta_h)
    if (
        R <= 0
        or chi < 0
        or g <= 0
        or model not in {'flat', 'parabolic_vortex', 'localized_scully_vortex'}
    ):
        raise ValueError('invalid meridional-circulation parameters')
    if model == 'localized_scully_vortex' and a <= 0:
        raise ValueError(
            'localized Scully meridional circulation requires a positive core radius'
        )
    vg = vortex_surface_geometry(model, float(zmean), R, D, a, g)
    omega = float(vg['angular_velocity_rad_s'])
    vtheta_peak = float(vg['vtheta_peak_m_s'])
    U = chi * vtheta_peak
    Apsi = 2.0 * U
    rr = np.clip(r, 0.0, R)
    rr2 = rr * rr
    a2 = a * a
    if model == 'parabolic_vortex' and D > 0.0:
        zsurf = float(zmean) + D * (rr2 / (R * R) - 0.5)
        hp = 2.0 * D * rr / (R * R)
    elif model == 'localized_scully_vortex' and D > 0.0:
        zsurf = float(zmean) - vg['h_bar'] + vg['h_inf'] * rr2 / (a2 + rr2)
        hp = 2.0 * vg['h_inf'] * a2 * rr / np.square(a2 + rr2)
    else:
        zsurf = np.full_like(rr, float(zmean))
        hp = np.zeros_like(rr)
    H = np.maximum(zsurf - float(zmin), 1e-15)
    eta = np.clip((z - float(zmin)) / H, 0.0, 1.0)
    s = (rr / R) ** 2
    f = eta * (1.0 - eta)
    fp = 1.0 - 2.0 * eta
    ur = Apsi * rr * np.square(1.0 - s) * fp / H
    uz = -Apsi * (
        2.0 * (1.0 - s) * (1.0 - 3.0 * s) * f
        - rr * np.square(1.0 - s) * fp * eta * hp / H
    )
    return (
        ur,
        uz,
        dict(
            free_surface_model=model,
            vtheta_peak_m_s=vtheta_peak,
            meridional_peak_speed_m_s=U,
            streamfunction_velocity_amplitude_m_s=Apsi,
            angular_velocity_rad_s=omega,
            omega_core_rad_s=(
                omega if model == 'localized_scully_vortex' else float('nan')
            ),
        ),
    )


def _schiller_naumann_velocity_host(
    stokes_velocity_m_s, diameter_m, water_density_kg_m3, water_dynamic_viscosity_pa_s
):
    """Solve the Schiller-Naumann corrected slip speed for a Stokes driving speed.

    The monotone equation is ``w*(1+0.15*Re(w)**0.687)=w_stokes``.  Newton's
    method is used with the Stokes value as an upper-bound initial estimate.
    The returned drag factor is ``w_stokes/w`` and tends exactly to one as
    Reynolds number tends to zero.
    """
    v0 = np.asarray(stokes_velocity_m_s, dtype=np.float64)
    d = np.asarray(diameter_m, dtype=np.float64)
    rho = float(water_density_kg_m3)
    mu = float(water_dynamic_viscosity_pa_s)
    if rho <= 0 or mu <= 0:
        raise ValueError('water density and viscosity must be positive')
    v = np.maximum(v0, 0.0).copy()
    K = rho * d / mu
    for _ in range(24):
        Re = np.maximum(K * v, 0.0)
        corr = 0.15 * np.power(Re, 0.687, where=Re > 0, out=np.zeros_like(Re))
        f = v * (1.0 + corr) - v0
        df = 1.0 + 1.687 * corr
        vn = np.maximum(v - f / df, 0.0)
        if np.all(np.abs(vn - v) <= 2e-14 * np.maximum(1.0, np.abs(v))):
            v = vn
            break
        v = vn
    Re = np.maximum(K * v, 0.0)
    drag = 1.0 + 0.15 * np.power(Re, 0.687, where=Re > 0, out=np.zeros_like(Re))
    return v, Re, drag


def sediment_transport_coefficients_host(
    diameter_m,
    zmean,
    zmin,
    radius,
    delta_h,
    core_radius,
    eddy_diffusivity_m2_s,
    particle_density_kg_m3=2600.0,
    water_density_kg_m3=997.0,
    water_dynamic_viscosity_pa_s=8.9e-4,
    gravity_m_s2=9.81,
    quadrature_points=8192,
    radial_lookup_points=2049,
    transport_model='schiller_naumann_drift_diffusion',
    free_surface_model='localized_scully_vortex',
):
    """Independent host reference for stationary sediment coefficients.

    ``uniform`` gives a homogeneous suspension. ``stokes_drift_diffusion``
    uses Stokes slip; ``schiller_naumann_drift_diffusion`` includes finite-Re
    drag. The meridional model uses the finite-Re coefficients as inputs,
    but its 2-D advection-diffusion field is solved on CUDA and is not returned
    by this host helper.
    """
    d = np.asarray(diameter_m, dtype=np.float64)
    tm = str(transport_model).strip().lower()
    if tm == 'uniform':
        one = np.ones_like(d)
        zero = np.zeros_like(d)
        return dict(
            response_time_s=zero,
            settling_velocity_m_s=zero,
            settling_reynolds=zero,
            settling_drag_factor=one,
            radial_log_coefficient=zero,
            vertical_inverse_length_m=zero,
            normalization_scaled=one,
            max_density_scale=one,
            radial_max_slip_velocity_m_s=zero,
            radial_max_reynolds=zero,
            radial_center_log_factor=zero,
            radial_log_lookup=np.zeros((d.size, 2)),
            radial_lookup_r_m=np.array([0.0, float(radius)]),
            omega_rad_s=0.0,
        )
    if tm not in {
        'stokes_drift_diffusion',
        'schiller_naumann_drift_diffusion',
        'schiller_naumann_meridional_advection_diffusion',
    }:
        raise ValueError(f'unknown sediment transport model {transport_model!r}')
    tm_base = (
        'schiller_naumann_drift_diffusion'
        if tm == 'schiller_naumann_meridional_advection_diffusion'
        else tm
    )
    fm = str(free_surface_model).strip().lower()
    if fm not in {'flat', 'parabolic_vortex', 'localized_scully_vortex'}:
        raise ValueError(f'unknown free-surface model {free_surface_model!r}')
    Dt = float(eddy_diffusivity_m2_s)
    rho_p = float(particle_density_kg_m3)
    rho_w = float(water_density_kg_m3)
    mu = float(water_dynamic_viscosity_pa_s)
    g = float(gravity_m_s2)
    R = float(radius)
    a = float(core_radius)
    if not (Dt > 0 and rho_p > rho_w > 0 and mu > 0 and g > 0 and R > 0):
        raise ValueError('invalid drift-diffusion physical parameters')
    if fm == 'localized_scully_vortex' and not (a > 0):
        raise ValueError('localized Scully transport requires a positive core radius')
    D = float(delta_h)
    vg = vortex_surface_geometry(fm, float(zmean), R, D, a, g)
    if fm == 'parabolic_vortex' and D > 0.0:
        omega2 = 2.0 * g * D / (R * R)
    elif fm == 'localized_scully_vortex' and D > 0.0:
        omega2 = 2.0 * g * vg['h_inf'] / (a * a)
    else:
        omega2 = 0.0
    tau = rho_p * d * d / (18.0 * mu)
    ws_stokes = (rho_p - rho_w) * g * d * d / (18.0 * mu)
    nr = max(65, int(radial_lookup_points))
    rnodes = np.linspace(0.0, R, nr, dtype=np.float64)
    a2 = a * a if a > 0.0 else 1.0
    accel_r = np.zeros_like(rnodes)
    if fm == 'parabolic_vortex' and omega2 > 0:
        accel_r = omega2 * rnodes
    elif fm == 'localized_scully_vortex' and omega2 > 0:
        accel_r = omega2 * rnodes / np.square(1.0 + (rnodes / a) ** 2)
    w0 = tau[:, None] * accel_r[None, :]
    fR = R * R / (a2 + R * R) if fm == 'localized_scully_vortex' else R * R
    if tm_base == 'stokes_drift_diffusion':
        ws = ws_stokes.copy()
        sett_re = rho_w * d * ws / mu
        sett_drag = np.ones_like(d)
        wr = w0
        radial_re = rho_w * d[:, None] * wr / mu
        if fm == 'localized_scully_vortex':
            Aeq = tau * omega2 * a2 / (2.0 * Dt)
            fnode = rnodes * rnodes / (a2 + rnodes * rnodes)
        else:
            Aeq = tau * omega2 / (2.0 * Dt)
            fnode = rnodes * rnodes
        radial_log = Aeq[:, None] * (fnode[None, :] - fR)
        radial_log[:, -1] = 0.0
    else:
        ws, sett_re, sett_drag = _schiller_naumann_velocity_host(
            ws_stokes, d, rho_w, mu
        )
        wr, radial_re, _ = _schiller_naumann_velocity_host(w0, d[:, None], rho_w, mu)
        dr = R / (nr - 1)
        cum = np.zeros_like(wr)
        if nr > 1:
            cum[:, 1:] = np.cumsum(0.5 * (wr[:, 1:] + wr[:, :-1]) * dr, axis=1)
        radial_log = (cum - cum[:, -1, None]) / Dt
        radial_log[:, -1] = 0.0
        Aeq = np.where(fR > 0, -radial_log[:, 0] / fR, 0.0)
    B = ws / Dt
    nquad = max(64, int(quadrature_points))
    dq = R / nquad
    rr = (np.arange(nquad, dtype=np.float64) + 0.5) * dq
    if tm_base == 'stokes_drift_diffusion':
        fmid = rr * rr / (a2 + rr * rr) if fm == 'localized_scully_vortex' else rr * rr
        radial_mid = Aeq[:, None] * (fmid[None, :] - fR)
    else:
        # interpolate the finite-Re radial potential onto the volume quadrature
        radial_mid = np.vstack(
            [np.interp(rr, rnodes, radial_log[i]) for i in range(d.size)]
        )
    Hinf = vg['h_inf']
    hbar = vg['h_bar']
    if fm == 'parabolic_vortex' and D > 0.0:
        zsurf = float(zmean) + D * (rr * rr / (R * R) - 0.5)
    elif fm == 'localized_scully_vortex' and D > 0.0:
        zsurf = float(zmean) - hbar + Hinf * rr * rr / (a2 + rr * rr)
    else:
        zsurf = np.full_like(rr, float(zmean))
    H = np.maximum(zsurf - float(zmin), 0.0)
    Hmean = float(zmean) - float(zmin)
    norm = np.empty_like(d)
    for i in range(d.size):
        radial = np.exp(np.minimum(radial_mid[i], 0.0))
        if abs(B[i]) < 1e-14:
            iz = radial * H
        else:
            iz = radial * (-np.expm1(-B[i] * H)) / B[i]
        norm[i] = (2.0 / (R * R * Hmean)) * np.sum(rr * iz) * dq
    return dict(
        response_time_s=tau,
        settling_velocity_m_s=ws,
        settling_reynolds=sett_re,
        settling_drag_factor=sett_drag,
        radial_log_coefficient=Aeq,
        vertical_inverse_length_m=B,
        normalization_scaled=norm,
        max_density_scale=1.0 / norm,
        radial_max_slip_velocity_m_s=np.max(wr, axis=1),
        radial_max_reynolds=np.max(radial_re, axis=1),
        radial_center_log_factor=radial_log[:, 0],
        radial_log_lookup=radial_log,
        radial_lookup_r_m=rnodes,
        omega_rad_s=math.sqrt(max(omega2, 0.0)),
        free_surface_model=fm,
    )


def get_material_optical_constants(
    material: str, profile_override=None, n_fallback=1.59
):
    m = material.strip().lower()
    if m == 'loess':
        d = LOESS_DIAMETER_M
        n = DEFAULT_LOESS_N_REAL.copy()
        k = DEFAULT_LOESS_K_IMAG.copy()
    elif m == 'kaolin':
        d = KAOLIN_DIAMETER_M
        n = DEFAULT_KAOLIN_N_REAL.copy()
        k = DEFAULT_KAOLIN_K_IMAG.copy()
    else:
        raise ValueError("material must be 'loess' or 'kaolin'")
    if profile_override:
        q = profile_override.get(m, profile_override.get(material, None))
        if q is not None:
            n = np.asarray(q.get('n_real', n), dtype=np.float64)
            k = np.asarray(q.get('k_imag', k), dtype=np.float64)
    if n.size != d.size or k.size != d.size:
        raise ValueError(
            f"{m} complex-optics profile must contain {d.size} n and k values"
        )
    if (
        np.any(~np.isfinite(n))
        or np.any(~np.isfinite(k))
        or np.any(n <= 0)
        or np.any(k < 0)
    ):
        raise ValueError(f"{m} complex-optics profile contains invalid n/k values")
    return n, k


CUDA_SRC = r'''extern "C" {

#define INF_F 3.402823466e+38F
#define PI_F 3.14159265358979323846f
#define EPS_F 2.0e-7f

__device__ unsigned int xorshift32_state(unsigned int* state) {
    unsigned int x = *state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *state = x; return x;
}

__device__ float rnd_uniform(unsigned int* state) {
    return (float)(xorshift32_state(state) * 2.3283064e-10f);
}


#define PI_D 3.141592653589793238462643383279502884
#define MIE_MAX_ORDER 4096

struct cpxd { double re; double im; };
__device__ cpxd CD(double re,double im){cpxd z;z.re=re;z.im=im;return z;}
__device__ cpxd cd_add(cpxd a,cpxd b){return CD(a.re+b.re,a.im+b.im);}
__device__ cpxd cd_sub(cpxd a,cpxd b){return CD(a.re-b.re,a.im-b.im);}
__device__ cpxd cd_mul(cpxd a,cpxd b){return CD(a.re*b.re-a.im*b.im,a.re*b.im+a.im*b.re);}
__device__ cpxd cd_scale(cpxd a,double s){return CD(a.re*s,a.im*s);}
__device__ cpxd cd_div(cpxd a,cpxd b){
    double d=b.re*b.re+b.im*b.im;if(d<1e-300)d=1e-300;
    return CD((a.re*b.re+a.im*b.im)/d,(a.im*b.re-a.re*b.im)/d);
}
__device__ double cd_abs2(cpxd a){return a.re*a.re+a.im*a.im;}
__device__ double cd_re_mul_conj(cpxd a,cpxd b){return a.re*b.re+a.im*b.im;}

__device__ float normal_box_muller(unsigned int* state){
    float u1=fmaxf(rnd_uniform(state),1e-12f),u2=rnd_uniform(state);
    return sqrtf(-2.0f*logf(u1))*cosf(2.0f*PI_F*u2);
}

__device__ float gamma_shape_sample(unsigned int* state,float shape){
    if(shape<=0.0f) return 0.0f;
    if(fabsf(shape-1.0f)<1e-7f) return -logf(fmaxf(rnd_uniform(state),1e-12f));
    float correction=1.0f;
    float a=shape;
    if(a<1.0f){
        float u=fmaxf(rnd_uniform(state),1e-12f);
        correction=powf(u,1.0f/a);
        a+=1.0f;
    }
    float d=a-0.3333333333333333f;
    float c=rsqrtf(9.0f*d);
    for(int it=0;it<128;++it){
        float x=normal_box_muller(state);
        float v=1.0f+c*x;
        if(v<=0.0f) continue;
        v=v*v*v;
        float u=rnd_uniform(state);
        if(u<1.0f-0.0331f*x*x*x*x) return correction*d*v;
        if(logf(fmaxf(u,1e-12f))<0.5f*x*x+d*(1.0f-v+logf(v))) return correction*d*v;
    }
    return correction*d;
}

__device__ float beta_sample(unsigned int* state,float a,float b){
    if(a<=0.0f||b<=0.0f) return 0.0f;
    // Exact inverse forms cover CLARITAS' default Beta(1,100) source efficiently.
    if(fabsf(a-1.0f)<1e-7f){
        float u=fminf(fmaxf(rnd_uniform(state),1e-12f),1.0f-1e-7f);
        return 1.0f-powf(1.0f-u,1.0f/b);
    }
    if(fabsf(b-1.0f)<1e-7f){
        float u=fminf(fmaxf(rnd_uniform(state),1e-12f),1.0f-1e-7f);
        return powf(u,1.0f/a);
    }
    float ga=gamma_shape_sample(state,a),gb=gamma_shape_sample(state,b);
    float s=ga+gb;return s>0.0f?ga/s:0.0f;
}

// The Beta law specifies angular radiance I(theta). In 3-D, polar-angle
// probability includes sin(theta), since dOmega = sin(theta) dtheta dphi.
// With u=theta/(pi/2), sample BetaPDF(u;a,b)*sin(pi*u/2) using
// q(u)=BetaPDF(u;a+1,b) and acceptance sin(c*u)/(c*u), c=pi/2.
// Acceptance is bounded below by 2/pi.
__device__ float beta_radiance_polar_sample(unsigned int* state,float a,float b){
    const float c=0.5f*PI_F;
    for(int it=0;it<128;++it){
        float u=beta_sample(state,a+1.0f,b);
        u=fminf(fmaxf(u,0.0f),1.0f);
        float x=c*u;
        float accept=(x<1.0e-7f)?1.0f:sinf(x)/x;
        if(rnd_uniform(state)<=accept) return x;
    }
    float u=beta_sample(state,a+1.0f,b);
    return c*fminf(fmaxf(u,0.0f),1.0f);
}

__device__ float dot3(float ax,float ay,float az,float bx,float by,float bz) {
    return ax*bx + ay*by + az*bz;
}

__device__ void normalize3(float* x,float* y,float* z) {
    float q=(*x)*(*x)+(*y)*(*y)+(*z)*(*z);
    if(q<=0.0f){*x=0.0f;*y=1.0f;*z=0.0f;return;}
    float s=rsqrtf(q); *x*=s;*y*=s;*z*=s;
}

__device__ void reflect3(float ix,float iy,float iz,float nx,float ny,float nz,
                         float* ox,float* oy,float* oz) {
    float d=dot3(ix,iy,iz,nx,ny,nz);
    *ox=ix-2.0f*d*nx; *oy=iy-2.0f*d*ny; *oz=iz-2.0f*d*nz;
    normalize3(ox,oy,oz);
}

__device__ int refract3(float ix,float iy,float iz,
                        float nx,float ny,float nz,
                        float n1,float n2,
                        float* ox,float* oy,float* oz) {
    // n points into the incident medium.
    float ci=-dot3(ix,iy,iz,nx,ny,nz);
    ci=fminf(fmaxf(ci,0.0f),1.0f);
    float eta=n1/n2;
    float k=1.0f-eta*eta*(1.0f-ci*ci);
    if(k<=0.0f) return 0;
    float ct=sqrtf(k);
    *ox=eta*ix+(eta*ci-ct)*nx;
    *oy=eta*iy+(eta*ci-ct)*ny;
    *oz=eta*iz+(eta*ci-ct)*nz;
    normalize3(ox,oy,oz); return 1;
}

__device__ float fresnel_R(float ci,float n1,float n2,int* tir) {
    ci=fminf(fmaxf(ci,0.0f),1.0f);
    float eta=n1/n2;
    float st2=eta*eta*fmaxf(0.0f,1.0f-ci*ci);
    if(st2>=1.0f){*tir=1;return 1.0f;}
    *tir=0;
    float ct=sqrtf(fmaxf(0.0f,1.0f-st2));
    float ds=n1*ci+n2*ct, dp=n1*ct+n2*ci;
    float rs=1.0f,rp=1.0f;
    if(fabsf(ds)>1e-12f){float a=(n1*ci-n2*ct)/ds;rs=a*a;}
    if(fabsf(dp)>1e-12f){float a=(n1*ct-n2*ci)/dp;rp=a*a;}
    return fminf(fmaxf(0.5f*(rs+rp),0.0f),1.0f);
}


// Nylon collimator and inner-ring-face models: 0=perfect absorption;
// 1=smooth opaque dielectric with unpolarised Fresnel reflection.
// Transmission terminates in the bulk ring; no diffuse-albedo multiplier is used.
__device__ float bore_side_hit(float q,float w,float vq,float vw,float radius){
    float A=vq*vq+vw*vw;
    if(A<=1.0e-20f)return INF_F;
    float B=2.0f*(q*vq+w*vw);
    float C=q*q+w*w-radius*radius;
    float D=B*B-4.0f*A*C;
    if(D<0.0f)return INF_F;
    float root=sqrtf(fmaxf(D,0.0f));
    float t1=(-B-root)/(2.0f*A),t2=(-B+root)/(2.0f*A);
    float t=INF_F;
    if(t1>EPS_F)t=t1;
    if(t2>EPS_F && t2<t)t=t2;
    return t;
}

__device__ int bore_wall_reflect(
    unsigned int* state,const int surface_model,const float n_air,const float n_bore,
    float q,float w,float* vs,float* vq,float* vw,int* interactions,int* reflections){
    (*interactions)++;
    if(surface_model==0)return 0;
    float rr=sqrtf(fmaxf(q*q+w*w,1.0e-30f));
    // Normal points from solid into the incident air cavity.
    float ns=0.0f,nq=-q/rr,nw=-w/rr;
    float ci=fminf(fmaxf(-((*vs)*ns+(*vq)*nq+(*vw)*nw),0.0f),1.0f);
    int tir=0;float R=fresnel_R(ci,n_air,n_bore,&tir);
    if(rnd_uniform(state)>=R)return 0; // transmitted into thick nylon -> loss
    float os,oq,ow;reflect3(*vs,*vq,*vw,ns,nq,nw,&os,&oq,&ow);
    *vs=os;*vq=oq;*vw=ow;(*reflections)++;
    return 1;
}

__device__ int trace_bore_segment(
    unsigned int* state,const int surface_model,const float n_air,const float n_bore,const int max_bounces,
    float s_end,const float radius,float* s,float* q,float* w,float* vs,float* vq,float* vw,
    int* interactions,int* reflections){
    for(int b=0;b<=max_bounces;++b){
        if(*vs<=1.0e-12f)return 0;
        float tp=(s_end-*s)/(*vs);
        if(tp<-EPS_F)return 0;
        if(tp<0.0f)tp=0.0f;
        float twall=bore_side_hit(*q,*w,*vq,*vw,radius);
        if(tp<=twall){
            *q+=tp*(*vq);*w+=tp*(*vw);*s=s_end;
            return ((*q)*(*q)+(*w)*(*w)<=radius*radius*(1.0f+2.0e-5f));
        }
        if(twall>=INF_F*0.5f)return 0;
        *q+=twall*(*vq);*w+=twall*(*vw);*s+=twall*(*vs);
        if(!bore_wall_reflect(state,surface_model,n_air,n_bore,*q,*w,vs,vq,vw,interactions,reflections))return 0;
        // Move back into the air cavity after the reflection.
        float rr=sqrtf(fmaxf((*q)*(*q)+(*w)*(*w),1.0e-30f));
        *q-=EPS_F*(*q)/rr;*w-=EPS_F*(*w)/rr;
    }
    return 0;
}

__device__ int source_collimator_transport(
    unsigned int* state,const int surface_model,const float n_air,const float n_bore,const int max_bounces,
    const float ring_in,const float throat_out,const float source_r,const float throat_r,const float counterbore_r,
    float* x,float* y,float* z,float* vx,float* vy,float* vz,int* interactions,int* reflections){
    // Local source axis is +y (toward the cell); q=x,w=z.
    if(*vy<=1.0e-12f)return 0;
    float s=*y,q=*x,w=*z,vs=*vy,vq=*vx,vw=*vz;
    if(q*q+w*w>counterbore_r*counterbore_r)return 0;
    // Traverse only the portion of the counterbore between the actual luminous
    // plane and the shoulder.  SOURCE_R may equal THROAT_OUT_R.
    float shoulder=-throat_out;
    if(s<shoulder-EPS_F){
        if(!trace_bore_segment(state,surface_model,n_air,n_bore,max_bounces,shoulder,counterbore_r,
                               &s,&q,&w,&vs,&vq,&vw,interactions,reflections))return 0;
    } else if(s>shoulder+EPS_F){
        return 0;
    }
    // The annular shoulder is opaque.  A ray must geometrically enter the 4-mm throat.
    if(q*q+w*w>throat_r*throat_r)return 0;
    if(!trace_bore_segment(state,surface_model,n_air,n_bore,max_bounces,-ring_in,throat_r,
                           &s,&q,&w,&vs,&vq,&vw,interactions,reflections))return 0;
    *x=q;*y=s;*z=w;*vx=vq;*vy=vs;*vz=vw;
    return 1;
}

__device__ int detector_collimator_path(
    unsigned int* state,float x,float y,float z,float vx,float vy,float vz,int mirror,
    const float ring_in,const float throat_out,const float ring_out,const float throat_r,const float counterbore_r,
    const int surface_model,const float n_air,const float n_bore,const int max_bounces,
    int* bounce_count){
    if(mirror){x=-x;vx=-vx;}
    float best_t=INF_F;int best=-1;
    // First identify the physically first detector entrance aperture reached.
    for(int j=0;j<18;++j){
        float th=(10.0f*(float)j)*(PI_F/180.0f),ux=sinf(th),uy=cosf(th);
        float den=vx*ux+vy*uy;if(den<=1.0e-10f)continue;
        float pdot=x*ux+y*uy;float t=(ring_in-pdot)/den;if(t<=0.0f)continue;
        float etx=cosf(th),ety=-sinf(th);
        float hx=x+t*vx,hy=y+t*vy,hz=z+t*vz;
        float q=hx*etx+hy*ety,w=hz;
        if(q*q+w*w<=throat_r*throat_r && t<best_t){best_t=t;best=j;}
    }
    if(best<0)return -1;
    float th=(10.0f*(float)best)*(PI_F/180.0f),ux=sinf(th),uy=cosf(th),etx=cosf(th),ety=-sinf(th);
    float hx=x+best_t*vx,hy=y+best_t*vy;
    float s=ring_in,q=hx*etx+hy*ety,w=z+best_t*vz;
    float vs=vx*ux+vy*uy,vq=vx*etx+vy*ety,vw=vz;
    int interactions=0,reflections=0;
    if(!trace_bore_segment(state,surface_model,n_air,n_bore,max_bounces,throat_out,throat_r,
                           &s,&q,&w,&vs,&vq,&vw,&interactions,&reflections))return -1;
    if(!trace_bore_segment(state,surface_model,n_air,n_bore,max_bounces,ring_out,counterbore_r,
                           &s,&q,&w,&vs,&vq,&vw,&interactions,&reflections))return -1;
    *bounce_count=reflections;return best;
}

// Lightweight complex arithmetic for absorbing-particle Fresnel coefficients.
struct cpx { float re; float im; };
__device__ cpx C(float re,float im){cpx z;z.re=re;z.im=im;return z;}
__device__ cpx cadd(cpx a,cpx b){return C(a.re+b.re,a.im+b.im);}
__device__ cpx csub(cpx a,cpx b){return C(a.re-b.re,a.im-b.im);}
__device__ cpx cmul(cpx a,cpx b){return C(a.re*b.re-a.im*b.im,a.re*b.im+a.im*b.re);}
__device__ cpx cscale(cpx a,float s){return C(a.re*s,a.im*s);}
__device__ cpx cdiv(cpx a,cpx b){
    float d=b.re*b.re+b.im*b.im;
    if(d<1e-30f)d=1e-30f;
    return C((a.re*b.re+a.im*b.im)/d,(a.im*b.re-a.re*b.im)/d);
}
__device__ float cabs2(cpx a){return a.re*a.re+a.im*a.im;}
__device__ cpx csqrtp(cpx z){
    float r=sqrtf(z.re*z.re+z.im*z.im);
    float u=sqrtf(fmaxf(0.0f,0.5f*(r+z.re)));
    float v=sqrtf(fmaxf(0.0f,0.5f*(r-z.re)));
    if(z.im<0.0f)v=-v;
    // choose branch with non-negative real part; for exactly-zero real part choose non-negative imag
    if(u<0.0f || (fabsf(u)<1e-20f && v<0.0f)){u=-u;v=-v;}
    return C(u,v);
}

// Unpolarised Fresnel reflectance with possibly complex incident/transmitted indices.
// Geometry/direction still uses real phase indices separately.
__device__ float fresnel_R_complex(float ci,float n1r,float n1k,float n2r,float n2k){
    ci=fminf(fmaxf(ci,0.0f),1.0f);
    float si2=fmaxf(0.0f,1.0f-ci*ci);
    cpx N1=C(n1r,n1k), N2=C(n2r,n2k);
    cpx eta=cdiv(N1,N2);
    cpx eta2=cmul(eta,eta);
    cpx one_minus=csub(C(1.0f,0.0f),cscale(eta2,si2));
    cpx ct=csqrtp(one_minus);
    cpx N1ci=cscale(N1,ci), N2ci=cscale(N2,ci);
    cpx N2ct=cmul(N2,ct), N1ct=cmul(N1,ct);
    cpx rs=cdiv(csub(N1ci,N2ct),cadd(N1ci,N2ct));
    cpx rp=cdiv(csub(N2ci,N1ct),cadd(N2ci,N1ct));
    float R=0.5f*(cabs2(rs)+cabs2(rp));
    return fminf(fmaxf(R,0.0f),1.0f);
}

__device__ float cylinder_hit(float x,float y,float vx,float vy,float R) {
    float a=vx*vx+vy*vy;
    if(a<1e-18f) return INF_F;
    float b=x*vx+y*vy;
    float c=x*x+y*y-R*R;
    float d=b*b-a*c;
    if(d<0.0f) return INF_F;
    float sd=sqrtf(fmaxf(d,0.0f));
    float t1=(-b-sd)/a, t2=(-b+sd)/a;
    float t=INF_F;
    if(t1>EPS_F) t=t1;
    if(t2>EPS_F && t2<t) t=t2;
    return t;
}

// At the top of the solid bottom disc, the water aperture is strictly r<Rin.
// Rin<=r<=Rout is contiguous acrylic sidewall, not an optical interface.
__device__ int bottom_disc_topology_at_top(float x,float y,float Rin,float Rout){
    double r2=(double)x*(double)x+(double)y*(double)y;
    double ri2=(double)Rin*(double)Rin,ro2=(double)Rout*(double)Rout;
    if(r2<ri2)return 1;
    if(r2<=ro2*(1.0+8.0*2.220446049250313080847263336181640625e-16))return 2;
    return 0;
}

__device__ void radial_normal(float x,float y,float* nx,float* ny) {
    float r=sqrtf(fmaxf(x*x+y*y,1e-30f)); *nx=x/r; *ny=y/r;
}


// FREE_SURFACE_MODEL: 0=flat, 1=volume-preserving paraboloid,
// 2=localized volume-preserving Scully vortex.
// DELTA_H is wall minus centre height; CORE_R is the Scully core radius a.
__device__ float localized_scully_hinf(float R,float DELTA_H,float CORE_R){
    float a2=CORE_R*CORE_R, R2=R*R;
    return DELTA_H*(a2+R2)/R2;
}

__device__ float localized_scully_hbar(float R,float DELTA_H,float CORE_R){
    float a2=CORE_R*CORE_R, R2=R*R;
    float H=localized_scully_hinf(R,DELTA_H,CORE_R);
    return H*(1.0f-(a2/R2)*log1pf(R2/a2));
}

__device__ float free_surface_z(float x,float y,float ZMEAN,float R,int MODEL,float DELTA_H,float CORE_R){
    if(MODEL==0 || DELTA_H<=0.0f) return ZMEAN;
    float q=x*x+y*y;
    if(MODEL==1){
        return ZMEAN + DELTA_H*(q/(R*R)-0.5f);
    }
    float a2=CORE_R*CORE_R;
    float H=localized_scully_hinf(R,DELTA_H,CORE_R);
    float hbar=localized_scully_hbar(R,DELTA_H,CORE_R);
    return ZMEAN - hbar + H*q/(a2+q);
}


// size-resolved stirred-sediment field.
// TRANSPORT_MODEL: 0=uniform, 1=Stokes drift-diffusion,
// 2=Schiller-Naumann finite-Re drift-diffusion,
// 3=finite-Re + explicit incompressible meridional advection-diffusion.
// Model 3 samples a CUDA-solved axisymmetric r-z finite-volume field.  The
// rectangular solver grid is cell-centred; inactive cells above the curved
// free surface are extended from the nearest liquid cell only for interpolation.
__device__ double sediment_grid_sample(
    const int pidx,const double* GRID,const int NR,const int NZ,
    const float x,const float y,const float z,const float R,const float ZMIN,const float ZGRIDMAX){
    if(NR<2 || NZ<2 || !(ZGRIDMAX>ZMIN)) return 1.0;
    double rr=sqrt(fmax((double)x*(double)x+(double)y*(double)y,0.0));
    if(rr<0.0)rr=0.0;if(rr>(double)R)rr=(double)R;
    double dr=(double)R/(double)NR,dz=((double)ZGRIDMAX-(double)ZMIN)/(double)NZ;
    double ur=rr/dr-0.5,uz=((double)z-(double)ZMIN)/dz-0.5;
    if(ur<0.0)ur=0.0;if(uz<0.0)uz=0.0;
    if(ur>(double)(NR-1))ur=(double)(NR-1);
    if(uz>(double)(NZ-1))uz=(double)(NZ-1);
    int i=(int)floor(ur),k=(int)floor(uz);double tr=ur-(double)i,tz=uz-(double)k;
    if(i>=NR-1){i=NR-2;tr=1.0;}if(k>=NZ-1){k=NZ-2;tz=1.0;}
    long long plane=(long long)NR*(long long)NZ;
    long long b=(long long)pidx*plane;
    double q00=GRID[b+(long long)i*NZ+k];
    double q10=GRID[b+(long long)(i+1)*NZ+k];
    double q01=GRID[b+(long long)i*NZ+k+1];
    double q11=GRID[b+(long long)(i+1)*NZ+k+1];
    double q0=(1.0-tr)*q00+tr*q10,q1=(1.0-tr)*q01+tr*q11;
    return fmax((1.0-tz)*q0+tz*q1,0.0);
}

__device__ double sediment_local_scale(
    const int TRANSPORT_MODEL,const int FREE_SURFACE_MODEL,const int pidx,const double* SED_A,const double* SED_B,
    const double* SED_NORM_SCALED,const double* SED_RADIAL_LOG,const int N_RADIAL,
    const double* SED_GRID,const int GRID_NR,const int GRID_NZ,const float GRID_ZMAX,
    const float x,const float y,const float z,const float R,const float ZMIN,const float CORE_R){
    if(TRANSPORT_MODEL==0) return 1.0;
    if(TRANSPORT_MODEL==3)
        return sediment_grid_sample(pidx,SED_GRID,GRID_NR,GRID_NZ,x,y,z,R,ZMIN,GRID_ZMAX);
    double rr2=(double)x*(double)x+(double)y*(double)y;
    double R2=(double)R*(double)R;if(rr2>R2)rr2=R2;
    double radial_log=0.0;
    if(TRANSPORT_MODEL==1){
        if(FREE_SURFACE_MODEL==1){
            radial_log=SED_A[pidx]*(rr2-R2);
        }else{
            double a2=(double)CORE_R*(double)CORE_R;
            double f=rr2/(a2+rr2),fR=R2/(a2+R2);
            radial_log=SED_A[pidx]*(f-fR);
        }
    } else {
        double rr=sqrt(fmax(rr2,0.0));
        double u=(R>0.0)?rr/(double)R*(double)(N_RADIAL-1):0.0;
        int j=(int)floor(u);if(j<0)j=0;if(j>N_RADIAL-2)j=N_RADIAL-2;
        double t=u-(double)j;
        long long base=(long long)pidx*(long long)N_RADIAL;
        radial_log=(1.0-t)*SED_RADIAL_LOG[base+j]+t*SED_RADIAL_LOG[base+j+1];
        if(radial_log>0.0)radial_log=0.0;
    }
    double h=fmax((double)z-(double)ZMIN,0.0);
    double logrel=radial_log-SED_B[pidx]*h;
    double ns=SED_NORM_SCALED[pidx];
    if(!(ns>0.0)) return 0.0;
    return exp(logrel)/ns;
}

__device__ int sample_local_mu_bin(
    unsigned int* state,const int TRANSPORT_MODEL,const int FREE_SURFACE_MODEL,const double* MU_BIN,const int n_particles,
    const double* SED_A,const double* SED_B,const double* SED_NORM_SCALED,const double* SED_RADIAL_LOG,const int N_RADIAL,
    const double* SED_GRID,const int GRID_NR,const int GRID_NZ,const float GRID_ZMAX,
    const float x,const float y,const float z,const float R,const float ZMIN,const float CORE_R,
    const double local_total){
    if(!(local_total>0.0))return 0;
    double target=(double)rnd_uniform(state)*local_total,cum=0.0;
    for(int i=0;i<n_particles;++i){
        double sc=sediment_local_scale(TRANSPORT_MODEL,FREE_SURFACE_MODEL,i,SED_A,SED_B,SED_NORM_SCALED,SED_RADIAL_LOG,N_RADIAL,SED_GRID,GRID_NR,GRID_NZ,GRID_ZMAX,x,y,z,R,ZMIN,CORE_R);
        cum+=MU_BIN[i]*sc;
        if(target<=cum)return i;
    }
    return n_particles>0?n_particles-1:0;
}

__device__ double quadratic_min_positive(double A,double B,double C){
    const double eps=0.0; // : geometric validity is independent of EPS_F displacement.
    if(fabs(A)<1e-24){
        if(fabs(B)<1e-24) return 1.0e300;
        double t=-C/B; return t>eps?t:1.0e300;
    }
    double D=B*B-4.0*A*C;
    if(D<0.0) return 1.0e300;
    double sd=sqrt(fmax(D,0.0));
    // Stable quadratic form, followed by the companion root from C/q.
    double q=-0.5*(B+copysign(sd,B));
    double t1=(fabs(q)>1e-300)?q/A:(-B-sd)/(2.0*A);
    double t2=(fabs(q)>1e-300)?C/q:(-B+sd)/(2.0*A);
    double best=1.0e300;
    if(t1>eps && t1<best)best=t1;
    if(t2>eps && t2<best)best=t2;
    return best;
}

__device__ double cubic_min_positive(double A3,double A2,double A1,double A0){
    const double eps=0.0; // : geometric validity is independent of EPS_F displacement.
    double scale=fmax(fmax(fabs(A3),fabs(A2)),fmax(fabs(A1),fabs(A0)));
    if(scale<=0.0) return 1.0e300;
    // Directions are normalized, so |A3| is O(|vz|). Treat near-horizontal
    // rays as the mathematically limiting quadratic to avoid cubic conditioning loss.
    if(fabs(A3)<=1e-12) return quadratic_min_positive(A2,A1,A0);
    double a=A2/A3,b=A1/A3,c=A0/A3;
    double p=b-a*a/3.0;
    double q=2.0*a*a*a/27.0-a*b/3.0+c;
    double halfq=0.5*q, thirdp=p/3.0;
    double d1=halfq*halfq,d2=thirdp*thirdp*thirdp;
    double disc=d1+d2;
    // Relative discriminant tolerance is essential here: ray-distance roots are
    // O(1e-2..1e-1 m), so an absolute O(1) tolerance would collapse genuinely
    // three-real-root cases into the repeated-root branch.
    double tol=1e-12*(fabs(d1)+fabs(d2)+1e-300);
    double best=1.0e300;
    if(disc>tol){
        double sd=sqrt(disc);
        double u=cbrt(-halfq+sd),v=cbrt(-halfq-sd);
        double t=u+v-a/3.0;
        if(t>eps)best=t;
    }else if(fabs(p)<1e-24 && fabs(q)<1e-24){
        double t=-a/3.0;if(t>eps)best=t;
    }else if(disc>=-tol){
        double u=cbrt(-halfq);
        double t1=2.0*u-a/3.0;
        double t2=-u-a/3.0;
        if(t1>eps && t1<best)best=t1;
        if(t2>eps && t2<best)best=t2;
    }else{
        double m=2.0*sqrt(fmax(-thirdp,0.0));
        double den=sqrt(fmax(-(thirdp*thirdp*thirdp),1e-300));
        double ca=fmin(fmax(-halfq/den,-1.0),1.0);
        double phi=acos(ca);
        for(int k=0;k<3;++k){
            double t=m*cos((phi+2.0*PI_D*(double)k)/3.0)-a/3.0;
            if(t>eps && t<best)best=t;
        }
    }
    return best;
}


// Bounded Scully intersection fallback. Split at the cubic's stationary points
// so each interval is monotone; refine sign-changing roots without Cardano cancellation.
__device__ double scully_bounded_root(double a3,double a2,double a1,double a0,
                                     double lo,double hi){
    if(!(hi>lo) || !isfinite(hi))return 1.0e300;
    double cuts[4];int n=1;cuts[0]=lo;
    double A=3.0*a3,B=2.0*a2,C=a1;
    if(A!=0.0){
        double disc=B*B-4.0*A*C;
        if(disc>=0.0){
            double q=-0.5*(B+copysign(sqrt(disc),B));
            double r1=(q!=0.0)?q/A:-B/(2.0*A);
            double r2=(q!=0.0)?C/q:r1;
            if(r1>lo&&r1<hi)cuts[n++]=r1;
            if(r2>lo&&r2<hi&&r2!=r1)cuts[n++]=r2;
        }
    }else if(B!=0.0){double r=-C/B;if(r>lo&&r<hi)cuts[n++]=r;}
    cuts[n++]=hi;
    for(int i=1;i<n;++i){double v=cuts[i];int j=i;while(j>0&&cuts[j-1]>v){cuts[j]=cuts[j-1];--j;}cuts[j]=v;}
    for(int i=0;i<n-1;++i){
        double l=cuts[i],r=cuts[i+1];
        double fl=((a3*l+a2)*l+a1)*l+a0;
        double fr=((a3*r+a2)*r+a1)*r+a0;
        if(l>0.0&&fl==0.0)return l;
        if((fl<0.0&&fr>0.0)||(fl>0.0&&fr<0.0)){
            for(int j=0;j<64;++j){
                double m=l+0.5*(r-l);if(m==l||m==r)break;
                double fm=((a3*m+a2)*m+a1)*m+a0;
                if(fm==0.0){l=r=m;break;}
                if((fl<0.0&&fm<0.0)||(fl>0.0&&fm>0.0)){l=m;fl=fm;}else r=m;
            }
            return l+0.5*(r-l);
        }
        if(r>0.0&&fr==0.0)return r;
    }
    return 1.0e300;
}

// First positive ray intersection with the selected free surface. For DELTA_H=0
// this is the exact flat expression and consumes no random numbers.
__device__ float free_surface_hit(float x,float y,float z,float vx,float vy,float vz,
                                  float ZMEAN,float R,int MODEL,float DELTA_H,float CORE_R){
    if(MODEL==0 || DELTA_H<=0.0f){
        if(fabsf(vz)<=1e-14f) return INF_F;
        float t=(ZMEAN-z)/vz;
        return (t>0.0f)?t:INF_F;
    }
    if(MODEL==1){
        float aa=DELTA_H/(R*R);
        double A=-(double)aa*((double)vx*vx+(double)vy*vy);
        double B=(double)vz-2.0*(double)aa*((double)x*vx+(double)y*vy);
        double C=(double)z-((double)ZMEAN-0.5*(double)DELTA_H)-(double)aa*((double)x*x+(double)y*y);
        double t=quadratic_min_positive(A,B,C);
        return (t<0.5*(double)INF_F)?(float)t:INF_F;
    }
    // Scully surface: z_s=C+H*q/(a^2+q), q=x^2+y^2.  Multiplication by
    // (a^2+q)>0 produces a cubic with no extraneous real roots.
    double X=x,Y=y,Z=z,VX=vx,VY=vy,VZ=vz;
    double RR=(double)R*(double)R, aa=(double)CORE_R*(double)CORE_R;
    double H=(double)DELTA_H*(aa+RR)/RR;
    double hbar=H*(1.0-(aa/RR)*log1p(RR/aa));
    double Csurf=(double)ZMEAN-hbar;
    double Q2=VX*VX+VY*VY;
    double Q1=2.0*(X*VX+Y*VY);
    double Q0=X*X+Y*Y;
    double L0=Z-Csurf;
    double c3=VZ*Q2;
    double c2=(L0-H)*Q2+VZ*Q1;
    double c1=(L0-H)*Q1+VZ*(aa+Q0);
    double c0=L0*aa+(L0-H)*Q0;
    double t=cubic_min_positive(c3,c2,c1,c0);
    // Validate the analytic root against the surface in physical metres.
    double xx=X+t*VX,yy=Y+t*VY,qq=xx*xx+yy*yy;
    double residual=(Z+t*VZ)-(Csurf+H*qq/(aa+qq));
    double tol=32.0*2.2204460492503131e-16*fmax(fmax(fabs(Z),fabs(Csurf)+fabs(H)),1.0e-3);
    if(!(t<0.5*(double)INF_F) || qq>RR*(1.0+2e-6) || !isfinite(residual) || fabs(residual)>tol){
        if(Q2>0.0){
            double dc=Q1*Q1-4.0*Q2*(Q0-RR);
            if(dc<0.0)return INF_F;
            double q=-0.5*(Q1+copysign(sqrt(dc),Q1));
            double ta=(q!=0.0)?q/Q2:-Q1/(2.0*Q2);
            double tb=(q!=0.0)?(Q0-RR)/q:ta;
            double lo=fmax(0.0,fmin(ta,tb)),hi=fmax(ta,tb);
            t=scully_bounded_root(c3,c2,c1,c0,lo,hi);
        }else{
            t=(VZ!=0.0)?((Csurf+H*Q0/(aa+Q0))-Z)/VZ:1.0e300;
            if(!(t>0.0) || Q0>RR*(1.0+2e-6))return INF_F;
        }
    }
    return (t>0.0&&t<0.5*(double)INF_F)?(float)t:INF_F;
}

// Unit normal pointing into the incident WATER medium.
__device__ void free_surface_normal_into_water(float x,float y,float R,int MODEL,float DELTA_H,float CORE_R,
                                               float* nx,float* ny,float* nz){
    if(MODEL==0 || DELTA_H<=0.0f){*nx=0.0f;*ny=0.0f;*nz=-1.0f;return;}
    if(MODEL==1){
        float aa=DELTA_H/(R*R);
        *nx=2.0f*aa*x;*ny=2.0f*aa*y;*nz=-1.0f;normalize3(nx,ny,nz);return;
    }
    float a2=CORE_R*CORE_R,q=x*x+y*y;
    float H=localized_scully_hinf(R,DELTA_H,CORE_R);
    float fac=2.0f*H*a2/((a2+q)*(a2+q));
    *nx=fac*x;*ny=fac*y;*nz=-1.0f;normalize3(nx,ny,nz);
}

__device__ void perpendicular_basis(float vx,float vy,float vz,
                                    float* e1x,float* e1y,float* e1z,
                                    float* e2x,float* e2y,float* e2z) {
    float rx,ry,rz;
    if(fabsf(vy)<0.9f){rx=0.0f;ry=1.0f;rz=0.0f;}else{rx=1.0f;ry=0.0f;rz=0.0f;}
    *e1x=ry*vz-rz*vy; *e1y=rz*vx-rx*vz; *e1z=rx*vy-ry*vx;
    normalize3(e1x,e1y,e1z);
    *e2x=vy*(*e1z)-vz*(*e1y); *e2y=vz*(*e1x)-vx*(*e1z); *e2z=vx*(*e1y)-vy*(*e1x);
    normalize3(e2x,e2y,e2z);
}

// Smith-Beckmann rough dielectric boundary: signed VNDF and stable
// Lambda/log-height random walk in a fixed TOP-oriented frame. At an
// inside-side encounter, wm = -sampleD_wi(-wi). Escape, intersection,
// numerical failure and watchdog failure are disjoint. The geometric normal
// is used only in the physical alpha->0 smooth-interface limit.
// Basis: Heitz, Hanika, d'Eon & Dachsbacher (2016), ACM TOG 35(4),
// DOI 10.1145/2897824.2925943; signed Beckmann slopes and dielectric walks.
// The configured RMS slope gives alpha=tan(25 deg).

enum {
    MICRO_INTERSECTION=1,
    MICRO_ESCAPE=2,
    MICRO_NUMERICAL_FAILURE=3,
    MICRO_WATCHDOG_FAILURE=4
};

// diagnostic subcodes.  These never alter physics; they identify the exact
// numerical branch if the fail-closed walker returns MICRO_NUMERICAL_FAILURE.
enum {
    MICRO_FAIL_NONE=0,
    VNDF_PROJECTED_AREA_INVALID=10,
    VNDF_CDF_INVALID=11,
    VNDF_INVERSE_CDF_INVALID=12,
    VNDF_VISIBILITY_FAILURE=13,
    VNDF_NORMALIZATION_FAILURE=14,
    HEIGHT_CDF_INVALID=20,              // aggregate validation code; kept for diagnostic compatibility
    HEIGHT_SAMPLE_INVALID=21,
    FREE_PATH_INVALID=22,
    HEIGHT_LAMBDA_INVALID=23,
    HEIGHT_C0_INVALID=24,
    HEIGHT_LOG_C0_INVALID=25,
    HEIGHT_G1_INVALID=26,
    HEIGHT_ESCAPE_PROB_INVALID=27,
    HEIGHT_RANDOM_INVALID=28,
    HEIGHT_LOG_CNEXT_INVALID=29,
    MICROFACET_NORMAL_INVALID=30,
    REFLECTION_VECTOR_INVALID=31,
    REFRACTION_VECTOR_INVALID=32,
    SIDE_STATE_INVALID=33,
    NONFINITE_DIRECTION=34,
    FRESNEL_INVALID=35,
    HEIGHT_CNEXT_RANGE_INVALID=36,
    HEIGHT_INVERSE_CDF_INVALID=37,
    HEIGHT_OUTPUT_INVALID=38,
    HEIGHT_FREE_PATH_INVALID=39,
    MICRO_WATCHDOG=40,
    MACRO_WATCHDOG=41
};
__device__ void micro_set_failure(int* code,int value){if(code)*code=value;}


// full-production execution-integrity codes.  These are reporting-only and never
// convert a failure into escape/absorption/success.  Existing top-level status=4 is retained
// for compatibility while production_failure_code identifies the exact failure stage.
enum {
    PROD_OK=0,
    PROD_PARTICLE_MICROSURFACE_NUMERICAL=1,
    PROD_PARTICLE_MICROSURFACE_WATCHDOG=2,
    PROD_PARTICLE_MACRO_GEOMETRY_FAILURE=3,
    PROD_PARTICLE_MACRO_BOUNCE_LIMIT=4,
    PROD_ACRYLIC_ANNULUS_NO_INTERSECTION=5,
    PROD_ACRYLIC_ANNULUS_BOUNCE_LIMIT=6,
    PROD_BOTTOM_DISC_GEOMETRY_FAILURE=7,
    PROD_BOTTOM_DISC_BOUNCE_LIMIT=8,
    PROD_WATER_BOUNDARY_NO_INTERSECTION=9,
    PROD_TRACE_ITERATION_LIMIT=10,
    PROD_NONFINITE_POSITION=11,
    PROD_NONFINITE_DIRECTION=12,
    PROD_INVALID_MEDIUM_STATE=13,
    PROD_INVALID_EVENT_STATE=14,
    PROD_OTHER_EXPLICIT_FAILURE=15,
    PROD_SPHEROID_GEOMETRY_FAILURE=16,
    PROD_HEADSPACE_BOUNDARY_NO_INTERSECTION=17,
    PROD_SPATIAL_MAJORANT_VIOLATION=18
};
enum {
    PROD_SUB_NONE=0,
    PROD_SUB_SPHERE_ENTRY_SIDE=1001,
    PROD_SUB_SPHERE_CHORD=1002,
    PROD_SUB_SPHERE_EXIT_SIDE=1003,
    PROD_SUB_SPHERE_INTERNAL_SIDE=1004,
    PROD_SUB_SPHERE_BOUNCE_LIMIT=1005,
    PROD_SUB_SPHEROID_ASPECT=1101,
    PROD_SUB_SPHEROID_AXES=1102,
    PROD_SUB_SPHEROID_ENTRY=1103,
    PROD_SUB_SPHEROID_CHORD=1104,
    PROD_SUB_SPHEROID_EXIT_SIDE=1105,
    PROD_SUB_SPHEROID_INTERNAL_SIDE=1106,
    PROD_SUB_SPHEROID_BOUNCE_LIMIT=1107,
    PROD_SUB_ACRYLIC_NO_INTERSECTION=2001,
    PROD_SUB_ACRYLIC_BOUNCE_LIMIT=2002,
    PROD_SUB_BOTTOM_NONPOSITIVE_T=2101,
    PROD_SUB_BOTTOM_TANGENTIAL=2102,
    PROD_SUB_BOTTOM_BOUNCE_LIMIT=2103,
    PROD_SUB_BOTTOM_TOPOLOGY_INVALID=2104,
    PROD_SUB_WATER_NO_BOUNDARY=3001,
    PROD_SUB_TRACE_ITERATION_LIMIT=3002,
    PROD_SUB_NONFINITE_POSITION=3003,
    PROD_SUB_NONFINITE_DIRECTION=3004,
    PROD_SUB_WATER_OUTSIDE_INNER_RADIUS=3005,
    PROD_SUB_HEADSPACE_NO_BOUNDARY=3006,
    PROD_SUB_SEDIMENT_MAJORANT_VIOLATION=4001
};
__device__ void prod_set_failure(int* code,int* subcode,int value,int sub){if(code)*code=value;if(subcode)*subcode=sub;}

// NVRTC-portable IEEE-754 quiet-NaN/+Inf sentinels; host C macros
// are not guaranteed to be defined in CuPy RawModule compilation.
__device__ double quiet_nan_double(){return __longlong_as_double(0x7ff8000000000000LL);}
__device__ double positive_infinity_double(){return __longlong_as_double(0x7ff0000000000000LL);}
__device__ float quiet_nan_float(){return __int_as_float(0x7fc00000);}
__device__ float positive_infinity_float(){return __int_as_float(0x7f800000);}

// Open-interval 32-bit uniform for the numerically delicate VNDF inversion.  The global
// rnd_uniform() returns float and can round the largest uint32 values to exactly 1.0f.
// Using the same xorshift state word with a half-ulp bin-centre mapping preserves the RNG
// sequence while guaranteeing 0<U<1 without clipping any physical VNDF tail.
__device__ double beckmann_open_uniform(unsigned int* state){
    if(!state)return -1.0;
    unsigned int r=xorshift32_state(state);
    return ((double)r+0.5)*2.3283064365386962890625e-10; // 2^-32
}

// B(x)=1-x*sqrt(pi)*erfcx(x), x>=0.  For deep Beckmann backside incidence the
// individual projected-area terms are exponentially tiny and nearly cancel.  Evaluating
// their ratio through B avoids both erff(a)+1 cancellation and underflow.  The large-x
// branch is the standard asymptotic expansion of erfcx and contains no probability cutoff.
__device__ double beckmann_tail_B(double x){
    if(!(x>=0.0)||!isfinite(x))return -1.0;
    const double SQRT_PI=1.77245385090551602729816748334114518;
    if(x<5.0){
        double e=exp(x*x)*erfc(x);
        double b=1.0-x*SQRT_PI*e;
        return (isfinite(b)&&b>0.0)?b:-1.0;
    }
    double y=1.0/(x*x);
    // 1/(2x^2)-3/(4x^4)+15/(8x^6)-... ; seven terms are ample at x>=5.
    double b=0.5*y-0.75*y*y+1.875*y*y*y-6.5625*y*y*y*y
            +29.53125*y*y*y*y*y-162.421875*y*y*y*y*y*y
            +1055.7421875*y*y*y*y*y*y*y;
    return (isfinite(b)&&b>0.0)?b:-1.0;
}

// Marginal CDF of the alpha=1 visible Beckmann slope for the numerically ordinary
// domain slope_i>=-1.  The complementary error function is used so that the negative
// Gaussian tail is never evaluated as 1+erf(x).  No arbitrary erf-domain clipping exists.
__device__ int beckmann_visible_cdf_ordinary(double slope,double slope_i,double* out){
    if(!out||!isfinite(slope)||!isfinite(slope_i))return 0;
    if(slope>=slope_i){*out=1.0;return 1;}
    const double INV_2_SQRT_PI_D=0.28209479177387814347403972578038629;
    double q=INV_2_SQRT_PI_D*exp(-slope_i*slope_i)+0.5*slope_i*erfc(-slope_i);
    if(!(q>0.0)||!isfinite(q))return 0;
    double n=INV_2_SQRT_PI_D*exp(-slope*slope)+0.5*slope_i*erfc(-slope);
    double f=n/q;
    if(!isfinite(f))return 0;
    if(f<0.0&&f>-1.0e-13)f=0.0;
    if(f>1.0&&f<1.0+1.0e-13)f=1.0;
    if(f<0.0||f>1.0)return 0;
    *out=f;return 1;
}

// Full-incidence alpha=1 Beckmann visible-slope sampler, parameterised by
// signed cot(theta) in double precision. For slope_i<-1, set x=-slope_i
// and slope=-(x+u), then invert the scaled complementary-tail CDF:
// F = exp(-(2*x*u+u^2))*[u/(x+u)+x/(x+u)*B(x+u)]/B(x).
// This preserves the full backside support as theta approaches pi.
__device__ int beckmann_sample11_signed_slope_d_ex(double slope_i,double U1,double U2,double* slope_x,double* slope_y,int* failure_code){
    micro_set_failure(failure_code,MICRO_FAIL_NONE);
    if(!slope_x||!slope_y||!isfinite(U1)||!isfinite(U2)){
        micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;
    }
    double u1=U1,u2=U2;
    if(!(u1>0.0&&u1<1.0)||!(u2>0.0&&u2<1.0)){
        micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;
    }
    // Preserve the near-+normal numerical limit (theta<1e-4 rad),
    // expressed directly in cot(theta) so production need not reconstruct theta.
    // In this limit the visible distribution is numerically indistinguishable from the NDF.
    const double COT_1E4=9999.999966666666;
    if(isinf(slope_i)||slope_i>COT_1E4){
        if(slope_i<0.0){micro_set_failure(failure_code,VNDF_PROJECTED_AREA_INVALID);return 0;}
        double r=sqrt(-log(u1)),phi=2.0*PI_D*u2;
        double sx=r*cos(phi),sy=r*sin(phi);
        if(!isfinite(sx)||!isfinite(sy)){micro_set_failure(failure_code,VNDF_INVERSE_CDF_INVALID);return 0;}
        *slope_x=sx;*slope_y=sy;return 1;
    }
    if(!isfinite(slope_i)){micro_set_failure(failure_code,VNDF_PROJECTED_AREA_INVALID);return 0;}

    double sx=0.0;
    if(slope_i < -1.0){
        // Deep-backside branch: invert the scaled complementary-tail CDF in u>=0.
        double x=-slope_i,Bx=beckmann_tail_B(x);
        if(!(Bx>0.0)||!isfinite(Bx)){micro_set_failure(failure_code,VNDF_PROJECTED_AREA_INVALID);return 0;}
        double target=log(u1),lo=0.0,hi=fmax(1.0/x,1.0e-12);
        int bracketed=0;
        for(int it=0;it<64;++it){
            double y=x+hi,By=beckmann_tail_B(y);
            if(!(By>0.0)||!isfinite(By)){micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;}
            double C=hi/y+(x/y)*By;
            if(!(C>0.0)||!isfinite(C)){micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;}
            double lr=-(2.0*x*hi+hi*hi)+log(C/Bx);
            if(!isfinite(lr)){micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;}
            if(lr<=target){bracketed=1;break;}
            hi*=2.0;
            if(!isfinite(hi)){micro_set_failure(failure_code,VNDF_INVERSE_CDF_INVALID);return 0;}
        }
        if(!bracketed){micro_set_failure(failure_code,VNDF_INVERSE_CDF_INVALID);return 0;}
        for(int it=0;it<52;++it){
            double mid=0.5*(lo+hi),y=x+mid,By=beckmann_tail_B(y);
            if(!(By>0.0)||!isfinite(By)){micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;}
            double C=mid/y+(x/y)*By;
            if(!(C>0.0)||!isfinite(C)){micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;}
            double lr=-(2.0*x*mid+mid*mid)+log(C/Bx);
            if(!isfinite(lr)){micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;}
            if(lr>target)lo=mid;else hi=mid;
        }
        sx=-(x+0.5*(lo+hi));
    }else{
        // Ordinary/front-side branch: monotone direct inversion in slope space.
        double lo=fmin(-1.0,slope_i-1.0),hi=slope_i,flo=1.0;
        int bracketed=0;
        for(int it=0;it<64;++it){
            if(!beckmann_visible_cdf_ordinary(lo,slope_i,&flo)){
                micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;
            }
            if(flo<=u1){bracketed=1;break;}
            lo=(lo<0.0)?2.0*lo:lo-1.0;
            if(!isfinite(lo)){micro_set_failure(failure_code,VNDF_INVERSE_CDF_INVALID);return 0;}
        }
        if(!bracketed){micro_set_failure(failure_code,VNDF_INVERSE_CDF_INVALID);return 0;}
        for(int it=0;it<48;++it){
            double mid=0.5*(lo+hi),fm=0.0;
            if(!beckmann_visible_cdf_ordinary(mid,slope_i,&fm)){
                micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;
            }
            if(fm<u1)lo=mid;else hi=mid;
        }
        sx=0.5*(lo+hi);
    }
    double ey=2.0*u2-1.0;
    if(!(ey>-1.0&&ey<1.0)||!isfinite(ey)){
        micro_set_failure(failure_code,VNDF_INVERSE_CDF_INVALID);return 0;
    }
    double sy=erfinv(ey);
    if(!isfinite(sx)||!isfinite(sy)){
        micro_set_failure(failure_code,VNDF_INVERSE_CDF_INVALID);return 0;
    }
    *slope_x=sx;*slope_y=sy;return 1;
}

// Compatibility angle API.  Theta remains double end-to-end; production bypasses
// this wrapper and supplies signed cot(theta) directly from the stretched direction.
__device__ int beckmann_sample11_signed_d_ex(double theta_i,double U1,double U2,double* slope_x,double* slope_y,int* failure_code){
    micro_set_failure(failure_code,MICRO_FAIL_NONE);
    if(!isfinite(theta_i)||!(theta_i>=0.0&&theta_i<PI_D)){
        micro_set_failure(failure_code,VNDF_CDF_INVALID);return 0;
    }
    if(theta_i==0.0)return beckmann_sample11_signed_slope_d_ex(positive_infinity_double(),U1,U2,slope_x,slope_y,failure_code);
    double st=sin(theta_i),ct=cos(theta_i);
    if(!(st>0.0)||!isfinite(st)||!isfinite(ct)){
        micro_set_failure(failure_code,VNDF_PROJECTED_AREA_INVALID);return 0;
    }
    return beckmann_sample11_signed_slope_d_ex(ct/st,U1,U2,slope_x,slope_y,failure_code);
}
__device__ int beckmann_sample11_signed_ex(float theta_i,double U1,double U2,float* slope_x,float* slope_y,int* failure_code){
    double sx=0.0,sy=0.0;int ok=beckmann_sample11_signed_d_ex(theta_i,U1,U2,&sx,&sy,failure_code);
    if(!ok)return 0;*slope_x=(float)sx;*slope_y=(float)sy;
    if(!isfinite(*slope_x)||!isfinite(*slope_y)){micro_set_failure(failure_code,VNDF_INVERSE_CDF_INVALID);return 0;}return 1;
}
__device__ int beckmann_sample11_signed(float theta_i,float U1,float U2,float* slope_x,float* slope_y){
    int fc=MICRO_FAIL_NONE;double u1=(double)U1,u2=(double)U2;if(u1<=0.0)u1=nextafter(0.0,1.0);if(u2<=0.0)u2=nextafter(0.0,1.0);if(u1>=1.0)u1=nextafter(1.0,0.0);if(u2>=1.0)u2=nextafter(1.0,0.0);return beckmann_sample11_signed_ex(theta_i,u1,u2,slope_x,slope_y,&fc);
}

// Signed Beckmann VNDF in a fixed TOP-oriented frame. Double-precision
// stretched slopes, reconstruction and visibility geometry prevent an
// ultra-grazing positive dot product from changing sign through rounding.
__device__ int beckmann_vndf_normal_signed_d_ex(
    unsigned int* state,float alpha,
    float bx,float by,float bz,
    float wx,float wy,float wz,
    double* mx,double* my,double* mz,int* failure_code){
    micro_set_failure(failure_code,MICRO_FAIL_NONE);
    if(!(alpha>=0.0f)||!isfinite(alpha)||!state||!mx||!my||!mz){
        micro_set_failure(failure_code,VNDF_NORMALIZATION_FAILURE);return 0;
    }
    normalize3(&bx,&by,&bz);normalize3(&wx,&wy,&wz);
    if(!isfinite(bx)||!isfinite(by)||!isfinite(bz)||!isfinite(wx)||!isfinite(wy)||!isfinite(wz)){
        micro_set_failure(failure_code,NONFINITE_DIRECTION);return 0;
    }
    if(alpha<=1.0e-8f){
        double vd=(double)wx*bx+(double)wy*by+(double)wz*bz;
        if(!(vd>0.0)){micro_set_failure(failure_code,VNDF_VISIBILITY_FAILURE);return 0;}
        *mx=(double)bx;*my=(double)by;*mz=(double)bz;return 1;
    }
    float t1x,t1y,t1z,t2x,t2y,t2z;
    perpendicular_basis(bx,by,bz,&t1x,&t1y,&t1z,&t2x,&t2y,&t2z);
    double lx=(double)wx*t1x+(double)wy*t1y+(double)wz*t1z;
    double ly=(double)wx*t2x+(double)wy*t2y+(double)wz*t2z;
    double lz=(double)wx*bx+(double)wy*by+(double)wz*bz; // SIGN IS PHYSICAL: never clamp to positive.
    double sxv=(double)alpha*lx,syv=(double)alpha*ly,szv=lz;
    double vl=sqrt(sxv*sxv+syv*syv+szv*szv);
    if(!(vl>0.0)||!isfinite(vl)){micro_set_failure(failure_code,VNDF_NORMALIZATION_FAILURE);return 0;}
    sxv/=vl;syv/=vl;szv/=vl;
    double radial=hypot(sxv,syv),slope_i=0.0;
    if(radial==0.0){
        if(szv>0.0)slope_i=positive_infinity_double();
        else{micro_set_failure(failure_code,VNDF_PROJECTED_AREA_INVALID);return 0;}
    }else{
        slope_i=szv/radial;
        if(!isfinite(slope_i)){micro_set_failure(failure_code,VNDF_PROJECTED_AREA_INVALID);return 0;}
    }
    double sx0=0.0,sy0=0.0;int fc=MICRO_FAIL_NONE;
    double U1=beckmann_open_uniform(state),U2=beckmann_open_uniform(state);
    if(!beckmann_sample11_signed_slope_d_ex(slope_i,U1,U2,&sx0,&sy0,&fc)){
        micro_set_failure(failure_code,fc?fc:VNDF_INVERSE_CDF_INVALID);return 0;
    }
    double phi=atan2(syv,sxv),cp=cos(phi),sp=sin(phi);
    double tx=cp*sx0-sp*sy0,ty=sp*sx0+cp*sy0;
    double sx=(double)alpha*tx,sy=(double)alpha*ty;
    if(!isfinite(sx)||!isfinite(sy)){micro_set_failure(failure_code,MICROFACET_NORMAL_INVALID);return 0;}
    double invn=1.0/sqrt(sx*sx+sy*sy+1.0);
    if(!(invn>0.0)||!isfinite(invn)){micro_set_failure(failure_code,VNDF_NORMALIZATION_FAILURE);return 0;}
    double nlx=-sx*invn,nly=-sy*invn,nlz=invn;
    double nx=nlx*t1x+nly*t2x+nlz*bx;
    double ny=nlx*t1y+nly*t2y+nlz*by;
    double nz=nlx*t1z+nly*t2z+nlz*bz;
    double nn=sqrt(nx*nx+ny*ny+nz*nz);
    if(!(nn>0.0)||!isfinite(nn)){micro_set_failure(failure_code,VNDF_NORMALIZATION_FAILURE);return 0;}
    nx/=nn;ny/=nn;nz/=nn;
    double top=nx*bx+ny*by+nz*bz;
    double vis=nx*wx+ny*wy+nz*wz;
    if(!(top>0.0)){micro_set_failure(failure_code,MICROFACET_NORMAL_INVALID);return 0;}
    if(!(vis>0.0)){micro_set_failure(failure_code,VNDF_VISIBILITY_FAILURE);return 0;}
    *mx=nx;*my=ny;*mz=nz;return 1;
}
__device__ int beckmann_vndf_normal_signed_ex(
    unsigned int* state,float alpha,
    float bx,float by,float bz,
    float wx,float wy,float wz,
    float* mx,float* my,float* mz,int* failure_code){
    double nx=0.0,ny=0.0,nz=0.0;int ok=beckmann_vndf_normal_signed_d_ex(state,alpha,bx,by,bz,wx,wy,wz,&nx,&ny,&nz,failure_code);
    if(!ok)return 0;*mx=(float)nx;*my=(float)ny;*mz=(float)nz;
    if(!isfinite(*mx)||!isfinite(*my)||!isfinite(*mz)){micro_set_failure(failure_code,MICROFACET_NORMAL_INVALID);return 0;}
    return 1;
}
__device__ int beckmann_vndf_normal_signed(
    unsigned int* state,float alpha,float bx,float by,float bz,float wx,float wy,float wz,float* mx,float* my,float* mz){
    int fc=MICRO_FAIL_NONE;return beckmann_vndf_normal_signed_ex(state,alpha,bx,by,bz,wx,wy,wz,mx,my,mz,&fc);
}

// Audit wrapper for the signed VNDF; no NDF or geometric-normal fallback.
__device__ int beckmann_microfacet_normal(
    unsigned int* state,float alpha,
    float bx,float by,float bz,
    float vx,float vy,float vz,int desired_dot_sign,
    float* mx,float* my,float* mz){
    float wx=(desired_dot_sign<0)?-vx:vx,wy=(desired_dot_sign<0)?-vy:vy,wz=(desired_dot_sign<0)?-vz:vz;
    return beckmann_vndf_normal_signed(state,alpha,bx,by,bz,wx,wy,wz,mx,my,mz);
}

__device__ float smith_uniform_C1(float h){return fminf(fmaxf(0.5f*(h+1.0f),0.0f),1.0f);}
__device__ float smith_uniform_invC1(float u){return 2.0f*fminf(fmaxf(u,0.0f),1.0f)-1.0f;}

// Signed Beckmann Smith Lambda over the full direction sphere. For a>0,
// Lambda = exp(-a^2)/(2*sqrt(pi)*a)*B(a),
// B(a)=1-a*sqrt(pi)*erfcx(a), evaluated without tail cancellation.
// Lambda(-a)=-1-Lambda(a); vertical limits are 0 and -1. No epsilon clamp.
__device__ double smith_beckmann_lambda_d(float wx,float wy,float wz,float bx,float by,float bz,float alpha){
    double c=(double)wx*bx+(double)wy*by+(double)wz*bz;
    if(!isfinite(c)||!isfinite(alpha)||!(alpha>=0.0f))return quiet_nan_double();
    c=fmax(-1.0,fmin(1.0,c));
    double ss2=fmax(0.0,1.0-c*c);
    if(ss2==0.0)return c>=0.0?0.0:-1.0;
    if(c==0.0)return copysign(positive_infinity_double(),c);
    if(alpha==0.0f)return c>=0.0?0.0:-1.0;
    double aa=fabs(c)/((double)alpha*sqrt(ss2));
    if(!(aa>0.0)||!isfinite(aa))return c>=0.0?0.0:-1.0;
    const double SQRT_PI=1.77245385090551602729816748334114518;
    double B=beckmann_tail_B(aa);
    if(!(B>0.0)||!isfinite(B))return quiet_nan_double();
    double logpref=-(aa*aa)-log(2.0*SQRT_PI*aa);
    // Direct exp() underflow simply rounds a physically positive Lambda below the smallest
    // representable double to zero; no empirical cutoff or epsilon clamp is introduced.
    double lp=exp(logpref)*B;
    if(!(lp>=0.0)||!isfinite(lp))return quiet_nan_double();
    return c>=0.0?lp:(-1.0-lp);
}
__device__ float smith_beckmann_lambda(float wx,float wy,float wz,float bx,float by,float bz,float alpha){
    double l=smith_beckmann_lambda_d(wx,wy,wz,bx,by,bz,alpha);
    if(isnan(l))return quiet_nan_float();
    if(isinf(l))return l>0.0?positive_infinity_float():-positive_infinity_float();
    return (float)l;
}

__device__ double smith_uniform_C1_d(double h){return fmin(1.0,fmax(0.0,0.5*(h+1.0)));}
__device__ double smith_uniform_invC1_d(double u){return 2.0*fmin(1.0,fmax(0.0,u))-1.0;}

__device__ int smith_G1_height_checked(float wx,float wy,float wz,float bx,float by,float bz,float alpha,float h,float* out_G1){
    if(!out_G1||!isfinite(h))return 0;
    double c=(double)wx*bx+(double)wy*by+(double)wz*bz;
    if(!isfinite(c))return 0;
    if(c<=0.0){*out_G1=0.0f;return 1;}
    double C1=smith_uniform_C1_d((double)h);
    if(C1<=0.0){*out_G1=0.0f;return 1;}
    if(C1>=1.0){*out_G1=1.0f;return 1;}
    double lam=smith_beckmann_lambda_d(wx,wy,wz,bx,by,bz,alpha);
    if(!isfinite(lam)||lam<0.0)return 0;
    if(lam==0.0){*out_G1=1.0f;return 1;}
    double lg=lam*log(C1),g=exp(lg);
    if(!(g>=0.0&&g<=1.0)||!isfinite(g))return 0;
    *out_G1=(float)g;return 1;
}

// Stable Smith height/free-path sampler.  It is algebraically identical to the Heitz
// uniform-height process but evaluates the escape probability as -expm1(Lambda*log(C0))
// and the next CDF in log domain.  This removes both 1-G cancellation and the unstable
// pow(1-U,1/Lambda) inversion.  Numerical failure and physical escape remain disjoint.
__device__ int smith_sample_height_u_ex(
    double U,float wx,float wy,float wz,float bx,float by,float bz,float alpha,float h,float* out_h,int* failure_code){
    micro_set_failure(failure_code,MICRO_FAIL_NONE);
    if(!out_h||!isfinite(U)||!isfinite(wx)||!isfinite(wy)||!isfinite(wz)||!isfinite(h)||!(alpha>=0.0f)||!isfinite(alpha)){
        micro_set_failure(failure_code,HEIGHT_RANDOM_INVALID);return MICRO_NUMERICAL_FAILURE;
    }
    if(!(U>0.0&&U<1.0)){micro_set_failure(failure_code,HEIGHT_RANDOM_INVALID);return MICRO_NUMERICAL_FAILURE;}
    double c=(double)wx*bx+(double)wy*by+(double)wz*bz;
    if(!isfinite(c)){micro_set_failure(failure_code,NONFINITE_DIRECTION);return MICRO_NUMERICAL_FAILURE;}
    c=fmax(-1.0,fmin(1.0,c));

    double C0=smith_uniform_C1_d((double)h);
    if(!(C0>=0.0&&C0<=1.0)||!isfinite(C0)){
        micro_set_failure(failure_code,HEIGHT_C0_INVALID);return MICRO_NUMERICAL_FAILURE;
    }

    // Exact vertical limits of the Smith process; no finite-angle exclusion cone.
    double ss2=fmax(0.0,1.0-c*c);
    if(ss2==0.0){
        if(c>0.0)return MICRO_ESCAPE;
        if(c<0.0){
            double Cnext=U*C0,hn=smith_uniform_invC1_d(Cnext);
            if(!isfinite(hn)){micro_set_failure(failure_code,HEIGHT_INVERSE_CDF_INVALID);return MICRO_NUMERICAL_FAILURE;}
            *out_h=(float)hn;return MICRO_INTERSECTION;
        }
        *out_h=h;return MICRO_INTERSECTION;
    }
    // Exact horizontal limit: Lambda magnitude diverges and the next interaction is at h.
    if(c==0.0){*out_h=h;return MICRO_INTERSECTION;}

    double lam=smith_beckmann_lambda_d(wx,wy,wz,bx,by,bz,alpha);
    if(isnan(lam)){micro_set_failure(failure_code,HEIGHT_LAMBDA_INVALID);return MICRO_NUMERICAL_FAILURE;}

    if(c>0.0){
        if(!(lam>=0.0)){micro_set_failure(failure_code,HEIGHT_LAMBDA_INVALID);return MICRO_NUMERICAL_FAILURE;}
        if(C0<=0.0)return MICRO_ESCAPE;
        if(C0>=1.0)return MICRO_ESCAPE;
        if(lam==0.0)return MICRO_ESCAPE; // exact representable Lambda->0 limit
        double logC0=log(C0);
        if(!isfinite(logC0)){micro_set_failure(failure_code,HEIGHT_LOG_C0_INVALID);return MICRO_NUMERICAL_FAILURE;}
        double logG1=lam*logC0;
        if(!(logG1<=0.0)||!isfinite(logG1)){micro_set_failure(failure_code,HEIGHT_G1_INVALID);return MICRO_NUMERICAL_FAILURE;}
        double p_intersect=-expm1(logG1);
        if(!(p_intersect>=0.0&&p_intersect<=1.0)||!isfinite(p_intersect)){
            micro_set_failure(failure_code,HEIGHT_ESCAPE_PROB_INVALID);return MICRO_NUMERICAL_FAILURE;
        }
        if(U>p_intersect)return MICRO_ESCAPE;
        double log1mU=log1p(-U);
        if(!isfinite(log1mU)){micro_set_failure(failure_code,HEIGHT_RANDOM_INVALID);return MICRO_NUMERICAL_FAILURE;}
        double logCnext=logC0-log1mU/lam;
        if(!isfinite(logCnext)){micro_set_failure(failure_code,HEIGHT_LOG_CNEXT_INVALID);return MICRO_NUMERICAL_FAILURE;}
        // Exact arithmetic guarantees logCnext<=0 for U<=p_intersect.  Permit only a
        // few double ulps of boundary-rounding before projecting the representable endpoint.
        const double BOUND_TOL=64.0*2.2204460492503130808472633361816e-16;
        if(logCnext>BOUND_TOL){micro_set_failure(failure_code,HEIGHT_CNEXT_RANGE_INVALID);return MICRO_NUMERICAL_FAILURE;}
        if(logCnext>0.0)logCnext=0.0;
        double Cnext=exp(logCnext);
        if(!(Cnext>=0.0&&Cnext<=1.0)||!isfinite(Cnext)){
            micro_set_failure(failure_code,HEIGHT_CNEXT_RANGE_INVALID);return MICRO_NUMERICAL_FAILURE;
        }
        double hn=smith_uniform_invC1_d(Cnext);
        if(!isfinite(hn)){micro_set_failure(failure_code,HEIGHT_INVERSE_CDF_INVALID);return MICRO_NUMERICAL_FAILURE;}
        *out_h=(float)hn;
        if(!isfinite(*out_h)){micro_set_failure(failure_code,HEIGHT_OUTPUT_INVALID);return MICRO_NUMERICAL_FAILURE;}
        return MICRO_INTERSECTION;
    }

    // Downward/signed side: Lambda<=-1, intersection is certain and the same log-CDF
    // inversion naturally moves C downward.  The exact negative-vertical limit is handled above.
    if(!(lam<=-1.0)||!isfinite(lam)){micro_set_failure(failure_code,HEIGHT_LAMBDA_INVALID);return MICRO_NUMERICAL_FAILURE;}
    if(C0<=0.0){*out_h=-1.0f;return MICRO_INTERSECTION;}
    double logC0=log(C0);
    if(!isfinite(logC0)){micro_set_failure(failure_code,HEIGHT_LOG_C0_INVALID);return MICRO_NUMERICAL_FAILURE;}
    double log1mU=log1p(-U);
    if(!isfinite(log1mU)){micro_set_failure(failure_code,HEIGHT_RANDOM_INVALID);return MICRO_NUMERICAL_FAILURE;}
    double logCnext=logC0-log1mU/lam;
    if(!isfinite(logCnext)){micro_set_failure(failure_code,HEIGHT_LOG_CNEXT_INVALID);return MICRO_NUMERICAL_FAILURE;}
    if(logCnext>0.0){micro_set_failure(failure_code,HEIGHT_CNEXT_RANGE_INVALID);return MICRO_NUMERICAL_FAILURE;}
    double Cnext=exp(logCnext);
    if(!(Cnext>=0.0&&Cnext<=C0*(1.0+8.0e-15))||!isfinite(Cnext)){
        micro_set_failure(failure_code,HEIGHT_CNEXT_RANGE_INVALID);return MICRO_NUMERICAL_FAILURE;
    }
    double hn=smith_uniform_invC1_d(Cnext);
    if(!isfinite(hn)){micro_set_failure(failure_code,HEIGHT_INVERSE_CDF_INVALID);return MICRO_NUMERICAL_FAILURE;}
    *out_h=(float)hn;
    if(!isfinite(*out_h)){micro_set_failure(failure_code,HEIGHT_OUTPUT_INVALID);return MICRO_NUMERICAL_FAILURE;}
    return MICRO_INTERSECTION;
}
__device__ int smith_sample_height_u(
    float U,float wx,float wy,float wz,float bx,float by,float bz,float alpha,float h,float* out_h){
    int fc=MICRO_FAIL_NONE;double u=(double)U;
    if(u<=0.0)u=nextafter(0.0,1.0);if(u>=1.0)u=nextafter(1.0,0.0);
    return smith_sample_height_u_ex(u,wx,wy,wz,bx,by,bz,alpha,h,out_h,&fc);
}
__device__ int smith_sample_height_ex(
    unsigned int* state,float wx,float wy,float wz,float bx,float by,float bz,float alpha,float h,float* out_h,int* failure_code){
    double U=beckmann_open_uniform(state);
    if(!(U>0.0&&U<1.0)||!isfinite(U)){micro_set_failure(failure_code,HEIGHT_RANDOM_INVALID);return MICRO_NUMERICAL_FAILURE;}
    return smith_sample_height_u_ex(U,wx,wy,wz,bx,by,bz,alpha,h,out_h,failure_code);
}
__device__ int smith_sample_height(
    unsigned int* state,float wx,float wy,float wz,float bx,float by,float bz,float alpha,float h,float* out_h){
    int fc=MICRO_FAIL_NONE;return smith_sample_height_ex(state,wx,wy,wz,bx,by,bz,alpha,h,out_h,&fc);
}

// Rough-dielectric Smith microsurface random walk with stable height
// arithmetic and double-precision geometry at ultra-grazing incidence.
__device__ int smith_microsurface_dielectric_walk_ex(
    unsigned int* state,float alpha,int micro_watchdog,
    float bx,float by,float bz,float n_top,float k_top,float n_bottom,float k_bottom,
    float* dx,float* dy,float* dz,int* out_top,int* scatter_order,int* failure_code){
    micro_set_failure(failure_code,MICRO_FAIL_NONE);
    normalize3(&bx,&by,&bz);normalize3(dx,dy,dz);*scatter_order=0;
    if(!isfinite(bx)||!isfinite(by)||!isfinite(bz)||!isfinite(*dx)||!isfinite(*dy)||!isfinite(*dz)){
        micro_set_failure(failure_code,NONFINITE_DIRECTION);return MICRO_NUMERICAL_FAILURE;
    }
    if(dot3(*dx,*dy,*dz,bx,by,bz)>=-1.0e-7f){micro_set_failure(failure_code,SIDE_STATE_INVALID);return MICRO_NUMERICAL_FAILURE;}
    if(micro_watchdog<1){micro_set_failure(failure_code,MICRO_WATCHDOG);return MICRO_WATCHDOG_FAILURE;}

    // Exact physical smooth-interface limit; not a stochastic fallback.
    if(alpha<=1.0e-7f){
        float wi_x=-(*dx),wi_y=-(*dy),wi_z=-(*dz);
        float ci=fminf(fmaxf(dot3(wi_x,wi_y,wi_z,bx,by,bz),0.0f),1.0f);
        float eta=n_top/n_bottom;
        if(!(n_bottom>0.0f)||!isfinite(eta)){micro_set_failure(failure_code,REFRACTION_VECTOR_INVALID);return MICRO_NUMERICAL_FAILURE;}
        int tir=(eta*eta*fmaxf(0.0f,1.0f-ci*ci)>=1.0f);
        float R=tir?1.0f:fresnel_R_complex(ci,n_top,k_top,n_bottom,k_bottom);
        if(!isfinite(R)){micro_set_failure(failure_code,FRESNEL_INVALID);return MICRO_NUMERICAL_FAILURE;}
        float ox,oy,oz;
        if(tir||rnd_uniform(state)<R){
            reflect3(*dx,*dy,*dz,bx,by,bz,&ox,&oy,&oz);*out_top=1;
            if(!isfinite(ox)||!isfinite(oy)||!isfinite(oz)){micro_set_failure(failure_code,REFLECTION_VECTOR_INVALID);return MICRO_NUMERICAL_FAILURE;}
        }else{
            if(!refract3(*dx,*dy,*dz,bx,by,bz,n_top,n_bottom,&ox,&oy,&oz)){
                micro_set_failure(failure_code,REFRACTION_VECTOR_INVALID);return MICRO_NUMERICAL_FAILURE;
            }
            *out_top=0;
        }
        if(!isfinite(ox)||!isfinite(oy)||!isfinite(oz)){micro_set_failure(failure_code,NONFINITE_DIRECTION);return MICRO_NUMERICAL_FAILURE;}
        *dx=ox;*dy=oy;*dz=oz;*scatter_order=1;return MICRO_ESCAPE;
    }

    float h=1.0f+smith_uniform_invC1(0.999f);int outside=1,order=0;
    for(int it=0;it<micro_watchdog;++it){
        float nh=0.0f;int hs,fc=MICRO_FAIL_NONE;
        if(outside){
            hs=smith_sample_height_ex(state,*dx,*dy,*dz,bx,by,bz,alpha,h,&nh,&fc);
            if(hs==MICRO_ESCAPE){*out_top=1;*scatter_order=order;return MICRO_ESCAPE;}
        }else{
            float tmp=0.0f;
            hs=smith_sample_height_ex(state,-(*dx),-(*dy),-(*dz),bx,by,bz,alpha,-h,&tmp,&fc);
            if(hs==MICRO_ESCAPE){*out_top=0;*scatter_order=order;return MICRO_ESCAPE;}
            if(hs==MICRO_INTERSECTION)nh=-tmp;
        }
        if(hs!=MICRO_INTERSECTION||!isfinite(nh)){
            *scatter_order=order;micro_set_failure(failure_code,fc?fc:HEIGHT_SAMPLE_INVALID);return MICRO_NUMERICAL_FAILURE;
        }
        h=nh;++order;
        float wix=-(*dx),wiy=-(*dy),wiz=-(*dz);double mx,my,mz;fc=MICRO_FAIL_NONE;
        if(outside){
            if(!beckmann_vndf_normal_signed_d_ex(state,alpha,bx,by,bz,wix,wiy,wiz,&mx,&my,&mz,&fc)){
                *scatter_order=order;micro_set_failure(failure_code,fc?fc:VNDF_INVERSE_CDF_INVALID);return MICRO_NUMERICAL_FAILURE;
            }
        }else{
            double tx,ty,tz;
            if(!beckmann_vndf_normal_signed_d_ex(state,alpha,bx,by,bz,-wix,-wiy,-wiz,&tx,&ty,&tz,&fc)){
                *scatter_order=order;micro_set_failure(failure_code,fc?fc:VNDF_INVERSE_CDF_INVALID);return MICRO_NUMERICAL_FAILURE;
            }
            mx=-tx;my=-ty;mz=-tz;
        }
        double vis=(double)wix*mx+(double)wiy*my+(double)wiz*mz;
        if(!isfinite(mx)||!isfinite(my)||!isfinite(mz)){
            *scatter_order=order;micro_set_failure(failure_code,MICROFACET_NORMAL_INVALID);return MICRO_NUMERICAL_FAILURE;
        }
        if(!(vis>0.0)){
            *scatter_order=order;micro_set_failure(failure_code,VNDF_VISIBILITY_FAILURE);return MICRO_NUMERICAL_FAILURE;
        }
        double ci_d=fmin(1.0,fmax(0.0,vis));float ci=(float)ci_d;
        float n1=outside?n_top:n_bottom,k1=outside?k_top:k_bottom,n2=outside?n_bottom:n_top,k2=outside?k_bottom:k_top;
        if(!(n1>0.0f&&n2>0.0f)){micro_set_failure(failure_code,REFRACTION_VECTOR_INVALID);return MICRO_NUMERICAL_FAILURE;}
        double eta=(double)n1/(double)n2;double kt=1.0-eta*eta*fmax(0.0,1.0-ci_d*ci_d);int tir=(kt<=0.0);
        float R=tir?1.0f:fresnel_R_complex(ci,n1,k1,n2,k2);
        if(!isfinite(R)){micro_set_failure(failure_code,FRESNEL_INVALID);return MICRO_NUMERICAL_FAILURE;}
        double ox_d,oy_d,oz_d;
        if(tir||rnd_uniform(state)<R){
            double dd=(double)(*dx)*mx+(double)(*dy)*my+(double)(*dz)*mz;
            ox_d=(double)(*dx)-2.0*dd*mx;oy_d=(double)(*dy)-2.0*dd*my;oz_d=(double)(*dz)-2.0*dd*mz;
        }else{
            double ct=sqrt(fmax(0.0,kt));
            ox_d=eta*(double)(*dx)+(eta*ci_d-ct)*mx;
            oy_d=eta*(double)(*dy)+(eta*ci_d-ct)*my;
            oz_d=eta*(double)(*dz)+(eta*ci_d-ct)*mz;
            outside=1-outside;
        }
        double qn=sqrt(ox_d*ox_d+oy_d*oy_d+oz_d*oz_d);
        if(!(qn>0.0)||!isfinite(qn)||!isfinite(ox_d)||!isfinite(oy_d)||!isfinite(oz_d)){
            *scatter_order=order;micro_set_failure(failure_code,tir?REFLECTION_VECTOR_INVALID:REFRACTION_VECTOR_INVALID);return MICRO_NUMERICAL_FAILURE;
        }
        ox_d/=qn;oy_d/=qn;oz_d/=qn;float ox=(float)ox_d,oy=(float)oy_d,oz=(float)oz_d;
        if(!isfinite(ox)||!isfinite(oy)||!isfinite(oz)){
            *scatter_order=order;micro_set_failure(failure_code,NONFINITE_DIRECTION);return MICRO_NUMERICAL_FAILURE;
        }
        normalize3(&ox,&oy,&oz);
        if(!isfinite(ox)||!isfinite(oy)||!isfinite(oz)){
            *scatter_order=order;micro_set_failure(failure_code,NONFINITE_DIRECTION);return MICRO_NUMERICAL_FAILURE;
        }
        *dx=ox;*dy=oy;*dz=oz;
    }
    *scatter_order=order;micro_set_failure(failure_code,MICRO_WATCHDOG);return MICRO_WATCHDOG_FAILURE;
}
__device__ int smith_microsurface_dielectric_walk(
    unsigned int* state,float alpha,int micro_watchdog,float bx,float by,float bz,
    float n_top,float k_top,float n_bottom,float k_bottom,float* dx,float* dy,float* dz,int* out_top,int* scatter_order){
    int fc=MICRO_FAIL_NONE;return smith_microsurface_dielectric_walk_ex(state,alpha,micro_watchdog,bx,by,bz,n_top,k_top,n_bottom,k_bottom,dx,dy,dz,out_top,scatter_order,&fc);
}

__device__ void sphere_sample_geometry(
    unsigned int* state,const float radius,const float x,const float y,const float z,
    const float vx,const float vy,const float vz,
    float* rho_out,float* phi_out,float* gnx_out,float* gny_out,float* gnz_out,
    float* cx_out,float* cy_out,float* cz_out){
    float rho=sqrtf(fminf(fmaxf(rnd_uniform(state),0.0f),0.99999994f));
    float ci=sqrtf(fmaxf(0.0f,1.0f-rho*rho));
    float phi=rnd_uniform(state)*2.0f*PI_F;
    float e1x,e1y,e1z,e2x,e2y,e2z;
    perpendicular_basis(vx,vy,vz,&e1x,&e1y,&e1z,&e2x,&e2y,&e2z);
    float qx=cosf(phi)*e1x+sinf(phi)*e2x;
    float qy=cosf(phi)*e1y+sinf(phi)*e2y;
    float qz=cosf(phi)*e1z+sinf(phi)*e2z;
    float gnx=-ci*vx+rho*qx,gny=-ci*vy+rho*qy,gnz=-ci*vz+rho*qz;
    normalize3(&gnx,&gny,&gnz);
    if(rho_out)*rho_out=rho;if(phi_out)*phi_out=phi;
    if(gnx_out)*gnx_out=gnx;if(gny_out)*gny_out=gny;if(gnz_out)*gnz_out=gnz;
    if(cx_out)*cx_out=x-radius*gnx;if(cy_out)*cy_out=y-radius*gny;if(cz_out)*cz_out=z-radius*gnz;
}

// finite-particle accessibility uses the ACTUAL sampled particle centre.
// The localized free-surface models are axisymmetric and monotone increasing with r,
// so the lowest free-surface height over a spherical footprint occurs at r=max(0,rc-radius).
__device__ int sphere_particle_accessible_in_water(
    const double cx,const double cy,const double cz,const double radius,
    const float Rin,const float Zmin,const float Zmean,const int free_surface_model,
    const float vortex_delta_h,const float vortex_core_radius,
    float* center_radial_clearance,float* finite_wall_clearance,float* bottom_clearance,float* top_clearance){
    // Carry the sampled particle centre in double precision so sub-micron
    // clearance is resolved independently of vessel-scale float32 displacement.
    // Promote CUDA geometry parameters from their exact ABI-delivered values.
    double rc=hypot(cx,cy),rr=radius;
    double cclear=(double)Rin-rc;
    double wall=cclear-rr;
    double bottom=cz-rr-(double)Zmin;
    double rmin=fmax(0.0,rc-rr);
    double zsurf=(double)Zmean;
    if(free_surface_model!=0 && vortex_delta_h>0.0f){
        double R=(double)Rin,dh=(double)vortex_delta_h,a=(double)vortex_core_radius;
        double q=rmin*rmin,R2=R*R;
        if(free_surface_model==1){
            zsurf=(double)Zmean+dh*(q/R2-0.5);
        }else{
            double a2=a*a;
            double H=dh*(a2+R2)/R2;
            double hbar=H*(1.0-(a2/R2)*log1p(R2/a2));
            zsurf=(double)Zmean-hbar+H*q/(a2+q);
        }
    }
    double top=zsurf-(cz+rr);
    // Clearance tolerance is arithmetic only. Double-conditioned placement
    // permits a few ulps without admitting sub-micron geometric overlap.
    double scale=fmax(fmax(fabs((double)Rin),fabs(cz)),fmax(rr,1.0e-6));
    double tol=128.0*2.220446049250313080847263336181640625e-16*scale;
    if(center_radial_clearance)*center_radial_clearance=(float)cclear;
    if(finite_wall_clearance)*finite_wall_clearance=(float)wall;
    if(bottom_clearance)*bottom_clearance=(float)bottom;
    if(top_clearance)*top_clearance=(float)top;
    return (wall>=-tol && bottom>=-tol && top>=-tol)?1:0;
}

__device__ int sphere_fresnel_interaction_3d_sampled_ex(
    unsigned int* state,float n_medium,float n_particle,float k_particle,float wavelength,float rough_alpha,float radius,int max_bounces,
    const float gnx,const float gny,const float gnz,const double cx,const double cy,const double cz,
    float* x,float* y,float* z,float* vx,float* vy,float* vz,float* internal_path,
    int* p_reflections,int* p_entry_reflections,int* p_internal_reflections,int* p_absorptions,
    int* prod_failure_code,int* prod_failure_subcode,int* out_macro_bounces,int* out_micro_order,
    float* out_entry_incidence_deg,float* out_exit_angle_deg) {
    prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_OK,PROD_SUB_NONE);
    if(out_macro_bounces)*out_macro_bounces=0;if(out_micro_order)*out_micro_order=0;
    if(out_entry_incidence_deg)*out_entry_incidence_deg=-1.0f;if(out_exit_angle_deg)*out_exit_angle_deg=-1.0f;
    if(radius<=0.0f)return 1;
    const int MICRO_WATCHDOG_LIMIT=4096;
    const float entry_cosi=fminf(fmaxf(-dot3(*vx,*vy,*vz,gnx,gny,gnz),0.0f),1.0f);
    if(out_entry_incidence_deg)*out_entry_incidence_deg=acosf(entry_cosi)*(180.0f/PI_F);
    int top=1,order=0,mfc=MICRO_FAIL_NONE;
    int ms=smith_microsurface_dielectric_walk_ex(state,rough_alpha,MICRO_WATCHDOG_LIMIT,gnx,gny,gnz,
            n_medium,0.0f,n_particle,k_particle,vx,vy,vz,&top,&order,&mfc);
    if(out_micro_order)*out_micro_order=order;
    if(ms!=MICRO_ESCAPE){
        prod_set_failure(prod_failure_code,prod_failure_subcode,
            ms==MICRO_WATCHDOG_FAILURE?PROD_PARTICLE_MICROSURFACE_WATCHDOG:PROD_PARTICLE_MICROSURFACE_NUMERICAL,mfc);
        return 0;
    }
    if(top){
        if(out_exit_angle_deg){float co=fminf(fmaxf(dot3(*vx,*vy,*vz,gnx,gny,gnz),0.0f),1.0f);*out_exit_angle_deg=acosf(co)*(180.0f/PI_F);}
        (*p_reflections)++;(*p_entry_reflections)++;return 1;
    }
    if(dot3(*vx,*vy,*vz,gnx,gny,gnz)>=0.0f){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_PARTICLE_MACRO_GEOMETRY_FAILURE,PROD_SUB_SPHERE_ENTRY_SIDE);return 0;}

    // Propagate inside a particle in local double-precision coordinates.
    // Centimetre-scale float32 world coordinates cannot resolve kaolin radii
    // near 1e-7 m accurately. Interface sampling retains its float RNG semantics.
    const double rr=(double)radius,rr2=rr*rr;
    double qx=rr*(double)gnx,qy=rr*(double)gny,qz=rr*(double)gnz;
    double qn0=sqrt(qx*qx+qy*qy+qz*qz);
    if(!(qn0>0.0)||!isfinite(qn0)){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_PARTICLE_MACRO_GEOMETRY_FAILURE,PROD_SUB_SPHERE_CHORD);return 0;}
    qx*=rr/qn0;qy*=rr/qn0;qz*=rr/qn0;
    float alpha_abs=(wavelength>0.0f && k_particle>0.0f)?(4.0f*PI_F*k_particle/wavelength):0.0f;
    for(int bounce=0;bounce<=max_bounces;++bounce){
        if(out_macro_bounces)*out_macro_bounces=bounce;
        double dvx=(double)(*vx),dvy=(double)(*vy),dvz=(double)(*vz);
        double A=dvx*dvx+dvy*dvy+dvz*dvz;
        double q2=qx*qx+qy*qy+qz*qz;
        if(!(A>0.0)||!isfinite(A)||!isfinite(q2)){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_PARTICLE_MACRO_GEOMETRY_FAILURE,PROD_SUB_SPHERE_CHORD);return 0;}
        // q is deliberately maintained on the particle sphere.  Re-project before
        // every chord using radius-scale float64 arithmetic, then select the mathematically
        // known NON-ZERO/FAR root.  A generic "smallest positive root" is wrong here because
        // the other exact root is t=0 and roundoff can make that self-root slightly positive.
        double qnorm=sqrt(q2);
        if(!(qnorm>0.0)||!isfinite(qnorm)){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_PARTICLE_MACRO_GEOMETRY_FAILURE,PROD_SUB_SPHERE_CHORD);return 0;}
        double qproj=rr/qnorm;qx*=qproj;qy*=qproj;qz*=qproj;
        double B=qx*dvx+qy*dvy+qz*dvz;
        if(!(B<0.0)||!isfinite(B)){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_PARTICLE_MACRO_GEOMETRY_FAILURE,PROD_SUB_SPHERE_INTERNAL_SIDE);return 0;}
        double chord=-2.0*B/A;
        if(!(chord>0.0)||!isfinite(chord)){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_PARTICLE_MACRO_GEOMETRY_FAILURE,PROD_SUB_SPHERE_CHORD);return 0;}
        // Arrival invariant: the far-side hit must lie on the sphere and the unchanged
        // incident direction must point outward there.  This makes a near-zero self-hit
        // impossible to pass silently.
        double tqx=qx+chord*dvx,tqy=qy+chord*dvy,tqz=qz+chord*dvz;
        double arrival_dot=tqx*dvx+tqy*dvy+tqz*dvz;
        double tq2=tqx*tqx+tqy*tqy+tqz*tqz;
        double sphere_err=fabs(tq2-rr2);
        double sphere_tol=512.0*2.220446049250313080847263336181640625e-16*fmax(rr2,1.0e-30);
        if(!(arrival_dot>0.0)||!isfinite(arrival_dot)||!isfinite(tq2)||sphere_err>sphere_tol){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_PARTICLE_MACRO_GEOMETRY_FAILURE,PROD_SUB_SPHERE_CHORD);return 0;}
        float chord_f=(float)chord;
        if(alpha_abs>0.0f){
            float survival=expf(-alpha_abs*chord_f);
            if(rnd_uniform(state)>survival){
                qx=tqx;qy=tqy;qz=tqz;
                *x=(float)(cx+qx);*y=(float)(cy+qy);*z=(float)(cz+qz);*internal_path+=chord_f;(*p_absorptions)++;return 2;
            }
        }
        qx=tqx;qy=tqy;qz=tqz;*internal_path+=chord_f;
        double qn=sqrt(qx*qx+qy*qy+qz*qz);
        if(!(qn>0.0)||!isfinite(qn)){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_PARTICLE_MACRO_GEOMETRY_FAILURE,PROD_SUB_SPHERE_CHORD);return 0;}
        // Projection removes only accumulated arithmetic roundoff; the correction is at
        // double precision and leaves the physical sphere/ray intersection unchanged.
        double proj=rr/qn;qx*=proj;qy*=proj;qz*=proj;
        *x=(float)(cx+qx);*y=(float)(cy+qy);*z=(float)(cz+qz);
        float nox=(float)(qx/rr),noy=(float)(qy/rr),noz=(float)(qz/rr);normalize3(&nox,&noy,&noz);
        float bx=-nox,by=-noy,bz=-noz;top=1;order=0;mfc=MICRO_FAIL_NONE;
        ms=smith_microsurface_dielectric_walk_ex(state,rough_alpha,MICRO_WATCHDOG_LIMIT,bx,by,bz,
                n_particle,k_particle,n_medium,0.0f,vx,vy,vz,&top,&order,&mfc);
        if(out_micro_order)*out_micro_order=order;
        if(ms!=MICRO_ESCAPE){
            prod_set_failure(prod_failure_code,prod_failure_subcode,
                ms==MICRO_WATCHDOG_FAILURE?PROD_PARTICLE_MICROSURFACE_WATCHDOG:PROD_PARTICLE_MICROSURFACE_NUMERICAL,mfc);
            return 0;
        }
        if(!top){
            if(dot3(*vx,*vy,*vz,nox,noy,noz)<=0.0f){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_PARTICLE_MACRO_GEOMETRY_FAILURE,PROD_SUB_SPHERE_EXIT_SIDE);return 0;}
            if(out_exit_angle_deg){float co=fminf(fmaxf(dot3(*vx,*vy,*vz,nox,noy,noz),0.0f),1.0f);*out_exit_angle_deg=acosf(co)*(180.0f/PI_F);}
            return 1;
        }
        if(dot3(*vx,*vy,*vz,nox,noy,noz)>=0.0f){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_PARTICLE_MACRO_GEOMETRY_FAILURE,PROD_SUB_SPHERE_INTERNAL_SIDE);return 0;}
        (*p_reflections)++;(*p_internal_reflections)++;
        if(bounce==max_bounces){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_PARTICLE_MACRO_BOUNCE_LIMIT,PROD_SUB_SPHERE_BOUNCE_LIMIT);return 0;}
    }
    prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_OTHER_EXPLICIT_FAILURE,PROD_SUB_NONE);return 0;
}

__device__ int sphere_fresnel_interaction_3d_ex(
    unsigned int* state,float n_medium,float n_particle,float k_particle,float wavelength,float rough_alpha,float radius,int max_bounces,
    float* x,float* y,float* z,float* vx,float* vy,float* vz,float* internal_path,
    int* p_reflections,int* p_entry_reflections,int* p_internal_reflections,int* p_absorptions,
    int* prod_failure_code,int* prod_failure_subcode,int* out_macro_bounces,int* out_micro_order,
    float* out_entry_incidence_deg,float* out_exit_angle_deg) {
    float rho=0.0f,phi=0.0f,gnx=0.0f,gny=0.0f,gnz=0.0f,cxf=*x,cyf=*y,czf=*z;
    sphere_sample_geometry(state,radius,*x,*y,*z,*vx,*vy,*vz,&rho,&phi,&gnx,&gny,&gnz,&cxf,&cyf,&czf);
    double cx=(double)(*x)-(double)radius*(double)gnx,cy=(double)(*y)-(double)radius*(double)gny,cz=(double)(*z)-(double)radius*(double)gnz;
    return sphere_fresnel_interaction_3d_sampled_ex(state,n_medium,n_particle,k_particle,wavelength,rough_alpha,radius,max_bounces,
        gnx,gny,gnz,cx,cy,cz,x,y,z,vx,vy,vz,internal_path,p_reflections,p_entry_reflections,p_internal_reflections,p_absorptions,
        prod_failure_code,prod_failure_subcode,out_macro_bounces,out_micro_order,out_entry_incidence_deg,out_exit_angle_deg);
}
__device__ int sphere_fresnel_interaction_3d(
    unsigned int* state,float n_medium,float n_particle,float k_particle,float wavelength,float rough_alpha,float radius,int max_bounces,
    float* x,float* y,float* z,float* vx,float* vy,float* vz,float* internal_path,
    int* p_reflections,int* p_entry_reflections,int* p_internal_reflections,int* p_absorptions) {
    int pc=PROD_OK,ps=PROD_SUB_NONE,mb=0,mo=0;float ei=-1.0f,eo=-1.0f;
    return sphere_fresnel_interaction_3d_ex(state,n_medium,n_particle,k_particle,wavelength,rough_alpha,radius,max_bounces,
        x,y,z,vx,vy,vz,internal_path,p_reflections,p_entry_reflections,p_internal_reflections,p_absorptions,&pc,&ps,&mb,&mo,&ei,&eo);
}


__device__ int sample_cdf_index(unsigned int* state,const double* cdf,int n) {
    if(n<=1) return 0;
    float u=rnd_uniform(state);
    int lo=0,hi=n-1;
    while(lo<hi){int mid=(lo+hi)>>1;if(u<=(float)cdf[mid])hi=mid;else lo=mid+1;}
    return lo;
}

__device__ float sample_mie_theta(unsigned int* state,const double* cdf_table,const double* theta_table,int ntheta,int pidx) {
    float u=rnd_uniform(state);int base=pidx*ntheta;int lo=0,hi=ntheta-1;
    while(lo<hi){int mid=(lo+hi)>>1;if(u<=(float)cdf_table[base+mid])hi=mid;else lo=mid+1;}
    int j=lo;if(j<=0)return (float)theta_table[0];
    float c0=(float)cdf_table[base+j-1],c1=(float)cdf_table[base+j];
    float t0=(float)theta_table[j-1],t1=(float)theta_table[j];
    float f=(c1>c0)?fminf(fmaxf((u-c0)/(c1-c0),0.0f),1.0f):0.0f;
    return t0+f*(t1-t0);
}

__device__ void mie_scatter_direction(unsigned int* state,const double* cdf_table,const double* theta_table,int ntheta,int pidx,
                                      float* vx,float* vy,float* vz) {
    float theta=sample_mie_theta(state,cdf_table,theta_table,ntheta,pidx);
    float phi=2.0f*PI_F*rnd_uniform(state);
    float e1x,e1y,e1z,e2x,e2y,e2z;perpendicular_basis(*vx,*vy,*vz,&e1x,&e1y,&e1z,&e2x,&e2y,&e2z);
    float st=sinf(theta),ct=cosf(theta),cp=cosf(phi),sp=sinf(phi);
    float ox=ct*(*vx)+st*(cp*e1x+sp*e2x);
    float oy=ct*(*vy)+st*(cp*e1y+sp*e2y);
    float oz=ct*(*vz)+st*(cp*e1z+sp*e2z);
    *vx=ox;*vy=oy;*vz=oz;normalize3(vx,vy,vz);
}


// Volume-preserving spheroid morphology applies only to the incoherent
// wave channel; the geometric channel uses the rough sphere. For q=a/b
// and volume-equivalent radius r, a=r*q^(2/3), b=r*q^(-1/3), a*b^2=r^3.
// q>1 is prolate and q<1 oblate. Configured q=1 with isotropic orientation.

__device__ void random_unit_axis(unsigned int* state,float* ux,float* uy,float* uz){
    float cz=2.0f*rnd_uniform(state)-1.0f;
    float az=2.0f*PI_F*rnd_uniform(state);
    float sz=sqrtf(fmaxf(0.0f,1.0f-cz*cz));
    *ux=sz*cosf(az);*uy=sz*sinf(az);*uz=cz;
}

// Axial orientation sampler.  The model code is:
//   0 = isotropic (random_unit_axis)
//   1 = local-tangential, head-tail-symmetric von-Mises-Fisher mixture.
// For model 1 and kappa=0 this calls random_unit_axis directly, preserving the
// exact RNG path.  For kappa>0, first sample the ordinary 3-D vMF law
// p(c)=kappa*exp(kappa*c)/(2*sinh(kappa)), c=cos(theta), by exact inverse CDF,
// then flip the sampled axis sign with probability 1/2.  The resulting axial
// density is proportional to cosh(kappa * u.flow), so +u and -u are equivalent.
__device__ void sample_spheroid_axis(
    unsigned int* state,int orientation_model,float kappa,
    float fx,float fy,float fz,float* ux,float* uy,float* uz){
    if(orientation_model==0 || !(kappa>1e-7f) || !isfinite(kappa)){
        random_unit_axis(state,ux,uy,uz);return;
    }
    normalize3(&fx,&fy,&fz);
    if(!(fx*fx+fy*fy+fz*fz>0.5f)){
        random_unit_axis(state,ux,uy,uz);return;
    }
    // Stable inverse CDF. exp(-2*kappa) safely underflows to zero at strong alignment.
    float u=fminf(fmaxf(rnd_uniform(state),1e-12f),1.0f-1e-7f);
    float em2=expf(-2.0f*kappa);
    float c=1.0f+logf(u+(1.0f-u)*em2)/kappa;
    c=fminf(fmaxf(c,-1.0f),1.0f);
    float e1x,e1y,e1z,e2x,e2y,e2z;
    perpendicular_basis(fx,fy,fz,&e1x,&e1y,&e1z,&e2x,&e2y,&e2z);
    float phi=2.0f*PI_F*rnd_uniform(state);
    float st=sqrtf(fmaxf(0.0f,1.0f-c*c));
    float cp=cosf(phi),sp=sinf(phi);
    *ux=c*fx+st*(cp*e1x+sp*e2x);
    *uy=c*fy+st*(cp*e1y+sp*e2y);
    *uz=c*fz+st*(cp*e1z+sp*e2z);
    if(rnd_uniform(state)<0.5f){*ux=-*ux;*uy=-*uy;*uz=-*uz;}
    normalize3(ux,uy,uz);
}

__global__ void orientation_sampler_test_kernel(
    const int n,const unsigned int seed,const int orientation_model,const float kappa,
    float* ux_out,float* uy_out,float* uz_out){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;if(tid>=n)return;
    unsigned int state=seed+(unsigned int)tid*104729u+29u;
    float ux,uy,uz;
    sample_spheroid_axis(&state,orientation_model,kappa,1.0f,0.0f,0.0f,&ux,&uy,&uz);
    ux_out[tid]=ux;uy_out[tid]=uy;uz_out[tid]=uz;
}

__device__ float spheroid_F(float x,float y,float z,float ux,float uy,float uz,float invb2,float delta){
    float pu=x*ux+y*uy+z*uz;
    return invb2*(x*x+y*y+z*z)+delta*pu*pu-1.0f;
}

__device__ void spheroid_normal(float x,float y,float z,float ux,float uy,float uz,float invb2,float delta,
                                 float* nx,float* ny,float* nz){
    float pu=x*ux+y*uy+z*uz;
    *nx=invb2*x+delta*pu*ux;
    *ny=invb2*y+delta*pu*uy;
    *nz=invb2*z+delta*pu*uz;
    normalize3(nx,ny,nz);
}

__device__ int spheroid_line_roots(float ox,float oy,float oz,float dx,float dy,float dz,
                                   float ux,float uy,float uz,float invb2,float delta,
                                   float* t0,float* t1){
    float du=dx*ux+dy*uy+dz*uz;
    float ou=ox*ux+oy*uy+oz*uz;
    float A=invb2*(dx*dx+dy*dy+dz*dz)+delta*du*du;
    float B=2.0f*(invb2*(ox*dx+oy*dy+oz*dz)+delta*ou*du);
    float Cc=invb2*(ox*ox+oy*oy+oz*oz)+delta*ou*ou-1.0f;
    float disc=B*B-4.0f*A*Cc;
    if(!(A>0.0f) || disc<0.0f)return 0;
    float sd=sqrtf(fmaxf(0.0f,disc));
    float r0=(-B-sd)/(2.0f*A),r1=(-B+sd)/(2.0f*A);
    if(r0>r1){float q=r0;r0=r1;r1=q;}
    *t0=r0;*t1=r1;return 1;
}

__device__ int spheroid_sample_entry_projected(unsigned int* state,float vx,float vy,float vz,float a,float b,
                                                float ux,float uy,float uz,
                                                float* x,float* y,float* z,float* gnx,float* gny,float* gnz){
    float invb2=1.0f/(b*b);
    float delta=1.0f/(a*a)-invb2;
    float e1x,e1y,e1z,e2x,e2y,e2z;
    perpendicular_basis(vx,vy,vz,&e1x,&e1y,&e1z,&e2x,&e2y,&e2z);
    float du=dot3(ux,uy,uz,vx,vy,vz);
    float rproj2=b*b+(a*a-b*b)*fmaxf(0.0f,1.0f-du*du);
    float rbound=sqrtf(fmaxf(b*b,rproj2));
    float rmax=fmaxf(a,b);
    // Rejection from a tight projected bounding disk gives an exactly uniform
    // offset over the spheroid's projected silhouette after conditioning on hit.
    for(int attempt=0;attempt<64;++attempt){
        float rr=rbound*sqrtf(rnd_uniform(state));
        float ph=2.0f*PI_F*rnd_uniform(state);
        float ox=rr*(cosf(ph)*e1x+sinf(ph)*e2x)-2.5f*rmax*vx;
        float oy=rr*(cosf(ph)*e1y+sinf(ph)*e2y)-2.5f*rmax*vy;
        float oz=rr*(cosf(ph)*e1z+sinf(ph)*e2z)-2.5f*rmax*vz;
        float t0,t1;
        if(!spheroid_line_roots(ox,oy,oz,vx,vy,vz,ux,uy,uz,invb2,delta,&t0,&t1))continue;
        float t=(t0>1e-9f*rmax)?t0:((t1>1e-9f*rmax)?t1:-1.0f);
        if(t<=0.0f)continue;
        *x=ox+t*vx;*y=oy+t*vy;*z=oz+t*vz;
        spheroid_normal(*x,*y,*z,ux,uy,uz,invb2,delta,gnx,gny,gnz);
        if(dot3(vx,vy,vz,*gnx,*gny,*gnz)<-1e-6f)return 1;
    }
    return 0;
}

__device__ float spheroid_next_boundary_distance(float x,float y,float z,float vx,float vy,float vz,
                                                  float ux,float uy,float uz,float invb2,float delta,float scale){
    float t0,t1;
    if(!spheroid_line_roots(x,y,z,vx,vy,vz,ux,uy,uz,invb2,delta,&t0,&t1))return -1.0f;
    float eps=fmaxf(1e-5f*scale,1e-12f);
    float t=INF_F;
    if(t0>eps)t=t0;
    if(t1>eps&&t1<t)t=t1;
    return t<INF_F*0.5f?t:-1.0f;
}

__device__ int spheroid_fresnel_scatter_direction_only_3d_ex(
    unsigned int* state,float n_medium,float n_particle,float k_particle,float wavelength,float rough_alpha,
    float r_equiv,float aspect_ratio,int orientation_model,float orientation_kappa,float flow_fx,float flow_fy,float flow_fz,
    int max_bounces,float* vx,float* vy,float* vz,float* internal_path,
    int* p_reflections,int* p_entry_reflections,int* p_internal_reflections,int* p_absorptions,
    int* prod_failure_code,int* prod_failure_subcode,int* out_macro_bounces,int* out_micro_order,
    float* out_entry_incidence_deg,float* out_exit_angle_deg){
    prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_OK,PROD_SUB_NONE);
    if(out_macro_bounces)*out_macro_bounces=0;if(out_micro_order)*out_micro_order=0;
    if(out_entry_incidence_deg)*out_entry_incidence_deg=-1.0f;if(out_exit_angle_deg)*out_exit_angle_deg=-1.0f;
    if(r_equiv<=0.0f)return 1;
    if(!(aspect_ratio>0.0f) || !isfinite(aspect_ratio)){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_SPHEROID_GEOMETRY_FAILURE,PROD_SUB_SPHEROID_ASPECT);return 0;}
    if(fabsf(aspect_ratio-1.0f)<1e-7f){
        float lx=0.0f,ly=0.0f,lz=0.0f,lvx=*vx,lvy=*vy,lvz=*vz,ip=0.0f;
        int ps=sphere_fresnel_interaction_3d_ex(state,n_medium,n_particle,k_particle,wavelength,rough_alpha,r_equiv,max_bounces,
            &lx,&ly,&lz,&lvx,&lvy,&lvz,&ip,p_reflections,p_entry_reflections,p_internal_reflections,p_absorptions,
            prod_failure_code,prod_failure_subcode,out_macro_bounces,out_micro_order,out_entry_incidence_deg,out_exit_angle_deg);
        *internal_path += ip;if(ps==1){*vx=lvx;*vy=lvy;*vz=lvz;}return ps;
    }
    const int MICRO_WATCHDOG_LIMIT=4096;
    float q13=powf(aspect_ratio,1.0f/3.0f);float a=r_equiv*q13*q13;float b=r_equiv/q13;
    if(!(a>0.0f&&b>0.0f&&isfinite(a)&&isfinite(b))){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_SPHEROID_GEOMETRY_FAILURE,PROD_SUB_SPHEROID_AXES);return 0;}
    float ux,uy,uz;sample_spheroid_axis(state,orientation_model,orientation_kappa,flow_fx,flow_fy,flow_fz,&ux,&uy,&uz);
    float x,y,z,gnx,gny,gnz;
    if(!spheroid_sample_entry_projected(state,*vx,*vy,*vz,a,b,ux,uy,uz,&x,&y,&z,&gnx,&gny,&gnz)){
        prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_SPHEROID_GEOMETRY_FAILURE,PROD_SUB_SPHEROID_ENTRY);return 0;
    }
    float invb2=1.0f/(b*b),delta=1.0f/(a*a)-invb2,scale=fmaxf(a,b);
    const float entry_cosi=fminf(fmaxf(-dot3(*vx,*vy,*vz,gnx,gny,gnz),0.0f),1.0f);
    if(out_entry_incidence_deg)*out_entry_incidence_deg=acosf(entry_cosi)*(180.0f/PI_F);
    int top=1,order=0,mfc=MICRO_FAIL_NONE;
    int ms=smith_microsurface_dielectric_walk_ex(state,rough_alpha,MICRO_WATCHDOG_LIMIT,gnx,gny,gnz,
            n_medium,0.0f,n_particle,k_particle,vx,vy,vz,&top,&order,&mfc);
    if(out_micro_order)*out_micro_order=order;
    if(ms!=MICRO_ESCAPE){prod_set_failure(prod_failure_code,prod_failure_subcode,ms==MICRO_WATCHDOG_FAILURE?PROD_PARTICLE_MICROSURFACE_WATCHDOG:PROD_PARTICLE_MICROSURFACE_NUMERICAL,mfc);return 0;}
    if(top){
        if(out_exit_angle_deg){float co=fminf(fmaxf(dot3(*vx,*vy,*vz,gnx,gny,gnz),0.0f),1.0f);*out_exit_angle_deg=acosf(co)*(180.0f/PI_F);}
        (*p_reflections)++;(*p_entry_reflections)++;return 1;
    }
    if(dot3(*vx,*vy,*vz,gnx,gny,gnz)>=0.0f){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_SPHEROID_GEOMETRY_FAILURE,PROD_SUB_SPHEROID_ENTRY);return 0;}
    float alpha_abs=(wavelength>0.0f&&k_particle>0.0f)?(4.0f*PI_F*k_particle/wavelength):0.0f;
    for(int bounce=0;bounce<=max_bounces;++bounce){
        if(out_macro_bounces)*out_macro_bounces=bounce;
        float chord=spheroid_next_boundary_distance(x,y,z,*vx,*vy,*vz,ux,uy,uz,invb2,delta,scale);
        if(chord<=0.0f){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_SPHEROID_GEOMETRY_FAILURE,PROD_SUB_SPHEROID_CHORD);return 0;}
        if(alpha_abs>0.0f){float survival=expf(-alpha_abs*chord);if(rnd_uniform(state)>survival){*internal_path+=chord;(*p_absorptions)++;return 2;}}
        x+=chord*(*vx);y+=chord*(*vy);z+=chord*(*vz);*internal_path+=chord;
        float nox,noy,noz;spheroid_normal(x,y,z,ux,uy,uz,invb2,delta,&nox,&noy,&noz);
        float bx=-nox,by=-noy,bz=-noz;top=1;order=0;mfc=MICRO_FAIL_NONE;
        ms=smith_microsurface_dielectric_walk_ex(state,rough_alpha,MICRO_WATCHDOG_LIMIT,bx,by,bz,
                n_particle,k_particle,n_medium,0.0f,vx,vy,vz,&top,&order,&mfc);
        if(out_micro_order)*out_micro_order=order;
        if(ms!=MICRO_ESCAPE){prod_set_failure(prod_failure_code,prod_failure_subcode,ms==MICRO_WATCHDOG_FAILURE?PROD_PARTICLE_MICROSURFACE_WATCHDOG:PROD_PARTICLE_MICROSURFACE_NUMERICAL,mfc);return 0;}
        if(!top){if(dot3(*vx,*vy,*vz,nox,noy,noz)<=0.0f){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_SPHEROID_GEOMETRY_FAILURE,PROD_SUB_SPHEROID_EXIT_SIDE);return 0;}if(out_exit_angle_deg){float co=fminf(fmaxf(dot3(*vx,*vy,*vz,nox,noy,noz),0.0f),1.0f);*out_exit_angle_deg=acosf(co)*(180.0f/PI_F);}return 1;}
        if(dot3(*vx,*vy,*vz,nox,noy,noz)>=0.0f){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_SPHEROID_GEOMETRY_FAILURE,PROD_SUB_SPHEROID_INTERNAL_SIDE);return 0;}
        (*p_reflections)++;(*p_internal_reflections)++;
        if(bounce==max_bounces){prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_PARTICLE_MACRO_BOUNCE_LIMIT,PROD_SUB_SPHEROID_BOUNCE_LIMIT);return 0;}
    }
    prod_set_failure(prod_failure_code,prod_failure_subcode,PROD_OTHER_EXPLICIT_FAILURE,PROD_SUB_NONE);return 0;
}
__device__ int spheroid_fresnel_scatter_direction_only_3d(
    unsigned int* state,float n_medium,float n_particle,float k_particle,float wavelength,float rough_alpha,
    float r_equiv,float aspect_ratio,int orientation_model,float orientation_kappa,float flow_fx,float flow_fy,float flow_fz,
    int max_bounces,float* vx,float* vy,float* vz,float* internal_path,
    int* p_reflections,int* p_entry_reflections,int* p_internal_reflections,int* p_absorptions){
    int pc=PROD_OK,ps=PROD_SUB_NONE,mb=0,mo=0;float ei=-1.0f,eo=-1.0f;
    return spheroid_fresnel_scatter_direction_only_3d_ex(state,n_medium,n_particle,k_particle,wavelength,rough_alpha,r_equiv,aspect_ratio,
        orientation_model,orientation_kappa,flow_fx,flow_fy,flow_fz,max_bounces,vx,vy,vz,internal_path,
        p_reflections,p_entry_reflections,p_internal_reflections,p_absorptions,&pc,&ps,&mb,&mo,&ei,&eo);
}

// The morphology-wave event obtains its outgoing direction from rough
// dielectric interface physics but keeps the macroscopic scattering site.
// A sphere-chord displacement is inappropriate when wave cross-section
// exceeds projected area.
__device__ int rough_particle_scatter_direction_only_3d(
    unsigned int* state,float n_medium,float n_particle,float k_particle,float wavelength,float rough_alpha,float radius,int max_bounces,
    float* vx,float* vy,float* vz,
    int* p_reflections,int* p_entry_reflections,int* p_internal_reflections,int* p_absorptions) {
    float lx=0.0f,ly=0.0f,lz=0.0f;
    float lvx=*vx,lvy=*vy,lvz=*vz,internal_path=0.0f;
    int ps=sphere_fresnel_interaction_3d(state,n_medium,n_particle,k_particle,wavelength,rough_alpha,radius,max_bounces,
        &lx,&ly,&lz,&lvx,&lvy,&lvz,&internal_path,p_reflections,p_entry_reflections,p_internal_reflections,p_absorptions);
    if(ps==1){*vx=lvx;*vy=lvy;*vz=lvz;}
    return ps;
}

// complete connected-cell acrylic topology.  The physical acrylic body is the UNION of
//   sidewall: Rin<=r<=Rout, Zmin<=z<=Zwall_top
//   bottom:   0<=r<=Rout, Zlow<=z<=Zmin
// It is NOT a rectangular cylindrical shell.  In particular r=Rin is an optical
// interface only for z>=Zmin; below Zmin both sides are the same bottom-disc acrylic.
// Surface codes: 1=inner cylindrical wall, 2=outer cylindrical wall,
// 3=bottom external face, 4=upper sidewall face, 5=bottom water aperture.
// Optical boundary table:
//   r=Rin,  Zmin<=z<=Zwall_top   acrylic <-> water/headspace   YES
//   r=Rout, Zlow<=z<=Zwall_top    acrylic <-> external air      YES
//   z=Zlow, r<=Rout              acrylic <-> bottom external   YES
//   z=Zwall_top, Rin<=r<=Rout    acrylic <-> external air      YES
//   z=Zmin, r<Rin                acrylic <-> water              YES
//   z=Zmin, Rin<=r<=Rout         bottom <-> sidewall acrylic    NO

// Forward declaration; implementation is below in the shared free-surface section.
__device__ float free_surface_z(float x,float y,float ZMEAN,float R,int MODEL,float DELTA_H,float CORE_R);

__device__ int acrylic_cylinder_positive_roots_d(
    float x,float y,float vx,float vy,float R,double* t0,double* t1){
    if(t0)*t0=1.0e300;if(t1)*t1=1.0e300;
    double X=(double)x,Y=(double)y,VX=(double)vx,VY=(double)vy,RR=(double)R;
    double a=VX*VX+VY*VY;if(!(a>1e-30)||!isfinite(a))return 0;
    double b=X*VX+Y*VY,c=X*X+Y*Y-RR*RR,d=b*b-a*c;
    double dscale=fabs(b*b)+fabs(a*c)+1e-300;
    double dtol=16.0*2.220446049250313080847263336181640625e-16*dscale;
    if(d<0.0){if(d>=-dtol)d=0.0;else return 0;}
    double sd=sqrt(fmax(d,0.0)),q=-b-copysign(sd,b);
    double r0=(fabs(q)>1e-300)?q/a:(-b-sd)/a;
    double r1=(fabs(q)>1e-300)?c/q:(-b+sd)/a;
    double a0=1.0e300,a1=1.0e300;
    if(r0>0.0&&isfinite(r0))a0=r0;
    if(r1>0.0&&isfinite(r1)){if(r1<a0){a1=a0;a0=r1;}else a1=r1;}
    int n=0;if(a0<1.0e299)n++;if(a1<1.0e299)n++;
    if(t0)*t0=a0;if(t1)*t1=a1;return n;
}

__device__ double acrylic_support_tol(float Rin,float Rout,float Zmin,float Zlow,float Zwall_top){
    double scale=fmax(fmax(fabs((double)Rout),fabs((double)Zlow)),fabs((double)Zwall_top));
    scale=fmax(scale,fmax(fabs((double)Rin),fabs((double)Zmin)));
    return 8.0*1.1920928955078125e-7*fmax(scale,1.0e-3);
}

// First positive finite-cylinder hit.  Mathematical cylinder roots are accepted only
// when the hit lies on the physical axial support.  Both roots are tested because the
// first mathematical crossing can be outside the finite tube while a later crossing is
// on a real surface.
__device__ double cell_cylinder_hit_finite_forward_d(
    float x,float y,float z,float vx,float vy,float vz,float R,float zlo,float zhi){
    double r0=1.0e300,r1=1.0e300;acrylic_cylinder_positive_roots_d(x,y,vx,vy,R,&r0,&r1);
    double tol=acrylic_support_tol(R,R,zlo,zlo,zhi),best=1.0e300;double roots[2]={r0,r1};
    for(int k=0;k<2;++k){double t=roots[k];if(!(t>0.0)||t>=1.0e299)continue;double zh=(double)z+t*(double)vz;
        if(zh>=(double)zlo-tol && zh<=(double)zhi+tol && t<best)best=t;}
    return best;
}


// Bore-side wall exit, unlike the acrylic-solid positive-root helper, includes
// an immediate interface for an outward/tangent boundary state. The existing
// water-state tolerance bounds all permissible reconciliation; no larger domain.
__device__ double bore_wall_hit_finite_d(
    float x,float y,float z,float vx,float vy,float vz,float R,float zlo,float zhi){
    double X=x,Y=y,VX=vx,VY=vy,RR=(double)R*R;
    double rr2=X*X+Y*Y,c=rr2-RR,b=X*VX+Y*VY,a=VX*VX+VY*VY;
    double radial_tol=8.0*1.1920928955078125e-7*RR;
    double ztol=acrylic_support_tol(R,R,zlo,zlo,zhi);
    if(c>radial_tol || !isfinite(c)||!isfinite(b)||!isfinite(a))return -1.0;
    if(c>=0.0){
        if(b>=0.0 && a>0.0 && z>=zlo-ztol && z<=zhi+ztol)return 0.0;
        // A rounding-scale outside state travelling inward belongs to the entry
        // interface, so calculate its subsequent exit from that interface.
        b*=sqrt(RR/rr2);c=0.0;
    }
    if(!(a>1e-30))return 1.0e300;
    double d=b*b-a*c;if(d<0.0)return 1.0e300;
    double sd=sqrt(d),q=-b-copysign(sd,b);
    double t=(b<0.0)?q/a:((q!=0.0)?c/q:0.0);
    if(!(t>=0.0)||!isfinite(t))return 1.0e300;
    double zh=(double)z+t*(double)vz;
    return (zh>=zlo-ztol&&zh<=zhi+ztol)?t:1.0e300;
}

__global__ void bore_wall_test_kernel(const float* rows,const int n,double* hits){
    int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
    const float* r=rows+9*i;
    hits[i]=bore_wall_hit_finite_d(r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[8]);
}


// Constrain ONLY the deliberate free-surface water-side offset at the bore rim.
// The original interface point must already satisfy the existing water tolerance.
// This never repairs a materially invalid transport position or consumes RNG.
__device__ int water_surface_bore_offset(float* x,float* y,float* z,
 float nx,float ny,float nz,float R,float ZMIN,float ZMEAN,int MODEL,float DH,float CORE){
    double rr2=(double)(*x)*(double)(*x)+(double)(*y)*(double)(*y),RR=(double)R*R;
    double tol=8.0*1.1920928955078125e-7*RR;
    if(!isfinite(rr2)||rr2>RR+tol)return 0;
    float xx=*x+EPS_F*nx,yy=*y+EPS_F*ny,zz=*z+EPS_F*nz;
    double rad=hypot((double)xx,(double)yy);
    if(rad>(double)R){
        // A small representational margin prevents float32 storage rounding out.
        double target=(double)R*(1.0-2.0*1.1920928955078125e-7);
        xx=(float)((double)xx*target/rad);yy=(float)((double)yy*target/rad);
        float surf=free_surface_z(xx,yy,ZMEAN,R,MODEL,DH,CORE);
        if(zz>=surf)zz=nextafterf(surf,-INF_F);
    }
    if(!isfinite(xx)||!isfinite(yy)||!isfinite(zz)||zz<=ZMIN ||
       zz>=free_surface_z(xx,yy,ZMEAN,R,MODEL,DH,CORE))return 0;
    *x=xx;*y=yy;*z=zz;return 1;
}
__global__ void water_surface_offset_test_kernel(const float* rows,int n,float* out){
    int i=blockDim.x*blockIdx.x+threadIdx.x;if(i>=n)return;
    const float* r=rows+12*i;float x=r[0],y=r[1],z=r[2];
    int ok=water_surface_bore_offset(&x,&y,&z,r[3],r[4],r[5],r[6],r[7],r[8],(int)r[9],r[10],r[11]);
    out[4*i]=x;out[4*i+1]=y;out[4*i+2]=z;out[4*i+3]=(float)ok;
}

// Dedicated diagnostic scratch words 7..27 retain the first invalid proposed
// water advance, even if diagnostic grids are disabled. No random draws.
__device__ void capture_water_advance_failure(unsigned int* record,unsigned int gid,int step,
 float x0,float y0,float z0,float vx,float vy,float vz,float travel,
 float side,float top,float bottom,float boundary,int particle,float x,float y,float z){
    unsigned int previous=atomicAdd(record+7,1U);if(previous!=0U)return;
    record[8]=gid;record[9]=(unsigned int)step;record[10]=(unsigned int)particle;
    float values[16]={x0,y0,z0,vx,vy,vz,travel,side,top,bottom,boundary,x,y,z,0.0f,0.0f};
    for(int j=0;j<16;++j)record[11+j]=__float_as_uint(values[j]);record[27]=1U;
}

// Complete apparatus region classifier.  Region IDs are intentionally distinct:
// 1=water, 2=acrylic sidewall, 3=acrylic bottom, 4=headspace, 5=external air,
// 0=ambiguous/interface.  Zmean defines the free-surface volume; Zwall_top defines
// the independent physical upper extent of the open acrylic tube.
__device__ int acrylic_region_classify(
    float x,float y,float z,float Rin,float Rout,float Zmin,float Zlow,float Zmean,float Zwall_top,
    int free_surface_model,float vortex_delta_h,float vortex_core_radius){
    double rr=hypot((double)x,(double)y),tol=acrylic_support_tol(Rin,Rout,Zmin,Zlow,Zwall_top);
    if((double)z>(double)Zlow+tol && (double)z<(double)Zmin-tol && rr<(double)Rout-tol)return 3;
    if((double)z>(double)Zmin+tol && (double)z<(double)Zwall_top-tol && rr>(double)Rin+tol && rr<(double)Rout-tol)return 2;
    if((double)z>=(double)Zmin-tol && (double)z<=(double)Zwall_top+tol && rr<(double)Rin-tol){
        float zs=free_surface_z(x,y,Zmean,Rin,free_surface_model,vortex_delta_h,vortex_core_radius);
        if((double)z<(double)zs-tol)return 1;
        if((double)z>(double)zs+tol)return 4;
        return 0;
    }
    if((double)z<(double)Zlow-tol || (double)z>(double)Zwall_top+tol || rr>(double)Rout+tol)return 5;
    return 0;
}

__device__ void acrylic_consider_candidate(double t,int surf,double* best,int* best_surf){
    if(t>0.0&&isfinite(t)&&t<*best){*best=t;*best_surf=surf;}
}

// Shared progress-watchdog primitive.  Real optical reflection count is not a
// failure criterion; only repeated same-surface state with effectively no motion
// increments the stuck counter.
__device__ int acrylic_progress_watchdog_update(int surf,int last_surf,float position_delta,float direction_delta,int stuck){
    return (surf==last_surf && position_delta<=1.0e-12f && direction_delta<=1.0e-7f)?(stuck+1):0;
}

// Return the nearest PHYSICAL boundary of the connected acrylic union:
// sidewall Rin<=r<=Rout, Zmin<=z<=Zwall_top plus bottom disc
// 0<=r<=Rout, Zlow<=z<=Zmin.  Surface codes:
// 1 inner cylinder; 2 outer cylinder; 3 bottom external face;
// 4 annular top face; 5 bottom water aperture.
__device__ int acrylic_connected_next_surface(
    float x,float y,float z,float vx,float vy,float vz,
    float Rin,float Rout,float Zmin,float Zlow,float Zwall_top,double* t_out){
    double best=1.0e300;int surf=0;
    double tol=acrylic_support_tol(Rin,Rout,Zmin,Zlow,Zwall_top);
    double r0=1.0e300,r1=1.0e300;acrylic_cylinder_positive_roots_d(x,y,vx,vy,Rin,&r0,&r1);
    double roots_i[2]={r0,r1};
    for(int k=0;k<2;++k){double t=roots_i[k];if(t>=1.0e299)continue;double zh=(double)z+t*(double)vz;
        if(zh>=(double)Zmin-tol && zh<=(double)Zwall_top+tol)acrylic_consider_candidate(t,1,&best,&surf);
    }
    r0=r1=1.0e300;acrylic_cylinder_positive_roots_d(x,y,vx,vy,Rout,&r0,&r1);double roots_o[2]={r0,r1};
    for(int k=0;k<2;++k){double t=roots_o[k];if(t>=1.0e299)continue;double zh=(double)z+t*(double)vz;
        if(zh>=(double)Zlow-tol && zh<=(double)Zwall_top+tol)acrylic_consider_candidate(t,2,&best,&surf);
    }
    if(vz<0.0f){
        double t=((double)Zlow-(double)z)/(double)vz;
        if(t>0.0&&isfinite(t)){double X=(double)x+t*(double)vx,Y=(double)y+t*(double)vy,rr=hypot(X,Y);
            if(rr<=(double)Rout+tol)acrylic_consider_candidate(t,3,&best,&surf);}
    }
    if(vz>0.0f){
        double t=((double)Zwall_top-(double)z)/(double)vz;
        if(t>0.0&&isfinite(t)){double X=(double)x+t*(double)vx,Y=(double)y+t*(double)vy,rr=hypot(X,Y);
            if(rr>=(double)Rin-tol && rr<=(double)Rout+tol)acrylic_consider_candidate(t,4,&best,&surf);}
        // The only top-of-bottom-disc optical patch is the water aperture r<Rin.
        double ta=((double)Zmin-(double)z)/(double)vz;
        if(ta>0.0&&isfinite(ta)){double X=(double)x+ta*(double)vx,Y=(double)y+ta*(double)vy,rr=hypot(X,Y);
            if(rr<(double)Rin-tol)acrylic_consider_candidate(ta,5,&best,&surf);}
    }
    if(t_out)*t_out=best;return surf;
}

// current-state trapped-orbit classifier.  This is deterministic and consumes no RNG.
// Specular reflections in coaxial cylinders / horizontal planes conserve |vz| and the
// cylindrical angular-momentum invariant Lz=x*vy-y*vx.  Those invariants determine the
// incidence magnitudes at all potentially reachable optical exits.  The classifier is
// deliberately sufficient/conservative: it returns trapped only when every reachable exit
// is necessarily TIR.  It does NOT rely on whether every bounce happened to be TIR.
__device__ int acrylic_tir_for_ci(double ci,double n1,double n2){
    if(!(n1>0.0)||!(n2>0.0)||n1<=n2)return 0;
    ci=fmin(fmax(ci,0.0),1.0);
    double st=(n1/n2)*sqrt(fmax(0.0,1.0-ci*ci));
    return st>=1.0;
}
__device__ int acrylic_current_state_trapped_tir(
    float x,float y,float z,float vx,float vy,float vz,
    float n_water,float n_acrylic,float n_air,float n_bottom_external,
    float Rin,float Rout,float Zmin,float Zlow,float Zwall_top){
    (void)z;(void)Zmin;(void)Zlow;(void)Zwall_top;
    double dx=(double)vx,dy=(double)vy,dz=(double)vz;
    double speed2=dx*dx+dy*dy+dz*dz;
    if(!(speed2>0.0)||!isfinite(speed2))return 0;
    double vxy2=dx*dx+dy*dy;
    double L=(double)x*dy-(double)y*dx;
    double root_speed=sqrt(speed2);
    const double reach_tol=4.0*(double)EPS_F; // undo only the deliberate post-interface displacement scale

    // Outer wall is reachable for any non-axial transverse orbit in the finite sidewall/bottom solid.
    if(vxy2>0.0){
        double tang=L/(double)Rout;
        double rad2=vxy2-tang*tang;
        if(rad2<0.0 && rad2>-1.0e-12*speed2)rad2=0.0;
        if(rad2<0.0)return 0;
        double ci_outer=sqrt(rad2)/root_speed;
        if(!acrylic_tir_for_ci(ci_outer,(double)n_acrylic,(double)n_air))return 0;

        // Inner cylinder is reachable when the conserved impact parameter is inside Rin.
        double impact=fabs(L)/sqrt(vxy2);
        if(impact<=(double)Rin+reach_tol){
            double ti=L/(double)Rin;
            double ri2=vxy2-ti*ti;
            if(ri2<0.0 && ri2>-1.0e-12*speed2)ri2=0.0;
            if(ri2>=0.0){
                double ci_inner=sqrt(ri2)/root_speed;
                // Water is the harder-to-trap inner-wall medium; TIR against water implies TIR against headspace air.
                if(!acrylic_tir_for_ci(ci_inner,(double)n_acrylic,(double)n_water))return 0;
            }
        }

        // A bottom-disc water-aperture encounter is possible only for an orbit whose physical
        // impact parameter is meaningfully inside Rin.  Near-tangent values within the explicit
        // EPS_F displacement scale are treated as the inner-wall tangent orbit, not as invented
        // aperture penetration caused by the numerical side offset.
        if(impact<(double)Rin-reach_tol && fabs(dz)>0.0){
            double ci_ax=fabs(dz)/root_speed;
            if(!acrylic_tir_for_ci(ci_ax,(double)n_acrylic,(double)n_water))return 0;
        }
    }

    // Axial outer/top/bottom exits are air (or bottom_external).  If vz!=0 they are potentially
    // reachable and must be TIR.  The water-aperture check above covers the harder axial water exit.
    if(fabs(dz)>0.0){
        double ci_ax=fabs(dz)/root_speed;
        // A purely axial orbit inside the r<Rin bottom-disc aperture can reach the water
        // patch directly; include the harder water criterion even though Lz/impact is undefined.
        if(vxy2<=1.0e-30*speed2 && hypot((double)x,(double)y)<(double)Rin-reach_tol){
            if(!acrylic_tir_for_ci(ci_ax,(double)n_acrylic,(double)n_water))return 0;
        }
        if(!acrylic_tir_for_ci(ci_ax,(double)n_acrylic,(double)n_air))return 0;
        if(!acrylic_tir_for_ci(ci_ax,(double)n_acrylic,(double)n_bottom_external))return 0;
    }
    return 1;
}

// Propagate a ray already INSIDE the connected acrylic solid. Return codes:
// 1=water; 2=external air through outer wall/top annulus; 3=bottom external;
// 4=headspace through the inner wall above the local free surface; 5=physically trapped all-TIR acrylic path; 0=failure.
__device__ int acrylic_connected_solid_ex(
    unsigned int* state,float n_water,float n_acrylic,float n_air,float n_bottom_external,
    float Rin,float Rout,float Zmin,float Zlow,float Zmean,float Zwall_top,
    int free_surface_model,float vortex_delta_h,float vortex_core_radius,int unused_bounce_limit,
    float* x,float* y,float* z,float* vx,float* vy,float* vz,float* acrylic_path,
    int* cell_inner_reflections,int* cell_outer_reflections,int* bottom_inner_reflections,int* bottom_outer_reflections,
    int* cell_tir,int* bottom_tir,int* failure_subcode,int* out_bounces){
    (void)unused_bounce_limit;
    if(failure_subcode)*failure_subcode=PROD_SUB_NONE;if(out_bounces)*out_bounces=0;
    const int emergency_cap=16384;int stuck=0,last_surf=-1;
    for(int interaction=0;interaction<emergency_cap;++interaction){
        if(out_bounces)*out_bounces=interaction;
        double td=1.0e300;int surf=acrylic_connected_next_surface(*x,*y,*z,*vx,*vy,*vz,Rin,Rout,Zmin,Zlow,Zwall_top,&td);
        if(!surf||!(td>0.0)||!isfinite(td)||td>=1.0e299){if(failure_subcode)*failure_subcode=PROD_SUB_ACRYLIC_NO_INTERSECTION;return 0;}
        float t=(float)td;float px=*x,py=*y,pz=*z,pvx=*vx,pvy=*vy,pvz=*vz;
        *x+=t*(*vx);*y+=t*(*vy);*z+=t*(*vz);*acrylic_path+=t;
        float a,b,c;int tir=0;float ci=0.0f,R=1.0f;
        if(surf==1){
            float nx,ny;radial_normal(*x,*y,&nx,&ny);
            float zsurf=free_surface_z(*x,*y,Zmean,Rin,free_surface_model,vortex_delta_h,vortex_core_radius);
            float n_inside=((*z<=zsurf+EPS_F)?n_water:n_air);
            ci=fminf(fmaxf(-dot3(*vx,*vy,*vz,nx,ny,0.0f),0.0f),1.0f);R=fresnel_R(ci,n_acrylic,n_inside,&tir);
            if(!tir&&rnd_uniform(state)>=R&&refract3(*vx,*vy,*vz,nx,ny,0.0f,n_acrylic,n_inside,&a,&b,&c)){
                *vx=a;*vy=b;*vz=c;*x-=EPS_F*nx;*y-=EPS_F*ny;return (n_inside==n_water)?1:4;}
            reflect3(*vx,*vy,*vz,nx,ny,0.0f,&a,&b,&c);*vx=a;*vy=b;*vz=c;(*cell_inner_reflections)++;if(tir)(*cell_tir)++;*x+=EPS_F*nx;*y+=EPS_F*ny;
        }else if(surf==2){
            float nx,ny;radial_normal(*x,*y,&nx,&ny);ci=fminf(fmaxf(dot3(*vx,*vy,*vz,nx,ny,0.0f),0.0f),1.0f);R=fresnel_R(ci,n_acrylic,n_air,&tir);
            if(!tir&&rnd_uniform(state)>=R&&refract3(*vx,*vy,*vz,-nx,-ny,0.0f,n_acrylic,n_air,&a,&b,&c)){
                *vx=a;*vy=b;*vz=c;*x+=EPS_F*nx;*y+=EPS_F*ny;return 2;}
            reflect3(*vx,*vy,*vz,nx,ny,0.0f,&a,&b,&c);*vx=a;*vy=b;*vz=c;(*cell_outer_reflections)++;if(tir)(*cell_tir)++;*x-=EPS_F*nx;*y-=EPS_F*ny;
        }else if(surf==3){
            *z=Zlow;ci=fminf(fmaxf(-(*vz),0.0f),1.0f);R=fresnel_R(ci,n_acrylic,n_bottom_external,&tir);
            if(!tir&&rnd_uniform(state)>=R&&refract3(*vx,*vy,*vz,0.0f,0.0f,1.0f,n_acrylic,n_bottom_external,&a,&b,&c)){
                *vx=a;*vy=b;*vz=c;*z=Zlow-EPS_F;return 3;}
            reflect3(*vx,*vy,*vz,0.0f,0.0f,1.0f,&a,&b,&c);*vx=a;*vy=b;*vz=c;(*bottom_outer_reflections)++;if(tir)(*bottom_tir)++;*z=Zlow+EPS_F;
        }else if(surf==4){
            *z=Zwall_top;ci=fminf(fmaxf(*vz,0.0f),1.0f);R=fresnel_R(ci,n_acrylic,n_air,&tir);
            if(!tir&&rnd_uniform(state)>=R&&refract3(*vx,*vy,*vz,0.0f,0.0f,-1.0f,n_acrylic,n_air,&a,&b,&c)){
                *vx=a;*vy=b;*vz=c;*z=Zwall_top+EPS_F;return 2;}
            reflect3(*vx,*vy,*vz,0.0f,0.0f,1.0f,&a,&b,&c);*vx=a;*vy=b;*vz=c;(*cell_outer_reflections)++;if(tir)(*cell_tir)++;*z=Zwall_top-EPS_F;
        }else if(surf==5){
            *z=Zmin;ci=fminf(fmaxf(*vz,0.0f),1.0f);R=fresnel_R(ci,n_acrylic,n_water,&tir);
            if(!tir&&rnd_uniform(state)>=R&&refract3(*vx,*vy,*vz,0.0f,0.0f,-1.0f,n_acrylic,n_water,&a,&b,&c)){
                *vx=a;*vy=b;*vz=c;*z=Zmin+EPS_F;return 1;}
            reflect3(*vx,*vy,*vz,0.0f,0.0f,1.0f,&a,&b,&c);*vx=a;*vy=b;*vz=c;(*bottom_inner_reflections)++;if(tir)(*bottom_tir)++;*z=Zmin-EPS_F;
        }else{if(failure_subcode)*failure_subcode=PROD_SUB_ACRYLIC_NO_INTERSECTION;return 0;}
        float pd=fabsf(*x-px)+fabsf(*y-py)+fabsf(*z-pz);
        float vd=fabsf(*vx-pvx)+fabsf(*vy-pvy)+fabsf(*vz-pvz);
        stuck=acrylic_progress_watchdog_update(surf,last_surf,pd,vd,stuck);
        if(stuck>=8){if(failure_subcode)*failure_subcode=PROD_SUB_ACRYLIC_BOUNCE_LIMIT;return 0;}
        last_surf=surf;
    }
    // At the emergency cap, classify the current conserved orbit. Cap reach
    // alone is insufficient to establish trapping; the test consumes no RNG.
    if(acrylic_current_state_trapped_tir(*x,*y,*z,*vx,*vy,*vz,n_water,n_acrylic,n_air,n_bottom_external,Rin,Rout,Zmin,Zlow,Zwall_top)){
        if(failure_subcode)*failure_subcode=PROD_SUB_NONE;if(out_bounces)*out_bounces=16384;return 5;
    }
    if(failure_subcode)*failure_subcode=PROD_SUB_ACRYLIC_BOUNCE_LIMIT;if(out_bounces)*out_bounces=16384;return 0;
}

__device__ int acrylic_annulus_ex(
    unsigned int* state,float n_water,float n_acrylic,float n_air,float n_bottom_external,
    float Rin,float Rout,float Zmin,float Zlow,float Zmean,float Zwall_top,int free_surface_model,float vortex_delta_h,float vortex_core_radius,int max_bounces,
    float* x,float* y,float* z,float* vx,float* vy,float* vz,float* acrylic_path,
    int* cell_inner_reflections,int* cell_outer_reflections,int* bottom_inner_reflections,int* bottom_outer_reflections,
    int* cell_tir,int* bottom_tir,int* failure_subcode,int* out_bounces){
    return acrylic_connected_solid_ex(state,n_water,n_acrylic,n_air,n_bottom_external,Rin,Rout,Zmin,Zlow,Zmean,Zwall_top,
        free_surface_model,vortex_delta_h,vortex_core_radius,max_bounces,x,y,z,vx,vy,vz,acrylic_path,
        cell_inner_reflections,cell_outer_reflections,bottom_inner_reflections,bottom_outer_reflections,cell_tir,bottom_tir,failure_subcode,out_bounces);
}

__device__ int bottom_acrylic_disc_ex(
    unsigned int* state,float n_water,float n_acrylic,float n_bottom_external,float n_side_external,
    float Rin,float Rout,float z_inner,float thickness,float zmean,float zwall_top,int free_surface_model,float vortex_delta_h,float vortex_core_radius,int max_bounces,
    float* x,float* y,float* z,float* vx,float* vy,float* vz,float* acrylic_path,
    int* cell_inner_reflections,int* cell_outer_reflections,int* bottom_inner_reflections,int* bottom_outer_reflections,
    int* cell_tir,int* bottom_tir,int* failure_subcode,int* out_bounces){
    (void)n_side_external;
    if(failure_subcode)*failure_subcode=PROD_SUB_NONE;if(out_bounces)*out_bounces=0;
    if(bottom_disc_topology_at_top(*x,*y,Rin,Rout)!=1){if(failure_subcode)*failure_subcode=PROD_SUB_BOTTOM_TOPOLOGY_INVALID;return 0;}
    float a,b,c;int tir=0;float ci=fminf(fmaxf(-dot3(*vx,*vy,*vz,0.0f,0.0f,1.0f),0.0f),1.0f);float R=fresnel_R(ci,n_water,n_acrylic,&tir);
    if(tir||rnd_uniform(state)<R||!refract3(*vx,*vy,*vz,0.0f,0.0f,1.0f,n_water,n_acrylic,&a,&b,&c)){
        reflect3(*vx,*vy,*vz,0.0f,0.0f,1.0f,&a,&b,&c);*vx=a;*vy=b;*vz=c;(*bottom_inner_reflections)++;if(tir)(*bottom_tir)++;*z=z_inner+EPS_F;return 1;}
    *vx=a;*vy=b;*vz=c;*z=z_inner-EPS_F;
    float zlow=z_inner-thickness;
    return acrylic_connected_solid_ex(state,n_water,n_acrylic,n_side_external,n_bottom_external,Rin,Rout,z_inner,zlow,zmean,zwall_top,
        free_surface_model,vortex_delta_h,vortex_core_radius,max_bounces,x,y,z,vx,vy,vz,acrylic_path,
        cell_inner_reflections,cell_outer_reflections,bottom_inner_reflections,bottom_outer_reflections,cell_tir,bottom_tir,failure_subcode,out_bounces);
}

// Full headspace transport for the open tube.  Rays transmitted through the water
// free surface remain inside the bore and may hit the acrylic inner wall, reflect at
// the free surface from above, re-enter water, or escape through the open tube top.
// Return: 1 water, 2 external via acrylic outer/top face, 3 bottom external,
// 4 open-top escape, 0 numerical/geometry failure.
__device__ int headspace_transport_ex(
    unsigned int* state,float n_water,float n_acrylic,float n_air,float n_bottom_external,
    float Rin,float Rout,float Zmin,float Zlow,float Zmean,float Zwall_top,
    int free_surface_model,float vortex_delta_h,float vortex_core_radius,int max_cell_bounces,
    float* x,float* y,float* z,float* vx,float* vy,float* vz,float* acrylic_path,
    int* cell_inner_reflections,int* cell_outer_reflections,int* bottom_inner_reflections,int* bottom_outer_reflections,
    int* cell_tir,int* bottom_tir,int* top_reflections,int* top_tir,int* failure_subcode,int* out_bounces){
    if(failure_subcode)*failure_subcode=PROD_SUB_NONE;if(out_bounces)*out_bounces=0;
    const int emergency_cap=16384;double tol=acrylic_support_tol(Rin,Rout,Zmin,Zlow,Zwall_top);
    for(int step=0;step<emergency_cap;++step){
        if(out_bounces)*out_bounces=step;
        if(!isfinite(*x)||!isfinite(*y)||!isfinite(*z)||!isfinite(*vx)||!isfinite(*vy)||!isfinite(*vz)){
            if(failure_subcode)*failure_subcode=PROD_SUB_HEADSPACE_NO_BOUNDARY;return 0;}
        float zs=free_surface_z(*x,*y,Zmean,Rin,free_surface_model,vortex_delta_h,vortex_core_radius);
        double rr=hypot((double)*x,(double)*y);
        if((double)*z<=(double)zs-tol && rr<(double)Rin-tol)return 1;
        if((double)*z>(double)Zwall_top+tol)return 4;
        double tside=bore_wall_hit_finite_d(*x,*y,*z,*vx,*vy,*vz,Rin,Zmin,Zwall_top);
        if(tside<0.0){if(failure_subcode)*failure_subcode=PROD_SUB_WATER_OUTSIDE_INNER_RADIUS;return 0;}
        double tsurf=(double)free_surface_hit(*x,*y,*z,*vx,*vy,*vz,Zmean,Rin,free_surface_model,vortex_delta_h,vortex_core_radius);
        double ttop=1.0e300;
        if(*vz>1.0e-14f){double t=((double)Zwall_top-(double)*z)/(double)*vz;if(t>0.0){double X=(double)*x+t*(double)*vx,Y=(double)*y+t*(double)*vy;if(hypot(X,Y)<(double)Rin+tol)ttop=t;}}
        int which=0;double best=1.0e300;
        if(tsurf>0.0&&tsurf<best){best=tsurf;which=1;}
        if(tside>=0.0&&tside<best){best=tside;which=2;}
        if(ttop>0.0&&ttop<best){best=ttop;which=3;}
        if(!which||!isfinite(best)||best>=1.0e299){if(failure_subcode)*failure_subcode=PROD_SUB_HEADSPACE_NO_BOUNDARY;return 0;}
        float t=(float)best;*x+=t*(*vx);*y+=t*(*vy);*z+=t*(*vz);
        if(which==3){*z=Zwall_top+EPS_F;return 4;}
        float a,b,c;int tir=0;
        if(which==1){
            float snx,sny,snz;free_surface_normal_into_water(*x,*y,Rin,free_surface_model,vortex_delta_h,vortex_core_radius,&snx,&sny,&snz);
            // Normal into incident headspace/air is the opposite of the water normal.
            float anx=-snx,any=-sny,anz=-snz;
            float ci=fminf(fmaxf(-dot3(*vx,*vy,*vz,anx,any,anz),0.0f),1.0f);float R=fresnel_R(ci,n_air,n_water,&tir);
            if(tir||rnd_uniform(state)<R||!refract3(*vx,*vy,*vz,anx,any,anz,n_air,n_water,&a,&b,&c)){
                reflect3(*vx,*vy,*vz,anx,any,anz,&a,&b,&c);*vx=a;*vy=b;*vz=c;(*top_reflections)++;if(tir)(*top_tir)++;
                *x+=EPS_F*anx;*y+=EPS_F*any;*z+=EPS_F*anz;continue;}
            *vx=a;*vy=b;*vz=c;
            if(!water_surface_bore_offset(x,y,z,snx,sny,snz,Rin,Zmin,Zmean,free_surface_model,vortex_delta_h,vortex_core_radius)){
                if(failure_subcode)*failure_subcode=PROD_SUB_WATER_OUTSIDE_INNER_RADIUS;return 0;
            }
            return 1;
        }
        // Headspace -> acrylic at the finite inner cylindrical wall.
        float nx,ny;radial_normal(*x,*y,&nx,&ny);float inx=-nx,iny=-ny;
        float ci=fminf(fmaxf(-dot3(*vx,*vy,*vz,inx,iny,0.0f),0.0f),1.0f);float R=fresnel_R(ci,n_air,n_acrylic,&tir);
        if(tir||rnd_uniform(state)<R||!refract3(*vx,*vy,*vz,inx,iny,0.0f,n_air,n_acrylic,&a,&b,&c)){
            reflect3(*vx,*vy,*vz,inx,iny,0.0f,&a,&b,&c);*vx=a;*vy=b;*vz=c;(*cell_inner_reflections)++;if(tir)(*cell_tir)++;
            *x-=EPS_F*nx;*y-=EPS_F*ny;continue;}
        *vx=a;*vy=b;*vz=c;*x+=EPS_F*nx;*y+=EPS_F*ny;
        int asub=PROD_SUB_NONE,ab=0;int where=acrylic_connected_solid_ex(state,n_water,n_acrylic,n_air,n_bottom_external,Rin,Rout,Zmin,Zlow,Zmean,Zwall_top,
            free_surface_model,vortex_delta_h,vortex_core_radius,max_cell_bounces,x,y,z,vx,vy,vz,acrylic_path,
            cell_inner_reflections,cell_outer_reflections,bottom_inner_reflections,bottom_outer_reflections,cell_tir,bottom_tir,&asub,&ab);
        if(where==4)continue;
        if(failure_subcode)*failure_subcode=asub;if(out_bounces)*out_bounces=ab;return where;
    }
    if(failure_subcode)*failure_subcode=PROD_SUB_HEADSPACE_NO_BOUNDARY;if(out_bounces)*out_bounces=emergency_cap;return 0;
}


__device__ void raster_water_segment(float x0,float y0,float z0,float vx,float vy,float vz,float dist,
    int vis,float Rin,float zmin,float zmax,float* hxy,float* hzy) {
    if(vis<=0 || dist<=0.0f) return;
    const float spacing=1.0e-5f;
    int n=(int)ceilf(dist/spacing); if(n<1)n=1; if(n>20000)n=20000;
    for(int k=1;k<=n;++k){
        float f=(float)k/(float)n;
        float x=x0+vx*dist*f,y=y0+vy*dist*f,z=z0+vz*dist*f;
        int ix=(int)(((x+Rin)/(2.0f*Rin))*vis);
        int iy=vis-1-(int)(((y+Rin)/(2.0f*Rin))*vis);
        int iz=(int)(((z-zmin)/(zmax-zmin))*vis);
        if(ix>=0&&ix<vis&&iy>=0&&iy<vis) atomicAdd(&hxy[iy*vis+ix],1.0f);
        if(iz>=0&&iz<vis&&iy>=0&&iy<vis) atomicAdd(&hzy[iy*vis+iz],1.0f);
    }
}

// diagnostic path: exact track-length deposition by 3-D voxel traversal.
// This code is observational only: it consumes no RNG and writes only dedicated
// diagnostic buffers. Array order is [z,y,x]. Benign floating-point endpoint
// snaps are counted separately from genuine traversal-integrity failures.
// Diagnostic geometry uses double precision to resolve short segments at fine voxel faces.
// The transport state and RNG remain untouched.
__device__ int light_diag_clip_axis(const double p,const double d,const double lo,const double hi,double* t0,double* t1){
    if(fabs(d)<1.0e-30f)return (p>=lo&&p<=hi)?1:0;
    double a=(lo-p)/d,b=(hi-p)/d;if(a>b){double q=a;a=b;b=q;}
    *t0=fmax(*t0,a);*t1=fmin(*t1,b);return *t1>=*t0;
}

__device__ void light_diag_record_segment(
    const unsigned int gid,const int active,const int nx,const int ny,const int nz,
    const float Rin,const float zmin,const float zmax,
    const float x0,const float y0,const float z0,const float vx,const float vy,const float vz,
    const float dist,const int particle_order,
    float* total_grid,float* post_grid,double* segment_total,double* deposited_total,
    unsigned long long* segment_count,unsigned int* error_count,unsigned int* boundary_adjustment_count,
    const int rep_rays,const int max_segments,
    float* tx0,float* ty0,float* tz0,float* tx1,float* ty1,float* tz1,
    int* torder,int* tevent,int* tcount,int* toverflow){
    if(!active || !(dist>0.0f) || !isfinite(dist)) return;
    if(nx<=0||ny<=0||nz<=0||!(zmax>zmin)) {atomicAdd(error_count,1U);return;}
    const double xmin=-(double)Rin,ymin=-(double)Rin;
    double c0=0.0,c1=1.0;
    if(!light_diag_clip_axis(x0,(double)vx*(double)dist,-Rin,Rin,&c0,&c1)||
       !light_diag_clip_axis(y0,(double)vy*(double)dist,-Rin,Rin,&c0,&c1)||
       !light_diag_clip_axis(z0,(double)vz*(double)dist,zmin,zmax,&c0,&c1))return;
    const double ax=x0+(double)vx*(double)dist*c0,ay=y0+(double)vy*(double)dist*c0,az=z0+(double)vz*(double)dist*c0;
    const double x1=x0+(double)vx*(double)dist*c1,y1=y0+(double)vy*(double)dist*c1,z1=z0+(double)vz*(double)dist*c1;
    const double clipped_dist=dist*(c1-c0);if(!(clipped_dist>0.0f))return;
    atomicAdd(segment_total,(double)clipped_dist);
    atomicAdd(segment_count,1ULL);
    if((int)gid<rep_rays && max_segments>0){
        int slot=tcount[gid];
        if(slot<max_segments){
            int q=((int)gid)*max_segments+slot;
            tx0[q]=ax;ty0[q]=ay;tz0[q]=az;tx1[q]=x1;ty1[q]=y1;tz1[q]=z1;
            torder[q]=particle_order;tevent[q]=0;tcount[gid]=slot+1;
        }else{toverflow[gid]=1;}
    }
    const double dx=(2.0*(double)Rin)/nx,dy=(2.0*(double)Rin)/ny,dz=((double)zmax-(double)zmin)/nz;
    int ix=(int)floor((ax-xmin)/dx),iy=(int)floor((ay-ymin)/dy),iz=(int)floor((az-zmin)/dz);
    if(ix<0)ix=0;if(ix>=nx)ix=nx-1;
    if(iy<0)iy=0;if(iy>=ny)iy=ny-1;
    if(iz<0)iz=0;if(iz>=nz)iz=nz-1;
    const double sx=x1-ax,sy=y1-ay,sz=z1-az;
    int stepx=(sx>0.0f)?1:((sx<0.0f)?-1:0);
    int stepy=(sy>0.0f)?1:((sy<0.0f)?-1:0);
    int stepz=(sz>0.0f)?1:((sz<0.0f)?-1:0);
    double bx=xmin+(double)(stepx>0?ix+1:ix)*dx;
    double by=ymin+(double)(stepy>0?iy+1:iy)*dy;
    double bz=zmin+(double)(stepz>0?iz+1:iz)*dz;
    double tmx=(stepx==0)?INF_F:(bx-ax)/sx;
    double tmy=(stepy==0)?INF_F:(by-ay)/sy;
    double tmz=(stepz==0)?INF_F:(bz-az)/sz;
    if(tmx<0.0f)tmx=0.0f;if(tmy<0.0f)tmy=0.0f;if(tmz<0.0f)tmz=0.0f;
    double u=0.0;double local_deposited=0.0;
    const float completion_tol=2.0e-6f;
    const int max_steps=nx+ny+nz+8;
    for(int k=0;k<max_steps && u<1.0f;++k){
        if(ix<0||ix>=nx||iy<0||iy>=ny||iz<0||iz>=nz){
            if(u>=1.0f-completion_tol){atomicAdd(boundary_adjustment_count,1U);u=1.0f;break;}
            atomicAdd(error_count,1U);break;
        }
        double un=fmin(1.0,fmin(tmx,fmin(tmy,tmz)));
        if(un<1.0f && un>=1.0f-completion_tol){un=1.0f;atomicAdd(boundary_adjustment_count,1U);}
        if(un<u){
            if(u-un<=completion_tol){un=u;atomicAdd(boundary_adjustment_count,1U);}
            else{atomicAdd(error_count,1U);break;}
        }
        double length=(un-u)*clipped_dist;
        if(length>0.0f){
            int q=(iz*ny+iy)*nx+ix;
            atomicAdd(&total_grid[q],(float)length);
            if(particle_order>0)atomicAdd(&post_grid[q],(float)length);
            local_deposited+=(double)length;
        }
        if(un>=1.0f){u=1.0f;break;}
        const double eps=8.0*2.2204460492503131e-16*fmax(1.0,fabs(un));
        if(tmx<=un+eps){ix+=stepx;tmx=(xmin+(double)(stepx>0?ix+1:ix)*dx-ax)/sx;}
        if(tmy<=un+eps){iy+=stepy;tmy=(ymin+(double)(stepy>0?iy+1:iy)*dy-ay)/sy;}
        if(tmz<=un+eps){iz+=stepz;tmz=((double)zmin+(double)(stepz>0?iz+1:iz)*dz-az)/sz;}
        u=un;
    }
    if(u<1.0f-completion_tol)atomicAdd(error_count,1U);
    atomicAdd(deposited_total,local_deposited);
}

__device__ void light_diag_mark_last_event(
    const unsigned int gid,const int active,const int rep_rays,const int max_segments,
    int* tevent,const int* tcount,const int event_code){
    if(!active||(int)gid>=rep_rays||max_segments<=0)return;
    int n=tcount[gid];if(n>0&&n<=max_segments)tevent[((int)gid)*max_segments+n-1]=event_code;
}

__device__ void light_diag_record_event(
    const int active,const int nx,const int ny,const int nz,const float Rin,
    const float zmin,const float zmax,const float x,const float y,const float z,const unsigned int gid,
    float* event_weight,unsigned int* event_count,unsigned long long* event_total,
    unsigned int* error_count,unsigned int* event_outside_count,unsigned int* boundary_adjustment_count){
    if(!active)return;
    atomicAdd(event_total,1ULL);
    const float coordinate_tol=2.0e-6f*fmaxf(2.0f*Rin,zmax-zmin);
    if(!isfinite(x)||!isfinite(y)||!isfinite(z)||x < -Rin-coordinate_tol || x > Rin+coordinate_tol ||
       y < -Rin-coordinate_tol || y > Rin+coordinate_tol ||
       z < zmin-coordinate_tol || z > zmax+coordinate_tol){
        unsigned int previous=atomicAdd(event_outside_count,1U);
        if(previous==0U){
            event_outside_count[1]=__float_as_uint(x);event_outside_count[2]=__float_as_uint(y);event_outside_count[3]=__float_as_uint(z);
            event_outside_count[4]=gid;event_outside_count[5]=__float_as_uint(coordinate_tol);event_outside_count[6]=1U;
        }
        atomicAdd(error_count,1U);return;
    }
    int ix=(int)floorf(((x+Rin)/(2.0f*Rin))*nx);
    int iy=(int)floorf(((y+Rin)/(2.0f*Rin))*ny);
    int iz=(int)floorf(((z-zmin)/(zmax-zmin))*nz);
    int adjusted=0;
    if(ix<0){ix=0;adjusted=1;}else if(ix>=nx){ix=nx-1;adjusted=1;}
    if(iy<0){iy=0;adjusted=1;}else if(iy>=ny){iy=ny-1;adjusted=1;}
    if(iz<0){iz=0;adjusted=1;}else if(iz>=nz){iz=nz-1;adjusted=1;}
    if(adjusted)atomicAdd(boundary_adjustment_count,1U);
    if(ix<0||ix>=nx||iy<0||iy>=ny||iz<0||iz>=nz){atomicAdd(event_outside_count,1U);atomicAdd(error_count,1U);return;}
    int q=(iz*ny+iy)*nx+ix;atomicAdd(&event_weight[q],1.0f);atomicAdd(&event_count[q],1U);
}

__global__ void light_diag_segment_test_kernel(
    const float* segments7,const int n,const int nx,const int ny,const int nz,
    const float Rin,const float zmin,const float zmax,float* total_grid,float* post_grid,
    double* segment_total,double* deposited_total,unsigned long long* segment_count,unsigned int* error_count,unsigned int* boundary_adjustment_count){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;if(tid>=n)return;const float* s=segments7+7*tid;
    light_diag_record_segment((unsigned int)tid,1,nx,ny,nz,Rin,zmin,zmax,s[0],s[1],s[2],s[3],s[4],s[5],s[6],tid&1,
        total_grid,post_grid,segment_total,deposited_total,segment_count,error_count,boundary_adjustment_count,0,1,total_grid,total_grid,total_grid,total_grid,total_grid,total_grid,
        (int*)error_count,(int*)error_count,(int*)error_count,(int*)error_count);
}

__global__ void light_diag_event_test_kernel(
    const float* xyz3,const int n,const int nx,const int ny,const int nz,const float Rin,const float zmin,const float zmax,
    float* event_weight,unsigned int* event_count,unsigned long long* event_total,unsigned int* error_count,unsigned int* event_outside_count,unsigned int* boundary_adjustment_count){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;if(tid>=n)return;const float* p=xyz3+3*tid;
    light_diag_record_event(1,nx,ny,nz,Rin,zmin,zmax,p[0],p[1],p[2],(unsigned int)tid,event_weight,event_count,event_total,error_count,event_outside_count,boundary_adjustment_count);
}


__global__ void build_theta_grid_kernel(double* theta,const int ntheta){
    int j=blockDim.x*blockIdx.x+threadIdx.x;if(j>=ntheta)return;
    if(j==0){theta[j]=0.0;return;}if(j==ntheta-1){theta[j]=PI_D;return;}
    double u=(double)j/(double)(ntheta-1);
    theta[j]=0.5*PI_D*(1.0-cos(PI_D*u));
}

__global__ void mie_coefficients_efficiencies_kernel(
    const double* diameter,const double* n_real,const double* k_imag,
    const double n_medium,const double wavelength,const int n_particles,
    double* D_re,double* D_im,double* a_re,double* a_im,double* b_re,double* b_im,
    int* nstop_out,double* size_parameter,double* qext,double* qsca,double* qabs,double* qback,double* asym_g,double* albedo,
    int* error_flag){
    int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=n_particles)return;
    double x=PI_D*n_medium*diameter[p]/wavelength;
    size_parameter[p]=x;
    if(!(x>0.0) || !(n_real[p]>0.0) || !(n_medium>0.0) || !(wavelength>0.0)){
        nstop_out[p]=0;qext[p]=qsca[p]=qabs[p]=qback[p]=asym_g[p]=0.0;albedo[p]=1.0;atomicExch(error_flag,2);return;
    }
    double mr=n_real[p]/n_medium,mi=k_imag[p]/n_medium;
    int nstop=(int)(x+4.0*cbrt(x)+2.0);if(nstop<1)nstop=1;
    double mz=hypot(mr*x,mi*x);
    int nmx=(int)fmax((double)nstop,mz)+15;if(nmx<nstop+1)nmx=nstop+1;
    if(nmx>=MIE_MAX_ORDER || nstop>=MIE_MAX_ORDER){nstop_out[p]=0;atomicExch(error_flag,1);return;}
    nstop_out[p]=nstop;
    int base=p*MIE_MAX_ORDER;
    D_re[base+nmx]=0.0;D_im[base+nmx]=0.0;
    cpxd z=CD(mr*x,mi*x);
    for(int n=nmx;n>=1;--n){
        cpxd nz=cd_div(CD((double)n,0.0),z);
        cpxd dn=CD(D_re[base+n],D_im[base+n]);
        cpxd val=cd_sub(nz,cd_div(CD(1.0,0.0),cd_add(dn,nz)));
        D_re[base+n-1]=val.re;D_im[base+n-1]=val.im;
    }
    double psi_nm1=sin(x),psi_n=sin(x)/x-cos(x);
    double chi_nm1=cos(x),chi_n=cos(x)/x+sin(x);
    cpxd m=CD(mr,mi);
    for(int j=0;j<nstop;++j){
        int n=j+1;
        cpxd xi_nm1=CD(psi_nm1,-chi_nm1),xi_n=CD(psi_n,-chi_n);
        cpxd dn=CD(D_re[base+n],D_im[base+n]);
        cpxd ca=cd_add(cd_div(dn,m),CD((double)n/x,0.0));
        cpxd cb=cd_add(cd_mul(m,dn),CD((double)n/x,0.0));
        cpxd aa=cd_div(cd_sub(cd_scale(ca,psi_n),CD(psi_nm1,0.0)),cd_sub(cd_mul(ca,xi_n),xi_nm1));
        cpxd bb=cd_div(cd_sub(cd_scale(cb,psi_n),CD(psi_nm1,0.0)),cd_sub(cd_mul(cb,xi_n),xi_nm1));
        a_re[base+j]=aa.re;a_im[base+j]=aa.im;b_re[base+j]=bb.re;b_im[base+j]=bb.im;
        double psi_np1=((2.0*n+1.0)/x)*psi_n-psi_nm1;
        double chi_np1=((2.0*n+1.0)/x)*chi_n-chi_nm1;
        psi_nm1=psi_n;psi_n=psi_np1;chi_nm1=chi_n;chi_n=chi_np1;
    }
    double se=0.0,ss=0.0,sbr=0.0,sbi=0.0,t1=0.0,t2=0.0;
    for(int j=0;j<nstop;++j){
        int n=j+1;double cn=2.0*n+1.0;
        cpxd aa=CD(a_re[base+j],a_im[base+j]),bb=CD(b_re[base+j],b_im[base+j]);
        se+=cn*(aa.re+bb.re);ss+=cn*(cd_abs2(aa)+cd_abs2(bb));
        double sign=(n&1)?-1.0:1.0;sbr+=cn*sign*(aa.re-bb.re);sbi+=cn*sign*(aa.im-bb.im);
        t2+=(cn/((double)n*(n+1.0)))*cd_re_mul_conj(aa,bb);
        if(j+1<nstop){
            cpxd an=CD(a_re[base+j+1],a_im[base+j+1]),bn=CD(b_re[base+j+1],b_im[base+j+1]);
            t1+=((double)n*(n+2.0)/(n+1.0))*(cd_re_mul_conj(aa,an)+cd_re_mul_conj(bb,bn));
        }
    }
    double x2=x*x;double qe=2.0*se/x2,qs=2.0*ss/x2,qb=(sbr*sbr+sbi*sbi)/x2;
    if(qe<0.0 && qe>-1e-12)qe=0.0;if(qs<0.0 && qs>-1e-12)qs=0.0;
    double qa=qe-qs;if(qa<0.0 && qa>-1e-10)qa=0.0;
    double gg=qs>0.0?4.0*(t1+t2)/(x2*qs):0.0;if(gg>1.0)gg=1.0;if(gg<-1.0)gg=-1.0;
    qext[p]=qe;qsca[p]=qs;qabs[p]=qa;qback[p]=qb;asym_g[p]=gg;albedo[p]=qe>0.0?qs/qe:1.0;
}

__global__ void mie_phase_pdf_kernel(
    const double* theta,const int ntheta,const int n_particles,const int* nstop,
    const double* a_re,const double* a_im,const double* b_re,const double* b_im,double* phase_pdf){
    long long idx=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    long long total=(long long)n_particles*ntheta;if(idx>=total)return;
    int p=(int)(idx/ntheta),j=(int)(idx-(long long)p*ntheta),base=p*MIE_MAX_ORDER;
    double th=theta[j],mu=cos(th),s1r=0.0,s1i=0.0,s2r=0.0,s2i=0.0;
    double pi_nm1=0.0,pi_n=1.0;
    for(int k=0;k<nstop[p];++k){
        int n=k+1;double nf=(double)n;
        double tau=nf*mu*pi_n-(nf+1.0)*pi_nm1;
        double c=(2.0*nf+1.0)/(nf*(nf+1.0));
        double ar=a_re[base+k],ai=a_im[base+k],br=b_re[base+k],bi=b_im[base+k];
        s1r+=c*(ar*pi_n+br*tau);s1i+=c*(ai*pi_n+bi*tau);
        s2r+=c*(ar*tau+br*pi_n);s2i+=c*(ai*tau+bi*pi_n);
        double pi_np1=((2.0*nf+1.0)/nf)*mu*pi_n-((nf+1.0)/nf)*pi_nm1;
        pi_nm1=pi_n;pi_n=pi_np1;
    }
    double intensity=0.5*(s1r*s1r+s1i*s1i+s2r*s2r+s2i*s2i);
    phase_pdf[idx]=intensity*sin(th);
}

__global__ void mie_phase_cdf_kernel(const double* theta,const double* pdf,const int ntheta,const int n_particles,double* cdf,int* error_flag){
    int p=blockDim.x*blockIdx.x+threadIdx.x;if(p>=n_particles)return;
    long long base=(long long)p*ntheta;cdf[base]=0.0;double s=0.0;
    for(int j=1;j<ntheta;++j){
        double area=0.5*(pdf[base+j]+pdf[base+j-1])*(theta[j]-theta[j-1]);
        if(area<0.0 && area>-1e-18)area=0.0;s+=area;cdf[base+j]=s;
    }
    if(!(s>0.0) || !isfinite(s)){atomicExch(error_flag,3);for(int j=0;j<ntheta;++j)cdf[base+j]=(double)j/(double)(ntheta-1);return;}
    double prev=0.0;for(int j=0;j<ntheta;++j){double v=cdf[base+j]/s;if(v<prev)v=prev;if(v>1.0)v=1.0;cdf[base+j]=v;prev=v;}cdf[base+ntheta-1]=1.0;
}

__global__ void dummy_phase_kernel(double* theta,double* cdf,const int n_particles){
    if(blockIdx.x==0&&threadIdx.x==0){theta[0]=0.0;theta[1]=PI_D;}
    int p=blockDim.x*blockIdx.x+threadIdx.x;if(p<n_particles){cdf[2*p]=0.0;cdf[2*p+1]=1.0;}
}

#define AGG_MAX_BINS 64

// Inherited orthokinetic compact-aggregation population balance (disabled by default).
// Independent variable exposure = alpha_stick * G * tau, dimensionless.
// The dimensional shear kernel K=(4/3)G(ri+rj)^3 therefore becomes
// K_E=(4/3)(ri+rj)^3 with respect to exposure E. Concentration enters
// through the initial number densities, so the PSD evolution is generated
// by collision physics rather than by a fitted concentration multiplier.
__global__ void orthokinetic_compact_aggregation_kernel(
    const double* diameter,const double* base_weights,const int n_particles,
    const double concentration,const double density,const double exposure,
    const double step_safety,const int max_steps,
    double* effective_weights,double* aggregation_scalars,int* error_flag){
    if(blockIdx.x||threadIdx.x)return;
    if(n_particles<1 || n_particles>AGG_MAX_BINS){atomicExch(error_flag,20);return;}
    if(!(density>0.0) || concentration<0.0 || exposure<0.0 || !(step_safety>0.0) || step_safety>0.25){atomicExch(error_flag,21);return;}
    double sw=0.0; for(int i=0;i<n_particles;++i) sw+=base_weights[i];
    if(!(sw>0.0)){atomicExch(error_flag,22);return;}
    double vol[AGG_MAX_BINS],mass[AGG_MAX_BINS],ncur[AGG_MAX_BINS],dn[AGG_MAX_BINS];
    double initial_n=0.0,initial_area=0.0;
    for(int i=0;i<n_particles;++i){
        double d=diameter[i],v=(PI_D/6.0)*d*d*d,m=density*v,w=base_weights[i]/sw;
        vol[i]=v;mass[i]=m;
        double ni=(concentration>0.0)?concentration*w/m:0.0;
        ncur[i]=ni;initial_n+=ni;initial_area+=ni*PI_D*0.25*d*d;
    }
    if(exposure<=0.0 || concentration<=0.0){
        for(int i=0;i<n_particles;++i) effective_weights[i]=base_weights[i]/sw;
        aggregation_scalars[0]=exposure; aggregation_scalars[1]=initial_n; aggregation_scalars[2]=initial_n;
        aggregation_scalars[3]=0.0; aggregation_scalars[4]=0.0; aggregation_scalars[5]=initial_area; aggregation_scalars[6]=initial_area;
        aggregation_scalars[7]=base_weights[n_particles-1]/sw; aggregation_scalars[8]=0.0; aggregation_scalars[9]=0.0;
        return;
    }
    double s=0.0; int steps=0; double overflow_events=0.0,total_events=0.0;
    while(s<exposure && steps<max_steps){
        double maxloss=0.0;
        for(int i=0;i<n_particles;++i){
            double Li=0.0,ri=0.5*diameter[i];
            for(int j=0;j<n_particles;++j){double q=ri+0.5*diameter[j]; Li+=(4.0/3.0)*q*q*q*ncur[j];}
            if(Li>maxloss)maxloss=Li;
        }
        double h=exposure-s;
        if(maxloss>0.0){double hs=step_safety/maxloss;if(h>hs)h=hs;}
        if(!(h>0.0) || !isfinite(h)){atomicExch(error_flag,23);return;}
        for(int i=0;i<n_particles;++i)dn[i]=0.0;
        for(int i=0;i<n_particles;++i){
            double ri=0.5*diameter[i];
            for(int j=i;j<n_particles;++j){
                double q=ri+0.5*diameter[j]; double K=(4.0/3.0)*q*q*q;
                double rate=(i==j)?0.5*K*ncur[i]*ncur[i]:K*ncur[i]*ncur[j];
                if(!(rate>0.0))continue;
                total_events+=rate*h;
                if(i==j)dn[i]-=2.0*rate; else {dn[i]-=rate;dn[j]-=rate;}
                double vn=vol[i]+vol[j];
                if(vn>=vol[n_particles-1]){
                    dn[n_particles-1]+=rate*(vn/vol[n_particles-1]);
                    if(vn>vol[n_particles-1]) overflow_events+=rate*h;
                }else{
                    int k=0; while(k+1<n_particles && vol[k+1]<vn)++k;
                    if(k+1>=n_particles){dn[n_particles-1]+=rate*(vn/vol[n_particles-1]);}
                    else{
                        double f=(vn-vol[k])/(vol[k+1]-vol[k]); if(f<0.0)f=0.0;if(f>1.0)f=1.0;
                        dn[k]+=rate*(1.0-f); dn[k+1]+=rate*f;
                    }
                }
            }
        }
        for(int i=0;i<n_particles;++i){
            double nv=ncur[i]+h*dn[i];
            if(nv<0.0 && nv>-1e-12*fmax(ncur[i],1.0))nv=0.0;
            if(nv<0.0 || !isfinite(nv)){atomicExch(error_flag,24);return;}
            ncur[i]=nv;
        }
        s+=h; ++steps;
    }
    if(s+1e-12<exposure){atomicExch(error_flag,25);return;}
    double final_mass=0.0,final_n=0.0,final_area=0.0;
    for(int i=0;i<n_particles;++i){final_mass+=ncur[i]*mass[i];final_n+=ncur[i];final_area+=ncur[i]*PI_D*0.25*diameter[i]*diameter[i];}
    if(!(final_mass>0.0)){atomicExch(error_flag,26);return;}
    for(int i=0;i<n_particles;++i)effective_weights[i]=(ncur[i]*mass[i])/final_mass;
    double mass_error=(concentration>0.0)?(final_mass-concentration)/concentration:0.0;
    double num_reduction=(initial_n>0.0)?1.0-final_n/initial_n:0.0;
    aggregation_scalars[0]=exposure; aggregation_scalars[1]=initial_n; aggregation_scalars[2]=final_n;
    aggregation_scalars[3]=num_reduction; aggregation_scalars[4]=mass_error; aggregation_scalars[5]=initial_area; aggregation_scalars[6]=final_area;
    aggregation_scalars[7]=effective_weights[n_particles-1]; aggregation_scalars[8]=(total_events>0.0)?overflow_events/total_events:0.0; aggregation_scalars[9]=(double)steps;
}


// Inherited coupled orthokinetic aggregation + shear-fragmentation population balance (disabled by default).
//
// Aggregation uses the Smoluchowski orthokinetic kernel with a common
// dimensionless exposure EA = alpha_stick * G * tau.
//
// Fragmentation is a first-order binary-breakage process with a common exposure
// EB defined at reference diameter d_ref:
//
//     B_i * tau = EB * (d_i / d_ref)^m
//
// Each resolved breakage event creates two equal-volume daughters. Daughter
// volume is conservatively mapped onto adjacent PSD volume bins. If the daughter
// size lies below the smallest resolved bin, the parent mass is retained in the
// smallest bin and the event is counted as a lower-grid underflow diagnostic.
//
// EA, EB, d_ref and m are shared across concentrations. Concentration enters
// only through number density in the aggregation term. Thus low concentration
// can be fragmentation-dominated while high concentration remains collision-
// dominated without any concentration-specific optical or PSD multiplier.
__global__ void orthokinetic_compact_aggregation_fragmentation_kernel(
    const double* diameter,const double* base_weights,const int n_particles,
    const double concentration,const double density,
    const double aggregation_exposure,const double breakup_exposure,
    const double breakup_reference_diameter,const double breakup_size_exponent,
    const double step_safety,const int max_steps,
    double* effective_weights,double* population_scalars,int* error_flag){
    if(blockIdx.x||threadIdx.x)return;
    if(n_particles<1 || n_particles>AGG_MAX_BINS){atomicExch(error_flag,30);return;}
    if(!(density>0.0) || concentration<0.0 || aggregation_exposure<0.0 || breakup_exposure<0.0 ||
       !(breakup_reference_diameter>0.0) || breakup_size_exponent<0.0 ||
       !(step_safety>0.0) || step_safety>0.25){atomicExch(error_flag,31);return;}
    double sw=0.0; for(int i=0;i<n_particles;++i)sw+=base_weights[i];
    if(!(sw>0.0)){atomicExch(error_flag,32);return;}
    double vol[AGG_MAX_BINS],mass[AGG_MAX_BINS],ncur[AGG_MAX_BINS],dn[AGG_MAX_BINS];
    double initial_n=0.0,initial_area=0.0;
    for(int i=0;i<n_particles;++i){
        double d=diameter[i],v=(PI_D/6.0)*d*d*d,mass_i=density*v,w=base_weights[i]/sw;
        vol[i]=v;mass[i]=mass_i;
        double ni=(concentration>0.0)?concentration*w/mass_i:0.0;
        ncur[i]=ni;initial_n+=ni;initial_area+=ni*PI_D*0.25*d*d;
    }
    if(concentration<=0.0 || (aggregation_exposure<=0.0 && breakup_exposure<=0.0)){
        for(int i=0;i<n_particles;++i)effective_weights[i]=base_weights[i]/sw;
        population_scalars[0]=aggregation_exposure;
        population_scalars[1]=breakup_exposure;
        population_scalars[2]=breakup_reference_diameter;
        population_scalars[3]=breakup_size_exponent;
        population_scalars[4]=initial_n;population_scalars[5]=initial_n;
        population_scalars[6]=0.0;population_scalars[7]=0.0;
        population_scalars[8]=initial_area;population_scalars[9]=initial_area;
        population_scalars[10]=base_weights[n_particles-1]/sw;
        population_scalars[11]=0.0;population_scalars[12]=0.0;
        population_scalars[13]=0.0;population_scalars[14]=0.0;population_scalars[15]=0.0;
        return;
    }
    double u=0.0;
    int steps=0;
    double aggregation_overflow_events=0.0,total_aggregation_events=0.0;
    double fragmentation_underflow_events=0.0,total_fragmentation_events=0.0;
    while(u<1.0 && steps<max_steps){
        double maxloss=0.0;
        for(int i=0;i<n_particles;++i){
            double ri=0.5*diameter[i],Lagg=0.0;
            if(aggregation_exposure>0.0){
                for(int j=0;j<n_particles;++j){
                    double q=ri+0.5*diameter[j];
                    Lagg+=(4.0/3.0)*q*q*q*ncur[j];
                }
                Lagg*=aggregation_exposure;
            }
            double Lbreak=0.0;
            if(i>0 && breakup_exposure>0.0){
                Lbreak=breakup_exposure*pow(diameter[i]/breakup_reference_diameter,breakup_size_exponent);
            }
            double Li=Lagg+Lbreak;if(Li>maxloss)maxloss=Li;
        }
        double h=1.0-u;
        if(maxloss>0.0){double hs=step_safety/maxloss;if(h>hs)h=hs;}
        if(!(h>0.0) || !isfinite(h)){atomicExch(error_flag,33);return;}
        for(int i=0;i<n_particles;++i)dn[i]=0.0;

        if(aggregation_exposure>0.0){
            for(int i=0;i<n_particles;++i){
                double ri=0.5*diameter[i];
                for(int j=i;j<n_particles;++j){
                    double q=ri+0.5*diameter[j];double K=(4.0/3.0)*q*q*q;
                    double rate=aggregation_exposure*((i==j)?0.5*K*ncur[i]*ncur[i]:K*ncur[i]*ncur[j]);
                    if(!(rate>0.0))continue;
                    total_aggregation_events+=rate*h;
                    if(i==j)dn[i]-=2.0*rate;else{dn[i]-=rate;dn[j]-=rate;}
                    double vn=vol[i]+vol[j];
                    if(vn>=vol[n_particles-1]){
                        dn[n_particles-1]+=rate*(vn/vol[n_particles-1]);
                        if(vn>vol[n_particles-1])aggregation_overflow_events+=rate*h;
                    }else{
                        int k=0;while(k+1<n_particles && vol[k+1]<vn)++k;
                        if(k+1>=n_particles){dn[n_particles-1]+=rate*(vn/vol[n_particles-1]);}
                        else{
                            double f=(vn-vol[k])/(vol[k+1]-vol[k]);if(f<0.0)f=0.0;if(f>1.0)f=1.0;
                            dn[k]+=rate*(1.0-f);dn[k+1]+=rate*f;
                        }
                    }
                }
            }
        }

        if(breakup_exposure>0.0){
            for(int i=1;i<n_particles;++i){
                double B=breakup_exposure*pow(diameter[i]/breakup_reference_diameter,breakup_size_exponent);
                double rate=B*ncur[i];if(!(rate>0.0))continue;
                total_fragmentation_events+=rate*h;dn[i]-=rate;
                double child_vol=0.5*vol[i];
                if(child_vol<=vol[0]){
                    // The equal-volume daughters are below the resolved grid.
                    // Retain total mass as an equivalent number of minimum-bin particles.
                    dn[0]+=rate*(vol[i]/vol[0]);
                    fragmentation_underflow_events+=rate*h;
                }else{
                    int k=0;while(k+1<n_particles && vol[k+1]<child_vol)++k;
                    if(k+1>=n_particles){atomicExch(error_flag,34);return;}
                    double f=(child_vol-vol[k])/(vol[k+1]-vol[k]);if(f<0.0)f=0.0;if(f>1.0)f=1.0;
                    dn[k]+=2.0*rate*(1.0-f);dn[k+1]+=2.0*rate*f;
                }
            }
        }

        for(int i=0;i<n_particles;++i){
            double nv=ncur[i]+h*dn[i];
            if(nv<0.0 && nv>-1e-12*fmax(ncur[i],1.0))nv=0.0;
            if(nv<0.0 || !isfinite(nv)){atomicExch(error_flag,35);return;}
            ncur[i]=nv;
        }
        u+=h;++steps;
    }
    if(u+1e-12<1.0){atomicExch(error_flag,36);return;}
    double final_mass=0.0,final_n=0.0,final_area=0.0;
    for(int i=0;i<n_particles;++i){
        final_mass+=ncur[i]*mass[i];final_n+=ncur[i];final_area+=ncur[i]*PI_D*0.25*diameter[i]*diameter[i];
    }
    if(!(final_mass>0.0)){atomicExch(error_flag,37);return;}
    for(int i=0;i<n_particles;++i)effective_weights[i]=(ncur[i]*mass[i])/final_mass;
    double mass_error=(concentration>0.0)?(final_mass-concentration)/concentration:0.0;
    double number_reduction=(initial_n>0.0)?1.0-final_n/initial_n:0.0;
    population_scalars[0]=aggregation_exposure;
    population_scalars[1]=breakup_exposure;
    population_scalars[2]=breakup_reference_diameter;
    population_scalars[3]=breakup_size_exponent;
    population_scalars[4]=initial_n;population_scalars[5]=final_n;
    population_scalars[6]=number_reduction;population_scalars[7]=mass_error;
    population_scalars[8]=initial_area;population_scalars[9]=final_area;
    population_scalars[10]=effective_weights[n_particles-1];
    population_scalars[11]=(total_aggregation_events>0.0)?aggregation_overflow_events/total_aggregation_events:0.0;
    population_scalars[12]=(total_fragmentation_events>0.0)?fragmentation_underflow_events/total_fragmentation_events:0.0;
    population_scalars[13]=(initial_n>0.0)?total_aggregation_events/initial_n:0.0;
    population_scalars[14]=(initial_n>0.0)?total_fragmentation_events/initial_n:0.0;
    population_scalars[15]=(double)steps;
}

__global__ void transport_precompute_kernel(
    const double* diameter,const double* weights,const double* qext,const double* qsca,const int n_particles,
    const double concentration,const double density,const int weight_mode,const int event_model,const double wave_scale,
    const double n_medium,const double wavelength,const double wave_rms_height_m,
    double* radius,double* particle_mass,double* source_weight,double* number_density,
    double* sigma_geom,double* sigma_sca,double* sigma_ext,double* sigma_wave,double* sigma_event,
    double* mu_geom_bin,double* mu_sca_bin,double* mu_ext_bin,double* mu_wave_bin,double* mu_event_bin,
    double* particle_event_weight,double* geometric_event_weight,double* wave_event_weight,
    double* geometric_cdf,double* wave_cdf,double* particle_event_cdf,double* mu_scalars,int* error_flag){
    if(blockIdx.x||threadIdx.x)return;
    double sw=0.0;for(int i=0;i<n_particles;++i)sw+=weights[i];if(!(sw>0.0)){atomicExch(error_flag,4);return;}
    double avg_mass=0.0;
    for(int i=0;i<n_particles;++i){
        double r=0.5*diameter[i];radius[i]=r;source_weight[i]=weights[i]/sw;
        double m=density*(4.0/3.0)*PI_D*r*r*r;particle_mass[i]=m;avg_mass+=source_weight[i]*m;
    }
    for(int i=0;i<n_particles;++i){
        double nd=0.0;
        if(weight_mode==0) nd=concentration*source_weight[i]/particle_mass[i];
        else nd=(avg_mass>0.0?concentration/avg_mass:0.0)*source_weight[i];
        number_density[i]=nd;
        double sg=PI_D*radius[i]*radius[i],ss=fmax(qsca[i],0.0)*sg,se=fmax(qext[i],0.0)*sg;
        double swv=0.0;
        if(event_model==3) swv=wave_scale*fmax(qsca[i]-1.0,0.0)*sg;
        else if(event_model==4 || event_model==5) swv=fmax(ss-sg,0.0);
        sigma_geom[i]=sg;sigma_sca[i]=ss;sigma_ext[i]=se;sigma_wave[i]=swv;
        mu_geom_bin[i]=nd*sg;mu_sca_bin[i]=nd*ss;mu_ext_bin[i]=nd*se;mu_wave_bin[i]=nd*swv;
    }
    double mg=0.0,ms=0.0,me=0.0,mw=0.0;
    for(int i=0;i<n_particles;++i){mg+=mu_geom_bin[i];ms+=mu_sca_bin[i];me+=mu_ext_bin[i];mw+=mu_wave_bin[i];}
    double mt=0.0;
    for(int i=0;i<n_particles;++i){
        double mb;
        if(event_model==0)mb=mu_geom_bin[i];
        else if(event_model==1)mb=mu_sca_bin[i];
        else if(event_model==2)mb=mu_ext_bin[i];
        else if(event_model==3)mb=mu_geom_bin[i]+mu_wave_bin[i];
        else mb=mu_sca_bin[i]; // exact-Mie total event rate.
        mu_event_bin[i]=mb;mt+=mb;
        if(event_model==0)sigma_event[i]=sigma_geom[i];
        else if(event_model==1)sigma_event[i]=sigma_sca[i];
        else if(event_model==2)sigma_event[i]=sigma_ext[i];
        else if(event_model==3)sigma_event[i]=sigma_geom[i]+sigma_wave[i];
        else sigma_event[i]=sigma_sca[i];
    }
    // For the geometric transport coefficient is the exact-rate remainder
    // mu_sca-mu_wave = nd*min(sigma_sca,sigma_geom).  The stored mu_geom_bin remains
    // the physical projected-area coefficient for diagnostics/CSA accounting.
    double mtrace_geom=((event_model==4)||(event_model==5))?fmax(mt-mw,0.0):((event_model==3)?mg:mt);
    double mtrace_wave=((event_model==3)||(event_model==4)||(event_model==5))?mw:0.0;
    double cg=0.0,cw=0.0,ct=0.0;
    for(int i=0;i<n_particles;++i){
        double mgi=((event_model==4)||(event_model==5))?fmax(mu_event_bin[i]-mu_wave_bin[i],0.0):mu_geom_bin[i];
        double pg=((event_model==3)||(event_model==4)||(event_model==5))?(mtrace_geom>0.0?mgi/mtrace_geom:0.0):(mt>0.0?mu_event_bin[i]/mt:0.0);
        double pw=(((event_model==3)||(event_model==4)||(event_model==5))&&mtrace_wave>0.0?mu_wave_bin[i]/mtrace_wave:0.0),pt=(mt>0.0?mu_event_bin[i]/mt:0.0);
        geometric_event_weight[i]=pg;wave_event_weight[i]=pw;particle_event_weight[i]=pt;
        cg+=pg;cw+=pw;ct+=pt;geometric_cdf[i]=cg;wave_cdf[i]=cw;particle_event_cdf[i]=ct;
    }
    if(n_particles>0){if(cg>0.0)geometric_cdf[n_particles-1]=1.0;if(cw>0.0)wave_cdf[n_particles-1]=1.0;if(ct>0.0)particle_event_cdf[n_particles-1]=1.0;}
    // [trace geometric channel, trace wave channel, physical geom, exact sca, ext, wave, total event, wave coherent fraction]
    mu_scalars[0]=mtrace_geom;mu_scalars[1]=mtrace_wave;
    mu_scalars[2]=mg;mu_scalars[3]=ms;mu_scalars[4]=me;mu_scalars[5]=mtrace_wave;mu_scalars[6]=mt;
    double cwave=1.0;
    if((event_model==3 || event_model==4 || event_model==5) && wave_rms_height_m>0.0 && wavelength>0.0 && n_medium>0.0){
        double a=4.0*PI_D*n_medium*wave_rms_height_m/wavelength;
        cwave=exp(-a*a);
        if(cwave<0.0)cwave=0.0;if(cwave>1.0)cwave=1.0;
    }
    mu_scalars[7]=cwave;
}


// Generate the drift-diffusion field entirely on CUDA.  One thread is
// intentional: there are only 37/38 PSD bins and this deterministic startup
// integration is negligible beside ray tracing while avoiding reduction races.
__device__ double schiller_naumann_velocity(const double v0,const double d,const double rho_w,const double mu,double* out_Re,double* out_drag){
    if(!(v0>0.0)){if(out_Re)*out_Re=0.0;if(out_drag)*out_drag=1.0;return 0.0;}
    double v=v0,K=rho_w*d/mu;
    for(int it=0;it<24;++it){
        double Re=fmax(K*v,0.0);double corr=0.15*pow(Re,0.687);
        double f=v*(1.0+corr)-v0;double df=1.0+1.687*corr;
        double vn=fmax(v-f/df,0.0);
        if(fabs(vn-v)<=2e-14*fmax(1.0,fabs(v))){v=vn;break;}v=vn;
    }
    double Re=fmax(K*v,0.0),drag=1.0+0.15*pow(Re,0.687);
    if(out_Re)*out_Re=Re;if(out_drag)*out_drag=drag;return v;
}

__global__ void sediment_transport_precompute_kernel(
    const double* diameter,const int n_particles,const int TRANSPORT_MODEL,
    const double particle_density,const double water_density,const double water_dynamic_viscosity,
    const double gravity,const double eddy_diffusivity,const double R,const double ZMIN,const double ZMEAN,
    const int FREE_SURFACE_MODEL,const double VORTEX_DELTA_H,const double VORTEX_CORE_RADIUS,const int NQUAD,const int N_RADIAL,
    const double* TRACE_GEOM_BIN,const double* TRACE_WAVE_BIN,
    double* RESPONSE_TIME,double* SETTLING_VELOCITY,double* SETTLING_RE,double* SETTLING_DRAG,
    double* SED_A,double* SED_B,double* SED_NORM_SCALED,double* SED_MAX_SCALE,
    double* RADIAL_MAX_SLIP,double* RADIAL_MAX_RE,double* RADIAL_CENTER_LOG,double* SED_RADIAL_LOG,double* SED_RADIAL_WR,
    double* SCALE_SENSOR_CENTER,double* SCALE_SENSOR_WALL,double* SCALE_BOTTOM_CENTER,double* SCALE_BOTTOM_WALL,
    double* MAJORANTS,int* error_flag){
    if(blockIdx.x||threadIdx.x)return;
    if(TRANSPORT_MODEL==0){
        double mg=0.0,mw=0.0;
        for(int i=0;i<n_particles;++i){
            RESPONSE_TIME[i]=0.0;SETTLING_VELOCITY[i]=0.0;SETTLING_RE[i]=0.0;SETTLING_DRAG[i]=1.0;
            SED_A[i]=0.0;SED_B[i]=0.0;SED_NORM_SCALED[i]=1.0;SED_MAX_SCALE[i]=1.0;
            RADIAL_MAX_SLIP[i]=0.0;RADIAL_MAX_RE[i]=0.0;RADIAL_CENTER_LOG[i]=0.0;
            SCALE_SENSOR_CENTER[i]=1.0;SCALE_SENSOR_WALL[i]=1.0;SCALE_BOTTOM_CENTER[i]=1.0;SCALE_BOTTOM_WALL[i]=1.0;
            for(int j=0;j<N_RADIAL;++j){SED_RADIAL_LOG[(long long)i*N_RADIAL+j]=0.0;SED_RADIAL_WR[(long long)i*N_RADIAL+j]=0.0;}
            mg+=TRACE_GEOM_BIN[i];mw+=TRACE_WAVE_BIN[i];
        }
        MAJORANTS[0]=mg;MAJORANTS[1]=mw;MAJORANTS[2]=mg+mw;MAJORANTS[3]=0.0;return;
    }
    if((TRANSPORT_MODEL!=1 && TRANSPORT_MODEL!=2 && TRANSPORT_MODEL!=3) || !(particle_density>water_density) || !(water_density>0.0) || !(water_dynamic_viscosity>0.0) || !(gravity>0.0) || !(eddy_diffusivity>0.0) || !(R>0.0) || NQUAD<64 || N_RADIAL<65){atomicExch(error_flag,31);return;}
    if(FREE_SURFACE_MODEL<0 || FREE_SURFACE_MODEL>2 || (VORTEX_DELTA_H>0.0 && FREE_SURFACE_MODEL==0) || (FREE_SURFACE_MODEL==2 && !(VORTEX_CORE_RADIUS>0.0))){atomicExch(error_flag,32);return;}
    double R2=R*R,a2=VORTEX_CORE_RADIUS*VORTEX_CORE_RADIUS;
    double Hinf=0.0,hbar=0.0,omega2=0.0;
    if(VORTEX_DELTA_H>0.0 && FREE_SURFACE_MODEL==1){
        Hinf=VORTEX_DELTA_H;hbar=0.5*VORTEX_DELTA_H;omega2=2.0*gravity*VORTEX_DELTA_H/R2;
    }else if(VORTEX_DELTA_H>0.0 && FREE_SURFACE_MODEL==2){
        Hinf=VORTEX_DELTA_H*(a2+R2)/R2;hbar=Hinf*(1.0-(a2/R2)*log1p(R2/a2));omega2=2.0*gravity*Hinf/a2;
    }
    double fR=(FREE_SURFACE_MODEL==2)?R2/(a2+R2):R2;
    double Hmean=ZMEAN-ZMIN,dr=R/(double)NQUAD,drad=R/(double)(N_RADIAL-1);
    if(!(Hmean>0.0)){atomicExch(error_flag,33);return;}
    double major_g=0.0,major_w=0.0,maxAeq=0.0;
    for(int i=0;i<n_particles;++i){
        double d=diameter[i];
        double tau=particle_density*d*d/(18.0*water_dynamic_viscosity);
        double ws0=(particle_density-water_density)*gravity*d*d/(18.0*water_dynamic_viscosity);
        double sre=0.0,sdrag=1.0;
        double ws=(TRANSPORT_MODEL==2 || TRANSPORT_MODEL==3)?schiller_naumann_velocity(ws0,d,water_density,water_dynamic_viscosity,&sre,&sdrag):ws0;
        if(TRANSPORT_MODEL==1){sre=water_density*d*ws/water_dynamic_viscosity;sdrag=1.0;}
        RESPONSE_TIME[i]=tau;SETTLING_VELOCITY[i]=ws;SETTLING_RE[i]=sre;SETTLING_DRAG[i]=sdrag;
        double B=ws/eddy_diffusivity;SED_B[i]=B;
        double maxwr=0.0,maxre=0.0;
        long long base=(long long)i*(long long)N_RADIAL;
        if(TRANSPORT_MODEL==1){
            double A=tau*omega2*((FREE_SURFACE_MODEL==2)?a2:1.0)/(2.0*eddy_diffusivity);SED_A[i]=A;if(A>maxAeq)maxAeq=A;
            for(int j=0;j<N_RADIAL;++j){double r=(double)j*drad,rr2=r*r,f=(FREE_SURFACE_MODEL==2)?rr2/(a2+rr2):rr2;SED_RADIAL_LOG[base+j]=A*(f-fR);}
            for(int j=0;j<N_RADIAL;++j){
                double r=(double)j*drad;double acc=0.0;if(r>0.0)acc=(FREE_SURFACE_MODEL==1)?omega2*r:((FREE_SURFACE_MODEL==2)?omega2*r/pow(1.0+r*r/a2,2.0):0.0);double wr=tau*acc;double re=water_density*d*wr/water_dynamic_viscosity;
                SED_RADIAL_WR[base+j]=wr;if(wr>maxwr)maxwr=wr;if(re>maxre)maxre=re;
            }
        } else {
            // First pass: integrate the corrected radial slip from centre to wall.
            double total=0.0,prev=0.0;
            for(int j=0;j<N_RADIAL;++j){
                double r=(double)j*drad;double acc=0.0;if(r>0.0)acc=(FREE_SURFACE_MODEL==1)?omega2*r:((FREE_SURFACE_MODEL==2)?omega2*r/pow(1.0+r*r/a2,2.0):0.0);double w0=tau*acc,re=0.0,drag=1.0;
                double wr=schiller_naumann_velocity(w0,d,water_density,water_dynamic_viscosity,&re,&drag);
                SED_RADIAL_WR[base+j]=wr;if(j>0)total+=0.5*(prev+wr)*drad;prev=wr;if(wr>maxwr)maxwr=wr;if(re>maxre)maxre=re;
            }
            // Second pass: store psi(r)=[integral_0^r w ds - integral_0^R w ds]/Dt.
            double cum=0.0;prev=0.0;
            for(int j=0;j<N_RADIAL;++j){
                double wr=SED_RADIAL_WR[base+j];
                if(j>0)cum+=0.5*(prev+wr)*drad;prev=wr;
                double psi=(cum-total)/eddy_diffusivity;if(psi>0.0)psi=0.0;SED_RADIAL_LOG[base+j]=psi;
            }
            SED_RADIAL_LOG[base+N_RADIAL-1]=0.0;
            double Aeq=(fR>0.0)?-SED_RADIAL_LOG[base]/fR:0.0;SED_A[i]=Aeq;if(Aeq>maxAeq)maxAeq=Aeq;
        }
        RADIAL_MAX_SLIP[i]=maxwr;RADIAL_MAX_RE[i]=maxre;RADIAL_CENTER_LOG[i]=SED_RADIAL_LOG[base];
        double integ=0.0;
        for(int j=0;j<NQUAD;++j){
            double r=((double)j+0.5)*dr,rr2=r*r;
            double zsurf=ZMEAN;if(FREE_SURFACE_MODEL==1 && VORTEX_DELTA_H>0.0)zsurf=ZMEAN+VORTEX_DELTA_H*(rr2/R2-0.5);else if(FREE_SURFACE_MODEL==2 && VORTEX_DELTA_H>0.0)zsurf=ZMEAN-hbar+Hinf*rr2/(a2+rr2);double H=fmax(zsurf-ZMIN,0.0);
            double radial_log;
            if(TRANSPORT_MODEL==1){double f=(FREE_SURFACE_MODEL==2)?rr2/(a2+rr2):rr2;radial_log=SED_A[i]*(f-fR);}else{
                double u=r/R*(double)(N_RADIAL-1);int k=(int)floor(u);if(k<0)k=0;if(k>N_RADIAL-2)k=N_RADIAL-2;double t=u-(double)k;
                radial_log=(1.0-t)*SED_RADIAL_LOG[base+k]+t*SED_RADIAL_LOG[base+k+1];if(radial_log>0.0)radial_log=0.0;
            }
            double radial=exp(radial_log);double iz=(fabs(B)<1e-14)?radial*H:radial*(-expm1(-B*H))/B;integ+=r*iz;
        }
        double norm=(2.0/(R2*Hmean))*integ*dr;
        if(!(norm>0.0) || !isfinite(norm)){atomicExch(error_flag,34);return;}
        double mx=1.0/norm;SED_NORM_SCALED[i]=norm;SED_MAX_SCALE[i]=mx;
        double rwall=0.95*R,uwall=rwall/R*(double)(N_RADIAL-1);int kw=(int)floor(uwall);if(kw>N_RADIAL-2)kw=N_RADIAL-2;double tw=uwall-(double)kw;
        double log0=(TRANSPORT_MODEL==1)?SED_A[i]*(0.0-fR):SED_RADIAL_LOG[base];
        double rrw2=rwall*rwall,fw=(FREE_SURFACE_MODEL==2)?rrw2/(a2+rrw2):rrw2;double logw=(TRANSPORT_MODEL==1)?SED_A[i]*(fw-fR):((1.0-tw)*SED_RADIAL_LOG[base+kw]+tw*SED_RADIAL_LOG[base+kw+1]);
        double hs=fmax(0.0-ZMIN,0.0);
        SCALE_SENSOR_CENTER[i]=exp(log0-B*hs)/norm;SCALE_SENSOR_WALL[i]=exp(logw-B*hs)/norm;
        SCALE_BOTTOM_CENTER[i]=exp(log0)/norm;SCALE_BOTTOM_WALL[i]=exp(logw)/norm;
        major_g+=TRACE_GEOM_BIN[i]*mx;major_w+=TRACE_WAVE_BIN[i]*mx;
    }
    MAJORANTS[0]=major_g;MAJORANTS[1]=major_w;MAJORANTS[2]=major_g+major_w;MAJORANTS[3]=maxAeq;
    if(!(MAJORANTS[2]>=0.0) || !isfinite(MAJORANTS[2]))atomicExch(error_flag,35);
}

// ---------------- explicit secondary meridional circulation ----------------
// The fluid streamfunction is
//   psi = -A r^2(1-r^2/R^2)^2 eta(1-eta),  eta=(z-ZMIN)/(zsurf(r)-ZMIN)
// with A=2*UPEAK.  UPEAK is chi times the maximum selected-vortex azimuthal speed.
// This gives outward bottom flow, outer-annulus upflow, inward surface flow and
// axial downflow while making psi constant on every continuous vessel boundary.
__device__ void meridional_surface_geometry_cuda(
    const int FREE_SURFACE_MODEL,const double r,const double ZMEAN,const double R,
    const double DELTA_H,const double CORE_R,double* zs,double* hp){
    double rr=fmin(fmax(r,0.0),R),rr2=rr*rr,R2=R*R;*zs=ZMEAN;*hp=0.0;
    if(!(DELTA_H>0.0) || FREE_SURFACE_MODEL==0)return;
    if(FREE_SURFACE_MODEL==1){
        *zs=ZMEAN+DELTA_H*(rr2/R2-0.5);*hp=2.0*DELTA_H*rr/R2;return;
    }
    double a2=CORE_R*CORE_R,Hinf=DELTA_H*(a2+R2)/R2;
    double hbar=Hinf*(1.0-(a2/R2)*log1p(R2/a2));
    *zs=ZMEAN-hbar+Hinf*rr2/(a2+rr2);*hp=2.0*Hinf*a2*rr/((a2+rr2)*(a2+rr2));
}

__device__ void meridional_velocity_cuda(
    const double r,const double z,const double ZMIN,const double ZMEAN,const double R,
    const int FREE_SURFACE_MODEL,const double DELTA_H,const double CORE_R,const double UPEAK,double* ur,double* uz){
    *ur=0.0;*uz=0.0;
    if(!(UPEAK>0.0) || !(R>0.0) || !(DELTA_H>0.0) || (FREE_SURFACE_MODEL==2 && !(CORE_R>0.0)))return;
    double rr=fmin(fmax(r,0.0),R),R2=R*R,rr2=rr*rr,zs=ZMEAN,hp=0.0;
    meridional_surface_geometry_cuda(FREE_SURFACE_MODEL,rr,ZMEAN,R,DELTA_H,CORE_R,&zs,&hp);
    double H=zs-ZMIN;
    if(!(H>0.0))return;
    double eta=(z-ZMIN)/H;if(eta<0.0)eta=0.0;if(eta>1.0)eta=1.0;
    double s=rr2/R2,f=eta*(1.0-eta),fp=1.0-2.0*eta;
    double Apsi=2.0*UPEAK,om=1.0-s;
    *ur=Apsi*rr*om*om*fp/H;
    *uz=-Apsi*(2.0*om*(1.0-3.0*s)*f-rr*om*om*fp*eta*hp/H);
}

__device__ double radial_slip_lookup_cuda(const int pidx,const double r,const double R,
    const double* WR,const int N_RADIAL){
    if(N_RADIAL<2 || !(R>0.0))return 0.0;
    double q=fmin(fmax(r/R,0.0),1.0)*(double)(N_RADIAL-1);
    int j=(int)floor(q);double t=q-(double)j;if(j>=N_RADIAL-1){j=N_RADIAL-2;t=1.0;}
    long long b=(long long)pidx*(long long)N_RADIAL;
    return (1.0-t)*WR[b+j]+t*WR[b+j+1];
}

__device__ int meridional_physical_fluid(const double r,const double z,const double ZMIN,const double ZMEAN,
    const double R,const int FREE_SURFACE_MODEL,const double DELTA_H,const double CORE_R){
    if(z<ZMIN)return 0;
    double zs=ZMEAN,hp=0.0;meridional_surface_geometry_cuda(FREE_SURFACE_MODEL,r,ZMEAN,R,DELTA_H,CORE_R,&zs,&hp);
    return z<zs;
}

__global__ void meridional_active_mask_kernel(
    const int NR,const int NZ,const double R,const double ZMIN,const double ZGRIDMAX,const double ZMEAN,
    const int FREE_SURFACE_MODEL,const double DELTA_H,const double CORE_R,unsigned char* ACTIVE){
    int t=blockDim.x*blockIdx.x+threadIdx.x,plane=NR*NZ;if(t>=plane)return;
    int i=t/NZ,k=t-i*NZ;double dr=R/(double)NR,dz=(ZGRIDMAX-ZMIN)/(double)NZ;
    double r=((double)i+0.5)*dr,z=ZMIN+((double)k+0.5)*dz;
    ACTIVE[t]=(unsigned char)(meridional_physical_fluid(r,z,ZMIN,ZMEAN,R,FREE_SURFACE_MODEL,DELTA_H,CORE_R)?1:0);
}

__global__ void meridional_grid_initialize_kernel(
    const int n_particles,const int NR,const int NZ,const double R,const double ZMIN,const double ZGRIDMAX,const int FREE_SURFACE_MODEL,const double CORE_R,
    const double* SED_A,const double* SED_B,const double* SED_NORM,const double* SED_RADIAL_LOG,const int N_RADIAL,
    const unsigned char* ACTIVE,double* GRID){
    long long plane=(long long)NR*(long long)NZ,total=(long long)n_particles*plane;
    long long t=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(t>=total)return;
    int p=(int)(t/plane),c=(int)(t-(long long)p*plane),i=c/NZ,k=c-i*NZ;
    if(!ACTIVE[c]){GRID[t]=0.0;return;}
    double dr=R/(double)NR,dz=(ZGRIDMAX-ZMIN)/(double)NZ;
    float x=(float)(((double)i+0.5)*dr),z=(float)(ZMIN+((double)k+0.5)*dz);
    GRID[t]=sediment_local_scale(2,FREE_SURFACE_MODEL,p,SED_A,SED_B,SED_NORM,SED_RADIAL_LOG,N_RADIAL,GRID,NR,NZ,(float)ZGRIDMAX,x,0.0f,z,(float)R,(float)ZMIN,(float)CORE_R);
}

__global__ void meridional_build_coefficients_kernel(
    const int n_particles,const int NR,const int NZ,const double R,const double ZMIN,const double ZGRIDMAX,const double ZMEAN,
    const int FREE_SURFACE_MODEL,const double DELTA_H,const double CORE_R,const double UPEAK,const double EDDY_DIFFUSIVITY,
    const double* SETTLING_VELOCITY,const double* RADIAL_WR,const int N_RADIAL,const unsigned char* ACTIVE,
    double* AP,double* AE,double* AW,double* AN,double* AS){
    long long plane=(long long)NR*(long long)NZ,total=(long long)n_particles*plane;
    long long t=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(t>=total)return;
    int p=(int)(t/plane),c=(int)(t-(long long)p*plane),i=c/NZ,k=c-i*NZ;
    AP[t]=AE[t]=AW[t]=AN[t]=AS[t]=0.0;if(!ACTIVE[c])return;
    double dr=R/(double)NR,dz=(ZGRIDMAX-ZMIN)/(double)NZ;
    double rc=((double)i+0.5)*dr,zc=ZMIN+((double)k+0.5)*dz;
    double rw=(double)i*dr,re=(double)(i+1)*dr;
    double Az=0.5*(re*re-rw*rw),ap=0.0,ur=0.0,uz=0.0,V=0.0,F=0.0,diff=0.0;
    // East radial face.
    if(i+1<NR && ACTIVE[(i+1)*NZ+k] && meridional_physical_fluid(re,zc,ZMIN,ZMEAN,R,FREE_SURFACE_MODEL,DELTA_H,CORE_R)){
        meridional_velocity_cuda(re,zc,ZMIN,ZMEAN,R,FREE_SURFACE_MODEL,DELTA_H,CORE_R,UPEAK,&ur,&uz);
        V=ur+radial_slip_lookup_cuda(p,re,R,RADIAL_WR,N_RADIAL);F=V*re*dz;diff=EDDY_DIFFUSIVITY*re*dz/dr;
        AE[t]=diff+fmax(-F,0.0);ap+=diff+fmax(F,0.0);
    }
    // West radial face; outward normal is -r.
    if(i>0 && ACTIVE[(i-1)*NZ+k] && meridional_physical_fluid(rw,zc,ZMIN,ZMEAN,R,FREE_SURFACE_MODEL,DELTA_H,CORE_R)){
        meridional_velocity_cuda(rw,zc,ZMIN,ZMEAN,R,FREE_SURFACE_MODEL,DELTA_H,CORE_R,UPEAK,&ur,&uz);
        V=ur+radial_slip_lookup_cuda(p,rw,R,RADIAL_WR,N_RADIAL);F=-V*rw*dz;diff=EDDY_DIFFUSIVITY*rw*dz/dr;
        AW[t]=diff+fmax(-F,0.0);ap+=diff+fmax(F,0.0);
    }
    // Top horizontal face.
    double zf=zc+0.5*dz;
    if(k+1<NZ && ACTIVE[i*NZ+k+1] && meridional_physical_fluid(rc,zf,ZMIN,ZMEAN,R,FREE_SURFACE_MODEL,DELTA_H,CORE_R)){
        meridional_velocity_cuda(rc,zf,ZMIN,ZMEAN,R,FREE_SURFACE_MODEL,DELTA_H,CORE_R,UPEAK,&ur,&uz);
        V=uz-SETTLING_VELOCITY[p];F=V*Az;diff=EDDY_DIFFUSIVITY*Az/dz;
        AN[t]=diff+fmax(-F,0.0);ap+=diff+fmax(F,0.0);
    }
    // Bottom horizontal face; outward normal is -z.
    zf=zc-0.5*dz;
    if(k>0 && ACTIVE[i*NZ+k-1] && meridional_physical_fluid(rc,zf,ZMIN,ZMEAN,R,FREE_SURFACE_MODEL,DELTA_H,CORE_R)){
        meridional_velocity_cuda(rc,zf,ZMIN,ZMEAN,R,FREE_SURFACE_MODEL,DELTA_H,CORE_R,UPEAK,&ur,&uz);
        V=uz-SETTLING_VELOCITY[p];F=-V*Az;diff=EDDY_DIFFUSIVITY*Az/dz;
        AS[t]=diff+fmax(-F,0.0);ap+=diff+fmax(F,0.0);
    }
    AP[t]=ap;
}

__global__ void meridional_relax_kernel(
    const int n_particles,const int NR,const int NZ,const int parity,const double OMEGA,
    const unsigned char* ACTIVE,const double* AP,const double* AE,const double* AW,const double* AN,const double* AS,double* GRID){
    long long plane=(long long)NR*(long long)NZ,total=(long long)n_particles*plane;
    long long t=(long long)blockDim.x*blockIdx.x+threadIdx.x;if(t>=total)return;
    int p=(int)(t/plane),c=(int)(t-(long long)p*plane),i=c/NZ,k=c-i*NZ;
    if(!ACTIVE[c] || ((i+k)&1)!=parity || !(AP[t]>0.0))return;
    long long b=(long long)p*plane;double num=0.0;
    if(i+1<NR)num+=AE[t]*GRID[b+(long long)(i+1)*NZ+k];
    if(i>0)num+=AW[t]*GRID[b+(long long)(i-1)*NZ+k];
    if(k+1<NZ)num+=AN[t]*GRID[b+(long long)i*NZ+k+1];
    if(k>0)num+=AS[t]*GRID[b+(long long)i*NZ+k-1];
    double target=num/AP[t],old=GRID[t],nv=(1.0-OMEGA)*old+OMEGA*target;
    if(!isfinite(nv) || nv<1e-14)nv=1e-14;GRID[t]=nv;
}

__global__ void meridional_normalize_kernel(
    const int n_particles,const int NR,const int NZ,const double R,const double ZMIN,const double ZGRIDMAX,const double ZMEAN,
    const unsigned char* ACTIVE,double* GRID,int* error_flag){
    int p=blockIdx.x;if(p>=n_particles || threadIdx.x)return;
    long long plane=(long long)NR*(long long)NZ,b=(long long)p*plane;double dr=R/(double)NR,dz=(ZGRIDMAX-ZMIN)/(double)NZ;
    double mass=0.0;
    for(int i=0;i<NR;++i){double rw=(double)i*dr,re=(double)(i+1)*dr,vol=0.5*(re*re-rw*rw)*dz;
        for(int k=0;k<NZ;++k){int c=i*NZ+k;if(ACTIVE[c])mass+=GRID[b+c]*vol;}}
    double target=0.5*R*R*(ZMEAN-ZMIN);if(!(mass>0.0) || !(target>0.0) || !isfinite(mass)){atomicExch(error_flag,41);return;}
    double scale=target/mass;
    for(int i=0;i<NR;++i)for(int k=0;k<NZ;++k){int c=i*NZ+k;if(ACTIVE[c])GRID[b+c]*=scale;}
}

__global__ void meridional_residual_kernel(
    const int n_particles,const int NR,const int NZ,const unsigned char* ACTIVE,
    const double* AP,const double* AE,const double* AW,const double* AN,const double* AS,const double* GRID,double* RESIDUAL){
    int p=blockIdx.x;if(p>=n_particles || threadIdx.x)return;long long plane=(long long)NR*(long long)NZ,b=(long long)p*plane;double mr=0.0;
    for(int i=0;i<NR;++i)for(int k=0;k<NZ;++k){int c=i*NZ+k;long long t=b+c;if(!ACTIVE[c] || !(AP[t]>0.0))continue;
        double num=0.0;if(i+1<NR)num+=AE[t]*GRID[b+(long long)(i+1)*NZ+k];if(i>0)num+=AW[t]*GRID[b+(long long)(i-1)*NZ+k];
        if(k+1<NZ)num+=AN[t]*GRID[b+(long long)i*NZ+k+1];if(k>0)num+=AS[t]*GRID[b+(long long)i*NZ+k-1];
        double den=fmax(fabs(AP[t]*GRID[t]),1e-30),q=fabs(AP[t]*GRID[t]-num)/den;if(q>mr)mr=q;}
    RESIDUAL[p]=mr;
}

__global__ void meridional_extend_grid_kernel(
    const int n_particles,const int NR,const int NZ,const unsigned char* ACTIVE,double* GRID){
    int t=blockDim.x*blockIdx.x+threadIdx.x,total=n_particles*NR;if(t>=total)return;int p=t/NR,i=t-p*NR;
    long long plane=(long long)NR*(long long)NZ,b=(long long)p*plane+(long long)i*NZ;int top=-1;
    for(int k=NZ-1;k>=0;--k){if(ACTIVE[i*NZ+k]){top=k;break;}}
    if(top<0){for(int k=0;k<NZ;++k)GRID[b+k]=1.0;return;}double q=GRID[b+top];for(int k=top+1;k<NZ;++k)GRID[b+k]=q;
}

__global__ void meridional_finalize_kernel(
    const int n_particles,const int NR,const int NZ,const double R,const double ZMIN,const double ZGRIDMAX,
    const unsigned char* ACTIVE,const double* GRID,double* SED_NORM,double* SED_MAX,double* SED_MIN,
    double* SCALE_SENSOR_CENTER,double* SCALE_SENSOR_WALL,double* SCALE_BOTTOM_CENTER,double* SCALE_BOTTOM_WALL){
    int p=blockIdx.x;if(p>=n_particles || threadIdx.x)return;long long plane=(long long)NR*(long long)NZ,b=(long long)p*plane;
    double mx=0.0,mn=1e300;for(int c=0;c<NR*NZ;++c)if(ACTIVE[c]){double q=GRID[b+c];if(q>mx)mx=q;if(q<mn)mn=q;}
    if(!(mx>0.0)){mx=1.0;mn=1.0;}SED_NORM[p]=1.0;SED_MAX[p]=mx;SED_MIN[p]=mn;
    SCALE_SENSOR_CENTER[p]=sediment_grid_sample(p,GRID,NR,NZ,0.0f,0.0f,0.0f,(float)R,(float)ZMIN,(float)ZGRIDMAX);
    SCALE_SENSOR_WALL[p]=sediment_grid_sample(p,GRID,NR,NZ,(float)(0.95*R),0.0f,0.0f,(float)R,(float)ZMIN,(float)ZGRIDMAX);
    SCALE_BOTTOM_CENTER[p]=sediment_grid_sample(p,GRID,NR,NZ,0.0f,0.0f,(float)ZMIN,(float)R,(float)ZMIN,(float)ZGRIDMAX);
    SCALE_BOTTOM_WALL[p]=sediment_grid_sample(p,GRID,NR,NZ,(float)(0.95*R),0.0f,(float)ZMIN,(float)R,(float)ZMIN,(float)ZGRIDMAX);
}

__global__ void meridional_majorants_kernel(const int n_particles,const double* TRACE_GEOM_BIN,const double* TRACE_WAVE_BIN,
    const double* SED_MAX,double* MAJORANTS){
    if(blockIdx.x||threadIdx.x)return;double mg=0.0,mw=0.0;for(int i=0;i<n_particles;++i){mg+=TRACE_GEOM_BIN[i]*SED_MAX[i];mw+=TRACE_WAVE_BIN[i]*SED_MAX[i];}
    MAJORANTS[0]=mg;MAJORANTS[1]=mw;MAJORANTS[2]=mg+mw;
}

__global__ void meridional_velocity_test_kernel(const float* rz,const int n,const float ZMIN,const float ZMEAN,const float R,
    const int FREE_SURFACE_MODEL,const float DELTA_H,const float CORE_R,const float UPEAK,float* ur,float* uz){
    int t=blockDim.x*blockIdx.x+threadIdx.x;if(t>=n)return;double a=0.0,b=0.0;meridional_velocity_cuda((double)rz[2*t],(double)rz[2*t+1],ZMIN,ZMEAN,R,FREE_SURFACE_MODEL,DELTA_H,CORE_R,UPEAK,&a,&b);ur[t]=(float)a;uz[t]=(float)b;
}


__global__ void sediment_field_test_kernel(
    const float* xyz,const int n_points,const int pidx,const int TRANSPORT_MODEL,const int FREE_SURFACE_MODEL,
    const double* SED_A,const double* SED_B,const double* SED_NORM_SCALED,const double* SED_RADIAL_LOG,const int N_RADIAL,
    const double* SED_GRID,const int GRID_NR,const int GRID_NZ,const float GRID_ZMAX,
    const float R,const float ZMIN,const float CORE_R,float* out_scale){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;if(tid>=n_points)return;
    out_scale[tid]=(float)sediment_local_scale(TRANSPORT_MODEL,FREE_SURFACE_MODEL,pidx,SED_A,SED_B,SED_NORM_SCALED,SED_RADIAL_LOG,N_RADIAL,SED_GRID,GRID_NR,GRID_NZ,GRID_ZMAX,xyz[3*tid],xyz[3*tid+1],xyz[3*tid+2],R,ZMIN,CORE_R);
}

__device__ int hardware_detector_id(float x,float y,float z,float vx,float vy,float vz,int mirror,
    float ring_in,float throat_out,float ring_out,float throat_r,float counterbore_r){
    if(mirror){x=-x;vx=-vx;}
    float best_t=INF_F;int best=-1;float rth2=throat_r*throat_r,rcb2=counterbore_r*counterbore_r;
    for(int j=0;j<18;++j){
        float th=(10.0f*(float)j)*(PI_F/180.0f),ux=sinf(th),uy=cosf(th);
        float den=vx*ux+vy*uy;if(den<=1e-10f)continue;
        float pdot=x*ux+y*uy;
        float t0=(ring_in-pdot)/den,t1=(throat_out-pdot)/den,t2=(ring_out-pdot)/den;
        if(!(t0>0.0f&&t1>t0&&t2>t1))continue;
        float q0x=x+t0*vx-ring_in*ux,q0y=y+t0*vy-ring_in*uy,q0z=z+t0*vz;
        float q1x=x+t1*vx-throat_out*ux,q1y=y+t1*vy-throat_out*uy,q1z=z+t1*vz;
        float q2x=x+t2*vx-ring_out*ux,q2y=y+t2*vy-ring_out*uy,q2z=z+t2*vz;
        if(q0x*q0x+q0y*q0y+q0z*q0z<=rth2 && q1x*q1x+q1y*q1y+q1z*q1z<=rth2 && q2x*q2x+q2y*q2y+q2z*q2z<=rcb2){
            if(t0<best_t){best_t=t0;best=j;}
        }
    }
    return best;
}


// external air-gap stray-light cavity.
// The 0.5-mm annulus between the acrylic tube outer radius and the nylon
// sensor-ring inner radius is traced explicitly only for rays that fail the
// established direct detector-bore acceptance.  The production baseline
// (RING_FACE_SURFACE_MODEL==0) bypasses this code exactly.
//
// Surface model 1 treats the broad inner cylindrical nylon face between bore
// openings as a smooth opaque dielectric.  Air->nylon Fresnel reflection is
// sampled; transmission into the thick nylon ring is loss.  After a face
// reflection the ray encounters the acrylic outer cylinder.  Air->acrylic
// Fresnel reflection keeps it in the external cavity; transmission into
// acrylic leaves this direct external-stray branch and is conservatively
// terminated rather than being misclassified as apparatus-only stray light.
// No diffuse albedo or fitted BRDF parameter is introduced.

// SFH213 packaged-photodiode angular response.
// TARDIIS uses the OSRAM/ams SFH213 5-mm clear-epoxy PIN photodiode.
// The manufacturer specifies a typical half angle phi_1/2 = 10 degrees and
// plots a relative directional response of approximately 0.10 at 20 degrees.
// We represent that published package-integrated response (lens + 1x1 mm chip)
// with the two-anchor generalized-Lorentz fit
//   S(phi) = 1 / [1 + (|phi|/10 deg)^p],  p = ln(9)/ln(2)
// which gives S(0)=1, S(10)=0.5 and S(20)=0.1 exactly.  No additional cos(phi)
// factor is applied because the datasheet directional curve already includes
// projected-area and package-lens effects.
__device__ float sfh213_relative_response_deg(float phi_deg){
    const float HALF_DEG=10.0f;
    const float P=3.169925001442312f; // ln(9)/ln(2)
    float a=fabsf(phi_deg);
    if(a>=90.0f)return 0.0f;
    float u=a/HALF_DEG;
    return 1.0f/(1.0f+powf(u,P));
}

__device__ float detector_incidence_angle_deg(float vx,float vy,float vz,int detector_id,int mirror){
    // NVRTC-portable invalid sentinel.  Do not rely on the host NAN macro: raw
    // CUDA source is compiled without a C math header by CuPy RawModule.
    if(detector_id<0||detector_id>=18)return -1.0f;
    if(mirror)vx=-vx;
    float th=(10.0f*(float)detector_id)*(PI_F/180.0f);
    float ux=sinf(th),uy=cosf(th);
    float c=vx*ux+vy*uy;
    c=fminf(fmaxf(c,-1.0f),1.0f);
    return acosf(c)*(180.0f/PI_F);
}

__device__ int hardware_throat_opening_id(float x,float y,float z,float vx,float vy,float vz,int mirror,
    float ring_in,float throat_r){
    if(mirror){x=-x;vx=-vx;}
    float best_t=INF_F;int best=-1;float r2=throat_r*throat_r;
    for(int j=0;j<19;++j){
        float th=(10.0f*(float)j)*(PI_F/180.0f),ux=sinf(th),uy=cosf(th);
        float den=vx*ux+vy*uy;if(den<=1.0e-10f)continue;
        float pdot=x*ux+y*uy;float t=(ring_in-pdot)/den;if(t<=0.0f)continue;
        float qx=x+t*vx-ring_in*ux,qy=y+t*vy-ring_in*uy,qz=z+t*vz;
        if(qx*qx+qy*qy+qz*qz<=r2 && t<best_t){best_t=t;best=j;}
    }
    return best; // 0..17 detector throat, 18 source throat, -1 solid face
}

__device__ int detector_once_from_air(
    unsigned int* state,float x,float y,float z,float vx,float vy,float vz,
    const float ring_in,const float throat_out,const float ring_out,const float throat_r,const float counterbore_r,
    const int detector_bore_surface_model,const float n_air,const float n_bore,const int max_bore_bounces,
    int* bore_bounces){
    *bore_bounces=0;
    if(detector_bore_surface_model==0)
        return hardware_detector_id(x,y,z,vx,vy,vz,0,ring_in,throat_out,ring_out,throat_r,counterbore_r);
    return detector_collimator_path(state,x,y,z,vx,vy,vz,0,ring_in,throat_out,ring_out,throat_r,counterbore_r,
                                    detector_bore_surface_model,n_air,n_bore,max_bore_bounces,bore_bounces);
}

__device__ int ring_face_cavity_detector_path(
    unsigned int* state,float x,float y,float z,float vx,float vy,float vz,int mirror,
    const float acrylic_outer_r,const float ring_in,const float throat_out,const float ring_out,
    const float throat_r,const float counterbore_r,
    const int ring_face_surface_model,const float n_air,const float n_ring,const float n_acrylic,const int max_ring_face_bounces,
    const int detector_bore_surface_model,const float n_bore,const int max_bore_bounces,
    int* bore_bounces,int* face_reflections){
    if(mirror){x=-x;vx=-vx;}
    *bore_bounces=0;*face_reflections=0;
    if(ring_face_surface_model==0)return -1;
    // The established direct path was already tested by the caller.  If this
    // leg enters any 4-mm throat but fails the complete bore acceptance, it is
    // absorbed by that bore rather than being allowed to reflect from solid face.
    if(hardware_throat_opening_id(x,y,z,vx,vy,vz,0,ring_in,throat_r)>=0)return -1;
    for(int b=0;b<max_ring_face_bounces;++b){
        float tr=cylinder_hit(x,y,vx,vy,ring_in);
        if(tr>=INF_F*0.5f)return -1;
        x+=tr*vx;y+=tr*vy;z+=tr*vz;
        float nx,ny;radial_normal(x,y,&nx,&ny);
        // Normal from nylon into the incident air cavity is -radial.
        float ci=fminf(fmaxf(vx*nx+vy*ny,0.0f),1.0f);
        int tir=0;float R=fresnel_R(ci,n_air,n_ring,&tir);
        if(rnd_uniform(state)>=R)return -1; // transmitted into bulk nylon -> loss
        float a,c,d;reflect3(vx,vy,vz,-nx,-ny,0.0f,&a,&c,&d);vx=a;vy=c;vz=d;(*face_reflections)++;
        x-=EPS_F*nx;y-=EPS_F*ny;

        float ta=cylinder_hit(x,y,vx,vy,acrylic_outer_r);
        if(ta>=INF_F*0.5f)return -1;
        x+=ta*vx;y+=ta*vy;z+=ta*vz;
        radial_normal(x,y,&nx,&ny);
        // At the acrylic outer surface, +radial points into incident air.
        ci=fminf(fmaxf(-dot3(vx,vy,vz,nx,ny,0.0f),0.0f),1.0f);
        R=fresnel_R(ci,n_air,n_acrylic,&tir);
        if(rnd_uniform(state)>=R)return -1; // re-enters acrylic: not an external-cavity-only stray path
        reflect3(vx,vy,vz,nx,ny,0.0f,&a,&c,&d);vx=a;vy=c;vz=d;
        x+=EPS_F*nx;y+=EPS_F*ny;

        int db=0;
        int det=detector_once_from_air(state,x,y,z,vx,vy,vz,ring_in,throat_out,ring_out,throat_r,counterbore_r,
                                       detector_bore_surface_model,n_air,n_bore,max_bore_bounces,&db);
        if(det>=0){*bore_bounces=db;return det;}
        // Detector/source throat interception without full acceptance is an
        // absorbing bore hit, not a reflection from the broad nylon face.
        if(hardware_throat_opening_id(x,y,z,vx,vy,vz,0,ring_in,throat_r)>=0)return -1;
    }
    return -1;
}

__global__ void score_detectors_kernel(
    const float* air_x,const float* air_y,const float* air_z,const float* air_vx,const float* air_vy,const float* air_vz,
    const int* status,const int* interaction_count,const int n,
    const float acrylic_outer_r,const float ring_in,const float throat_out,const float ring_out,const float throat_r,const float counterbore_r,const float legacy_accept_deg,
    const int detector_bore_surface_model,const float n_air,const float n_bore,const int max_bore_bounces,
    const int ring_face_surface_model,const float n_ring_face,const float n_acrylic,const int max_ring_face_bounces,
    const int photodiode_response_model,
    const unsigned int detector_seed,const unsigned int ray_offset,
    int* hardware_native,int* hardware_mirror,int* hardware_native_direct,int* hardware_mirror_direct,
    int* hardware_native_stray,int* hardware_mirror_stray,int* hardware_native_face_stray,int* hardware_mirror_face_stray,
    float* photodiode_native,float* photodiode_mirror,float* incidence_native_sum,float* incidence_mirror_sum,
    int* legacy_native,int* legacy_mirror,int* native_id,int* mirror_id,int* native_bounces,int* mirror_bounces,int* native_face_bounces,int* mirror_face_bounces){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;if(tid>=n)return;
    native_id[tid]=-1;mirror_id[tid]=-1;native_bounces[tid]=0;mirror_bounces[tid]=0;native_face_bounces[tid]=0;mirror_face_bounces[tid]=0;
    if(status[tid]!=1)return;
    float x=air_x[tid],y=air_y[tid],z=air_z[tid],vx=air_vx[tid],vy=air_vy[tid],vz=air_vz[tid];
    if(!(isfinite(x)&&isfinite(y)&&isfinite(z)&&isfinite(vx)&&isfinite(vy)&&isfinite(vz)))return;
    unsigned int gid=ray_offset+(unsigned int)tid;
    unsigned int stateN=detector_seed+gid*2246822519u+3266489917u;
    unsigned int stateM=detector_seed+gid*3266489917u+668265263u;
    int nb=0,mb=0,nfb=0,mfb=0,ni=-1,mi=-1;
    // Test direct detector acceptance first. With RING_FACE_SURFACE_MODEL=0,
    // the face cavity is bypassed; otherwise it is tried only after direct failure.
    if(detector_bore_surface_model==0){
        ni=hardware_detector_id(x,y,z,vx,vy,vz,0,ring_in,throat_out,ring_out,throat_r,counterbore_r);
        mi=hardware_detector_id(x,y,z,vx,vy,vz,1,ring_in,throat_out,ring_out,throat_r,counterbore_r);
    }else{
        ni=detector_collimator_path(&stateN,x,y,z,vx,vy,vz,0,ring_in,throat_out,ring_out,throat_r,counterbore_r,detector_bore_surface_model,n_air,n_bore,max_bore_bounces,&nb);
        mi=detector_collimator_path(&stateM,x,y,z,vx,vy,vz,1,ring_in,throat_out,ring_out,throat_r,counterbore_r,detector_bore_surface_model,n_air,n_bore,max_bore_bounces,&mb);
    }
    if(ni<0 && ring_face_surface_model!=0)
        ni=ring_face_cavity_detector_path(&stateN,x,y,z,vx,vy,vz,0,acrylic_outer_r,ring_in,throat_out,ring_out,throat_r,counterbore_r,
            ring_face_surface_model,n_air,n_ring_face,n_acrylic,max_ring_face_bounces,detector_bore_surface_model,n_bore,max_bore_bounces,&nb,&nfb);
    if(mi<0 && ring_face_surface_model!=0)
        mi=ring_face_cavity_detector_path(&stateM,x,y,z,vx,vy,vz,1,acrylic_outer_r,ring_in,throat_out,ring_out,throat_r,counterbore_r,
            ring_face_surface_model,n_air,n_ring_face,n_acrylic,max_ring_face_bounces,detector_bore_surface_model,n_bore,max_bore_bounces,&mb,&mfb);
    native_id[tid]=ni;mirror_id[tid]=mi;native_bounces[tid]=nb;mirror_bounces[tid]=mb;native_face_bounces[tid]=nfb;mirror_face_bounces[tid]=mfb;
    if(ni>=0){
        atomicAdd(&hardware_native[ni],1);if(nb>0||nfb>0)atomicAdd(&hardware_native_stray[ni],1);else atomicAdd(&hardware_native_direct[ni],1);if(nfb>0)atomicAdd(&hardware_native_face_stray[ni],1);
        float phi=detector_incidence_angle_deg(vx,vy,vz,ni,0);
        if(phi>=0.0f && isfinite(phi)){
            float resp=(photodiode_response_model==0)?1.0f:sfh213_relative_response_deg(phi);
            atomicAdd(&photodiode_native[ni],resp);atomicAdd(&incidence_native_sum[ni],phi);
        }
    }
    if(mi>=0){
        atomicAdd(&hardware_mirror[mi],1);if(mb>0||mfb>0)atomicAdd(&hardware_mirror_stray[mi],1);else atomicAdd(&hardware_mirror_direct[mi],1);if(mfb>0)atomicAdd(&hardware_mirror_face_stray[mi],1);
        float phi=detector_incidence_angle_deg(vx,vy,vz,mi,1);
        if(phi>=0.0f && isfinite(phi)){
            float resp=(photodiode_response_model==0)?1.0f:sfh213_relative_response_deg(phi);
            atomicAdd(&photodiode_mirror[mi],resp);atomicAdd(&incidence_mirror_sum[mi],phi);
        }
    }
    for(int mirror=0;mirror<2;++mirror){
        float xx=mirror?-x:x;float ang=atan2f(xx,y)*(180.0f/PI_F);if(ang<0.0f)ang+=360.0f;
        if(ang<0.0f||ang>180.0f)continue;
        for(int j=0;j<18;++j){float deg=10.0f*j;if(interaction_count[tid]<=0&&deg>=90.0f)continue;if(fabsf(ang-deg)<=legacy_accept_deg){if(mirror)atomicAdd(&legacy_mirror[j],1);else atomicAdd(&legacy_native[j],1);}}
    }
}

__global__ void free_surface_geometry_test_kernel(
    const float* rays6,const int N,const float ZMEAN,const float R,const int MODEL,
    const float DELTA_H,const float CORE_R,
    float* surface_z_start,float* t_hit,float* hit_surface_z,float* normal_x,float* normal_y,float* normal_z){
    int tid=blockDim.x*blockIdx.x+threadIdx.x;if(tid>=N)return;
    const float* q=rays6+6*tid;
    float x=q[0],y=q[1],z=q[2],vx=q[3],vy=q[4],vz=q[5];
    surface_z_start[tid]=free_surface_z(x,y,ZMEAN,R,MODEL,DELTA_H,CORE_R);
    float t=free_surface_hit(x,y,z,vx,vy,vz,ZMEAN,R,MODEL,DELTA_H,CORE_R);
    t_hit[tid]=t;
    if(t<INF_F*0.5f){
        float hx=x+t*vx,hy=y+t*vy;
        hit_surface_z[tid]=free_surface_z(hx,hy,ZMEAN,R,MODEL,DELTA_H,CORE_R);
        free_surface_normal_into_water(hx,hy,R,MODEL,DELTA_H,CORE_R,&normal_x[tid],&normal_y[tid],&normal_z[tid]);
    }else{
        hit_surface_z[tid]=INF_F;normal_x[tid]=0.0f;normal_y[tid]=0.0f;normal_z[tid]=0.0f;
    }
}


// ================= Qsca>=1 MIE-TWO-MOMENT ANGULAR CONSTRAINT =================
__device__ void apply_mie_moment_constraint(
    const float vinx,const float viny,const float vinz,
    float* vx,float* vy,float* vz,const float A,const float B){
    float c=fminf(fmaxf(dot3(vinx,viny,vinz,*vx,*vy,*vz),-1.0f),1.0f);
    float d=fminf(fmaxf(1.0f-c,0.0f),2.0f);
    float dm=fminf(fmaxf(A*d+B*d*d,0.0f),2.0f);
    float cp=1.0f-dm;
    float px=*vx-c*vinx,py=*vy-c*viny,pz=*vz-c*vinz;
    float pn=sqrtf(fmaxf(px*px+py*py+pz*pz,0.0f));
    if(pn>1.0e-8f){px/=pn;py/=pn;pz/=pn;}
    else{
        float e1x,e1y,e1z,e2x,e2y,e2z;
        perpendicular_basis(vinx,viny,vinz,&e1x,&e1y,&e1z,&e2x,&e2y,&e2z);
        px=e1x;py=e1y;pz=e1z;
    }
    float sp=sqrtf(fmaxf(1.0f-cp*cp,0.0f));
    *vx=cp*vinx+sp*px;*vy=cp*viny+sp*py;*vz=cp*vinz+sp*pz;normalize3(vx,vy,vz);
}

__global__ void trace_kernel(
    const float MAX_ITERATIONS,const double* MU_CHANNELS,
    const float N_WATER,const float N_ACRYLIC,const float N_AIR,const float PARTICLE_WAVELENGTH,const float PARTICLE_SURFACE_ROUGH_ALPHA,const float WAVE_SPHEROID_ASPECT_RATIO,
    const int WAVE_SPHEROID_ORIENTATION_MODEL,const float WAVE_SPHEROID_ORIENTATION_KAPPA,
    const int MAX_PARTICLE_BOUNCES,const int MAX_CELL_BOUNCES,
    const float RIN,const float ROUT,const float ZMIN,const float ZMEAN,const float ZWALL_TOP,const int FREE_SURFACE_MODEL,const float VORTEX_DELTA_H,const float VORTEX_CORE_RADIUS,const float BOTTOM_DISC_THICKNESS,const float BOTTOM_EXTERNAL_N,const int MAX_AXIAL_BOUNCES,
    const float RING_IN_R,const float THROAT_OUT_R,const float RING_OUT_R,const float SOURCE_R,
    const float THROAT_R,const float COUNTERBORE_R,const float BEAM_SIGMA,
    const int VIS_SIZE,const float SOURCE_ALPHA1,const float SOURCE_ALPHA2,const int SOURCE_ANGULAR_MODEL,
    const int SOURCE_BORE_SURFACE_MODEL,const float BORE_N,const int MAX_BORE_BOUNCES,
    const int N_rays,
    const double* geometric_cdf,const double* wave_cdf,const double* mie_phase_cdf,const double* mie_theta,const int mie_ntheta,
    const double* particle_n,const double* particle_k,const double* particle_radius,const int n_particles,
    const double* MOMENT_A,const double* MOMENT_B,const unsigned char* MOMENT_MODE,
    const int SEDIMENT_TRANSPORT_MODEL,const double* TRACE_GEOM_BIN,const double* TRACE_WAVE_BIN,
    const double* SED_A,const double* SED_B,const double* SED_NORM_SCALED,const double* SED_RADIAL_LOG,const int SED_N_RADIAL,const double* SED_MAJORANTS,
    const double* SED_GRID,const int SED_GRID_NR,const int SED_GRID_NZ,const float SED_GRID_ZMAX,
    float* heat_xy,float* heat_zy,
    const int LIGHT_DIAG_ENABLED,const unsigned int LIGHT_DIAG_MAX_RAYS,
    const int LIGHT_DIAG_NX,const int LIGHT_DIAG_NY,const int LIGHT_DIAG_NZ,const float LIGHT_DIAG_ZMAX,
    float* light_total_grid,float* light_post_grid,float* light_event_weight,unsigned int* light_event_count,
    double* light_segment_total,double* light_deposited_total,unsigned long long* light_segment_count,
    unsigned long long* light_event_total,unsigned int* light_error_count,unsigned int* light_boundary_adjustment_count,unsigned int* light_event_outside_count,
    const int LIGHT_DIAG_REP_RAYS,const int LIGHT_DIAG_MAX_SEGMENTS,
    float* light_tx0,float* light_ty0,float* light_tz0,float* light_tx1,float* light_ty1,float* light_tz1,
    int* light_torder,int* light_tevent,int* light_tcount,int* light_toverflow,
    float* air_x,float* air_y,float* air_z,float* air_vx,float* air_vy,float* air_vz,
    float* water_path,float* acrylic_path,
    float* first_particle_deflection_deg,float* last_particle_deflection_deg,float* internal_particle_path_total,float* internal_particle_opl_total,float* water_path_after_last_particle_event,
    float* cumulative_particle_deflection_deg,float* cumulative_particle_deflection_sq_deg2,float* max_particle_deflection_deg,
    float* max_deflection_event_x,float* max_deflection_event_y,float* max_deflection_event_z,
    float* first_particle_x,float* first_particle_y,float* first_particle_z,float* last_particle_x,float* last_particle_y,float* last_particle_z,
    float* particle_event_mean_r_m,float* particle_event_mean_z_m,float* particle_event_min_r_m,float* particle_event_max_r_m,float* particle_event_min_z_m,float* particle_event_max_z_m,
    float* water_path_at_first_particle_event_m,float* water_path_at_last_particle_event_m,
    float* failure_x,float* failure_y,float* failure_z,float* failure_vx,float* failure_vy,float* failure_vz,
    float* particle_diag_radius,float* particle_diag_rho,float* particle_diag_phi,float* particle_diag_cx,float* particle_diag_cy,float* particle_diag_cz,
    float* particle_diag_entry_x,float* particle_diag_entry_y,float* particle_diag_entry_z,float* particle_diag_exit_x,float* particle_diag_exit_y,float* particle_diag_exit_z,
    float* particle_diag_pre_x,float* particle_diag_pre_y,float* particle_diag_pre_z,float* particle_diag_pre_vx,float* particle_diag_pre_vy,float* particle_diag_pre_vz,
    float* particle_diag_post_x,float* particle_diag_post_y,float* particle_diag_post_z,float* particle_diag_post_vx,float* particle_diag_post_vy,float* particle_diag_post_vz,
    float* particle_diag_center_radial_clearance,float* particle_diag_finite_wall_clearance,float* particle_diag_bottom_clearance,float* particle_diag_top_clearance,
    float* max_deflection_event_internal_path_m,float* max_deflection_event_internal_path_over_diameter,
    float* max_deflection_event_incidence_deg,float* max_deflection_event_exit_deg,
    int* interaction_count,int* geometric_interactions,int* wave_interactions,int* coherent_wave_interactions,int* morphology_wave_interactions,
    int* particle_reflections,int* particle_entry_reflections,int* particle_internal_reflections,int* particle_absorptions,
    int* cell_inner_reflections,int* cell_outer_reflections,int* cell_tir,
    int* top_reflections,int* top_tir,int* bottom_inner_reflections,int* bottom_outer_reflections,int* bottom_tir,
    int* source_bore_interactions,int* source_bore_reflections,int* entered_water,int* status,
    int* production_failure_code,int* production_failure_subcode,int* production_failure_trace_iteration,int* production_failure_macro_bounces,int* production_failure_micro_order,int* production_failure_medium,int* production_failure_event_channel,
    int* particle_diag_bin,int* particle_diag_previous_event,int* particle_diag_overlap_class,int* particle_diag_proposal_accepted,int* particle_proposal_count,int* particle_accessible_count,int* particle_null_count,
    int* first_particle_event_type,int* last_particle_event_type,int* last_particle_event_order,int* first_particle_backscatter,int* first_particle_source_hemisphere,
    int* first_particle_bin,int* last_particle_bin,int* first_geometric_bin,int* last_geometric_bin,int* first_wave_bin,int* last_wave_bin,
    int* acrylic_return_after_particle_count,int* particle_after_acrylic_return_count,
    int* particle_branch_transition_count,int* particle_event_type_transition_count,
    int* particle_small_deflection_count,int* particle_medium_deflection_count,int* particle_large_deflection_count,
    int* max_deflection_event_type,int* max_deflection_event_order,int* max_deflection_event_bin,
    int* max_deflection_event_macro_bounces,int* max_deflection_event_terminal_micro_order,
    unsigned int seed0,unsigned int seed1,unsigned int seed2,const unsigned int ray_offset) {
    int tid=blockDim.x*blockIdx.x+threadIdx.x; if(tid>=N_rays)return;
    unsigned int gid=ray_offset+(unsigned int)tid;
    int light_diag_active=(LIGHT_DIAG_ENABLED!=0 && gid<LIGHT_DIAG_MAX_RAYS);
    unsigned int state=seed0+gid*74729u+13u, stateOPT=seed1+gid*104729u+29u, stateSRC=seed2+gid*130363u+43u;
    float MU_GEOM=(float)MU_CHANNELS[0],MU_WAVE=(float)MU_CHANNELS[1];
    float MU_MAJOR_GEOM=(SEDIMENT_TRANSPORT_MODEL==0)?MU_GEOM:(float)SED_MAJORANTS[0];
    float MU_MAJOR_WAVE=(SEDIMENT_TRANSPORT_MODEL==0)?MU_WAVE:(float)SED_MAJORANTS[1];
    float WAVE_COHERENT_FRACTION=fminf(fmaxf((float)MU_CHANNELS[7],0.0f),1.0f);
    int ic=0,gic=0,wic=0,wcoh=0,wmorph=0,pr=0,per=0,pir=0,pabs=0,cir=0,cor=0,ctir=0,tr=0,ttir=0,bir=0,bor=0,btir=0,sbi=0,sbr=0,entered=0,st=0;
    int pf=PROD_OK,pfs=PROD_SUB_NONE,pfstep=-1,pfmacro=-1,pfmicro=-1,pfmedium=0,pfevent=0;
    int pd_bin=-1,pd_prev_evt=0,pd_overlap=0,pd_accept=-1,pd_proposals=0,pd_accessible=0,pd_null=0;
    float pd_radius=0.0f,pd_rho=0.0f,pd_phi=0.0f,pd_cx=0.0f,pd_cy=0.0f,pd_cz=0.0f;
    float pd_ex=0.0f,pd_ey=0.0f,pd_ez=0.0f,pd_xx=0.0f,pd_xy=0.0f,pd_xz=0.0f;
    float pd_prex=0.0f,pd_prey=0.0f,pd_prez=0.0f,pd_prevx=0.0f,pd_prevy=0.0f,pd_prevz=0.0f;
    float pd_postx=0.0f,pd_posty=0.0f,pd_postz=0.0f,pd_postvx=0.0f,pd_postvy=0.0f,pd_postvz=0.0f;
    float pd_center_clear=0.0f,pd_wall_clear=0.0f,pd_bottom_clear=0.0f,pd_top_clear=0.0f;
    // Diagnostic state. Event type codes are host-documented and do not alter transport:
    // 1 geom external reflection, 2 geom transmit/refract, 3 geom internal-reflection path,
    // 4 coherent Mie wave, 5 morphology-wave external reflection,
    // 6 morphology-wave transmit/refract, 7 morphology-wave internal-reflection path.
    int first_evt=0,last_evt=0,last_evt_order=0,first_back=0,first_source_hemi=0;
    // diagnostic-only causal provenance. These counters consume no RNG and do not influence transport.
    int first_pbin=-1,last_pbin=-1,first_gbin=-1,last_gbin=-1,first_wbin=-1,last_wbin=-1;
    int acrylic_returns_after_particle=0,particle_events_after_acrylic_return=0;
    float first_defl=-1.0f,last_defl=-1.0f,internal_path_total=0.0f,internal_opl_total=0.0f,wp_at_last_evt=-1.0f;
    // diagnostic-only angular-diffusion, branch-sequence and collision-location state.
    // Maximum-deflection provenance is observational. No random numbers are consumed and none of these variables feeds a transport decision.
    float cumulative_defl=0.0f,cumulative_defl_sq=0.0f,max_defl=-1.0f;
    // diagnostic-only provenance for the largest single particle turn in each ray history.
    // Strict > retains the first event on an exact floating-point tie. No RNG or transport state is modified.
    float max_evt_x=0.0f,max_evt_y=0.0f,max_evt_z=0.0f;
    float max_evt_internal_path=0.0f,max_evt_internal_path_over_diameter=0.0f;
    float max_evt_incidence_deg=-1.0f,max_evt_exit_deg=-1.0f;
    int max_evt_type=0,max_evt_order=0,max_evt_bin=-1,max_evt_macro_bounces=0,max_evt_terminal_micro_order=0;
    float first_evt_x=0.0f,first_evt_y=0.0f,first_evt_z=0.0f,last_evt_x=0.0f,last_evt_y=0.0f,last_evt_z=0.0f;
    float event_sum_r=0.0f,event_sum_z=0.0f,event_min_r=INF_F,event_max_r=-INF_F,event_min_z=INF_F,event_max_z=-INF_F;
    float wp_first_evt=-1.0f,wp_last_evt=-1.0f;
    int branch_transitions=0,event_type_transitions=0,small_deflections=0,medium_deflections=0,large_deflections=0;
    int previous_branch=0,previous_event_type=0;
    float wp=0.0f,ap=0.0f;

    // The LED launches at 180 degrees from a narrow Gaussian emitting spot.
    // The ring collimator has a counterbore followed by a through-bore; its
    // dimensions are passed from the configured engineering geometry.
    float u1=fmaxf(rnd_uniform(&stateSRC),1e-12f),u2=rnd_uniform(&stateSRC);
    float mag=sqrtf(-2.0f*logf(u1));
    float x=BEAM_SIGMA*mag*cosf(2.0f*PI_F*u2);
    float z=BEAM_SIGMA*mag*sinf(2.0f*PI_F*u2);
    float y=-SOURCE_R;
    float polar=0.0f,az=0.0f,vx=0.0f,vy=1.0f,vz=0.0f;
    if(SOURCE_ANGULAR_MODEL==0){
        // Comparison source: the Beta law is the polar-angle probability density.
        polar=beta_sample(&stateSRC,SOURCE_ALPHA1,SOURCE_ALPHA2)*(PI_F/2.0f);
        az=2.0f*PI_F*rnd_uniform(&stateSRC);
        float sp=sinf(polar);vx=sp*cosf(az);vy=cosf(polar);vz=sp*sinf(az);
    } else if(SOURCE_ANGULAR_MODEL==1){
        // Production source: the Beta law specifies angular radiance;
        // include the 3-D solid-angle Jacobian before uniform azimuth sampling.
        polar=beta_radiance_polar_sample(&stateSRC,SOURCE_ALPHA1,SOURCE_ALPHA2);
        az=2.0f*PI_F*rnd_uniform(&stateSRC);
        float sp=sinf(polar);vx=sp*cosf(az);vy=cosf(polar);vz=sp*sinf(az);
    } else {
        // Diagnostic reproduction of the thesis CLARITAS planar source:
        // signed horizontal Beta-distributed angle and zero vertical angle.
        float h=beta_sample(&stateSRC,SOURCE_ALPHA1,SOURCE_ALPHA2)*(PI_F/2.0f);
        if(rnd_uniform(&stateSRC)<0.5f) h=-h;
        vx=sinf(h);vy=cosf(h);vz=0.0f;
    }
    normalize3(&vx,&vy,&vz);
    if(vy<=1e-12f){st=3;goto WRITE_OUT;}

    // Absorbing bores use hard-aperture acceptance. The reflective branch
    // propagates through the piecewise cylindrical collimator with Fresnel walls.
    if(SOURCE_BORE_SURFACE_MODEL==0){
        if(x*x+z*z>COUNTERBORE_R*COUNTERBORE_R){st=3;goto WRITE_OUT;}
        float t=(-THROAT_OUT_R-y)/vy;
        if(t<-EPS_F){st=3;goto WRITE_OUT;}
        if(t<0.0f)t=0.0f;
        float xt=x+t*vx,zt=z+t*vz;
        if(xt*xt+zt*zt>THROAT_R*THROAT_R){st=3;goto WRITE_OUT;}
        t=(-RING_IN_R-y)/vy;
        if(t<=0.0f){st=3;goto WRITE_OUT;}
        x+=t*vx;y+=t*vy;z+=t*vz;
        if(x*x+z*z>THROAT_R*THROAT_R){st=3;goto WRITE_OUT;}
    }else{
        if(!source_collimator_transport(&stateSRC,SOURCE_BORE_SURFACE_MODEL,N_AIR,BORE_N,MAX_BORE_BOUNCES,
                RING_IN_R,THROAT_OUT_R,SOURCE_R,THROAT_R,COUNTERBORE_R,
                &x,&y,&z,&vx,&vy,&vz,&sbi,&sbr)){st=3;goto WRITE_OUT;}
    }

    // Reach outer acrylic cylinder from the air gap and apply air -> acrylic Fresnel/Snell.
    {
        double tdwall=cell_cylinder_hit_finite_forward_d(x,y,z,vx,vy,vz,ROUT,ZMIN-BOTTOM_DISC_THICKNESS,ZWALL_TOP);
        float t=(tdwall<0.5*(double)INF_F)?(float)tdwall:INF_F;
        if(t>=INF_F*0.5f){st=3;goto WRITE_OUT;}
        x+=t*vx;y+=t*vy;z+=t*vz;
        float nx,ny;radial_normal(x,y,&nx,&ny);
        float ci=fminf(fmaxf(-dot3(vx,vy,vz,nx,ny,0.0f),0.0f),1.0f);
        int tir=0;float R=fresnel_R(ci,N_AIR,N_ACRYLIC,&tir);
        float a,b,c;
        if(tir || rnd_uniform(&state)<R || !refract3(vx,vy,vz,nx,ny,0.0f,N_AIR,N_ACRYLIC,&a,&b,&c)){
            reflect3(vx,vy,vz,nx,ny,0.0f,&a,&b,&c);vx=a;vy=b;vz=c;cor++;if(tir)ctir++;
            x+=EPS_F*nx;y+=EPS_F*ny;st=1;goto AIR_EXIT;
        }
        vx=a;vy=b;vz=c;x-=EPS_F*nx;y-=EPS_F*ny;
        int ahsub=PROD_SUB_NONE,ahb=0;int where=acrylic_annulus_ex(&state,N_WATER,N_ACRYLIC,N_AIR,BOTTOM_EXTERNAL_N,RIN,ROUT,ZMIN,ZMIN-BOTTOM_DISC_THICKNESS,ZMEAN,ZWALL_TOP,FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS,MAX_CELL_BOUNCES,
            &x,&y,&z,&vx,&vy,&vz,&ap,&cir,&cor,&bir,&bor,&ctir,&btir,&ahsub,&ahb);
        if(where==2){st=1;goto AIR_EXIT;}
        if(where==3){st=7;goto WRITE_OUT;}
        if(where==5){st=8;goto WRITE_OUT;}
        if(where==4){
            int hsub=PROD_SUB_NONE,hb=0;
            int hw=headspace_transport_ex(&state,N_WATER,N_ACRYLIC,N_AIR,BOTTOM_EXTERNAL_N,RIN,ROUT,ZMIN,ZMIN-BOTTOM_DISC_THICKNESS,ZMEAN,ZWALL_TOP,
                FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS,MAX_CELL_BOUNCES,&x,&y,&z,&vx,&vy,&vz,&ap,
                &cir,&cor,&bir,&bor,&ctir,&btir,&tr,&ttir,&hsub,&hb);
            if(hw==1){entered=1;}
            else if(hw==2){st=1;goto AIR_EXIT;}
            else if(hw==3){st=7;goto WRITE_OUT;}
            else if(hw==4){st=6;goto WRITE_OUT;}
            else if(hw==5){st=8;goto WRITE_OUT;}
            else{pf=PROD_HEADSPACE_BOUNDARY_NO_INTERSECTION;pfs=hsub;pfmacro=hb;pfmedium=5;st=4;goto WRITE_OUT;}
        } else if(where!=1){pf=(ahsub==PROD_SUB_ACRYLIC_BOUNCE_LIMIT)?PROD_ACRYLIC_ANNULUS_BOUNCE_LIMIT:PROD_ACRYLIC_ANNULUS_NO_INTERSECTION;pfs=ahsub;pfmacro=ahb;pfmedium=2;st=4;goto WRITE_OUT;}
        entered=1;
    }

    for(int step=0;step<(int)MAX_ITERATIONS;++step){
        if(!isfinite(x)||!isfinite(y)||!isfinite(z)){pf=PROD_NONFINITE_POSITION;pfs=PROD_SUB_NONFINITE_POSITION;pfstep=step;pfmedium=1;st=4;break;}
        if(!isfinite(vx)||!isfinite(vy)||!isfinite(vz)){pf=PROD_NONFINITE_DIRECTION;pfs=PROD_SUB_NONFINITE_DIRECTION;pfstep=step;pfmedium=1;st=4;break;}
        float zsurf_here=free_surface_z(x,y,ZMEAN,RIN,FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS);
        if(z<=ZMIN || z>=zsurf_here){st=2;break;}
        // Defensive topology check.  Correct interface handling keeps water rays inside
        // Rin; a materially outside state is reported, never silently snapped inward.
        double rr2w=(double)x*(double)x+(double)y*(double)y,ri2w=(double)RIN*(double)RIN;
        double water_tol=8.0*1.1920928955078125e-7*ri2w;
        if(rr2w>ri2w+water_tol){capture_water_advance_failure(light_event_outside_count,gid,step,x,y,z,vx,vy,vz,0.0f,INF_F,INF_F,INF_F,INF_F,0,x,y,z);pf=PROD_INVALID_MEDIUM_STATE;pfs=PROD_SUB_WATER_OUTSIDE_INNER_RADIUS;pfstep=step;pfmedium=1;st=4;break;}
        double tsided=bore_wall_hit_finite_d(x,y,z,vx,vy,vz,RIN,ZMIN,ZWALL_TOP);
        if(tsided<0.0){capture_water_advance_failure(light_event_outside_count,gid,step,x,y,z,vx,vy,vz,0.0f,(float)tsided,INF_F,INF_F,INF_F,0,x,y,z);pf=PROD_INVALID_MEDIUM_STATE;pfs=PROD_SUB_WATER_OUTSIDE_INNER_RADIUS;pfstep=step;pfmedium=1;st=4;break;}
        float tside=(tsided<0.5*(double)INF_F)?(float)tsided:INF_F;
        float ttop=INF_F,tbottom=INF_F;
        ttop=free_surface_hit(x,y,z,vx,vy,vz,ZMEAN,RIN,FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS);
        if(vz<-1e-14f) tbottom=(ZMIN-z)/vz;
        if(!(ttop>0.0f)) ttop=INF_F;
        if(!(tbottom>0.0f)) tbottom=INF_F;
        int axial_top=(ttop<=tbottom && ttop<tside);
        int axial_bottom=(tbottom<ttop && tbottom<tside);
        int axial=axial_top||axial_bottom;
        float tbound=axial_top?ttop:(axial_bottom?tbottom:tside);
        if(tbound>=INF_F*0.5f){pf=PROD_WATER_BOUNDARY_NO_INTERSECTION;pfs=PROD_SUB_WATER_NO_BOUNDARY;pfstep=step;pfmedium=1;st=4;break;}
        float freep=INF_F;
        float MU_TOTAL=MU_GEOM+MU_WAVE;
        float MU_TRACK=(SEDIMENT_TRANSPORT_MODEL==0)?MU_TOTAL:(MU_MAJOR_GEOM+MU_MAJOR_WAVE);
        if(tbound>0.0f && MU_TRACK>0.0f){float u=fmaxf(rnd_uniform(&state),1e-12f);freep=-logf(u)/MU_TRACK;}
        int do_particle=freep<tbound;
        float travel=do_particle?freep:tbound;
        float x0=x,y0=y,z0=z;
        float zvismax=free_surface_z(RIN,0.0f,ZMEAN,RIN,FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS);
        raster_water_segment(x0,y0,z0,vx,vy,vz,travel,VIS_SIZE,RIN,ZMIN,zvismax,heat_xy,heat_zy);
        light_diag_record_segment(gid,light_diag_active,LIGHT_DIAG_NX,LIGHT_DIAG_NY,LIGHT_DIAG_NZ,
            RIN,ZMIN,LIGHT_DIAG_ZMAX,x0,y0,z0,vx,vy,vz,travel,ic,
            light_total_grid,light_post_grid,light_segment_total,light_deposited_total,
            light_segment_count,light_error_count,light_boundary_adjustment_count,LIGHT_DIAG_REP_RAYS,LIGHT_DIAG_MAX_SEGMENTS,
            light_tx0,light_ty0,light_tz0,light_tx1,light_ty1,light_tz1,
            light_torder,light_tevent,light_tcount,light_toverflow);
        x+=travel*vx;y+=travel*vy;z+=travel*vz;wp+=travel;
        double endpoint_r2=(double)x*x+(double)y*y;
        if(do_particle && (endpoint_r2>ri2w+water_tol || z<ZMIN-EPS_F ||
            z>free_surface_z(x,y,ZMEAN,RIN,FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS)+EPS_F)){
            capture_water_advance_failure(light_event_outside_count,gid,step,x0,y0,z0,vx,vy,vz,travel,tside,ttop,tbottom,tbound,1,x,y,z);
            pf=PROD_INVALID_MEDIUM_STATE;pfs=PROD_SUB_WATER_OUTSIDE_INNER_RADIUS;pfstep=step;pfmedium=1;pfevent=3;st=4;break;
        }
        if(do_particle){
            float LOCAL_GEOM=MU_GEOM,LOCAL_WAVE=MU_WAVE,LOCAL_TOTAL=MU_TOTAL;
            if(SEDIMENT_TRANSPORT_MODEL!=0){
                double lg=0.0,lw=0.0;
                for(int i=0;i<n_particles;++i){
                    double sc=sediment_local_scale(SEDIMENT_TRANSPORT_MODEL,FREE_SURFACE_MODEL,i,SED_A,SED_B,SED_NORM_SCALED,SED_RADIAL_LOG,SED_N_RADIAL,SED_GRID,SED_GRID_NR,SED_GRID_NZ,SED_GRID_ZMAX,x,y,z,RIN,ZMIN,VORTEX_CORE_RADIUS);
                    lg+=TRACE_GEOM_BIN[i]*sc;lw+=TRACE_WAVE_BIN[i]*sc;
                }
                LOCAL_GEOM=(float)lg;LOCAL_WAVE=(float)lw;LOCAL_TOTAL=LOCAL_GEOM+LOCAL_WAVE;
                if(!(LOCAL_TOTAL>0.0f))continue;
                // the Woodcock majorant is a mathematical validity condition, not a
                // quantity to silently clamp.  Permit only float32 accumulation/rounding slack:
                // 64 ulps at the scale of max(|MU_TRACK|,1), then fail closed.
                float majorant_slack=64.0f*1.1920928955078125e-7f*fmaxf(fabsf(MU_TRACK),1.0f);
                if(LOCAL_TOTAL>MU_TRACK+majorant_slack){
                    pf=PROD_SPATIAL_MAJORANT_VIOLATION;pfs=PROD_SUB_SEDIMENT_MAJORANT_VIOLATION;pfstep=step;pfmedium=1;st=4;break;
                }
                float pacc=LOCAL_TOTAL/fmaxf(MU_TRACK,1e-30f);
                if(pacc>1.0f)pacc=1.0f; // only the explicitly bounded round-off interval reaches here.
                if(rnd_uniform(&state)>=pacc)continue; // unbiased Woodcock null collision.
            }
            ic++;
            const float vinx=vx,viny=vy,vinz=vz;
            const float event_site_x=x,event_site_y=y,event_site_z=z;
            const float event_site_r=sqrtf(fmaxf(event_site_x*event_site_x+event_site_y*event_site_y,0.0f));
            const int per_before=per,pir_before=pir;
            int event_type=0,event_pidx=-1;
            float event_internal_path=0.0f;
            float event_incidence_deg=-1.0f,event_exit_deg=-1.0f;
            // zero-RNG drill-down of the finite-particle path that produces the largest turn.
            int event_macro_bounces=0,event_terminal_micro_order=0;
            int use_wave=(LOCAL_WAVE>0.0f && rnd_uniform(&state)*LOCAL_TOTAL>=LOCAL_GEOM);
            if(use_wave){
                wic++;
                int pidx=(SEDIMENT_TRANSPORT_MODEL==0)?sample_cdf_index(&state,wave_cdf,n_particles):sample_local_mu_bin(&state,SEDIMENT_TRANSPORT_MODEL,FREE_SURFACE_MODEL,TRACE_WAVE_BIN,n_particles,SED_A,SED_B,SED_NORM_SCALED,SED_RADIAL_LOG,SED_N_RADIAL,SED_GRID,SED_GRID_NR,SED_GRID_NZ,SED_GRID_ZMAX,x,y,z,RIN,ZMIN,VORTEX_CORE_RADIUS,(double)LOCAL_WAVE);
                event_pidx=pidx;
                // sigma_h=0 gives C=1 without an extra RNG draw in the coherent branch.
                int coherent=(WAVE_COHERENT_FRACTION>=1.0f) || (rnd_uniform(&stateOPT)<WAVE_COHERENT_FRACTION);
                if(coherent){
                    wcoh++;
                    mie_scatter_direction(&stateOPT,mie_phase_cdf,mie_theta,mie_ntheta,pidx,&vx,&vy,&vz);
                    event_type=4;
                } else {
                    wmorph++;
                    float fr=sqrtf(x*x+y*y);
                    float ffx=1.0f,ffy=0.0f,ffz=0.0f;
                    if(fr>1e-12f){ffx=-y/fr;ffy=x/fr;}
                    int ppc=PROD_OK,ppsub=PROD_SUB_NONE,ppmb=0,ppmo=0;
                    int ps=spheroid_fresnel_scatter_direction_only_3d_ex(&stateOPT,N_WATER,(float)particle_n[pidx],(float)particle_k[pidx],PARTICLE_WAVELENGTH,PARTICLE_SURFACE_ROUGH_ALPHA,
                        (float)particle_radius[pidx],WAVE_SPHEROID_ASPECT_RATIO,WAVE_SPHEROID_ORIENTATION_MODEL,WAVE_SPHEROID_ORIENTATION_KAPPA,ffx,ffy,ffz,
                        MAX_PARTICLE_BOUNCES,&vx,&vy,&vz,&event_internal_path,&pr,&per,&pir,&pabs,&ppc,&ppsub,&ppmb,&ppmo,&event_incidence_deg,&event_exit_deg);
                    if(ps==2){light_diag_mark_last_event(gid,light_diag_active,LIGHT_DIAG_REP_RAYS,LIGHT_DIAG_MAX_SEGMENTS,light_tevent,light_tcount,9);st=5;break;}
                    if(ps!=1){light_diag_mark_last_event(gid,light_diag_active,LIGHT_DIAG_REP_RAYS,LIGHT_DIAG_MAX_SEGMENTS,light_tevent,light_tcount,10);pf=ppc?ppc:PROD_OTHER_EXPLICIT_FAILURE;pfs=ppsub;pfstep=step;pfmacro=ppmb;pfmicro=ppmo;pfmedium=4;pfevent=2;st=4;break;}
                    event_macro_bounces=ppmb;event_terminal_micro_order=ppmo;
                    event_type=(per>per_before)?5:((pir>pir_before)?7:6);
                }
            } else {
                gic++;
                int pidx=(SEDIMENT_TRANSPORT_MODEL==0)?sample_cdf_index(&state,geometric_cdf,n_particles):sample_local_mu_bin(&state,SEDIMENT_TRANSPORT_MODEL,FREE_SURFACE_MODEL,TRACE_GEOM_BIN,n_particles,SED_A,SED_B,SED_NORM_SCALED,SED_RADIAL_LOG,SED_N_RADIAL,SED_GRID,SED_GRID_NR,SED_GRID_NZ,SED_GRID_ZMAX,x,y,z,RIN,ZMIN,VORTEX_CORE_RADIUS,(double)LOCAL_GEOM);
                event_pidx=pidx;
                int ppc=PROD_OK,ppsub=PROD_SUB_NONE,ppmb=0,ppmo=0;
                // sample the finite sphere placement ONCE, validate the ACTUAL sampled centre,
                // then execute the interaction with exactly that geometry.  Inaccessible proposals are
                // null collisions in the bulk proposal process.
                float proposed_radius=(float)particle_radius[pidx];
                float srho=0.0f,sphi=0.0f,sgnx=0.0f,sgny=0.0f,sgnz=0.0f,scx=x,scy=y,scz=z;
                const float prepx=x,prepy=y,prepz=z,prepvx=vx,prevpy=vy,prevpz=vz;
                sphere_sample_geometry(&stateOPT,proposed_radius,x,y,z,vx,vy,vz,&srho,&sphi,&sgnx,&sgny,&sgnz,&scx,&scy,&scz);
                // Preserve the exact sampled particle centre in double precision for the
                // particle-scale accessibility and internal transport calculations.
                double dscx=(double)x-(double)proposed_radius*(double)sgnx;
                double dscy=(double)y-(double)proposed_radius*(double)sgny;
                double dscz=(double)z-(double)proposed_radius*(double)sgnz;
                scx=(float)dscx;scy=(float)dscy;scz=(float)dscz;
                float center_clear=0.0f,wall_clear=0.0f,bottom_clear=0.0f,top_clear=0.0f;
                int accessible=sphere_particle_accessible_in_water(dscx,dscy,dscz,(double)proposed_radius,RIN,ZMIN,ZMEAN,FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS,
                    &center_clear,&wall_clear,&bottom_clear,&top_clear);
                pd_bin=pidx;pd_prev_evt=last_evt;pd_overlap=accessible?0:1;pd_accept=accessible?1:0;pd_proposals++;
                pd_radius=proposed_radius;pd_rho=srho;pd_phi=sphi;pd_cx=scx;pd_cy=scy;pd_cz=scz;
                pd_ex=x;pd_ey=y;pd_ez=z;pd_prex=prepx;pd_prey=prepy;pd_prez=prepz;pd_prevx=prepvx;pd_prevy=prevpy;pd_prevz=prevpz;
                pd_center_clear=center_clear;pd_wall_clear=wall_clear;pd_bottom_clear=bottom_clear;pd_top_clear=top_clear;
                if(!accessible){pd_null++;pd_xx=x;pd_xy=y;pd_xz=z;pd_postx=x;pd_posty=y;pd_postz=z;pd_postvx=vx;pd_postvy=vy;pd_postvz=vz;ic--;gic--;light_diag_mark_last_event(gid,light_diag_active,LIGHT_DIAG_REP_RAYS,LIGHT_DIAG_MAX_SEGMENTS,light_tevent,light_tcount,8);continue;}
                pd_accessible++;
                int ps=sphere_fresnel_interaction_3d_sampled_ex(&stateOPT,N_WATER,(float)particle_n[pidx],(float)particle_k[pidx],PARTICLE_WAVELENGTH,PARTICLE_SURFACE_ROUGH_ALPHA,proposed_radius,
                    MAX_PARTICLE_BOUNCES,sgnx,sgny,sgnz,dscx,dscy,dscz,&x,&y,&z,&vx,&vy,&vz,&event_internal_path,&pr,&per,&pir,&pabs,&ppc,&ppsub,&ppmb,&ppmo,&event_incidence_deg,&event_exit_deg);
                pd_xx=x;pd_xy=y;pd_xz=z;pd_postx=x;pd_posty=y;pd_postz=z;pd_postvx=vx;pd_postvy=vy;pd_postvz=vz;
                if(ps==2){light_diag_mark_last_event(gid,light_diag_active,LIGHT_DIAG_REP_RAYS,LIGHT_DIAG_MAX_SEGMENTS,light_tevent,light_tcount,9);st=5;break;}
                if(ps!=1){light_diag_mark_last_event(gid,light_diag_active,LIGHT_DIAG_REP_RAYS,LIGHT_DIAG_MAX_SEGMENTS,light_tevent,light_tcount,10);pf=ppc?ppc:PROD_OTHER_EXPLICIT_FAILURE;pfs=ppsub;pfstep=step;pfmacro=ppmb;pfmicro=ppmo;pfmedium=4;pfevent=1;st=4;break;}
                event_macro_bounces=ppmb;event_terminal_micro_order=ppmo;
                event_type=(per>per_before)?1:((pir>pir_before)?3:2);pd_prev_evt=event_type;
            }
            // Optional Mie-two-moment constraints change only the post-outcome angle.
            // Event rate, branch selection, Fresnel outcome, internal path and event code
            // remain those of the base sampler; coefficients are derived at runtime.
            if(event_pidx>=0 && MOMENT_MODE[event_pidx]){
                apply_mie_moment_constraint(vinx,viny,vinz,&vx,&vy,&vz,(float)MOMENT_A[event_pidx],(float)MOMENT_B[event_pidx]);
            }
            light_diag_record_event(light_diag_active,LIGHT_DIAG_NX,LIGHT_DIAG_NY,LIGHT_DIAG_NZ,RIN,ZMIN,LIGHT_DIAG_ZMAX,
                event_site_x,event_site_y,event_site_z,gid,light_event_weight,light_event_count,light_event_total,light_error_count,light_event_outside_count,light_boundary_adjustment_count);
            light_diag_mark_last_event(gid,light_diag_active,LIGHT_DIAG_REP_RAYS,LIGHT_DIAG_MAX_SEGMENTS,light_tevent,light_tcount,event_type);
            // Causal diagnostic is evaluated only after a survived particle event.
            float dcos=fminf(fmaxf(dot3(vinx,viny,vinz,vx,vy,vz),-1.0f),1.0f);
            float defl=acosf(dcos)*(180.0f/PI_F);
            internal_path_total+=event_internal_path;
            if(event_pidx>=0) internal_opl_total+=event_internal_path*(float)particle_n[event_pidx];
            // zero-RNG diagnostic accumulation, committed only after a survived particle event.
            cumulative_defl+=defl;cumulative_defl_sq+=defl*defl;
            if(defl>max_defl){
                max_defl=defl;max_evt_x=event_site_x;max_evt_y=event_site_y;max_evt_z=event_site_z;
                max_evt_type=event_type;max_evt_order=ic;max_evt_bin=event_pidx;
                max_evt_macro_bounces=event_macro_bounces;max_evt_terminal_micro_order=event_terminal_micro_order;
                max_evt_internal_path=event_internal_path;
                max_evt_internal_path_over_diameter=(event_pidx>=0 && particle_radius[event_pidx]>0.0)?event_internal_path/(2.0f*(float)particle_radius[event_pidx]):0.0f;
                max_evt_incidence_deg=event_incidence_deg;max_evt_exit_deg=event_exit_deg;
            }
            if(defl<30.0f)small_deflections++;else if(defl<90.0f)medium_deflections++;else large_deflections++;
            int current_branch=(event_type>=1 && event_type<=3)?1:2;
            if(previous_branch>0 && current_branch!=previous_branch)branch_transitions++;
            if(previous_event_type>0 && event_type!=previous_event_type)event_type_transitions++;
            previous_branch=current_branch;previous_event_type=event_type;
            event_sum_r+=event_site_r;event_sum_z+=event_site_z;
            if(event_site_r<event_min_r)event_min_r=event_site_r;if(event_site_r>event_max_r)event_max_r=event_site_r;
            if(event_site_z<event_min_z)event_min_z=event_site_z;if(event_site_z>event_max_z)event_max_z=event_site_z;
            if(ic==1){first_evt_x=event_site_x;first_evt_y=event_site_y;first_evt_z=event_site_z;wp_first_evt=wp;}
            last_evt_x=event_site_x;last_evt_y=event_site_y;last_evt_z=event_site_z;wp_last_evt=wp;
            if(event_pidx>=0){
                if(first_pbin<0) first_pbin=event_pidx; last_pbin=event_pidx;
                if(event_type>=1 && event_type<=3){if(first_gbin<0) first_gbin=event_pidx; last_gbin=event_pidx;}
                else if(event_type>=4 && event_type<=7){if(first_wbin<0) first_wbin=event_pidx; last_wbin=event_pidx;}
                if(acrylic_returns_after_particle>0) particle_events_after_acrylic_return++;
            }
            last_evt=event_type;last_evt_order=ic;last_defl=defl;wp_at_last_evt=wp;
            if(ic==1){first_evt=event_type;first_defl=defl;first_back=(dcos<0.0f)?1:0;first_source_hemi=(vy<0.0f)?1:0;}
            continue;
        }
        if(axial){
            float a,b,c; int tir=0;
            if(axial_top){
                // Water -> air at the local flat/parabolic/localized free surface.  The normal
                // is defined into the incident water medium, matching refract3().
                float snx=0.0f,sny=0.0f,snz=-1.0f;
                free_surface_normal_into_water(x,y,RIN,FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS,&snx,&sny,&snz);
                float ci=fminf(fmaxf(-dot3(vx,vy,vz,snx,sny,snz),0.0f),1.0f);
                float R=fresnel_R(ci,N_WATER,N_AIR,&tir);
                if(tir || rnd_uniform(&state)<R || !refract3(vx,vy,vz,snx,sny,snz,N_WATER,N_AIR,&a,&b,&c)){
                    reflect3(vx,vy,vz,snx,sny,snz,&a,&b,&c);vx=a;vy=b;vz=c;tr++;if(tir)ttir++;
                    if(!water_surface_bore_offset(&x,&y,&z,snx,sny,snz,RIN,ZMIN,ZMEAN,FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS)){
                        capture_water_advance_failure(light_event_outside_count,gid,step,x,y,z,snx,sny,snz,EPS_F,tside,ttop,tbottom,tbound,0,x+EPS_F*snx,y+EPS_F*sny,z+EPS_F*snz);
                        pf=PROD_INVALID_MEDIUM_STATE;pfs=PROD_SUB_WATER_OUTSIDE_INNER_RADIUS;pfstep=step;pfmedium=1;st=4;break;
                    }
                    continue;
                }
                // complete apparatus topology: transmission through the free surface
                // enters the physical headspace inside the open acrylic tube.  It is NOT a
                // detector/external-air exit until the ray reaches the open bore or exits the
                // acrylic vessel.
                vx=a;vy=b;vz=c;x-=EPS_F*snx;y-=EPS_F*sny;z-=EPS_F*snz;
                {
                    int hsub=PROD_SUB_NONE,hb=0;
                    int hw=headspace_transport_ex(&state,N_WATER,N_ACRYLIC,N_AIR,BOTTOM_EXTERNAL_N,RIN,ROUT,ZMIN,ZMIN-BOTTOM_DISC_THICKNESS,ZMEAN,ZWALL_TOP,
                        FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS,MAX_CELL_BOUNCES,&x,&y,&z,&vx,&vy,&vz,&ap,
                        &cir,&cor,&bir,&bor,&ctir,&btir,&tr,&ttir,&hsub,&hb);
                    if(hw==1){if(ic>0) acrylic_returns_after_particle++;continue;}
                    if(hw==2){st=1;goto AIR_EXIT;}
                    if(hw==3){st=7;goto WRITE_OUT;}
                    if(hw==4){st=6;goto WRITE_OUT;}
                    pf=PROD_HEADSPACE_BOUNDARY_NO_INTERSECTION;pfs=hsub;pfstep=step;pfmacro=hb;pfmedium=5;st=4;goto WRITE_OUT;
                }
            } else {
                // Water -> finite acrylic bottom disc, including repeated acrylic-slab Fresnel/TIR.
                int bhsub=PROD_SUB_NONE,bhb=0;int where=bottom_acrylic_disc_ex(&state,N_WATER,N_ACRYLIC,BOTTOM_EXTERNAL_N,N_AIR,RIN,ROUT,ZMIN,BOTTOM_DISC_THICKNESS,ZMEAN,ZWALL_TOP,FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS,MAX_AXIAL_BOUNCES,
                    &x,&y,&z,&vx,&vy,&vz,&ap,&cir,&cor,&bir,&bor,&ctir,&btir,&bhsub,&bhb);
                if(where==1){if(ic>0) acrylic_returns_after_particle++;continue;}
                if(where==2){st=1;goto AIR_EXIT;}
                if(where==3){st=7;break;}
                if(where==5){st=8;break;}
                if(where==4){
                    int hsub=PROD_SUB_NONE,hb=0;
                    int hw=headspace_transport_ex(&state,N_WATER,N_ACRYLIC,N_AIR,BOTTOM_EXTERNAL_N,RIN,ROUT,ZMIN,ZMIN-BOTTOM_DISC_THICKNESS,ZMEAN,ZWALL_TOP,
                        FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS,MAX_CELL_BOUNCES,&x,&y,&z,&vx,&vy,&vz,&ap,
                        &cir,&cor,&bir,&bor,&ctir,&btir,&tr,&ttir,&hsub,&hb);
                    if(hw==1){if(ic>0) acrylic_returns_after_particle++;continue;}
                    if(hw==2){st=1;goto AIR_EXIT;}
                    if(hw==3){st=7;goto WRITE_OUT;}
                    if(hw==4){st=6;goto WRITE_OUT;}
                    pf=PROD_HEADSPACE_BOUNDARY_NO_INTERSECTION;pfs=hsub;pfstep=step;pfmacro=hb;pfmedium=5;st=4;goto WRITE_OUT;
                }
                pf=(bhsub==PROD_SUB_ACRYLIC_BOUNCE_LIMIT)?PROD_ACRYLIC_ANNULUS_BOUNCE_LIMIT:PROD_BOTTOM_DISC_GEOMETRY_FAILURE;pfs=bhsub;pfstep=step;pfmacro=bhb;pfmedium=3;st=4;break;
            }
        }

        // water -> acrylic at the cylindrical sample-cell wall.
        float nx,ny;radial_normal(x,y,&nx,&ny);
        float ci=fminf(fmaxf(dot3(vx,vy,vz,nx,ny,0.0f),0.0f),1.0f);
        int tir=0;float R=fresnel_R(ci,N_WATER,N_ACRYLIC,&tir);float a,b,c;
        if(tir || rnd_uniform(&state)<R || !refract3(vx,vy,vz,-nx,-ny,0.0f,N_WATER,N_ACRYLIC,&a,&b,&c)){
            reflect3(vx,vy,vz,nx,ny,0.0f,&a,&b,&c);vx=a;vy=b;vz=c;cir++;if(tir)ctir++;
            x-=EPS_F*nx;y-=EPS_F*ny;continue;
        }
        vx=a;vy=b;vz=c;x+=EPS_F*nx;y+=EPS_F*ny;
        int ahsub=PROD_SUB_NONE,ahb=0;int where=acrylic_annulus_ex(&state,N_WATER,N_ACRYLIC,N_AIR,BOTTOM_EXTERNAL_N,RIN,ROUT,ZMIN,ZMIN-BOTTOM_DISC_THICKNESS,ZMEAN,ZWALL_TOP,FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS,MAX_CELL_BOUNCES,
            &x,&y,&z,&vx,&vy,&vz,&ap,&cir,&cor,&bir,&bor,&ctir,&btir,&ahsub,&ahb);
        if(where==1){if(ic>0) acrylic_returns_after_particle++;continue;}
        if(where==2){st=1;goto AIR_EXIT;}
        if(where==3){st=7;break;}
        if(where==5){st=8;break;}
        if(where==4){
            int hsub=PROD_SUB_NONE,hb=0;
            int hw=headspace_transport_ex(&state,N_WATER,N_ACRYLIC,N_AIR,BOTTOM_EXTERNAL_N,RIN,ROUT,ZMIN,ZMIN-BOTTOM_DISC_THICKNESS,ZMEAN,ZWALL_TOP,
                FREE_SURFACE_MODEL,VORTEX_DELTA_H,VORTEX_CORE_RADIUS,MAX_CELL_BOUNCES,&x,&y,&z,&vx,&vy,&vz,&ap,
                &cir,&cor,&bir,&bor,&ctir,&btir,&tr,&ttir,&hsub,&hb);
            if(hw==1){if(ic>0) acrylic_returns_after_particle++;continue;}
            if(hw==2){st=1;goto AIR_EXIT;}
            if(hw==3){st=7;goto WRITE_OUT;}
            if(hw==4){st=6;goto WRITE_OUT;}
            if(hw==5){st=8;goto WRITE_OUT;}
            pf=PROD_HEADSPACE_BOUNDARY_NO_INTERSECTION;pfs=hsub;pfstep=step;pfmacro=hb;pfmedium=5;st=4;goto WRITE_OUT;
        }
        pf=(ahsub==PROD_SUB_ACRYLIC_BOUNCE_LIMIT)?PROD_ACRYLIC_ANNULUS_BOUNCE_LIMIT:PROD_ACRYLIC_ANNULUS_NO_INTERSECTION;pfs=ahsub;pfstep=step;pfmacro=ahb;pfmedium=2;st=4;break;
    }
    if(st==0){pf=PROD_TRACE_ITERATION_LIMIT;pfs=PROD_SUB_TRACE_ITERATION_LIMIT;pfstep=(int)MAX_ITERATIONS;pfmedium=1;st=4;}
    goto WRITE_OUT;

AIR_EXIT:
    air_x[tid]=x;air_y[tid]=y;air_z[tid]=z;air_vx[tid]=vx;air_vy[tid]=vy;air_vz[tid]=vz;
WRITE_OUT:
    water_path[tid]=wp;acrylic_path[tid]=ap;
    production_failure_code[tid]=pf;production_failure_subcode[tid]=pfs;production_failure_trace_iteration[tid]=pfstep;production_failure_macro_bounces[tid]=pfmacro;production_failure_micro_order[tid]=pfmicro;production_failure_medium[tid]=pfmedium;production_failure_event_channel[tid]=pfevent;
    if(pf!=PROD_OK){failure_x[tid]=x;failure_y[tid]=y;failure_z[tid]=z;failure_vx[tid]=vx;failure_vy[tid]=vy;failure_vz[tid]=vz;}
    particle_diag_radius[tid]=pd_radius;particle_diag_rho[tid]=pd_rho;particle_diag_phi[tid]=pd_phi;particle_diag_cx[tid]=pd_cx;particle_diag_cy[tid]=pd_cy;particle_diag_cz[tid]=pd_cz;
    particle_diag_entry_x[tid]=pd_ex;particle_diag_entry_y[tid]=pd_ey;particle_diag_entry_z[tid]=pd_ez;particle_diag_exit_x[tid]=pd_xx;particle_diag_exit_y[tid]=pd_xy;particle_diag_exit_z[tid]=pd_xz;
    particle_diag_pre_x[tid]=pd_prex;particle_diag_pre_y[tid]=pd_prey;particle_diag_pre_z[tid]=pd_prez;particle_diag_pre_vx[tid]=pd_prevx;particle_diag_pre_vy[tid]=pd_prevy;particle_diag_pre_vz[tid]=pd_prevz;
    particle_diag_post_x[tid]=pd_postx;particle_diag_post_y[tid]=pd_posty;particle_diag_post_z[tid]=pd_postz;particle_diag_post_vx[tid]=pd_postvx;particle_diag_post_vy[tid]=pd_postvy;particle_diag_post_vz[tid]=pd_postvz;
    particle_diag_center_radial_clearance[tid]=pd_center_clear;particle_diag_finite_wall_clearance[tid]=pd_wall_clear;particle_diag_bottom_clearance[tid]=pd_bottom_clear;particle_diag_top_clearance[tid]=pd_top_clear;
    max_deflection_event_internal_path_m[tid]=(ic>0)?max_evt_internal_path:0.0f;max_deflection_event_internal_path_over_diameter[tid]=(ic>0)?max_evt_internal_path_over_diameter:0.0f;
    max_deflection_event_incidence_deg[tid]=(ic>0)?max_evt_incidence_deg:-1.0f;max_deflection_event_exit_deg[tid]=(ic>0)?max_evt_exit_deg:-1.0f;
    particle_diag_bin[tid]=pd_bin;particle_diag_previous_event[tid]=pd_prev_evt;particle_diag_overlap_class[tid]=pd_overlap;particle_diag_proposal_accepted[tid]=pd_accept;particle_proposal_count[tid]=pd_proposals;particle_accessible_count[tid]=pd_accessible;particle_null_count[tid]=pd_null;
    first_particle_deflection_deg[tid]=first_defl;last_particle_deflection_deg[tid]=last_defl;internal_particle_path_total[tid]=internal_path_total;internal_particle_opl_total[tid]=internal_opl_total;
    water_path_after_last_particle_event[tid]=(last_evt_order>0 && wp_at_last_evt>=0.0f)?fmaxf(wp-wp_at_last_evt,0.0f):-1.0f;
    cumulative_particle_deflection_deg[tid]=(ic>0)?cumulative_defl:-1.0f;cumulative_particle_deflection_sq_deg2[tid]=(ic>0)?cumulative_defl_sq:-1.0f;max_particle_deflection_deg[tid]=(ic>0)?max_defl:-1.0f;
    max_deflection_event_x[tid]=(ic>0)?max_evt_x:0.0f;max_deflection_event_y[tid]=(ic>0)?max_evt_y:0.0f;max_deflection_event_z[tid]=(ic>0)?max_evt_z:0.0f;
    first_particle_x[tid]=(ic>0)?first_evt_x:0.0f;first_particle_y[tid]=(ic>0)?first_evt_y:0.0f;first_particle_z[tid]=(ic>0)?first_evt_z:0.0f;
    last_particle_x[tid]=(ic>0)?last_evt_x:0.0f;last_particle_y[tid]=(ic>0)?last_evt_y:0.0f;last_particle_z[tid]=(ic>0)?last_evt_z:0.0f;
    particle_event_mean_r_m[tid]=(ic>0)?event_sum_r/(float)ic:-1.0f;particle_event_mean_z_m[tid]=(ic>0)?event_sum_z/(float)ic:-1.0f;
    particle_event_min_r_m[tid]=(ic>0)?event_min_r:-1.0f;particle_event_max_r_m[tid]=(ic>0)?event_max_r:-1.0f;particle_event_min_z_m[tid]=(ic>0)?event_min_z:-1.0f;particle_event_max_z_m[tid]=(ic>0)?event_max_z:-1.0f;
    water_path_at_first_particle_event_m[tid]=(ic>0)?wp_first_evt:-1.0f;water_path_at_last_particle_event_m[tid]=(ic>0)?wp_last_evt:-1.0f;
    interaction_count[tid]=ic;geometric_interactions[tid]=gic;wave_interactions[tid]=wic;coherent_wave_interactions[tid]=wcoh;morphology_wave_interactions[tid]=wmorph;particle_reflections[tid]=pr;particle_entry_reflections[tid]=per;particle_internal_reflections[tid]=pir;particle_absorptions[tid]=pabs;
    cell_inner_reflections[tid]=cir;cell_outer_reflections[tid]=cor;cell_tir[tid]=ctir;
    top_reflections[tid]=tr;top_tir[tid]=ttir;bottom_inner_reflections[tid]=bir;bottom_outer_reflections[tid]=bor;bottom_tir[tid]=btir;
    source_bore_interactions[tid]=sbi;source_bore_reflections[tid]=sbr;entered_water[tid]=entered;status[tid]=st;
    first_particle_event_type[tid]=first_evt;last_particle_event_type[tid]=last_evt;last_particle_event_order[tid]=last_evt_order;first_particle_backscatter[tid]=first_back;first_particle_source_hemisphere[tid]=first_source_hemi;
    first_particle_bin[tid]=first_pbin;last_particle_bin[tid]=last_pbin;first_geometric_bin[tid]=first_gbin;last_geometric_bin[tid]=last_gbin;first_wave_bin[tid]=first_wbin;last_wave_bin[tid]=last_wbin;
    acrylic_return_after_particle_count[tid]=acrylic_returns_after_particle;particle_after_acrylic_return_count[tid]=particle_events_after_acrylic_return;
    particle_branch_transition_count[tid]=branch_transitions;particle_event_type_transition_count[tid]=event_type_transitions;
    particle_small_deflection_count[tid]=small_deflections;particle_medium_deflection_count[tid]=medium_deflections;particle_large_deflection_count[tid]=large_deflections;
    max_deflection_event_type[tid]=(ic>0)?max_evt_type:0;max_deflection_event_order[tid]=(ic>0)?max_evt_order:0;max_deflection_event_bin[tid]=(ic>0)?max_evt_bin:-1;
    max_deflection_event_macro_bounces[tid]=(ic>0)?max_evt_macro_bounces:0;max_deflection_event_terminal_micro_order[tid]=(ic>0)?max_evt_terminal_micro_order:0;
}
// ================= DIAGNOSTIC-ONLY SINGLE-PARTICLE AUDIT =================
// Samples the particle angular law in isolation with a fixed +y
// incident direction. It is never called by trace_kernel and cannot alter production transport.
__global__ void single_particle_angular_audit_kernel(
    const int SAMPLES_PER_BIN,const int N_PARTICLES,
    const float N_MEDIUM,const float WAVELENGTH,const float ROUGH_ALPHA,
    const float WAVE_SPHEROID_ASPECT_RATIO,const int WAVE_ORIENTATION_MODEL,const float WAVE_ORIENTATION_KAPPA,
    const int MAX_PARTICLE_BOUNCES,const double* PARTICLE_N,const double* PARTICLE_K,const double* PARTICLE_RADIUS,
    const double* QSCA,const double* MIE_PHASE_CDF,const double* MIE_THETA,const int MIE_NTHETA,
    const float WAVE_COHERENT_FRACTION,const unsigned int SEED,
    float* INTERFACE_COS,unsigned char* INTERFACE_TYPE,unsigned char* INTERFACE_STATUS,
    float* PRODUCTION_COS,unsigned char* PRODUCTION_TYPE,unsigned char* PRODUCTION_STATUS){
    long long tid=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    long long total=(long long)SAMPLES_PER_BIN*(long long)N_PARTICLES;if(tid>=total)return;
    int p=(int)(tid/(long long)SAMPLES_PER_BIN);
    unsigned int sid=(unsigned int)(tid-(long long)p*(long long)SAMPLES_PER_BIN);
    unsigned int stateI=SEED+(unsigned int)p*2654435761u+sid*74729u+13u;
    unsigned int stateP=SEED^0x9e3779b9u;stateP+=(unsigned int)p*2246822519u+sid*104729u+29u;

    // Finite-particle interface kernel.
    float vx=0.0f,vy=1.0f,vz=0.0f,x=0.0f,y=0.0f,z=0.0f,ip=0.0f;int pr=0,per=0,pir=0,pabs=0;
    int ps=sphere_fresnel_interaction_3d(&stateI,N_MEDIUM,(float)PARTICLE_N[p],(float)PARTICLE_K[p],WAVELENGTH,ROUGH_ALPHA,
        (float)PARTICLE_RADIUS[p],MAX_PARTICLE_BOUNCES,&x,&y,&z,&vx,&vy,&vz,&ip,&pr,&per,&pir,&pabs);
    INTERFACE_STATUS[tid]=(unsigned char)ps;
    if(ps==1){INTERFACE_COS[tid]=fminf(fmaxf(vy,-1.0f),1.0f);INTERFACE_TYPE[tid]=(unsigned char)((per>0)?1:((pir>0)?3:2));}
    else {INTERFACE_COS[tid]=0.0f;INTERFACE_TYPE[tid]=0;} // status!=1 makes payload invalid

    // Exact per-bin branch mixture. For Qsca<1 the exact event-rate remainder is
    // entirely the geometric/interface branch. For Qsca>=1 the projected-area branch has
    // probability 1/Qsca and the Mie-excess wave branch has probability (Qsca-1)/Qsca.
    float pvx=0.0f,pvy=1.0f,pvz=0.0f;int ppr=0,pper=0,ppir=0,ppabs=0;float pip=0.0f;
    float q=(float)fmax(QSCA[p],0.0);float geom_fraction=(q>1.0f)?(1.0f/q):1.0f;
    int use_wave=(q>1.0f && rnd_uniform(&stateP)>=geom_fraction);
    if(!use_wave){
        float px=0.0f,py=0.0f,pz=0.0f;
        ps=sphere_fresnel_interaction_3d(&stateP,N_MEDIUM,(float)PARTICLE_N[p],(float)PARTICLE_K[p],WAVELENGTH,ROUGH_ALPHA,
            (float)PARTICLE_RADIUS[p],MAX_PARTICLE_BOUNCES,&px,&py,&pz,&pvx,&pvy,&pvz,&pip,&ppr,&pper,&ppir,&ppabs);
        PRODUCTION_STATUS[tid]=(unsigned char)ps;
        if(ps==1){PRODUCTION_COS[tid]=fminf(fmaxf(pvy,-1.0f),1.0f);PRODUCTION_TYPE[tid]=(unsigned char)((pper>0)?1:((ppir>0)?3:2));}
        else {PRODUCTION_COS[tid]=0.0f;PRODUCTION_TYPE[tid]=0;} // status!=1 makes payload invalid
        return;
    }
    int coherent=(WAVE_COHERENT_FRACTION>=1.0f)||(rnd_uniform(&stateP)<WAVE_COHERENT_FRACTION);
    if(coherent){
        mie_scatter_direction(&stateP,MIE_PHASE_CDF,MIE_THETA,MIE_NTHETA,p,&pvx,&pvy,&pvz);
        PRODUCTION_STATUS[tid]=1;PRODUCTION_COS[tid]=fminf(fmaxf(pvy,-1.0f),1.0f);PRODUCTION_TYPE[tid]=4;return;
    }
    ps=spheroid_fresnel_scatter_direction_only_3d(&stateP,N_MEDIUM,(float)PARTICLE_N[p],(float)PARTICLE_K[p],WAVELENGTH,ROUGH_ALPHA,
        (float)PARTICLE_RADIUS[p],WAVE_SPHEROID_ASPECT_RATIO,WAVE_ORIENTATION_MODEL,WAVE_ORIENTATION_KAPPA,1.0f,0.0f,0.0f,
        MAX_PARTICLE_BOUNCES,&pvx,&pvy,&pvz,&pip,&ppr,&pper,&ppir,&ppabs);
    PRODUCTION_STATUS[tid]=(unsigned char)ps;
    if(ps==1){PRODUCTION_COS[tid]=fminf(fmaxf(pvy,-1.0f),1.0f);PRODUCTION_TYPE[tid]=(unsigned char)((pper>0)?5:((ppir>0)?7:6));}
    else {PRODUCTION_COS[tid]=0.0f;PRODUCTION_TYPE[tid]=0;} // status!=1 makes payload invalid
}

// ================= MOMENT-CONSTRAINED SINGLE-PARTICLE AUDIT =================
__global__ void single_particle_moment_constrained_audit_kernel(
    const int SAMPLES_PER_BIN,const int N_PARTICLES,
    const float N_MEDIUM,const float WAVELENGTH,const float ROUGH_ALPHA,
    const float WAVE_SPHEROID_ASPECT_RATIO,const int WAVE_ORIENTATION_MODEL,const float WAVE_ORIENTATION_KAPPA,
    const int MAX_PARTICLE_BOUNCES,const double* PARTICLE_N,const double* PARTICLE_K,const double* PARTICLE_RADIUS,
    const double* QSCA,const double* MIE_PHASE_CDF,const double* MIE_THETA,const int MIE_NTHETA,
    const float WAVE_COHERENT_FRACTION,const unsigned int SEED,
    const double* MOMENT_A,const double* MOMENT_B,const unsigned char* MOMENT_MODE,
    float* BASE_COS,float* CONSTRAINED_COS,unsigned char* EVENT_TYPE,unsigned char* STATUS){
    long long tid=(long long)blockDim.x*blockIdx.x+threadIdx.x;
    long long total=(long long)SAMPLES_PER_BIN*(long long)N_PARTICLES;if(tid>=total)return;
    int p=(int)(tid/(long long)SAMPLES_PER_BIN);
    unsigned int sid=(unsigned int)(tid-(long long)p*(long long)SAMPLES_PER_BIN);
    unsigned int state=SEED^0x517cc1b7u;state+=(unsigned int)p*2246822519u+sid*104729u+31u;
    float vx=0.0f,vy=1.0f,vz=0.0f;int pr=0,per=0,pir=0,pabs=0;float ip=0.0f;
    float q=(float)fmax(QSCA[p],0.0);float geom_fraction=(q>1.0f)?(1.0f/q):1.0f;
    int use_wave=(q>1.0f && rnd_uniform(&state)>=geom_fraction);int ps=0;unsigned char typ=0;
    if(!use_wave){
        float x=0.0f,y=0.0f,z=0.0f;
        ps=sphere_fresnel_interaction_3d(&state,N_MEDIUM,(float)PARTICLE_N[p],(float)PARTICLE_K[p],WAVELENGTH,ROUGH_ALPHA,
            (float)PARTICLE_RADIUS[p],MAX_PARTICLE_BOUNCES,&x,&y,&z,&vx,&vy,&vz,&ip,&pr,&per,&pir,&pabs);
        typ=(unsigned char)((per>0)?1:((pir>0)?3:2));
    } else {
        int coherent=(WAVE_COHERENT_FRACTION>=1.0f)||(rnd_uniform(&state)<WAVE_COHERENT_FRACTION);
        if(coherent){mie_scatter_direction(&state,MIE_PHASE_CDF,MIE_THETA,MIE_NTHETA,p,&vx,&vy,&vz);ps=1;typ=4;}
        else{
            ps=spheroid_fresnel_scatter_direction_only_3d(&state,N_MEDIUM,(float)PARTICLE_N[p],(float)PARTICLE_K[p],WAVELENGTH,ROUGH_ALPHA,
                (float)PARTICLE_RADIUS[p],WAVE_SPHEROID_ASPECT_RATIO,WAVE_ORIENTATION_MODEL,WAVE_ORIENTATION_KAPPA,1.0f,0.0f,0.0f,
                MAX_PARTICLE_BOUNCES,&vx,&vy,&vz,&ip,&pr,&per,&pir,&pabs);
            typ=(unsigned char)((per>0)?5:((pir>0)?7:6));
        }
    }
    STATUS[tid]=(unsigned char)ps;
    if(ps!=1){BASE_COS[tid]=0.0f;CONSTRAINED_COS[tid]=0.0f;EVENT_TYPE[tid]=0;return;}
    float base=fminf(fmaxf(vy,-1.0f),1.0f);BASE_COS[tid]=base;EVENT_TYPE[tid]=typ;
    if(MOMENT_MODE[p]){float ox=vx,oy=vy,oz=vz;apply_mie_moment_constraint(0.0f,1.0f,0.0f,&ox,&oy,&oz,(float)MOMENT_A[p],(float)MOMENT_B[p]);CONSTRAINED_COS[tid]=fminf(fmaxf(oy,-1.0f),1.0f);}
    else CONSTRAINED_COS[tid]=base;
}

}
'''


def get_material_psd_raw(material: str):
    """Return authoritative diameter bins and raw source weights without host normalization."""
    m = material.strip().lower()
    if m == 'loess':
        return LOESS_DIAMETER_M.copy(), LOESS_WEIGHTS.copy()
    if m == 'kaolin':
        return KAOLIN_DIAMETER_M.copy(), KAOLIN_WEIGHTS.copy()
    raise ValueError("material must be 'loess' or 'kaolin'")


def get_material_psd(material: str):
    """Compatibility view with normalized weights; forward CUDA normalizes its own inputs."""
    d, w = get_material_psd_raw(material)
    return d, w / w.sum()


def _histogram_median(hist):
    hist = np.asarray(hist, dtype=np.int64)
    n = int(hist.sum())
    if n <= 0:
        return float('nan')
    cs = np.cumsum(hist)
    if n % 2:
        return float(np.searchsorted(cs, n // 2 + 1))
    a = int(np.searchsorted(cs, n // 2))
    b = int(np.searchsorted(cs, n // 2 + 1))
    return 0.5 * (a + b)


def _jackknife_normalized_se(batch_scores):
    """Batch-jackknife SE for the normalized 18-channel detector response."""
    if len(batch_scores) < 3:
        return np.full(DETECTOR_ANGLES_DEG.size, np.nan, dtype=float)
    s = np.asarray(batch_scores, dtype=float)
    total = s.sum(axis=0)
    loo = []
    for b in range(s.shape[0]):
        q = total - s[b]
        den = q.sum()
        if den > 0:
            loo.append(q / den)
    if len(loo) < 3:
        return np.full(DETECTOR_ANGLES_DEG.size, np.nan, dtype=float)
    loo = np.asarray(loo)
    m = loo.mean(axis=0)
    B = loo.shape[0]
    return np.sqrt((B - 1.0) / B * np.sum((loo - m) ** 2, axis=0))


# ============================ EVENT-HISTORY DECOMPOSITION ============================
def _event_history_masks(ints):
    """Return exact per-ray mechanism/order masks from CUDA-emitted primitive counters.

    The masks are diagnostic only.  They do not modify ray weights or transport.  The
    interaction-order bins are mutually exclusive.  The single-event mechanism partition
    is also mutually exclusive for detected, non-absorbed rays:
      external entry reflection | internal-reflection path | surface transmission/refraction
      | coherent Mie-wave event.
    Morphology-wave events use the same explicit particle-surface Fresnel primitive as the
    geometric branch and therefore enter the surface categories.
    """
    ic = np.asarray(ints['interaction_count'])
    gic = np.asarray(ints['geometric_interactions'])
    wic = np.asarray(ints['wave_interactions'])
    wcoh = np.asarray(ints['coherent_wave_interactions'])
    wmorph = np.asarray(ints['morphology_wave_interactions'])
    per = np.asarray(ints['particle_entry_reflections'])
    pir = np.asarray(ints['particle_internal_reflections'])
    pabs = np.asarray(ints['particle_absorptions'])
    single = ic == 1
    multiple = ic >= 2
    surface_touch = (gic > 0) | (wmorph > 0)
    masks = {
        # Broad mechanism categories overlap; order partitions below do not.
        'any_particle': ic > 0,
        'any_geometric': gic > 0,
        'any_wave': wic > 0,
        'any_coherent_wave': wcoh > 0,
        'any_morphology_wave': wmorph > 0,
        'any_entry_reflection': per > 0,
        'any_internal_reflection': pir > 0,
        'ballistic': ic == 0,
        'single_interaction': single,
        'single_entry_reflection': single & (per > 0),
        'multiple_interaction': multiple,
        # exact interaction order.
        'interaction_order_2': ic == 2,
        'interaction_order_3': ic == 3,
        'interaction_order_4plus': ic >= 4,
        # single-event branch identity.
        'exactly_one_geometric': single & (gic == 1) & (wic == 0),
        'exactly_one_wave': single & (gic == 0) & (wic == 1),
        'single_coherent_wave_escape': single & (wcoh == 1),
        'single_external_entry_reflection_escape': single & (per > 0),
        'single_internal_reflection_escape': single & (pir > 0),
        'single_surface_transmission_escape': single
        & surface_touch
        & (per == 0)
        & (pir == 0)
        & (pabs == 0),
        # multi-event branch identity.
        'mixed_geometric_wave': multiple & (gic > 0) & (wic > 0),
        'geometric_only_multiple': multiple & (gic > 0) & (wic == 0),
        'wave_only_multiple': multiple & (gic == 0) & (wic > 0),
        'multiple_with_entry_reflection': multiple & (per > 0),
        'multiple_with_internal_reflection': multiple & (pir > 0),
    }
    return masks


PARTICLE_EVENT_TYPE_NAMES = {
    0: 'none',
    1: 'geometric_external_entry_reflection',
    2: 'geometric_surface_transmission_refraction',
    3: 'geometric_internal_reflection_path',
    4: 'coherent_mie_wave',
    5: 'morphology_wave_external_entry_reflection',
    6: 'morphology_wave_surface_transmission_refraction',
    7: 'morphology_wave_internal_reflection_path',
}


def _safe_fraction(num, den):
    num = np.asarray(num, dtype=np.float64)
    den = np.asarray(den, dtype=np.float64)
    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


@dataclass
class ForwardResult:
    detector_angles_deg: np.ndarray
    raw_hits: np.ndarray  # compatibility alias: native reconstructed-hardware hits
    normalized_response: (
        np.ndarray
    )  # selected: symmetry-averaged reconstructed hardware
    native_exact_hits: np.ndarray  # compatibility alias: hardware native
    mirror_exact_hits: np.ndarray  # compatibility alias: hardware mirror
    symmetry_exact_scores: np.ndarray  # compatibility alias: hardware symmetry score
    normalized_native_exact_response: np.ndarray
    normalized_mirror_exact_response: np.ndarray
    normalized_symmetry_exact_response: np.ndarray
    normalized_response_jackknife_se: np.ndarray
    hardware_native_hits: np.ndarray
    hardware_mirror_hits: np.ndarray
    hardware_symmetry_scores: np.ndarray
    normalized_hardware_native_response: np.ndarray
    normalized_hardware_mirror_response: np.ndarray
    normalized_hardware_symmetry_response: np.ndarray
    hardware_normalized_jackknife_se: np.ndarray
    legacy_native_channel_counts: np.ndarray
    legacy_mirror_channel_counts: np.ndarray
    legacy_symmetry_channel_scores: np.ndarray
    normalized_legacy_native_response: np.ndarray
    normalized_legacy_mirror_response: np.ndarray
    normalized_legacy_symmetry_response: np.ndarray
    legacy_normalized_jackknife_se: np.ndarray
    legacy_acceptance_deg: float
    valid_exit_count: int
    numerical_failure_count: int  # compatibility: total production integrity failures
    numerical_failure_fraction: float
    production_failure_count: int
    production_failure_code_counts: dict
    production_failure_subcode_counts: dict
    production_watchdog_count: int
    production_geometry_failure_count: int
    production_nonfinite_count: int
    production_failure_diagnostics: dict
    n_rays: int
    requested_n_rays: int
    stopped_adaptively: bool
    stop_reason: str
    particle_event_model: str
    particle_surface_rms_slope_deg: float
    particle_surface_beckmann_alpha: float
    wave_surface_rms_height_nm: float
    wave_spheroid_aspect_ratio: float
    wave_spheroid_orientation_model: str
    wave_spheroid_orientation_kappa: float
    aggregation_model: str
    aggregation_collision_exposure: float
    aggregation_number_reduction_fraction: float
    aggregation_mass_conservation_error: float
    aggregation_projected_area_ratio: float
    aggregation_largest_bin_mass_fraction: float
    aggregation_overflow_event_fraction: float
    aggregation_steps: int
    population_number_reduction_fraction: float
    fragmentation_breakup_exposure: float
    fragmentation_reference_diameter_m: float
    fragmentation_size_exponent: float
    fragmentation_underflow_event_fraction: float
    aggregation_events_per_initial_particle: float
    fragmentation_events_per_initial_particle: float
    wave_coherent_fraction_nominal_cuda: float
    mean_coherent_wave_interactions: float
    mean_morphology_wave_interactions: float
    morphology_fraction_of_wave_interactions: float
    mu_geom_per_m: float
    mu_wave_per_m: float
    mu_event_per_m: float
    mu_sca_per_m: float
    mu_ext_per_m: float
    geometric_optical_depth_diameter: float
    optical_depth_diameter: float
    mean_interactions: float
    mean_geometric_interactions: float
    mean_wave_interactions: float
    median_interactions: float
    ballistic_fraction: float
    mean_fresnel_reflections: float
    entry_reflection_fraction: float
    internal_reflection_fraction: float
    particle_absorption_fraction: float
    particle_absorbed_ray_count: int
    mean_cell_inner_reflections: float
    mean_cell_outer_reflections: float
    cell_tir_fraction: float
    mean_top_surface_reflections: float
    top_surface_tir_fraction: float
    mean_top_surface_interactions: float
    free_surface_model: str
    vortex_wall_center_height_difference_mm: float
    vortex_core_radius_mm: float
    vortex_asymptotic_height_parameter_mm: float
    vortex_area_mean_uncorrected_rise_mm: float
    vortex_equivalent_core_rpm: float
    vortex_equivalent_rigid_body_rpm: float
    vortex_center_surface_z_mm: float
    vortex_wall_surface_z_mm: float
    sediment_transport_model: str
    sediment_eddy_diffusivity_m2_s: float
    sediment_majorant_geom_per_m: float
    sediment_majorant_wave_per_m: float
    sediment_majorant_total_per_m: float
    sediment_max_bin_density_scale: float
    meridional_circulation_ratio: float
    meridional_peak_speed_m_s: float
    meridional_grid_nr: int
    meridional_grid_nz: int
    meridional_solver_iterations: int
    meridional_solver_max_residual: float
    mean_bottom_inner_reflections: float
    mean_bottom_outer_reflections: float
    bottom_tir_fraction: float
    top_exit_fraction: float
    bottom_exit_fraction: float
    source_bore_surface_model: str
    detector_bore_surface_model: str
    bore_refractive_index: float
    ring_inner_face_surface_model: str
    ring_inner_face_refractive_index: float
    max_ring_face_bounces: int
    source_collimator_acceptance_fraction: float
    source_direct_acceptance_fraction: float
    source_reflected_acceptance_fraction: float
    source_bore_interactions_per_ray: float
    source_bore_reflections_per_ray: float
    hardware_native_direct_hits: np.ndarray
    hardware_mirror_direct_hits: np.ndarray
    hardware_symmetry_direct_scores: np.ndarray
    hardware_native_stray_hits: np.ndarray
    hardware_mirror_stray_hits: np.ndarray
    hardware_symmetry_stray_scores: np.ndarray
    detector_stray_fraction_by_channel: np.ndarray
    detector_stray_detection_fraction: float
    hardware_native_ring_face_stray_hits: np.ndarray
    hardware_mirror_ring_face_stray_hits: np.ndarray
    hardware_symmetry_ring_face_stray_scores: np.ndarray
    detector_ring_face_stray_fraction_by_channel: np.ndarray
    detector_ring_face_stray_detection_fraction: float
    photodiode_response_model: str
    sfh213_half_angle_deg: float
    sfh213_directional_fit_exponent: float
    photodiode_native_weighted_scores: np.ndarray
    photodiode_mirror_weighted_scores: np.ndarray
    photodiode_symmetry_weighted_scores: np.ndarray
    normalized_photodiode_native_response: np.ndarray
    normalized_photodiode_mirror_response: np.ndarray
    normalized_photodiode_symmetry_response: np.ndarray
    photodiode_relative_response_by_channel: np.ndarray
    detector_mean_incidence_angle_deg_by_channel: np.ndarray
    hardware_candidate_detection_fraction: float
    entered_water_fraction: float
    air_exit_fraction: float
    axial_loss_fraction: float
    detector_detection_fraction: float  # selected symmetry-equivalent score / ray
    native_detector_detection_fraction: float
    mirror_detector_detection_fraction: float
    legacy_channel_score_per_ray: float
    symmetry_variance_reduction: bool
    batch_diagnostics: Optional[list] = None
    particle_diagnostics: Optional[Dict[str, np.ndarray]] = None
    detector_particle_path_diagnostics: Optional[Dict[str, np.ndarray]] = None
    ray_data: Optional[Dict[str, np.ndarray]] = None
    heatmap_xy: Optional[np.ndarray] = None
    heatmap_zy: Optional[np.ndarray] = None
    light_transport_diagnostics: Optional[dict] = None

    def to_dict(self):
        d = {
            k: v
            for k, v in self.__dict__.items()
            if k
            not in {
                'particle_diagnostics',
                'detector_particle_path_diagnostics',
                'ray_data',
                'heatmap_xy',
                'heatmap_zy',
                'light_transport_diagnostics',
            }
        }
        for k, v in list(d.items()):
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, np.generic):
                d[k] = v.item()
        return d


class ClaritasForwardModel:
    """CUDA light-transport model with selectable event laws and angular constraints.

    Host code performs configuration, array transfer, diagnostics aggregation and
    I/O only. Mie/event/phase/source/detector physics that determine ray outcomes
    are implemented in CUDA kernels in ``CUDA_SRC``.
    """

    MIE_MAX_ORDER = 4096
    EVENT_MODEL_CODES = {
        'geometric': 0,
        'mie_qsca': 1,
        'mie_qext': 2,
        'hybrid_mie_excess': 3,
        'hybrid_mie_exact_rate': 4,
        'hybrid_mie_exact_rate_moment_constrained': 5,
    }
    WEIGHT_MODE_CODES = {'mass_fraction': 0, 'number_fraction': 1}
    SOURCE_ANGULAR_MODEL_CODES = {
        'legacy_polar_pdf': 0,
        'beta_radiance_3d': 1,
        'thesis_planar_beta': 2,
    }
    BORE_SURFACE_MODEL_CODES = {'absorbing': 0, 'smooth_fresnel_nylon': 1}
    RING_FACE_SURFACE_MODEL_CODES = {'absorbing': 0, 'smooth_fresnel_nylon': 1}
    PHOTODIODE_RESPONSE_MODEL_CODES = {
        'ideal_hard_aperture': 0,
        'sfh213_datasheet_fit': 1,
    }
    SPHEROID_ORIENTATION_MODEL_CODES = {'isotropic': 0, 'local_tangential_axial_vmf': 1}
    FREE_SURFACE_MODEL_CODES = {
        'flat': 0,
        'parabolic_vortex': 1,
        'localized_scully_vortex': 2,
    }
    SEDIMENT_TRANSPORT_MODEL_CODES = {
        'uniform': 0,
        'stokes_drift_diffusion': 1,
        'schiller_naumann_drift_diffusion': 2,
        'schiller_naumann_meridional_advection_diffusion': 3,
    }

    def __init__(
        self,
        n_water=1.33,
        n_particle=1.59,
        n_acrylic=1.4906,
        n_air=1.0,
        tube_inner_radius_m=0.0465,
        tube_outer_radius_m=0.0500,
        tube_length_m=0.491,
        water_height_m=0.142,
        sensor_height_above_bottom_m=0.093,
        sensor_ring_inner_radius_m=0.0505,
        sensor_ring_outer_radius_m=0.0655,
        counterbore_depth_m=0.008,
        through_bore_diameter_m=0.004,
        counterbore_diameter_m=0.0087,
        source_launch_radius_m=0.0655,
        legacy_detector_acceptance_deg=6.5,
        source_beam_sigma_m=1.0e-5,
        source_bore_surface_model='absorbing',
        detector_bore_surface_model='absorbing',
        bore_refractive_index=1.53,
        max_bore_bounces=16,
        ring_inner_face_surface_model='absorbing',
        ring_inner_face_refractive_index=1.53,
        max_ring_face_bounces=16,
        photodiode_response_model='ideal_hard_aperture',
        particle_wavelength_m=DEFAULT_PARTICLE_WAVELENGTH_M,
        particle_optical_constants_by_material=None,
        particle_event_model='geometric',
        hybrid_wave_scale=1.0,
        mie_phase_grid_size=2049,
        moment_constraint_by_material=None,
        particle_surface_rms_slope_deg=25.0,
        wave_surface_rms_height_m=100e-9,
        wave_spheroid_aspect_ratio=1.5,
        wave_spheroid_orientation_model='isotropic',
        wave_spheroid_orientation_kappa=0.0,
        free_surface_model='localized_scully_vortex',
        vortex_wall_center_height_difference_m=0.050,
        vortex_core_radius_m=0.020,
        sediment_transport_model='schiller_naumann_drift_diffusion',
        sediment_eddy_diffusivity_m2_s=1.0e-3,
        water_density_kg_per_m3=997.0,
        water_dynamic_viscosity_pa_s=8.9e-4,
        gravity_m_s2=9.81,
        sediment_transport_quadrature_points=4096,
        sediment_radial_lookup_points=2049,
        meridional_circulation_ratio=0.10,
        meridional_grid_nr=48,
        meridional_grid_nz=96,
        meridional_solver_max_iterations=8000,
        meridional_solver_check_interval=250,
        meridional_solver_tolerance=1.0e-6,
        meridional_solver_sor=1.75,
        aggregation_model='none',
        aggregation_collision_exposure=0.0,
        fragmentation_breakup_exposure=0.0,
        fragmentation_reference_diameter_m=100e-6,
        fragmentation_size_exponent=1.0,
        aggregation_step_safety=0.2,
        aggregation_max_steps=200000,
        alpha1=0.45,
        alpha2=5.0,
        source_angular_model='beta_radiance_3d',
        density_kg_per_m3=2600.0,
        bottom_disc_thickness_m=0.003,
        bottom_external_refractive_index=1.0,
        max_axial_bounces=64,
        max_iterations=int(1e6),
        max_internal_bounces=64,
        max_cell_bounces=64,
        chunk_size=250000,
        statistics_batch_rays=250000,
        symmetry_average_exact=True,
    ):
        try:
            import cupy as cp
        except Exception as exc:
            raise RuntimeError('CuPy/CUDA is required for CLARITAS .') from exc
        self.cp = cp
        self.n_water = float(n_water)
        self.n_particle = float(n_particle)
        self.n_acrylic = float(n_acrylic)
        self.n_air = float(n_air)
        self.rin = float(tube_inner_radius_m)
        self.rout = float(tube_outer_radius_m)
        self.tube_length = float(tube_length_m)
        self.water_height = float(water_height_m)
        self.sensor_height = float(sensor_height_above_bottom_m)
        self.zmin = -self.sensor_height
        self.zmean = self.water_height - self.sensor_height
        self.zwall_top = self.tube_length - self.sensor_height
        # Compatibility alias retained for sediment/free-surface code; it is NOT the acrylic wall top.
        self.zmax = self.zmean
        self.ring_in = float(sensor_ring_inner_radius_m)
        self.ring_out = float(sensor_ring_outer_radius_m)
        self.counterbore_depth = float(counterbore_depth_m)
        self.throat_out = self.ring_out - self.counterbore_depth
        self.throat_radius = 0.5 * float(through_bore_diameter_m)
        self.counterbore_radius = 0.5 * float(counterbore_diameter_m)
        self.source_launch_radius = float(source_launch_radius_m)
        self.legacy_acceptance_deg = float(legacy_detector_acceptance_deg)
        self.beam_sigma = float(source_beam_sigma_m)
        self.source_bore_surface_model = str(source_bore_surface_model).strip().lower()
        self.detector_bore_surface_model = (
            str(detector_bore_surface_model).strip().lower()
        )
        self.bore_refractive_index = float(bore_refractive_index)
        self.max_bore_bounces = int(max_bore_bounces)
        self.ring_inner_face_surface_model = (
            str(ring_inner_face_surface_model).strip().lower()
        )
        self.ring_inner_face_refractive_index = float(ring_inner_face_refractive_index)
        self.max_ring_face_bounces = int(max_ring_face_bounces)
        self.photodiode_response_model = str(photodiode_response_model).strip().lower()
        self.particle_wavelength = float(particle_wavelength_m)
        self.particle_optical_constants_by_material = (
            particle_optical_constants_by_material or {}
        )
        self.particle_event_model = str(particle_event_model).strip().lower()
        self.hybrid_wave_scale = float(hybrid_wave_scale)
        self.mie_phase_grid_size = int(mie_phase_grid_size)
        self.moment_constraint_by_material = moment_constraint_by_material or {}
        self.particle_surface_rms_slope_deg = float(particle_surface_rms_slope_deg)
        self.particle_surface_beckmann_alpha = float(
            np.tan(np.deg2rad(self.particle_surface_rms_slope_deg))
        )
        self.wave_surface_rms_height_m = float(wave_surface_rms_height_m)
        self.wave_spheroid_aspect_ratio = float(wave_spheroid_aspect_ratio)
        self.wave_spheroid_orientation_model = (
            str(wave_spheroid_orientation_model).strip().lower()
        )
        self.wave_spheroid_orientation_kappa = float(wave_spheroid_orientation_kappa)
        self.free_surface_model = str(free_surface_model).strip().lower()
        self.vortex_delta_h = float(vortex_wall_center_height_difference_m)
        self.vortex_core_radius = float(vortex_core_radius_m)
        self.sediment_transport_model = str(sediment_transport_model).strip().lower()
        self.sediment_eddy_diffusivity = float(sediment_eddy_diffusivity_m2_s)
        self.water_density = float(water_density_kg_per_m3)
        self.water_dynamic_viscosity = float(water_dynamic_viscosity_pa_s)
        self.gravity = float(gravity_m_s2)
        self.sediment_transport_quadrature_points = int(
            sediment_transport_quadrature_points
        )
        self.sediment_radial_lookup_points = int(sediment_radial_lookup_points)
        self.meridional_circulation_ratio = float(meridional_circulation_ratio)
        self.meridional_grid_nr = int(meridional_grid_nr)
        self.meridional_grid_nz = int(meridional_grid_nz)
        self.meridional_solver_max_iterations = int(meridional_solver_max_iterations)
        self.meridional_solver_check_interval = int(meridional_solver_check_interval)
        self.meridional_solver_tolerance = float(meridional_solver_tolerance)
        self.meridional_solver_sor = float(meridional_solver_sor)
        self.meridional_peak_speed = 0.0
        self.meridional_vtheta_peak = 0.0
        self.aggregation_model = str(aggregation_model).strip().lower()
        self.aggregation_collision_exposure = float(aggregation_collision_exposure)
        self.fragmentation_breakup_exposure = float(fragmentation_breakup_exposure)
        self.fragmentation_reference_diameter_m = float(
            fragmentation_reference_diameter_m
        )
        self.fragmentation_size_exponent = float(fragmentation_size_exponent)
        self.aggregation_step_safety = float(aggregation_step_safety)
        self.aggregation_max_steps = int(aggregation_max_steps)
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.source_angular_model = str(source_angular_model).strip().lower()
        self.density = float(density_kg_per_m3)
        self.bottom_disc_thickness = float(bottom_disc_thickness_m)
        self.bottom_external_n = float(bottom_external_refractive_index)
        self.max_axial_bounces = int(max_axial_bounces)
        self.max_iterations = int(max_iterations)
        self.max_internal_bounces = int(max_internal_bounces)
        self.max_cell_bounces = int(max_cell_bounces)
        self.chunk_size = int(chunk_size)
        self.statistics_batch_rays = int(statistics_batch_rays)
        self.symmetry_average_exact = bool(symmetry_average_exact)
        if self.chunk_size < 1 or self.statistics_batch_rays < 1:
            raise ValueError('chunk/batch sizes must be positive')
        if not (
            0
            < self.rin
            < self.rout
            < self.ring_in
            < self.throat_out
            <= self.source_launch_radius
            <= self.ring_out
        ):
            raise ValueError('TARDIIS radial/source geometry is inconsistent')
        if not (self.tube_length > 0.0 and np.isfinite(self.tube_length)):
            raise ValueError('tube_length_m must be finite and positive')
        if not (self.zwall_top > self.zmean > self.zmin):
            raise ValueError(
                'physical wall top must lie above the complete water/free-surface domain and bottom'
            )
        if self.photodiode_response_model not in self.PHOTODIODE_RESPONSE_MODEL_CODES:
            raise ValueError(
                f'unknown photodiode_response_model {self.photodiode_response_model!r}'
            )
        if self.photodiode_response_model != 'ideal_hard_aperture' and (
            self.detector_bore_surface_model != 'absorbing'
            or self.ring_inner_face_surface_model != 'absorbing'
        ):
            raise ValueError(
                'SFH213 package-response audit requires absorbing detector bores and absorbing ring inner face so incidence angle is defined on the validated direct path'
            )
        if not (0 < self.throat_radius < self.counterbore_radius):
            raise ValueError('through-bore must be narrower than counterbore')
        if not (0.0 < self.legacy_acceptance_deg < 90.0):
            raise ValueError(
                'legacy detector acceptance must be between 0 and 90 degrees'
            )
        if self.particle_wavelength <= 0:
            raise ValueError('particle_wavelength_m must be positive')
        if self.particle_event_model not in self.EVENT_MODEL_CODES:
            raise ValueError('invalid particle_event_model')
        if self.hybrid_wave_scale < 0 or not np.isfinite(self.hybrid_wave_scale):
            raise ValueError('hybrid_wave_scale must be finite and >=0')
        if self.mie_phase_grid_size < 257:
            raise ValueError('mie_phase_grid_size must be >=257')
        if not np.isfinite(self.particle_surface_rms_slope_deg) or not (
            0.0 <= self.particle_surface_rms_slope_deg < 80.0
        ):
            raise ValueError(
                'particle_surface_rms_slope_deg must be finite and in [0,80) degrees'
            )
        if (
            not np.isfinite(self.wave_surface_rms_height_m)
            or self.wave_surface_rms_height_m < 0.0
        ):
            raise ValueError('wave_surface_rms_height_m must be finite and >=0')
        if not np.isfinite(self.wave_spheroid_aspect_ratio) or not (
            0.2 <= self.wave_spheroid_aspect_ratio <= 5.0
        ):
            raise ValueError(
                'wave_spheroid_aspect_ratio must be finite and in [0.2,5.0]'
            )
        if (
            self.wave_spheroid_orientation_model
            not in self.SPHEROID_ORIENTATION_MODEL_CODES
        ):
            raise ValueError(
                "wave_spheroid_orientation_model must be 'isotropic' or 'local_tangential_axial_vmf'"
            )
        if not np.isfinite(self.wave_spheroid_orientation_kappa) or not (
            0.0 <= self.wave_spheroid_orientation_kappa <= 100.0
        ):
            raise ValueError(
                'wave_spheroid_orientation_kappa must be finite and in [0,100]'
            )
        if self.free_surface_model not in self.FREE_SURFACE_MODEL_CODES:
            raise ValueError(
                "free_surface_model must be 'flat', 'parabolic_vortex', or 'localized_scully_vortex'"
            )
        if not np.isfinite(self.vortex_delta_h) or self.vortex_delta_h < 0.0:
            raise ValueError(
                'vortex_wall_center_height_difference_m must be finite and >=0'
            )
        if self.free_surface_model == 'localized_scully_vortex' and (
            not np.isfinite(self.vortex_core_radius) or self.vortex_core_radius <= 0.0
        ):
            raise ValueError(
                'vortex_core_radius_m must be finite and >0 for localized_scully_vortex'
            )
        if self.sediment_transport_model not in self.SEDIMENT_TRANSPORT_MODEL_CODES:
            raise ValueError(
                "sediment_transport_model must be 'uniform', 'stokes_drift_diffusion', 'schiller_naumann_drift_diffusion', or 'schiller_naumann_meridional_advection_diffusion'"
            )
        if (
            not np.isfinite(self.sediment_eddy_diffusivity)
            or self.sediment_eddy_diffusivity <= 0.0
        ):
            raise ValueError('sediment_eddy_diffusivity_m2_s must be finite and >0')
        if not (
            np.isfinite(self.water_density) and 0.0 < self.water_density < self.density
        ):
            raise ValueError(
                'water_density_kg_per_m3 must be positive and below particle density'
            )
        if (
            not np.isfinite(self.water_dynamic_viscosity)
            or self.water_dynamic_viscosity <= 0.0
        ):
            raise ValueError('water_dynamic_viscosity_pa_s must be finite and >0')
        if not np.isfinite(self.gravity) or self.gravity <= 0.0:
            raise ValueError('gravity_m_s2 must be finite and >0')
        if self.sediment_transport_quadrature_points < 64:
            raise ValueError('sediment_transport_quadrature_points must be >=64')
        if self.sediment_radial_lookup_points < 65:
            raise ValueError('sediment_radial_lookup_points must be >=65')
        if not np.isfinite(self.meridional_circulation_ratio) or not (
            0.0 <= self.meridional_circulation_ratio <= 0.5
        ):
            raise ValueError(
                'meridional_circulation_ratio must be finite and in [0,0.5]'
            )
        if self.meridional_grid_nr < 16 or self.meridional_grid_nz < 32:
            raise ValueError('meridional solver grid must be at least 16 x 32')
        if (
            self.meridional_solver_max_iterations < 250
            or self.meridional_solver_check_interval < 10
        ):
            raise ValueError('invalid meridional solver iteration controls')
        if not np.isfinite(self.meridional_solver_tolerance) or not (
            0.0 < self.meridional_solver_tolerance < 1e-2
        ):
            raise ValueError('meridional_solver_tolerance must be in (0,1e-2)')
        if not np.isfinite(self.meridional_solver_sor) or not (
            1.0 <= self.meridional_solver_sor < 1.95
        ):
            raise ValueError('meridional_solver_sor must be in [1,1.95)')
        vg = vortex_surface_geometry(
            self.free_surface_model,
            self.zmean,
            self.rin,
            self.vortex_delta_h,
            self.vortex_core_radius,
            self.gravity,
        )
        if not (float(vg['wall_z']) < self.zwall_top):
            raise ValueError(
                'free surface reaches or exceeds the physical acrylic wall top'
            )
        if (
            self.free_surface_model in {'localized_scully_vortex', 'parabolic_vortex'}
            and self.vortex_delta_h > 0.0
        ):
            self.meridional_vtheta_peak = float(vg['vtheta_peak_m_s'])
            self.meridional_peak_speed = (
                self.meridional_circulation_ratio * self.meridional_vtheta_peak
            )
        if (
            self.sediment_transport_model
            == 'schiller_naumann_meridional_advection_diffusion'
            and (
                self.free_surface_model
                not in {'localized_scully_vortex', 'parabolic_vortex'}
                or self.vortex_delta_h <= 0.0
            )
        ):
            raise ValueError(
                'meridional transport requires a non-zero supported vortex'
            )
        if vg['center_z'] <= self.zmin:
            raise ValueError(
                'vortex depression reaches or passes the acrylic bottom; reduce vortex height difference or increase core radius'
            )
        if self.aggregation_model not in {
            'none',
            'orthokinetic_compact',
            'orthokinetic_compact_fragmentation',
        }:
            raise ValueError(
                "aggregation_model must be 'none', 'orthokinetic_compact', or 'orthokinetic_compact_fragmentation'"
            )
        if (
            not np.isfinite(self.aggregation_collision_exposure)
            or self.aggregation_collision_exposure < 0.0
        ):
            raise ValueError('aggregation_collision_exposure must be finite and >=0')
        if (
            not np.isfinite(self.fragmentation_breakup_exposure)
            or self.fragmentation_breakup_exposure < 0.0
        ):
            raise ValueError('fragmentation_breakup_exposure must be finite and >=0')
        if (
            not np.isfinite(self.fragmentation_reference_diameter_m)
            or self.fragmentation_reference_diameter_m <= 0.0
        ):
            raise ValueError('fragmentation_reference_diameter_m must be finite and >0')
        if not np.isfinite(self.fragmentation_size_exponent) or not (
            0.0 <= self.fragmentation_size_exponent <= 6.0
        ):
            raise ValueError('fragmentation_size_exponent must be finite and in [0,6]')
        if (
            self.aggregation_model != 'orthokinetic_compact_fragmentation'
            and self.fragmentation_breakup_exposure > 0.0
        ):
            raise ValueError(
                'fragmentation_breakup_exposure requires aggregation_model=orthokinetic_compact_fragmentation'
            )
        if not (0.0 < self.aggregation_step_safety <= 0.25):
            raise ValueError('aggregation_step_safety must be in (0,0.25]')
        if self.aggregation_max_steps < 1:
            raise ValueError('aggregation_max_steps must be >=1')
        if not (self.alpha1 > 0.0 and self.alpha2 > 0.0):
            raise ValueError('source Beta alpha1/alpha2 must be positive')
        if self.source_angular_model not in self.SOURCE_ANGULAR_MODEL_CODES:
            raise ValueError(
                "source_angular_model must be one of: legacy_polar_pdf, beta_radiance_3d, thesis_planar_beta"
            )
        if (
            self.source_bore_surface_model not in self.BORE_SURFACE_MODEL_CODES
            or self.detector_bore_surface_model not in self.BORE_SURFACE_MODEL_CODES
        ):
            raise ValueError(
                "bore surface model must be absorbing or smooth_fresnel_nylon"
            )
        if self.ring_inner_face_surface_model not in self.RING_FACE_SURFACE_MODEL_CODES:
            raise ValueError(
                "ring inner face surface model must be absorbing or smooth_fresnel_nylon"
            )
        if not (
            self.ring_inner_face_refractive_index > 0.0
            and np.isfinite(self.ring_inner_face_refractive_index)
        ):
            raise ValueError(
                'ring_inner_face_refractive_index must be finite and positive'
            )
        if self.max_ring_face_bounces < 1:
            raise ValueError('max_ring_face_bounces must be >=1')
        if not (self.bore_refractive_index > 1.0):
            raise ValueError('bore_refractive_index must be >1')
        if self.max_bore_bounces < 0:
            raise ValueError('max_bore_bounces must be >=0')
        if not (self.zmin < 0 < self.zmax):
            raise ValueError('sensor plane must lie within water column')
        if self.bottom_disc_thickness <= 0.0:
            raise ValueError('bottom_disc_thickness_m must be positive')
        if self.bottom_external_n <= 0.0:
            raise ValueError('bottom_external_refractive_index must be positive')
        if self.max_axial_bounces < 1:
            raise ValueError('max_axial_bounces must be >=1')
        self.module = cp.RawModule(code=CUDA_SRC, options=('-std=c++11',))
        names = [
            'water_surface_offset_test_kernel',
            'bore_wall_test_kernel',
            'trace_kernel',
            'light_diag_segment_test_kernel',
            'light_diag_event_test_kernel',
            'free_surface_geometry_test_kernel',
            'sediment_field_test_kernel',
            'meridional_velocity_test_kernel',
            'meridional_active_mask_kernel',
            'meridional_grid_initialize_kernel',
            'meridional_build_coefficients_kernel',
            'meridional_relax_kernel',
            'meridional_normalize_kernel',
            'meridional_residual_kernel',
            'meridional_extend_grid_kernel',
            'meridional_finalize_kernel',
            'meridional_majorants_kernel',
            'orientation_sampler_test_kernel',
            'build_theta_grid_kernel',
            'mie_coefficients_efficiencies_kernel',
            'mie_phase_pdf_kernel',
            'mie_phase_cdf_kernel',
            'dummy_phase_kernel',
            'orthokinetic_compact_aggregation_kernel',
            'orthokinetic_compact_aggregation_fragmentation_kernel',
            'transport_precompute_kernel',
            'sediment_transport_precompute_kernel',
            'score_detectors_kernel',
            'single_particle_angular_audit_kernel',
            'single_particle_moment_constrained_audit_kernel',
        ]
        self.kernels = {n: self.module.get_function(n) for n in names}
        self.kernel = self.kernels['trace_kernel']

    def _prepare_gpu_particle_physics(
        self, material, concentration_g_per_L, weights=None, weight_mode='mass_fraction'
    ):
        cp = self.cp
        d, bw = get_material_psd_raw(material)
        w = bw if weights is None else np.asarray(weights, dtype=np.float64)
        if w.ndim != 1 or w.size != d.size:
            raise ValueError(f'{material} requires {d.size} PSD weights')
        if np.any(~np.isfinite(w)) or np.any(w < 0.0) or not np.any(w > 0.0):
            raise ValueError('invalid PSD weights')
        if weight_mode not in self.WEIGHT_MODE_CODES:
            raise ValueError("weight_mode must be 'mass_fraction' or 'number_fraction'")
        conc = float(concentration_g_per_L)
        if not np.isfinite(conc) or conc < 0.0:
            raise ValueError('concentration must be finite and >=0')
        nidx, kidx = get_material_optical_constants(
            material, self.particle_optical_constants_by_material, self.n_particle
        )
        n = int(d.size)
        M = self.MIE_MAX_ORDER
        dd = cp.asarray(d, dtype=cp.float64)
        wd_input = cp.asarray(w, dtype=cp.float64)
        nd = cp.asarray(nidx, dtype=cp.float64)
        kd = cp.asarray(kidx, dtype=cp.float64)
        wd = cp.empty_like(wd_input)
        agg_scalars = cp.zeros(10, dtype=cp.float64)
        population_scalars = cp.zeros(16, dtype=cp.float64)
        agg_err = cp.zeros(1, dtype=cp.int32)
        used_coupled_population_kernel = False
        if self.aggregation_model == 'orthokinetic_compact':
            if weight_mode != 'mass_fraction':
                raise ValueError(
                    'orthokinetic_compact aggregation currently requires mass_fraction PSD input'
                )
            self.kernels['orthokinetic_compact_aggregation_kernel'](
                (1,),
                (1,),
                (
                    dd,
                    wd_input,
                    np.int32(n),
                    np.float64(conc),
                    np.float64(self.density),
                    np.float64(self.aggregation_collision_exposure),
                    np.float64(self.aggregation_step_safety),
                    np.int32(self.aggregation_max_steps),
                    wd,
                    agg_scalars,
                    agg_err,
                ),
            )
            cp.cuda.Stream.null.synchronize()
            ecode = int(cp.asnumpy(agg_err)[0])
            if ecode:
                raise RuntimeError(
                    f'CUDA aggregation precompute failed with error code {ecode}'
                )
        elif self.aggregation_model == 'orthokinetic_compact_fragmentation':
            if weight_mode != 'mass_fraction':
                raise ValueError(
                    'orthokinetic_compact_fragmentation currently requires mass_fraction PSD input'
                )
            # Zero breakup exposure uses aggregation alone; nonzero exposure uses
            # the coupled aggregation-fragmentation CUDA kernel.
            if self.fragmentation_breakup_exposure <= 0.0:
                self.kernels['orthokinetic_compact_aggregation_kernel'](
                    (1,),
                    (1,),
                    (
                        dd,
                        wd_input,
                        np.int32(n),
                        np.float64(conc),
                        np.float64(self.density),
                        np.float64(self.aggregation_collision_exposure),
                        np.float64(self.aggregation_step_safety),
                        np.int32(self.aggregation_max_steps),
                        wd,
                        agg_scalars,
                        agg_err,
                    ),
                )
            else:
                used_coupled_population_kernel = True
                self.kernels['orthokinetic_compact_aggregation_fragmentation_kernel'](
                    (1,),
                    (1,),
                    (
                        dd,
                        wd_input,
                        np.int32(n),
                        np.float64(conc),
                        np.float64(self.density),
                        np.float64(self.aggregation_collision_exposure),
                        np.float64(self.fragmentation_breakup_exposure),
                        np.float64(self.fragmentation_reference_diameter_m),
                        np.float64(self.fragmentation_size_exponent),
                        np.float64(self.aggregation_step_safety),
                        np.int32(self.aggregation_max_steps),
                        wd,
                        population_scalars,
                        agg_err,
                    ),
                )
            cp.cuda.Stream.null.synchronize()
            ecode = int(cp.asnumpy(agg_err)[0])
            if ecode:
                raise RuntimeError(
                    f'CUDA aggregation-fragmentation precompute failed with error code {ecode}'
                )
        else:
            self.kernels['orthokinetic_compact_aggregation_kernel'](
                (1,),
                (1,),
                (
                    dd,
                    wd_input,
                    np.int32(n),
                    np.float64(conc),
                    np.float64(self.density),
                    np.float64(0.0),
                    np.float64(self.aggregation_step_safety),
                    np.int32(self.aggregation_max_steps),
                    wd,
                    agg_scalars,
                    agg_err,
                ),
            )
            cp.cuda.Stream.null.synchronize()
            ecode = int(cp.asnumpy(agg_err)[0])
            if ecode:
                raise RuntimeError(
                    f'CUDA PSD normalization precompute failed with error code {ecode}'
                )
        # Mie coefficient scratch and outputs. Physics is computed only by CUDA.
        Dre = cp.empty(n * M, dtype=cp.float64)
        Dim = cp.empty_like(Dre)
        ar = cp.empty_like(Dre)
        ai = cp.empty_like(Dre)
        br = cp.empty_like(Dre)
        bi = cp.empty_like(Dre)
        nstop = cp.zeros(n, dtype=cp.int32)
        x = cp.empty(n, dtype=cp.float64)
        qext = cp.empty_like(x)
        qsca = cp.empty_like(x)
        qabs = cp.empty_like(x)
        qback = cp.empty_like(x)
        g = cp.empty_like(x)
        albedo = cp.empty_like(x)
        err = cp.zeros(1, dtype=cp.int32)
        threads = 128
        blocks = (n + threads - 1) // threads
        self.kernels['mie_coefficients_efficiencies_kernel'](
            (blocks,),
            (threads,),
            (
                dd,
                nd,
                kd,
                np.float64(self.n_water),
                np.float64(self.particle_wavelength),
                np.int32(n),
                Dre,
                Dim,
                ar,
                ai,
                br,
                bi,
                nstop,
                x,
                qext,
                qsca,
                qabs,
                qback,
                g,
                albedo,
                err,
            ),
        )
        cp.cuda.Stream.null.synchronize()
        ecode = int(cp.asnumpy(err)[0])
        if ecode:
            raise RuntimeError(
                f'CUDA Mie precompute failed with error code {ecode}; MIE_MAX_ORDER={M}'
            )
        if self.particle_event_model in {
            'hybrid_mie_excess',
            'hybrid_mie_exact_rate',
            'hybrid_mie_exact_rate_moment_constrained',
        }:
            nt = int(self.mie_phase_grid_size)
            theta = cp.empty(nt, dtype=cp.float64)
            phase_pdf = cp.empty(n * nt, dtype=cp.float64)
            phase_cdf = cp.empty(n * nt, dtype=cp.float64)
            tb = (nt + threads - 1) // threads
            self.kernels['build_theta_grid_kernel'](
                (tb,), (threads,), (theta, np.int32(nt))
            )
            total = n * nt
            pb = (total + threads - 1) // threads
            self.kernels['mie_phase_pdf_kernel'](
                (pb,),
                (threads,),
                (theta, np.int32(nt), np.int32(n), nstop, ar, ai, br, bi, phase_pdf),
            )
            self.kernels['mie_phase_cdf_kernel'](
                (n,),
                (1,),
                (theta, phase_pdf, np.int32(nt), np.int32(n), phase_cdf, err),
            )
        else:
            nt = 2
            theta = cp.empty(2, dtype=cp.float64)
            phase_cdf = cp.empty(2 * n, dtype=cp.float64)
            self.kernels['dummy_phase_kernel'](
                (max(1, (n + threads - 1) // threads),),
                (threads,),
                (theta, phase_cdf, np.int32(n)),
            )
        # Transport arrays/CDFs.
        arr = lambda: cp.empty(n, dtype=cp.float64)
        radius = arr()
        pmass = arr()
        srcw = arr()
        nden = arr()
        sg = arr()
        ssca = arr()
        sext = arr()
        swave = arr()
        sevent = arr()
        mug = arr()
        mus = arr()
        mue = arr()
        muw = arr()
        muev = arr()
        pew = arr()
        gew = arr()
        wew = arr()
        gcdf = arr()
        wcdf = arr()
        pcdf = arr()
        muscal = cp.empty(8, dtype=cp.float64)
        self.kernels['transport_precompute_kernel'](
            (1,),
            (1,),
            (
                dd,
                wd,
                qext,
                qsca,
                np.int32(n),
                np.float64(conc),
                np.float64(self.density),
                np.int32(self.WEIGHT_MODE_CODES[weight_mode]),
                np.int32(self.EVENT_MODEL_CODES[self.particle_event_model]),
                np.float64(self.hybrid_wave_scale),
                np.float64(self.n_water),
                np.float64(self.particle_wavelength),
                np.float64(self.wave_surface_rms_height_m),
                radius,
                pmass,
                srcw,
                nden,
                sg,
                ssca,
                sext,
                swave,
                sevent,
                mug,
                mus,
                mue,
                muw,
                muev,
                pew,
                gew,
                wew,
                gcdf,
                wcdf,
                pcdf,
                muscal,
                err,
            ),
        )
        cp.cuda.Stream.null.synchronize()
        ecode = int(cp.asnumpy(err)[0])
        if ecode:
            raise RuntimeError(
                f'CUDA particle transport precompute failed with error code {ecode}'
            )
        # Per-bin angular constraints derive from the base sampler and exact Mie
        # moments, never detector data.
        moment_a = cp.ones(n, dtype=cp.float64)
        moment_b = cp.zeros(n, dtype=cp.float64)
        moment_mode = cp.zeros(n, dtype=cp.uint8)
        if self.particle_event_model == 'hybrid_mie_exact_rate_moment_constrained':
            key = str(material).strip().lower()
            spec = self.moment_constraint_by_material.get(key)
            if spec is None:
                raise RuntimeError(f'moment constraints missing for material {key!r}')
            ah = np.asarray(spec.get('a'), dtype=np.float64)
            bh = np.asarray(spec.get('b'), dtype=np.float64)
            mh = np.asarray(spec.get('mode'), dtype=np.uint8)
            if ah.shape != (n,) or bh.shape != (n,) or mh.shape != (n,):
                raise RuntimeError(
                    f'moment constraint shape mismatch for {key}: {(ah.shape,bh.shape,mh.shape)} expected {(n,)}'
                )
            if np.any(~np.isfinite(ah)) or np.any(~np.isfinite(bh)):
                raise RuntimeError('moment constraints contain non-finite coefficients')
            if np.any((mh != 0) & (mh != 1)):
                raise RuntimeError('moment constraint mode must be 0/1')
            qh = cp.asnumpy(qsca)
            if np.any(mh[qh < 1.0] != 0):
                raise RuntimeError('Qsca<1 bins must remain unconstrained')
            # Monotone bounded quadratic map T(d)=a*d+b*d^2 over d in [0,2].
            active = mh.astype(bool)
            if np.any(ah[active] < -1e-12) or np.any(
                ah[active] + 4.0 * bh[active] < -1e-10
            ):
                raise RuntimeError('moment map is non-monotone on [0,2]')
            if np.any(2.0 * ah[active] + 4.0 * bh[active] > 2.000001):
                raise RuntimeError('moment map exceeds d=2 bound before CUDA clipping')
            moment_a = cp.asarray(ah)
            moment_b = cp.asarray(bh)
            moment_mode = cp.asarray(mh)
        # Non-hybrid laws trace mu_event; hybrid laws trace a geometric channel
        # plus the separate Mie-excess wave channel.
        if self.particle_event_model == 'hybrid_mie_excess':
            trace_geom_bin = mug
            trace_wave_bin = muw
        elif self.particle_event_model in {
            'hybrid_mie_exact_rate',
            'hybrid_mie_exact_rate_moment_constrained',
        }:
            trace_geom_bin = cp.maximum(muev - muw, 0.0)
            trace_wave_bin = muw
        else:
            trace_geom_bin = muev
            trace_wave_bin = cp.zeros_like(muw)
        sed_tau = arr()
        sed_ws = arr()
        sed_sre = arr()
        sed_sdrag = arr()
        sed_A = arr()
        sed_B = arr()
        sed_norm = arr()
        sed_max = arr()
        sed_min = arr()
        sed_rmax = arr()
        sed_rre = arr()
        sed_rcenter = arr()
        sed_radial_log = cp.empty(
            n * self.sediment_radial_lookup_points, dtype=cp.float64
        )
        sed_radial_wr = cp.empty_like(sed_radial_log)
        sed_sc = arr()
        sed_sw = arr()
        sed_bc = arr()
        sed_bw = arr()
        sed_major = cp.zeros(4, dtype=cp.float64)
        sed_min.fill(np.nan)
        err.fill(0)
        self.kernels['sediment_transport_precompute_kernel'](
            (1,),
            (1,),
            (
                dd,
                np.int32(n),
                np.int32(
                    self.SEDIMENT_TRANSPORT_MODEL_CODES[self.sediment_transport_model]
                ),
                np.float64(self.density),
                np.float64(self.water_density),
                np.float64(self.water_dynamic_viscosity),
                np.float64(self.gravity),
                np.float64(self.sediment_eddy_diffusivity),
                np.float64(self.rin),
                np.float64(self.zmin),
                np.float64(self.zmax),
                np.int32(self.FREE_SURFACE_MODEL_CODES[self.free_surface_model]),
                np.float64(self.vortex_delta_h),
                np.float64(self.vortex_core_radius),
                np.int32(self.sediment_transport_quadrature_points),
                np.int32(self.sediment_radial_lookup_points),
                trace_geom_bin,
                trace_wave_bin,
                sed_tau,
                sed_ws,
                sed_sre,
                sed_sdrag,
                sed_A,
                sed_B,
                sed_norm,
                sed_max,
                sed_rmax,
                sed_rre,
                sed_rcenter,
                sed_radial_log,
                sed_radial_wr,
                sed_sc,
                sed_sw,
                sed_bc,
                sed_bw,
                sed_major,
                err,
            ),
        )
        cp.cuda.Stream.null.synchronize()
        ecode = int(cp.asnumpy(err)[0])
        if ecode:
            raise RuntimeError(
                f'CUDA sediment transport precompute failed with error code {ecode}'
            )

        # Model 3 solves stationary axisymmetric advection-diffusion on CUDA.
        # Finite-Re slips are particle-relative velocities; the streamfunction adds
        # bulk meridional advection. Conservative finite volumes use first-order
        # upwind advection, centred diffusion, no-flux boundaries and per-bin mass
        # renormalisation.
        vg = vortex_surface_geometry(
            self.free_surface_model,
            self.zmean,
            self.rin,
            self.vortex_delta_h,
            self.vortex_core_radius,
            self.gravity,
        )
        sed_grid = cp.ones(1, dtype=cp.float64)
        grid_nr = 2
        grid_nz = 2
        grid_zmax = float(vg['wall_z'])
        solver_iterations = 0
        solver_residual = np.zeros(n, dtype=np.float64)
        if (
            self.sediment_transport_model
            == 'schiller_naumann_meridional_advection_diffusion'
        ):
            grid_nr = self.meridional_grid_nr
            grid_nz = self.meridional_grid_nz
            plane = grid_nr * grid_nz
            total = n * plane
            sed_grid = cp.empty(total, dtype=cp.float64)
            active = cp.empty(plane, dtype=cp.uint8)
            ap = cp.empty(total, dtype=cp.float64)
            ae = cp.empty_like(ap)
            aw = cp.empty_like(ap)
            an = cp.empty_like(ap)
            ass = cp.empty_like(ap)
            residual = cp.empty(n, dtype=cp.float64)
            threads_solver = 256
            blocks_plane = (plane + threads_solver - 1) // threads_solver
            blocks_total = (total + threads_solver - 1) // threads_solver
            self.kernels['meridional_active_mask_kernel'](
                (blocks_plane,),
                (threads_solver,),
                (
                    np.int32(grid_nr),
                    np.int32(grid_nz),
                    np.float64(self.rin),
                    np.float64(self.zmin),
                    np.float64(grid_zmax),
                    np.float64(self.zmax),
                    np.int32(self.FREE_SURFACE_MODEL_CODES[self.free_surface_model]),
                    np.float64(self.vortex_delta_h),
                    np.float64(self.vortex_core_radius),
                    active,
                ),
            )
            self.kernels['meridional_grid_initialize_kernel'](
                (blocks_total,),
                (threads_solver,),
                (
                    np.int32(n),
                    np.int32(grid_nr),
                    np.int32(grid_nz),
                    np.float64(self.rin),
                    np.float64(self.zmin),
                    np.float64(grid_zmax),
                    np.int32(self.FREE_SURFACE_MODEL_CODES[self.free_surface_model]),
                    np.float64(self.vortex_core_radius),
                    sed_A,
                    sed_B,
                    sed_norm,
                    sed_radial_log,
                    np.int32(self.sediment_radial_lookup_points),
                    active,
                    sed_grid,
                ),
            )
            self.kernels['meridional_build_coefficients_kernel'](
                (blocks_total,),
                (threads_solver,),
                (
                    np.int32(n),
                    np.int32(grid_nr),
                    np.int32(grid_nz),
                    np.float64(self.rin),
                    np.float64(self.zmin),
                    np.float64(grid_zmax),
                    np.float64(self.zmax),
                    np.int32(self.FREE_SURFACE_MODEL_CODES[self.free_surface_model]),
                    np.float64(self.vortex_delta_h),
                    np.float64(self.vortex_core_radius),
                    np.float64(self.meridional_peak_speed),
                    np.float64(self.sediment_eddy_diffusivity),
                    sed_ws,
                    sed_radial_wr,
                    np.int32(self.sediment_radial_lookup_points),
                    active,
                    ap,
                    ae,
                    aw,
                    an,
                    ass,
                ),
            )
            err.fill(0)
            self.kernels['meridional_normalize_kernel'](
                (n,),
                (1,),
                (
                    np.int32(n),
                    np.int32(grid_nr),
                    np.int32(grid_nz),
                    np.float64(self.rin),
                    np.float64(self.zmin),
                    np.float64(grid_zmax),
                    np.float64(self.zmax),
                    active,
                    sed_grid,
                    err,
                ),
            )
            cp.cuda.Stream.null.synchronize()
            ecode = int(cp.asnumpy(err)[0])
            if ecode:
                raise RuntimeError(
                    f'CUDA meridional initial normalization failed with error code {ecode}'
                )
            min_solver_iterations = min(1000, self.meridional_solver_max_iterations)
            last_checked = 0
            for it in range(1, self.meridional_solver_max_iterations + 1):
                self.kernels['meridional_relax_kernel'](
                    (blocks_total,),
                    (threads_solver,),
                    (
                        np.int32(n),
                        np.int32(grid_nr),
                        np.int32(grid_nz),
                        np.int32(0),
                        np.float64(self.meridional_solver_sor),
                        active,
                        ap,
                        ae,
                        aw,
                        an,
                        ass,
                        sed_grid,
                    ),
                )
                self.kernels['meridional_relax_kernel'](
                    (blocks_total,),
                    (threads_solver,),
                    (
                        np.int32(n),
                        np.int32(grid_nr),
                        np.int32(grid_nz),
                        np.int32(1),
                        np.float64(self.meridional_solver_sor),
                        active,
                        ap,
                        ae,
                        aw,
                        an,
                        ass,
                        sed_grid,
                    ),
                )
                if it % 20 == 0:
                    self.kernels['meridional_normalize_kernel'](
                        (n,),
                        (1,),
                        (
                            np.int32(n),
                            np.int32(grid_nr),
                            np.int32(grid_nz),
                            np.float64(self.rin),
                            np.float64(self.zmin),
                            np.float64(grid_zmax),
                            np.float64(self.zmax),
                            active,
                            sed_grid,
                            err,
                        ),
                    )
                if (
                    it % self.meridional_solver_check_interval == 0
                    or it == self.meridional_solver_max_iterations
                ):
                    self.kernels['meridional_normalize_kernel'](
                        (n,),
                        (1,),
                        (
                            np.int32(n),
                            np.int32(grid_nr),
                            np.int32(grid_nz),
                            np.float64(self.rin),
                            np.float64(self.zmin),
                            np.float64(grid_zmax),
                            np.float64(self.zmax),
                            active,
                            sed_grid,
                            err,
                        ),
                    )
                    self.kernels['meridional_residual_kernel'](
                        (n,),
                        (1,),
                        (
                            np.int32(n),
                            np.int32(grid_nr),
                            np.int32(grid_nz),
                            active,
                            ap,
                            ae,
                            aw,
                            an,
                            ass,
                            sed_grid,
                            residual,
                        ),
                    )
                    cp.cuda.Stream.null.synchronize()
                    ecode = int(cp.asnumpy(err)[0])
                    solver_residual = cp.asnumpy(residual)
                    last_checked = it
                    if ecode:
                        raise RuntimeError(
                            f'CUDA meridional normalization failed with error code {ecode}'
                        )
                    if (
                        it >= min_solver_iterations
                        and float(np.max(solver_residual))
                        <= self.meridional_solver_tolerance
                    ):
                        solver_iterations = it
                        break
            if solver_iterations == 0:
                solver_iterations = (
                    last_checked or self.meridional_solver_max_iterations
                )
            if float(np.max(solver_residual)) > self.meridional_solver_tolerance:
                raise RuntimeError(
                    f'meridional transport solver did not converge: max residual={float(np.max(solver_residual)):.6g} after {solver_iterations} iterations'
                )
            # Extend only for interpolation above the curved boundary; active-cell
            # values and mass are unchanged. Then rebuild majorants/point diagnostics.
            self.kernels['meridional_extend_grid_kernel'](
                ((n * grid_nr + threads_solver - 1) // threads_solver,),
                (threads_solver,),
                (np.int32(n), np.int32(grid_nr), np.int32(grid_nz), active, sed_grid),
            )
            self.kernels['meridional_finalize_kernel'](
                (n,),
                (1,),
                (
                    np.int32(n),
                    np.int32(grid_nr),
                    np.int32(grid_nz),
                    np.float64(self.rin),
                    np.float64(self.zmin),
                    np.float64(grid_zmax),
                    active,
                    sed_grid,
                    sed_norm,
                    sed_max,
                    sed_min,
                    sed_sc,
                    sed_sw,
                    sed_bc,
                    sed_bw,
                ),
            )
            self.kernels['meridional_majorants_kernel'](
                (1,),
                (1,),
                (np.int32(n), trace_geom_bin, trace_wave_bin, sed_max, sed_major),
            )
            cp.cuda.Stream.null.synchronize()
        # Copy only diagnostic products; device fields/rates remain authoritative inputs to ray transport.
        keys = {
            'diameter_m': dd,
            'input_source_weight': wd_input,
            'effective_aggregation_weight': wd,
            'source_weight': srcw,
            'radius_m': radius,
            'particle_mass_kg': pmass,
            'number_density_by_bin': nden,
            'n_real': nd,
            'k_imag': kd,
            'size_parameter': x,
            'qext': qext,
            'qsca': qsca,
            'qabs': qabs,
            'qback': qback,
            'asymmetry_g': g,
            'single_scattering_albedo': albedo,
            'geometric_cross_section_m2': sg,
            'scattering_cross_section_m2': ssca,
            'extinction_cross_section_m2': sext,
            'wave_cross_section_m2': swave,
            'event_cross_section_m2': sevent,
            'mu_geom_by_bin': mug,
            'mu_sca_by_bin': mus,
            'mu_ext_by_bin': mue,
            'mu_wave_by_bin': muw,
            'mu_event_by_bin': muev,
            'particle_event_weights': pew,
            'geometric_event_weights': gew,
            'wave_event_weights': wew,
            'sediment_response_time_s': sed_tau,
            'sediment_settling_velocity_m_s': sed_ws,
            'sediment_settling_reynolds': sed_sre,
            'sediment_settling_drag_factor': sed_sdrag,
            'sediment_radial_log_coefficient_A': sed_A,
            'sediment_vertical_inverse_length_B_per_m': sed_B,
            'sediment_radial_max_slip_velocity_m_s': sed_rmax,
            'sediment_radial_max_reynolds': sed_rre,
            'sediment_radial_center_log_factor': sed_rcenter,
            'sediment_normalization_scaled': sed_norm,
            'sediment_max_density_scale': sed_max,
            'sediment_min_density_scale': sed_min,
            'sediment_scale_sensor_center': sed_sc,
            'sediment_scale_sensor_wall_r0p95R': sed_sw,
            'sediment_scale_bottom_center': sed_bc,
            'sediment_scale_bottom_wall_r0p95R': sed_bw,
        }
        host = {k: cp.asnumpy(v) for k, v in keys.items()}
        mu_host = cp.asnumpy(muscal)
        agg_host = cp.asnumpy(agg_scalars)
        # host-only diagnostic: integrate the already-CUDA-generated Mie
        # angular CDF over narrow near-backscatter bands.  This does not alter
        # event selection or ray directions; it exposes the existing phase law.
        if self.particle_event_model == 'hybrid_mie_excess' and nt > 2:
            theta_host = cp.asnumpy(theta)
            cdf_host = cp.asnumpy(phase_cdf).reshape(n, nt)

            def _cdf_at_deg(deg):
                q = math.radians(float(deg))
                return np.asarray(
                    [np.interp(q, theta_host, row) for row in cdf_host],
                    dtype=np.float64,
                )

            host['mie_phase_probability_165_175_by_bin'] = np.maximum(
                _cdf_at_deg(175.0) - _cdf_at_deg(165.0), 0.0
            )
            host['mie_phase_probability_169_171_by_bin'] = np.maximum(
                _cdf_at_deg(171.0) - _cdf_at_deg(169.0), 0.0
            )
        else:
            host['mie_phase_probability_165_175_by_bin'] = np.zeros(n, dtype=np.float64)
            host['mie_phase_probability_169_171_by_bin'] = np.zeros(n, dtype=np.float64)
        if used_coupled_population_kernel:
            pop_host = cp.asnumpy(population_scalars)
        else:
            # Compatibility mapping from the unchanged aggregation kernel.
            # Fragmentation fields are zero because the coupled kernel was not active.
            pop_host = np.zeros(16, dtype=np.float64)
            pop_host[0] = float(agg_host[0])
            pop_host[1] = 0.0
            pop_host[2] = self.fragmentation_reference_diameter_m
            pop_host[3] = self.fragmentation_size_exponent
            pop_host[4] = float(agg_host[1])
            pop_host[5] = float(agg_host[2])
            pop_host[6] = float(agg_host[3])
            pop_host[7] = float(agg_host[4])
            pop_host[8] = float(agg_host[5])
            pop_host[9] = float(agg_host[6])
            pop_host[10] = float(agg_host[7])
            pop_host[11] = float(agg_host[8])
            pop_host[12] = 0.0
            pop_host[13] = 0.0
            pop_host[14] = 0.0
            pop_host[15] = float(agg_host[9])
        input_sw = float(np.sum(host['input_source_weight']))
        if input_sw > 0:
            host['input_source_weight'] = host['input_source_weight'] / input_sw
        host['effective_population_weight'] = host[
            'effective_aggregation_weight'
        ].copy()
        if self.particle_event_model in {
            'hybrid_mie_exact_rate',
            'hybrid_mie_exact_rate_moment_constrained',
        }:
            host['mu_effective_geometric_event_by_bin'] = np.maximum(
                host['mu_event_by_bin'] - host['mu_wave_by_bin'], 0.0
            )
            host['effective_geometric_event_cross_section_m2'] = np.divide(
                host['mu_effective_geometric_event_by_bin'],
                host['number_density_by_bin'],
                out=np.zeros(n, dtype=np.float64),
                where=host['number_density_by_bin'] > 0,
            )
            host['legacy_hybrid_mu_event_by_bin'] = (
                host['mu_geom_by_bin']
                + np.maximum(host['qsca'] - 1.0, 0.0) * host['mu_geom_by_bin']
            )
            host['legacy_hybrid_event_overcount_by_bin'] = (
                host['legacy_hybrid_mu_event_by_bin'] - host['mu_sca_by_bin']
            )
            host['qsca_lt1_mask'] = (host['qsca'] < 1.0).astype(np.float64)
        else:
            host['mu_effective_geometric_event_by_bin'] = host['mu_geom_by_bin'].copy()
            host['effective_geometric_event_cross_section_m2'] = host[
                'geometric_cross_section_m2'
            ].copy()
            host['legacy_hybrid_mu_event_by_bin'] = host['mu_event_by_bin'].copy()
            host['legacy_hybrid_event_overcount_by_bin'] = np.zeros(n, dtype=np.float64)
            host['qsca_lt1_mask'] = np.zeros(n, dtype=np.float64)
        host['moment_constraint_a_by_bin'] = cp.asnumpy(moment_a)
        host['moment_constraint_b_by_bin'] = cp.asnumpy(moment_b)
        host['moment_constraint_mode_by_bin'] = cp.asnumpy(moment_mode).astype(np.int32)
        host.update(
            {
                'mu_trace_geometric': float(mu_host[0]),
                'mu_trace_wave': float(mu_host[1]),
                'mu_geom': float(mu_host[2]),
                'mu_sca': float(mu_host[3]),
                'mu_ext': float(mu_host[4]),
                'mu_wave': float(mu_host[5]),
                'mu_event': float(mu_host[6]),
                'wave_coherent_fraction_nominal_cuda': float(mu_host[7]),
                'event_model': self.particle_event_model,
                'wave_scale': self.hybrid_wave_scale,
                'wave_surface_rms_height_m': self.wave_surface_rms_height_m,
                'aggregation_model': self.aggregation_model,
                'aggregation_collision_exposure': float(pop_host[0]),
                'fragmentation_breakup_exposure': float(pop_host[1]),
                'fragmentation_reference_diameter_m': float(pop_host[2]),
                'fragmentation_size_exponent': float(pop_host[3]),
                'aggregation_initial_number_density': float(pop_host[4]),
                'aggregation_final_number_density': float(pop_host[5]),
                'aggregation_number_reduction_fraction': float(pop_host[6]),
                'population_number_reduction_fraction': float(pop_host[6]),
                'aggregation_mass_conservation_error': float(pop_host[7]),
                'aggregation_initial_projected_area_per_m': float(pop_host[8]),
                'aggregation_final_projected_area_per_m': float(pop_host[9]),
                'aggregation_projected_area_ratio': (
                    float(pop_host[9] / pop_host[8]) if pop_host[8] > 0 else 1.0
                ),
                'aggregation_largest_bin_mass_fraction': float(pop_host[10]),
                'aggregation_overflow_event_fraction': float(pop_host[11]),
                'fragmentation_underflow_event_fraction': float(pop_host[12]),
                'aggregation_events_per_initial_particle': float(pop_host[13]),
                'fragmentation_events_per_initial_particle': float(pop_host[14]),
                'aggregation_steps': int(round(float(pop_host[15]))),
                'population_kernel_coupled_fragmentation_active': bool(
                    used_coupled_population_kernel
                ),
            }
        )
        sed_major_host = cp.asnumpy(sed_major)
        host.update(
            {
                'sediment_transport_model': self.sediment_transport_model,
                'sediment_eddy_diffusivity_m2_s': self.sediment_eddy_diffusivity,
                'sediment_water_density_kg_m3': self.water_density,
                'sediment_water_dynamic_viscosity_pa_s': self.water_dynamic_viscosity,
                'sediment_gravity_m_s2': self.gravity,
                'sediment_transport_quadrature_points': self.sediment_transport_quadrature_points,
                'sediment_radial_lookup_points': self.sediment_radial_lookup_points,
                'sediment_majorant_geom_per_m': float(sed_major_host[0]),
                'sediment_majorant_wave_per_m': float(sed_major_host[1]),
                'sediment_majorant_total_per_m': float(sed_major_host[2]),
                'sediment_max_radial_log_coefficient': float(sed_major_host[3]),
                'meridional_circulation_ratio': self.meridional_circulation_ratio,
                'meridional_vtheta_peak_m_s': self.meridional_vtheta_peak,
                'meridional_peak_speed_m_s': self.meridional_peak_speed,
                'meridional_grid_nr': int(grid_nr),
                'meridional_grid_nz': int(grid_nz),
                'meridional_grid_zmax_m': float(grid_zmax),
                'meridional_solver_iterations': int(solver_iterations),
                'meridional_solver_tolerance': self.meridional_solver_tolerance,
                'meridional_solver_max_residual': (
                    float(np.max(solver_residual)) if solver_residual.size else 0.0
                ),
                'meridional_solver_residual_by_bin': np.asarray(
                    solver_residual, dtype=np.float64
                ),
            }
        )
        return {
            'd': d,
            'w': w,
            'nidx': nidx,
            'kidx': kidx,
            'n': n,
            'device': {
                'radius': radius,
                'n': nd,
                'k': kd,
                'gcdf': gcdf,
                'wcdf': wcdf,
                'phase_cdf': phase_cdf,
                'theta': theta,
                'qsca': qsca,
                'moment_a': moment_a,
                'moment_b': moment_b,
                'moment_mode': moment_mode,
                'mu_channels': muscal,
                'trace_geom_bin': trace_geom_bin,
                'trace_wave_bin': trace_wave_bin,
                'sed_A': sed_A,
                'sed_B': sed_B,
                'sed_norm': sed_norm,
                'sed_radial_log': sed_radial_log,
                'sed_majorants': sed_major,
                'sed_grid': sed_grid,
                'sed_grid_nr': int(grid_nr),
                'sed_grid_nz': int(grid_nz),
                'sed_grid_zmax': float(grid_zmax),
            },
            'host': host,
            'ntheta': nt,
        }

    def single_particle_angular_audit(
        self,
        material,
        samples_per_bin=100000,
        seed=243500,
        concentration_g_per_L=1.0,
        weight_mode='mass_fraction',
    ):
        """Diagnostic-only Monte Carlo audit of the per-event angular kernel.

        The fixed incident direction is +y. For a spherical/isotropic kernel this loses no
        generality. Returned samples contain cos(theta) relative to the incident direction,
        event mechanism codes, status codes, and the CUDA-generated exact-Mie phase table.
        trace_kernel is not invoked.
        """
        cp = self.cp
        samples = int(samples_per_bin)
        if samples < 1:
            raise ValueError('samples_per_bin must be >=1')
        if samples > 2000000:
            raise ValueError('samples_per_bin must be <=2000000')
        prep = self._prepare_gpu_particle_physics(
            material, float(concentration_g_per_L), None, weight_mode
        )
        n = prep['n']
        dev = prep['device']
        total = n * samples
        ic = cp.empty(total, dtype=cp.float32)
        it = cp.zeros(total, dtype=cp.uint8)
        ist = cp.zeros(total, dtype=cp.uint8)
        pc = cp.empty(total, dtype=cp.float32)
        pt = cp.zeros(total, dtype=cp.uint8)
        pst = cp.zeros(total, dtype=cp.uint8)
        coh = np.float32(prep['host']['wave_coherent_fraction_nominal_cuda'])
        threads = 128
        blocks = (total + threads - 1) // threads
        self.kernels['single_particle_angular_audit_kernel'](
            (blocks,),
            (threads,),
            (
                np.int32(samples),
                np.int32(n),
                np.float32(self.n_water),
                np.float32(self.particle_wavelength),
                np.float32(self.particle_surface_beckmann_alpha),
                np.float32(self.wave_spheroid_aspect_ratio),
                np.int32(
                    self.SPHEROID_ORIENTATION_MODEL_CODES[
                        self.wave_spheroid_orientation_model
                    ]
                ),
                np.float32(self.wave_spheroid_orientation_kappa),
                np.int32(self.max_internal_bounces),
                dev['n'],
                dev['k'],
                dev['radius'],
                dev['qsca'],
                dev['phase_cdf'],
                dev['theta'],
                np.int32(prep['ntheta']),
                coh,
                np.uint32(seed),
                ic,
                it,
                ist,
                pc,
                pt,
                pst,
            ),
        )
        cp.cuda.Stream.null.synchronize()
        theta = cp.asnumpy(dev['theta'])
        cdf = cp.asnumpy(dev['phase_cdf']).reshape(n, prep['ntheta'])
        return {
            'material': material,
            'samples_per_bin': samples,
            'host': prep['host'],
            'theta': theta,
            'mie_cdf': cdf,
            'interface_cos': cp.asnumpy(ic).reshape(n, samples),
            'interface_type': cp.asnumpy(it).reshape(n, samples),
            'interface_status': cp.asnumpy(ist).reshape(n, samples),
            'production_cos': cp.asnumpy(pc).reshape(n, samples),
            'production_type': cp.asnumpy(pt).reshape(n, samples),
            'production_status': cp.asnumpy(pst).reshape(n, samples),
        }

    def single_particle_moment_constrained_audit(
        self,
        material,
        samples_per_bin=50000,
        seed=243600,
        concentration_g_per_L=1.0,
        weight_mode='mass_fraction',
    ):
        """Independent CUDA audit of the post-outcome angular map using the base sampler."""
        cp = self.cp
        samples = int(samples_per_bin)
        if self.particle_event_model != 'hybrid_mie_exact_rate_moment_constrained':
            raise RuntimeError(
                'constrained audit requires moment-constrained event model'
            )
        if samples < 1 or samples > 2000000:
            raise ValueError('samples_per_bin out of range')
        prep = self._prepare_gpu_particle_physics(
            material, float(concentration_g_per_L), None, weight_mode
        )
        n = prep['n']
        dev = prep['device']
        total = n * samples
        bc = cp.empty(total, dtype=cp.float32)
        cc = cp.empty(total, dtype=cp.float32)
        typ = cp.zeros(total, dtype=cp.uint8)
        st = cp.zeros(total, dtype=cp.uint8)
        coh = np.float32(prep['host']['wave_coherent_fraction_nominal_cuda'])
        threads = 128
        blocks = (total + threads - 1) // threads
        self.kernels['single_particle_moment_constrained_audit_kernel'](
            (blocks,),
            (threads,),
            (
                np.int32(samples),
                np.int32(n),
                np.float32(self.n_water),
                np.float32(self.particle_wavelength),
                np.float32(self.particle_surface_beckmann_alpha),
                np.float32(self.wave_spheroid_aspect_ratio),
                np.int32(
                    self.SPHEROID_ORIENTATION_MODEL_CODES[
                        self.wave_spheroid_orientation_model
                    ]
                ),
                np.float32(self.wave_spheroid_orientation_kappa),
                np.int32(self.max_internal_bounces),
                dev['n'],
                dev['k'],
                dev['radius'],
                dev['qsca'],
                dev['phase_cdf'],
                dev['theta'],
                np.int32(prep['ntheta']),
                coh,
                np.uint32(seed),
                dev['moment_a'],
                dev['moment_b'],
                dev['moment_mode'],
                bc,
                cc,
                typ,
                st,
            ),
        )
        cp.cuda.Stream.null.synchronize()
        theta = cp.asnumpy(dev['theta'])
        cdf = cp.asnumpy(dev['phase_cdf']).reshape(n, prep['ntheta'])
        return {
            'material': material,
            'samples_per_bin': samples,
            'host': prep['host'],
            'theta': theta,
            'mie_cdf': cdf,
            'base_cos': cp.asnumpy(bc).reshape(n, samples),
            'constrained_cos': cp.asnumpy(cc).reshape(n, samples),
            'event_type': cp.asnumpy(typ).reshape(n, samples),
            'status': cp.asnumpy(st).reshape(n, samples),
        }

    def particle_physics_diagnostics(
        self, material, concentration_g_per_L, weights=None, weight_mode='mass_fraction'
    ):
        """Run CUDA precompute and return diagnostic copies without tracing rays."""
        return self._prepare_gpu_particle_physics(
            material, concentration_g_per_L, weights, weight_mode
        )['host']

    def phase_table_diagnostics(
        self, material, concentration_g_per_L, weights=None, weight_mode='mass_fraction'
    ):
        """Diagnostic-only copy of the CUDA-generated Mie theta/CDF table."""
        prep = self._prepare_gpu_particle_physics(
            material, concentration_g_per_L, weights, weight_mode
        )
        cp = self.cp
        nt = prep['ntheta']
        n = prep['n']
        return cp.asnumpy(prep['device']['theta']), cp.asnumpy(
            prep['device']['phase_cdf']
        ).reshape(n, nt)

    def simulate(
        self,
        material,
        concentration_g_per_L,
        weights=None,
        n_rays=100000,
        seed=12345,
        weight_mode='mass_fraction',
        collect_rays=False,
        heatmap_size=0,
        light_transport_config=None,
        target_detector_score=None,
        min_rays=None,
        max_rays=None,
        min_rear_channel_score=None,
        min_h170_score=None,
        stability_l1_tolerance=None,
        stability_window=3,
        ray_id_start=0,
    ):
        cp = self.cp
        prep = self._prepare_gpu_particle_physics(
            material, concentration_g_per_L, weights, weight_mode
        )
        ph = prep['host']
        dev = prep['device']
        npart = prep['n']
        ntheta = prep['ntheta']
        requested = int(n_rays)
        adaptive = target_detector_score is not None
        ray_id_start = int(ray_id_start)
        if ray_id_start < 0 or ray_id_start + requested > 2**32:
            raise ValueError("ray_id_start range exceeds uint32 global IDs")
        if adaptive:
            target = float(target_detector_score)
            if target <= 0:
                raise ValueError('target_detector_score must be >0')
            rear_target = (
                None
                if min_rear_channel_score is None
                else float(min_rear_channel_score)
            )
            h170_target = None if min_h170_score is None else float(min_h170_score)
            if rear_target is not None and rear_target <= 0:
                raise ValueError('min_rear_channel_score must be >0')
            if h170_target is not None and h170_target <= 0:
                raise ValueError('min_h170_score must be >0')
            maxN = int(
                max_rays
                if max_rays is not None
                else max(requested, self.statistics_batch_rays)
            )
            minN = int(
                min_rays
                if min_rays is not None
                else min(self.statistics_batch_rays, maxN)
            )
            if minN < 0 or maxN < minN:
                raise ValueError('invalid min_rays/max_rays')
        else:
            target = None
            rear_target = None
            h170_target = None
            maxN = requested
            minN = requested
        if maxN < 1:
            raise ValueError('number of rays must be positive')
        if ray_id_start + maxN > 2**32:
            raise ValueError('global ray range exceeds uint32')
        vis = int(heatmap_size)
        hxy = cp.zeros(max(1, vis * vis), dtype=cp.float32)
        hzy = cp.zeros(max(1, vis * vis), dtype=cp.float32)
        ldc = dict(light_transport_config or {})
        light_enabled = bool(ldc.get('enabled', False))
        light_max_rays = int(ldc.get('max_rays', 0)) if light_enabled else 0
        light_nx = int(ldc.get('nx', 1)) if light_enabled else 1
        light_ny = int(ldc.get('ny', 1)) if light_enabled else 1
        light_nz = int(ldc.get('nz', 1)) if light_enabled else 1
        light_rep = (
            min(int(ldc.get('representative_rays', 0)), light_max_rays)
            if light_enabled
            else 0
        )
        light_max_segments = (
            int(ldc.get('max_segments_per_ray', 0)) if light_enabled else 0
        )
        if light_enabled:
            if light_max_rays < 1:
                raise ValueError('light diagnostic max_rays must be >=1')
            if min(light_nx, light_ny, light_nz) < 4:
                raise ValueError('light diagnostic grid dimensions must each be >=4')
            if light_rep < 0 or light_max_segments < 1:
                raise ValueError('invalid representative-ray diagnostic configuration')
        light_vg = vortex_surface_geometry(
            self.free_surface_model,
            self.zmean,
            self.rin,
            self.vortex_delta_h,
            self.vortex_core_radius,
            self.gravity,
        )
        light_zmax = (
            max(float(light_vg['center_z']), float(light_vg['wall_z']))
            if light_enabled
            else self.zmean
        )
        light_voxels = light_nx * light_ny * light_nz if light_enabled else 1
        # Four voxel fields plus bounded trajectories are explicit GPU allocations.
        light_bytes = (
            light_voxels * 16
            + max(light_rep * light_max_segments, 1) * (6 * 4 + 2 * 4)
            + max(light_rep, 1) * 8
            + 8
        )
        if light_enabled and light_bytes > 1_000_000_000:
            raise ValueError(
                f'light diagnostic allocation is too large ({light_bytes/1e9:.3g} GB); reduce grid dimensions, representative rays, or segments per ray'
            )
        light_diag_t0 = time.perf_counter() if light_enabled else None
        light_total = cp.zeros(light_voxels, dtype=cp.float32)
        light_post = cp.zeros(light_voxels, dtype=cp.float32)
        light_event_weight = cp.zeros(light_voxels, dtype=cp.float32)
        light_event_count = cp.zeros(light_voxels, dtype=cp.uint32)
        light_segment_total = cp.zeros(1, dtype=cp.float64)
        light_deposited_total = cp.zeros(1, dtype=cp.float64)
        light_segment_count = cp.zeros(1, dtype=cp.uint64)
        light_event_total = cp.zeros(1, dtype=cp.uint64)
        light_error_count = cp.zeros(1, dtype=cp.uint32)
        light_boundary_adjustment_count = cp.zeros(1, dtype=cp.uint32)
        light_event_outside_count = cp.zeros(32, dtype=cp.uint32)
        light_traj_size = max(light_rep * light_max_segments, 1)
        light_traj_float = [
            cp.full(light_traj_size, cp.nan, dtype=cp.float32) for _ in range(6)
        ]
        light_torder = cp.full(light_traj_size, -1, dtype=cp.int32)
        light_tevent = cp.full(light_traj_size, -1, dtype=cp.int32)
        light_tcount = cp.zeros(max(light_rep, 1), dtype=cp.int32)
        light_toverflow = cp.zeros(max(light_rep, 1), dtype=cp.int32)
        if light_enabled:
            print(
                f'  Light-history CUDA buffers: {light_nx}x{light_ny}x{light_nz}; {light_bytes/1048576:.1f} MiB; {max(0,min(maxN,light_max_rays-ray_id_start)):,} diagnostic rays; global ID start {ray_id_start:,}'
            )
        seed0 = np.uint32((int(seed) * 1664525 + 1013904223) & 0x7FFFFFFF or 1)
        seed1 = np.uint32((int(seed) * 22695477 + 1) & 0x7FFFFFFF or 7)
        seed2 = np.uint32((int(seed) * 1103515245 + 12345) & 0x7FFFFFFF or 11)
        threads = 256
        native_total = np.zeros(18, dtype=np.int64)
        mirror_total = np.zeros(18, dtype=np.int64)
        sym_total = np.zeros(18, dtype=np.float64)
        native_direct_total = np.zeros(18, dtype=np.int64)
        mirror_direct_total = np.zeros(18, dtype=np.int64)
        native_stray_total = np.zeros(18, dtype=np.int64)
        mirror_stray_total = np.zeros(18, dtype=np.int64)
        native_face_total = np.zeros(18, dtype=np.int64)
        mirror_face_total = np.zeros(18, dtype=np.int64)
        photodiode_native_total = np.zeros(18, dtype=np.float64)
        photodiode_mirror_total = np.zeros(18, dtype=np.float64)
        incidence_native_sum_total = np.zeros(18, dtype=np.float64)
        incidence_mirror_sum_total = np.zeros(18, dtype=np.float64)
        legacy_native_total = np.zeros(18, dtype=np.int64)
        legacy_mirror_total = np.zeros(18, dtype=np.int64)
        legacy_sym_total = np.zeros(18, dtype=np.float64)
        batch_scores = []
        hardware_batch_scores = []
        photodiode_batch_scores = []
        legacy_batch_scores = []
        batch_rows = []
        prev_cum = None
        recent_l1 = []
        total_rays = 0
        valid_air_total = 0
        numerical_failure_total = 0
        entered_total = 0
        source_accept_total = 0
        source_direct_accept_total = 0
        source_reflected_accept_total = 0
        source_bore_interactions_total = 0
        source_bore_reflections_total = 0
        axial_total = 0
        top_exit_total = 0
        bottom_exit_total = 0
        sum_ic = sum_gic = sum_wic = sum_wcoh = sum_wmorph = 0.0
        ballistic_total = 0
        ic_hist = np.zeros(1, dtype=np.int64)
        sum_pr = 0.0
        rays_entry_pr = rays_internal_pr = particle_absorbed_total = 0
        sum_cir = sum_cor = 0.0
        rays_ctir = 0
        sum_top_ref = sum_bottom_inner = sum_bottom_outer = 0.0
        rays_top_tir = rays_bottom_tir = 0
        # H170 and event-history partitions use GPU-emitted counters without
        # changing ray probabilities; interaction-order and branch closure is checked.
        path_keys = [
            'any_particle',
            'any_geometric',
            'any_wave',
            'any_coherent_wave',
            'any_morphology_wave',
            'any_entry_reflection',
            'any_internal_reflection',
            'ballistic',
            'single_interaction',
            'single_entry_reflection',
            'multiple_interaction',
            'interaction_order_2',
            'interaction_order_3',
            'interaction_order_4plus',
            'exactly_one_geometric',
            'exactly_one_wave',
            'single_coherent_wave_escape',
            'single_external_entry_reflection_escape',
            'single_internal_reflection_escape',
            'single_surface_transmission_escape',
            'mixed_geometric_wave',
            'geometric_only_multiple',
            'wave_only_multiple',
            'multiple_with_entry_reflection',
            'multiple_with_internal_reflection',
        ]
        path_scores = {k: np.zeros(18, dtype=np.float64) for k in path_keys}
        path_interaction_sums = np.zeros(18, dtype=np.float64)
        path_geometric_interaction_sums = np.zeros(18, dtype=np.float64)
        path_wave_interaction_sums = np.zeros(18, dtype=np.float64)
        path_coherent_wave_interaction_sums = np.zeros(18, dtype=np.float64)
        path_morphology_wave_interaction_sums = np.zeros(18, dtype=np.float64)
        path_entry_reflection_sums = np.zeros(18, dtype=np.float64)
        path_internal_reflection_sums = np.zeros(18, dtype=np.float64)
        # Accumulate causal final-event and path-length diagnostics online
        # without modifying ray weights or transport decisions.
        causal_event_codes = tuple(
            sorted(k for k in PARTICLE_EVENT_TYPE_NAMES if k > 0)
        )
        causal_last_event_scores = {
            code: np.zeros(18, dtype=np.float64) for code in causal_event_codes
        }
        causal_first_event_scores = {
            code: np.zeros(18, dtype=np.float64) for code in causal_event_codes
        }
        causal_last_event_angle_sums = {
            code: np.zeros(18, dtype=np.float64) for code in causal_event_codes
        }
        causal_last_event_angle_counts = {
            code: np.zeros(18, dtype=np.float64) for code in causal_event_codes
        }
        causal_first_backscatter_scores = np.zeros(18, dtype=np.float64)
        causal_first_source_hemi_scores = np.zeros(18, dtype=np.float64)
        causal_first_angle_sums = np.zeros(18, dtype=np.float64)
        causal_first_angle_counts = np.zeros(18, dtype=np.float64)
        causal_last_angle_sums = np.zeros(18, dtype=np.float64)
        causal_last_angle_counts = np.zeros(18, dtype=np.float64)
        causal_internal_path_sums = np.zeros(18, dtype=np.float64)
        causal_internal_opl_sums = np.zeros(18, dtype=np.float64)
        causal_post_last_water_path_sums = np.zeros(18, dtype=np.float64)
        causal_path_counts = np.zeros(18, dtype=np.float64)
        causal_last_order_sums = np.zeros(18, dtype=np.float64)
        causal_h170_angle_edges = np.asarray(
            [0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0001], dtype=np.float64
        )
        causal_h170_last_angle_hist = np.zeros((8, 6), dtype=np.float64)
        causal_h170_first_angle_hist = np.zeros(6, dtype=np.float64)
        # exact-order, PSD provenance, and acrylic-assisted causal partitions.
        # Order rows are [0,1,2,3,4,5+] and are mutually exclusive by construction.
        exact_particle_order_scores = np.zeros((6, 18), dtype=np.float64)
        exact_geometric_order_scores = np.zeros((6, 18), dtype=np.float64)
        first_particle_bin_scores = np.zeros((npart, 18), dtype=np.float64)
        last_particle_bin_scores = np.zeros((npart, 18), dtype=np.float64)
        first_geometric_bin_scores = np.zeros((npart, 18), dtype=np.float64)
        last_geometric_bin_scores = np.zeros((npart, 18), dtype=np.float64)
        first_wave_bin_scores = np.zeros((npart, 18), dtype=np.float64)
        last_wave_bin_scores = np.zeros((npart, 18), dtype=np.float64)
        wall_path_scores = {
            k: np.zeros(18, dtype=np.float64)
            for k in (
                'zero_particle',
                'particle_no_acrylic_return',
                'particle_with_acrylic_return',
            )
        }
        particle_after_acrylic_return_scores = np.zeros(18, dtype=np.float64)
        acrylic_reflection_order_scores = np.zeros(
            (5, 18), dtype=np.float64
        )  # [0,1,2,3,4+]
        # zero-RNG angular-diffusion / sequence / spatial diagnostics.
        history_mean_metric_names = (
            'cumulative_particle_deflection_deg',
            'cumulative_particle_deflection_sq_deg2',
            'max_particle_deflection_deg',
            'first_particle_r_m',
            'first_particle_z_m',
            'last_particle_r_m',
            'last_particle_z_m',
            'particle_event_mean_r_m',
            'particle_event_mean_z_m',
            'particle_event_min_r_m',
            'particle_event_max_r_m',
            'particle_event_min_z_m',
            'particle_event_max_z_m',
            'water_path_at_first_particle_event_m',
            'water_path_at_last_particle_event_m',
            'particle_water_path_span_first_to_last_m',
            'particle_radial_span_m',
            'particle_axial_span_m',
            'particle_branch_transition_count',
            'particle_event_type_transition_count',
            'particle_small_deflection_count',
            'particle_medium_deflection_count',
            'particle_large_deflection_count',
        )
        history_metric_sums = {
            k: np.zeros(18, dtype=np.float64) for k in history_mean_metric_names
        }
        history_metric_counts = {
            k: np.zeros(18, dtype=np.float64) for k in history_mean_metric_names
        }
        history_deflection_history_scores = {
            k: np.zeros(18, dtype=np.float64)
            for k in ('zero_particle', 'all_lt30', 'has_30_90_no_ge90', 'has_ge90')
        }
        history_branch_sequence_scores = {
            k: np.zeros(18, dtype=np.float64)
            for k in (
                'zero_particle',
                'geometric_only',
                'wave_only',
                'mixed_g_to_g',
                'mixed_g_to_w',
                'mixed_w_to_g',
                'mixed_w_to_w',
            )
        }
        history_branch_transition_scores = np.zeros(
            (6, 18), dtype=np.float64
        )  # [0,1,2,3,4,5+], particle histories only
        history_event_transition_scores = np.zeros((6, 18), dtype=np.float64)
        history_h170_cumulative_deflection_edges = np.asarray(
            [
                0.0,
                30.0,
                60.0,
                90.0,
                120.0,
                180.0,
                240.0,
                360.0,
                540.0,
                720.0,
                1080.0,
                1800.0,
                3600.0,
                1.0e9,
            ],
            dtype=np.float64,
        )
        history_h170_cumulative_deflection_hist = np.zeros(
            len(history_h170_cumulative_deflection_edges) - 1, dtype=np.float64
        )
        # Spatial bins cover the actual water domain rather than the much taller acrylic wall.
        # The vortex wall/centre extrema bracket the free surface for every supported surface model.
        history_surface = vortex_surface_geometry(
            self.free_surface_model,
            self.zmean,
            self.rin,
            self.vortex_delta_h,
            self.vortex_core_radius,
            self.gravity,
        )
        history_water_ztop = (
            max(float(history_surface['wall_z']), float(history_surface['center_z']))
            + 1.0e-9
        )
        history_r_edges_m = np.linspace(
            0.0, float(self.rin) + 1.0e-9, 6, dtype=np.float64
        )
        history_z_edges_m = np.linspace(
            float(self.zmin), history_water_ztop, 6, dtype=np.float64
        )
        history_h170_first_spatial = np.zeros(
            (len(history_r_edges_m) - 1, len(history_z_edges_m) - 1), dtype=np.float64
        )
        history_h170_last_spatial = np.zeros_like(history_h170_first_spatial)
        # max-deflection-event provenance. These are mutually exclusive score partitions.
        deflection_max_event_type_scores = {
            code: np.zeros(18, dtype=np.float64) for code in causal_event_codes
        }
        deflection_max_event_order_scores = np.zeros(
            (5, 18), dtype=np.float64
        )  # 1,2,3,4,5+
        deflection_max_event_bin_scores = np.zeros((npart, 18), dtype=np.float64)
        deflection_h170_max_spatial = np.zeros_like(history_h170_first_spatial)
        # finite-particle drill-down for the maximum-deflection event.
        # These diagnostics consume no RNG and do not feed any transport decision.
        internal_path_max_macro_bounce_scores = np.zeros(
            (5, 18), dtype=np.float64
        )  # 0,1,2,3,4+
        internal_path_max_terminal_micro_order_scores = np.zeros(
            (6, 18), dtype=np.float64
        )  # 0,1,2,3,4,5+
        internal_path_max_length_ratio_sums = np.zeros(18, dtype=np.float64)
        internal_path_max_macro_sums = np.zeros(18, dtype=np.float64)
        internal_path_max_micro_sums = np.zeros(18, dtype=np.float64)
        internal_path_h170_length_ratio_edges = np.asarray(
            [0.0, 1.0e-12, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0, 1.0e9],
            dtype=np.float64,
        )
        internal_path_h170_length_ratio_hist = np.zeros(
            len(internal_path_h170_length_ratio_edges) - 1, dtype=np.float64
        )
        internal_path_h170_type_by_macro = np.zeros(
            (8, 5), dtype=np.float64
        )  # event codes 1..7 x macro classes 0,1,2,3,4+
        # zero-RNG maximum-turn size/branch/interface-angle correlation.
        # Angle class 0 is "undefined" (coherent Mie has no finite-particle interface);
        # classes 1..6 are 0-15,15-30,30-45,45-60,60-75,75-90 degrees.
        interface_angle_edges = np.asarray(
            [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0001], dtype=np.float64
        )
        interface_angle_labels = [
            'undefined',
            '0-15',
            '15-30',
            '30-45',
            '45-60',
            '60-75',
            '75-90',
        ]
        interface_h170_incidence_hist = np.zeros(7, dtype=np.float64)
        interface_h170_exit_hist = np.zeros(7, dtype=np.float64)
        interface_h170_type_by_psd = np.zeros((8, npart), dtype=np.float64)
        interface_h170_type_by_incidence = np.zeros((8, 7), dtype=np.float64)
        interface_h170_type_by_exit = np.zeros((8, 7), dtype=np.float64)
        interface_h170_psd_by_incidence = np.zeros((npart, 7), dtype=np.float64)
        interface_h170_psd_by_exit = np.zeros((npart, 7), dtype=np.float64)
        interface_incidence_sums = np.zeros(18, dtype=np.float64)
        interface_incidence_defined_scores = np.zeros(18, dtype=np.float64)
        interface_exit_sums = np.zeros(18, dtype=np.float64)
        interface_exit_defined_scores = np.zeros(18, dtype=np.float64)
        ray_chunks = [] if collect_rays else None
        stop_reason = 'fixed_n_rays'
        production_failure_code_counts = np.zeros(
            max(PRODUCTION_FAILURE_NAMES) + 1, dtype=np.int64
        )
        production_failure_subcode_counts = {}
        production_failure_samples = []
        production_failure_sample_limit = 512
        particle_proposal_total = 0
        particle_accessible_total = 0
        particle_null_total = 0
        acrylic_trapped_tir_total = 0
        while total_rays < maxN:
            batchN = min(self.statistics_batch_rays, maxN - total_rays)
            batch_native = np.zeros(18, dtype=np.int64)
            batch_mirror = np.zeros(18, dtype=np.int64)
            batch_native_direct = np.zeros(18, dtype=np.int64)
            batch_mirror_direct = np.zeros(18, dtype=np.int64)
            batch_native_stray = np.zeros(18, dtype=np.int64)
            batch_mirror_stray = np.zeros(18, dtype=np.int64)
            batch_native_face = np.zeros(18, dtype=np.int64)
            batch_mirror_face = np.zeros(18, dtype=np.int64)
            batch_photodiode_native = np.zeros(18, dtype=np.float64)
            batch_photodiode_mirror = np.zeros(18, dtype=np.float64)
            batch_incidence_native_sum = np.zeros(18, dtype=np.float64)
            batch_incidence_mirror_sum = np.zeros(18, dtype=np.float64)
            batch_legacy_native = np.zeros(18, dtype=np.int64)
            batch_legacy_mirror = np.zeros(18, dtype=np.int64)
            batch_start = total_rays
            for local_start in range(0, batchN, self.chunk_size):
                local_end = min(batchN, local_start + self.chunk_size)
                n = local_end - local_start
                global_start = ray_id_start + batch_start + local_start
                outf = [cp.full(n, cp.nan, dtype=cp.float32) for _ in range(71)]
                outi = [cp.zeros(n, dtype=cp.int32) for _ in range(58)]
                blocks = (n + threads - 1) // threads
                self.kernel(
                    (blocks,),
                    (threads,),
                    (
                        np.float32(self.max_iterations),
                        dev['mu_channels'],
                        np.float32(self.n_water),
                        np.float32(self.n_acrylic),
                        np.float32(self.n_air),
                        np.float32(self.particle_wavelength),
                        np.float32(self.particle_surface_beckmann_alpha),
                        np.float32(self.wave_spheroid_aspect_ratio),
                        np.int32(
                            self.SPHEROID_ORIENTATION_MODEL_CODES[
                                self.wave_spheroid_orientation_model
                            ]
                        ),
                        np.float32(self.wave_spheroid_orientation_kappa),
                        np.int32(self.max_internal_bounces),
                        np.int32(self.max_cell_bounces),
                        np.float32(self.rin),
                        np.float32(self.rout),
                        np.float32(self.zmin),
                        np.float32(self.zmean),
                        np.float32(self.zwall_top),
                        np.int32(
                            self.FREE_SURFACE_MODEL_CODES[self.free_surface_model]
                        ),
                        np.float32(self.vortex_delta_h),
                        np.float32(self.vortex_core_radius),
                        np.float32(self.bottom_disc_thickness),
                        np.float32(self.bottom_external_n),
                        np.int32(self.max_axial_bounces),
                        np.float32(self.ring_in),
                        np.float32(self.throat_out),
                        np.float32(self.ring_out),
                        np.float32(self.source_launch_radius),
                        np.float32(self.throat_radius),
                        np.float32(self.counterbore_radius),
                        np.float32(self.beam_sigma),
                        np.int32(vis),
                        np.float32(self.alpha1),
                        np.float32(self.alpha2),
                        np.int32(
                            self.SOURCE_ANGULAR_MODEL_CODES[self.source_angular_model]
                        ),
                        np.int32(
                            self.BORE_SURFACE_MODEL_CODES[
                                self.source_bore_surface_model
                            ]
                        ),
                        np.float32(self.bore_refractive_index),
                        np.int32(self.max_bore_bounces),
                        np.int32(n),
                        dev['gcdf'],
                        dev['wcdf'],
                        dev['phase_cdf'],
                        dev['theta'],
                        np.int32(ntheta),
                        dev['n'],
                        dev['k'],
                        dev['radius'],
                        np.int32(npart),
                        dev['moment_a'],
                        dev['moment_b'],
                        dev['moment_mode'],
                        np.int32(
                            self.SEDIMENT_TRANSPORT_MODEL_CODES[
                                self.sediment_transport_model
                            ]
                        ),
                        dev['trace_geom_bin'],
                        dev['trace_wave_bin'],
                        dev['sed_A'],
                        dev['sed_B'],
                        dev['sed_norm'],
                        dev['sed_radial_log'],
                        np.int32(self.sediment_radial_lookup_points),
                        dev['sed_majorants'],
                        dev['sed_grid'],
                        np.int32(dev['sed_grid_nr']),
                        np.int32(dev['sed_grid_nz']),
                        np.float32(dev['sed_grid_zmax']),
                        hxy,
                        hzy,
                        np.int32(1 if light_enabled else 0),
                        np.uint32(light_max_rays),
                        np.int32(light_nx),
                        np.int32(light_ny),
                        np.int32(light_nz),
                        np.float32(light_zmax),
                        light_total,
                        light_post,
                        light_event_weight,
                        light_event_count,
                        light_segment_total,
                        light_deposited_total,
                        light_segment_count,
                        light_event_total,
                        light_error_count,
                        light_boundary_adjustment_count,
                        light_event_outside_count,
                        np.int32(light_rep),
                        np.int32(light_max_segments),
                        *light_traj_float,
                        light_torder,
                        light_tevent,
                        light_tcount,
                        light_toverflow,
                        *outf,
                        *outi,
                        seed0,
                        seed1,
                        seed2,
                        np.uint32(global_start),
                    ),
                )
                # Detector geometry/scoring is CUDA-resident. Only counts/indices are copied back.
                hnat = cp.zeros(18, dtype=cp.int32)
                hmir = cp.zeros(18, dtype=cp.int32)
                hnd = cp.zeros(18, dtype=cp.int32)
                hmd = cp.zeros(18, dtype=cp.int32)
                hns = cp.zeros(18, dtype=cp.int32)
                hms = cp.zeros(18, dtype=cp.int32)
                hnf = cp.zeros(18, dtype=cp.int32)
                hmf = cp.zeros(18, dtype=cp.int32)
                pdnat = cp.zeros(18, dtype=cp.float32)
                pdmir = cp.zeros(18, dtype=cp.float32)
                iangn = cp.zeros(18, dtype=cp.float32)
                iangm = cp.zeros(18, dtype=cp.float32)
                lnat = cp.zeros(18, dtype=cp.int32)
                lmir = cp.zeros(18, dtype=cp.int32)
                nid = cp.empty(n, dtype=cp.int32)
                mid = cp.empty(n, dtype=cp.int32)
                nbc = cp.empty(n, dtype=cp.int32)
                mbc = cp.empty(n, dtype=cp.int32)
                nfc = cp.empty(n, dtype=cp.int32)
                mfc = cp.empty(n, dtype=cp.int32)
                detector_seed = np.uint32(
                    (int(seed) * 2654435761 + 97531) & 0x7FFFFFFF or 17
                )
                self.kernels['score_detectors_kernel'](
                    (blocks,),
                    (threads,),
                    (
                        outf[0],
                        outf[1],
                        outf[2],
                        outf[3],
                        outf[4],
                        outf[5],
                        outi[20],
                        outi[0],
                        np.int32(n),
                        np.float32(self.rout),
                        np.float32(self.ring_in),
                        np.float32(self.throat_out),
                        np.float32(self.ring_out),
                        np.float32(self.throat_radius),
                        np.float32(self.counterbore_radius),
                        np.float32(self.legacy_acceptance_deg),
                        np.int32(
                            self.BORE_SURFACE_MODEL_CODES[
                                self.detector_bore_surface_model
                            ]
                        ),
                        np.float32(self.n_air),
                        np.float32(self.bore_refractive_index),
                        np.int32(self.max_bore_bounces),
                        np.int32(
                            self.RING_FACE_SURFACE_MODEL_CODES[
                                self.ring_inner_face_surface_model
                            ]
                        ),
                        np.float32(self.ring_inner_face_refractive_index),
                        np.float32(self.n_acrylic),
                        np.int32(self.max_ring_face_bounces),
                        np.int32(
                            self.PHOTODIODE_RESPONSE_MODEL_CODES[
                                self.photodiode_response_model
                            ]
                        ),
                        detector_seed,
                        np.uint32(global_start),
                        hnat,
                        hmir,
                        hnd,
                        hmd,
                        hns,
                        hms,
                        hnf,
                        hmf,
                        pdnat,
                        pdmir,
                        iangn,
                        iangm,
                        lnat,
                        lmir,
                        nid,
                        mid,
                        nbc,
                        mbc,
                        nfc,
                        mfc,
                    ),
                )
                cp.cuda.Stream.null.synchronize()
                batch_native += cp.asnumpy(hnat).astype(np.int64)
                batch_mirror += cp.asnumpy(hmir).astype(np.int64)
                batch_native_direct += cp.asnumpy(hnd).astype(np.int64)
                batch_mirror_direct += cp.asnumpy(hmd).astype(np.int64)
                batch_native_stray += cp.asnumpy(hns).astype(np.int64)
                batch_mirror_stray += cp.asnumpy(hms).astype(np.int64)
                batch_native_face += cp.asnumpy(hnf).astype(np.int64)
                batch_mirror_face += cp.asnumpy(hmf).astype(np.int64)
                batch_photodiode_native += cp.asnumpy(pdnat).astype(np.float64)
                batch_photodiode_mirror += cp.asnumpy(pdmir).astype(np.float64)
                batch_incidence_native_sum += cp.asnumpy(iangn).astype(np.float64)
                batch_incidence_mirror_sum += cp.asnumpy(iangm).astype(np.float64)
                batch_legacy_native += cp.asnumpy(lnat).astype(np.int64)
                batch_legacy_mirror += cp.asnumpy(lmir).astype(np.int64)
                af = [cp.asnumpy(a) for a in outf]
                ai = [cp.asnumpy(a) for a in outi]
                arrays = dict(
                    zip(
                        [
                            'air_x',
                            'air_y',
                            'air_z',
                            'air_vx',
                            'air_vy',
                            'air_vz',
                            'water_path',
                            'acrylic_path',
                            'first_particle_deflection_deg',
                            'last_particle_deflection_deg',
                            'internal_particle_path_total_m',
                            'internal_particle_opl_total_m',
                            'water_path_after_last_particle_event_m',
                            'cumulative_particle_deflection_deg',
                            'cumulative_particle_deflection_sq_deg2',
                            'max_particle_deflection_deg',
                            'max_deflection_event_x',
                            'max_deflection_event_y',
                            'max_deflection_event_z',
                            'first_particle_x',
                            'first_particle_y',
                            'first_particle_z',
                            'last_particle_x',
                            'last_particle_y',
                            'last_particle_z',
                            'particle_event_mean_r_m',
                            'particle_event_mean_z_m',
                            'particle_event_min_r_m',
                            'particle_event_max_r_m',
                            'particle_event_min_z_m',
                            'particle_event_max_z_m',
                            'water_path_at_first_particle_event_m',
                            'water_path_at_last_particle_event_m',
                            'failure_x',
                            'failure_y',
                            'failure_z',
                            'failure_vx',
                            'failure_vy',
                            'failure_vz',
                            'particle_diag_radius',
                            'particle_diag_rho',
                            'particle_diag_phi',
                            'particle_diag_cx',
                            'particle_diag_cy',
                            'particle_diag_cz',
                            'particle_diag_entry_x',
                            'particle_diag_entry_y',
                            'particle_diag_entry_z',
                            'particle_diag_exit_x',
                            'particle_diag_exit_y',
                            'particle_diag_exit_z',
                            'particle_diag_pre_x',
                            'particle_diag_pre_y',
                            'particle_diag_pre_z',
                            'particle_diag_pre_vx',
                            'particle_diag_pre_vy',
                            'particle_diag_pre_vz',
                            'particle_diag_post_x',
                            'particle_diag_post_y',
                            'particle_diag_post_z',
                            'particle_diag_post_vx',
                            'particle_diag_post_vy',
                            'particle_diag_post_vz',
                            'particle_diag_center_radial_clearance',
                            'particle_diag_finite_wall_clearance',
                            'particle_diag_bottom_clearance',
                            'particle_diag_top_clearance',
                            'max_deflection_event_internal_path_m',
                            'max_deflection_event_internal_path_over_diameter',
                            'max_deflection_event_incidence_deg',
                            'max_deflection_event_exit_deg',
                        ],
                        af,
                    )
                )
                ints = dict(
                    zip(
                        [
                            'interaction_count',
                            'geometric_interactions',
                            'wave_interactions',
                            'coherent_wave_interactions',
                            'morphology_wave_interactions',
                            'particle_reflections',
                            'particle_entry_reflections',
                            'particle_internal_reflections',
                            'particle_absorptions',
                            'cell_inner_reflections',
                            'cell_outer_reflections',
                            'cell_tir',
                            'top_reflections',
                            'top_tir',
                            'bottom_inner_reflections',
                            'bottom_outer_reflections',
                            'bottom_tir',
                            'source_bore_interactions',
                            'source_bore_reflections',
                            'entered_water',
                            'status',
                            'production_failure_code',
                            'production_failure_subcode',
                            'production_failure_trace_iteration',
                            'production_failure_macro_bounces',
                            'production_failure_micro_order',
                            'production_failure_medium',
                            'production_failure_event_channel',
                            'particle_diag_bin',
                            'particle_diag_previous_event',
                            'particle_diag_overlap_class',
                            'particle_diag_proposal_accepted',
                            'particle_proposal_count',
                            'particle_accessible_count',
                            'particle_null_count',
                            'first_particle_event_type',
                            'last_particle_event_type',
                            'last_particle_event_order',
                            'first_particle_backscatter',
                            'first_particle_source_hemisphere',
                            'first_particle_bin',
                            'last_particle_bin',
                            'first_geometric_bin',
                            'last_geometric_bin',
                            'first_wave_bin',
                            'last_wave_bin',
                            'acrylic_return_after_particle_count',
                            'particle_after_acrylic_return_count',
                            'particle_branch_transition_count',
                            'particle_event_type_transition_count',
                            'particle_small_deflection_count',
                            'particle_medium_deflection_count',
                            'particle_large_deflection_count',
                            'max_deflection_event_type',
                            'max_deflection_event_order',
                            'max_deflection_event_bin',
                            'max_deflection_event_macro_bounces',
                            'max_deflection_event_terminal_micro_order',
                        ],
                        ai,
                    )
                )
                particle_proposal_total += int(
                    np.asarray(ints['particle_proposal_count'], dtype=np.int64).sum()
                )
                particle_accessible_total += int(
                    np.asarray(ints['particle_accessible_count'], dtype=np.int64).sum()
                )
                particle_null_total += int(
                    np.asarray(ints['particle_null_count'], dtype=np.int64).sum()
                )
                nid_h = cp.asnumpy(nid)
                mid_h = cp.asnumpy(mid)

                def _det_hist(ids, mask, weights=None):
                    vv = (ids >= 0) & (ids < 18) & mask
                    if not np.any(vv):
                        return np.zeros(18, dtype=np.float64)
                    ww = (
                        None
                        if weights is None
                        else np.asarray(weights, dtype=np.float64)[vv]
                    )
                    return np.bincount(
                        ids[vv].astype(np.int64), weights=ww, minlength=18
                    ).astype(np.float64)

                ic_all = ints['interaction_count']
                per_all = ints['particle_entry_reflections']
                pir_all = ints['particle_internal_reflections']
                conds = _event_history_masks(ints)
                for pk, pmask in conds.items():
                    nn = _det_hist(nid_h, pmask)
                    mm = _det_hist(mid_h, pmask)
                    path_scores[pk] += (
                        0.5 * (nn + mm) if self.symmetry_average_exact else nn
                    )
                anymask = np.ones(n, dtype=bool)
                for vals, accumulator in [
                    (ic_all, path_interaction_sums),
                    (ints['geometric_interactions'], path_geometric_interaction_sums),
                    (ints['wave_interactions'], path_wave_interaction_sums),
                    (
                        ints['coherent_wave_interactions'],
                        path_coherent_wave_interaction_sums,
                    ),
                    (
                        ints['morphology_wave_interactions'],
                        path_morphology_wave_interaction_sums,
                    ),
                    (per_all, path_entry_reflection_sums),
                    (pir_all, path_internal_reflection_sums),
                ]:
                    nn = _det_hist(nid_h, anymask, vals)
                    mm = _det_hist(mid_h, anymask, vals)
                    accumulator += (
                        0.5 * (nn + mm) if self.symmetry_average_exact else nn
                    )
                # Causal aggregation. Only successfully detected rays contribute;
                # native/mirror symmetry weighting is identical to the production estimator.
                first_type = ints['first_particle_event_type']
                last_type = ints['last_particle_event_type']
                first_angle = arrays['first_particle_deflection_deg']
                last_angle = arrays['last_particle_deflection_deg']
                valid_particle = last_type > 0

                def _sym_hist(mask, weights=None):
                    nn = _det_hist(nid_h, mask, weights)
                    mm = _det_hist(mid_h, mask, weights)
                    return 0.5 * (nn + mm) if self.symmetry_average_exact else nn

                # Exact interaction-order partitions, including ballistic order zero.
                for oi in range(6):
                    om = (ic_all == oi) if oi < 5 else (ic_all >= 5)
                    gm = (
                        (ints['geometric_interactions'] == oi)
                        if oi < 5
                        else (ints['geometric_interactions'] >= 5)
                    )
                    exact_particle_order_scores[oi] += _sym_hist(om)
                    exact_geometric_order_scores[oi] += _sym_hist(gm)
                # First/last PSD-bin attribution. Last-bin attribution is the particle event immediately
                # preceding the detected exit and is therefore the primary high-angle redirection provenance.
                for bi in range(npart):
                    first_particle_bin_scores[bi] += _sym_hist(
                        ints['first_particle_bin'] == bi
                    )
                    last_particle_bin_scores[bi] += _sym_hist(
                        ints['last_particle_bin'] == bi
                    )
                    first_geometric_bin_scores[bi] += _sym_hist(
                        ints['first_geometric_bin'] == bi
                    )
                    last_geometric_bin_scores[bi] += _sym_hist(
                        ints['last_geometric_bin'] == bi
                    )
                    first_wave_bin_scores[bi] += _sym_hist(ints['first_wave_bin'] == bi)
                    last_wave_bin_scores[bi] += _sym_hist(ints['last_wave_bin'] == bi)
                acr_ret = ints['acrylic_return_after_particle_count'] > 0
                wall_path_scores['zero_particle'] += _sym_hist(ic_all == 0)
                wall_path_scores['particle_no_acrylic_return'] += _sym_hist(
                    (ic_all > 0) & (~acr_ret)
                )
                wall_path_scores['particle_with_acrylic_return'] += _sym_hist(
                    (ic_all > 0) & acr_ret
                )
                particle_after_acrylic_return_scores += _sym_hist(
                    ints['particle_after_acrylic_return_count'] > 0
                )
                acr_n = (
                    ints['cell_inner_reflections']
                    + ints['cell_outer_reflections']
                    + ints['bottom_inner_reflections']
                    + ints['bottom_outer_reflections']
                )
                for ai_ in range(5):
                    am_ = (acr_n == ai_) if ai_ < 4 else (acr_n >= 4)
                    acrylic_reflection_order_scores[ai_] += _sym_hist(am_)
                for code in causal_event_codes:
                    lm = last_type == code
                    fm = first_type == code
                    causal_last_event_scores[code] += _sym_hist(lm)
                    causal_first_event_scores[code] += _sym_hist(fm)
                    lav = lm & np.isfinite(last_angle) & (last_angle >= 0.0)
                    causal_last_event_angle_sums[code] += _sym_hist(lav, last_angle)
                    causal_last_event_angle_counts[code] += _sym_hist(lav)
                causal_first_backscatter_scores += _sym_hist(
                    valid_particle & (ints['first_particle_backscatter'] > 0)
                )
                causal_first_source_hemi_scores += _sym_hist(
                    valid_particle & (ints['first_particle_source_hemisphere'] > 0)
                )
                fav = valid_particle & np.isfinite(first_angle) & (first_angle >= 0.0)
                lav = valid_particle & np.isfinite(last_angle) & (last_angle >= 0.0)
                causal_first_angle_sums += _sym_hist(fav, first_angle)
                causal_first_angle_counts += _sym_hist(fav)
                causal_last_angle_sums += _sym_hist(lav, last_angle)
                causal_last_angle_counts += _sym_hist(lav)
                path_valid = (
                    valid_particle
                    & np.isfinite(arrays['internal_particle_path_total_m'])
                    & np.isfinite(arrays['internal_particle_opl_total_m'])
                    & np.isfinite(arrays['water_path_after_last_particle_event_m'])
                    & (arrays['water_path_after_last_particle_event_m'] >= 0.0)
                )
                causal_internal_path_sums += _sym_hist(
                    path_valid, arrays['internal_particle_path_total_m']
                )
                causal_internal_opl_sums += _sym_hist(
                    path_valid, arrays['internal_particle_opl_total_m']
                )
                causal_post_last_water_path_sums += _sym_hist(
                    path_valid, arrays['water_path_after_last_particle_event_m']
                )
                causal_last_order_sums += _sym_hist(
                    path_valid, ints['last_particle_event_order']
                )
                causal_path_counts += _sym_hist(path_valid)
                # angular-diffusion, branch-sequence and collision-location attribution.
                first_r = np.sqrt(
                    np.maximum(
                        arrays['first_particle_x'] ** 2
                        + arrays['first_particle_y'] ** 2,
                        0.0,
                    )
                )
                last_r = np.sqrt(
                    np.maximum(
                        arrays['last_particle_x'] ** 2 + arrays['last_particle_y'] ** 2,
                        0.0,
                    )
                )
                history_values = {
                    'cumulative_particle_deflection_deg': arrays[
                        'cumulative_particle_deflection_deg'
                    ],
                    'cumulative_particle_deflection_sq_deg2': arrays[
                        'cumulative_particle_deflection_sq_deg2'
                    ],
                    'max_particle_deflection_deg': arrays[
                        'max_particle_deflection_deg'
                    ],
                    'first_particle_r_m': first_r,
                    'first_particle_z_m': arrays['first_particle_z'],
                    'last_particle_r_m': last_r,
                    'last_particle_z_m': arrays['last_particle_z'],
                    'particle_event_mean_r_m': arrays['particle_event_mean_r_m'],
                    'particle_event_mean_z_m': arrays['particle_event_mean_z_m'],
                    'particle_event_min_r_m': arrays['particle_event_min_r_m'],
                    'particle_event_max_r_m': arrays['particle_event_max_r_m'],
                    'particle_event_min_z_m': arrays['particle_event_min_z_m'],
                    'particle_event_max_z_m': arrays['particle_event_max_z_m'],
                    'water_path_at_first_particle_event_m': arrays[
                        'water_path_at_first_particle_event_m'
                    ],
                    'water_path_at_last_particle_event_m': arrays[
                        'water_path_at_last_particle_event_m'
                    ],
                    'particle_water_path_span_first_to_last_m': arrays[
                        'water_path_at_last_particle_event_m'
                    ]
                    - arrays['water_path_at_first_particle_event_m'],
                    # Cast before subtraction: failed/absorbed events may retain finite
                    # float32 +/-INF_F sentinels. Their valid_particle mask is false,
                    # but subtracting in float32 first would still emit an overflow warning.
                    'particle_radial_span_m': arrays['particle_event_max_r_m'].astype(
                        np.float64
                    )
                    - arrays['particle_event_min_r_m'].astype(np.float64),
                    'particle_axial_span_m': arrays['particle_event_max_z_m'].astype(
                        np.float64
                    )
                    - arrays['particle_event_min_z_m'].astype(np.float64),
                    'particle_branch_transition_count': ints[
                        'particle_branch_transition_count'
                    ],
                    'particle_event_type_transition_count': ints[
                        'particle_event_type_transition_count'
                    ],
                    'particle_small_deflection_count': ints[
                        'particle_small_deflection_count'
                    ],
                    'particle_medium_deflection_count': ints[
                        'particle_medium_deflection_count'
                    ],
                    'particle_large_deflection_count': ints[
                        'particle_large_deflection_count'
                    ],
                }
                for mk, mv in history_values.items():
                    vv = valid_particle & np.isfinite(mv)
                    history_metric_sums[mk] += _sym_hist(vv, mv)
                    history_metric_counts[mk] += _sym_hist(vv)
                # Mutually exclusive deflection-history classes.
                small = ints['particle_small_deflection_count']
                medium = ints['particle_medium_deflection_count']
                large = ints['particle_large_deflection_count']
                history_deflection_history_scores['zero_particle'] += _sym_hist(
                    ic_all == 0
                )
                history_deflection_history_scores['all_lt30'] += _sym_hist(
                    valid_particle & (medium == 0) & (large == 0)
                )
                history_deflection_history_scores['has_30_90_no_ge90'] += _sym_hist(
                    valid_particle & (medium > 0) & (large == 0)
                )
                history_deflection_history_scores['has_ge90'] += _sym_hist(
                    valid_particle & (large > 0)
                )
                # Branch-sequence classes. Branch codes are derived from the already-recorded event type.
                first_branch = np.where(
                    (first_type >= 1) & (first_type <= 3),
                    1,
                    np.where(first_type >= 4, 2, 0),
                )
                last_branch = np.where(
                    (last_type >= 1) & (last_type <= 3),
                    1,
                    np.where(last_type >= 4, 2, 0),
                )
                gcount = ints['geometric_interactions']
                wcount = ints['wave_interactions']
                history_branch_sequence_scores['zero_particle'] += _sym_hist(
                    ic_all == 0
                )
                history_branch_sequence_scores['geometric_only'] += _sym_hist(
                    valid_particle & (gcount > 0) & (wcount == 0)
                )
                history_branch_sequence_scores['wave_only'] += _sym_hist(
                    valid_particle & (wcount > 0) & (gcount == 0)
                )
                mixed = (gcount > 0) & (wcount > 0) & valid_particle
                for fb, lb, nm in (
                    (1, 1, 'mixed_g_to_g'),
                    (1, 2, 'mixed_g_to_w'),
                    (2, 1, 'mixed_w_to_g'),
                    (2, 2, 'mixed_w_to_w'),
                ):
                    history_branch_sequence_scores[nm] += _sym_hist(
                        mixed & (first_branch == fb) & (last_branch == lb)
                    )
                for ti in range(6):
                    bm = (
                        (ints['particle_branch_transition_count'] == ti)
                        if ti < 5
                        else (ints['particle_branch_transition_count'] >= 5)
                    )
                    em = (
                        (ints['particle_event_type_transition_count'] == ti)
                        if ti < 5
                        else (ints['particle_event_type_transition_count'] >= 5)
                    )
                    history_branch_transition_scores[ti] += _sym_hist(
                        valid_particle & bm
                    )
                    history_event_transition_scores[ti] += _sym_hist(
                        valid_particle & em
                    )
                # H170 cumulative-deflection histogram and first/last collision spatial maps.
                cumdef = arrays['cumulative_particle_deflection_deg']
                for b in range(len(history_h170_cumulative_deflection_edges) - 1):
                    cm = (
                        valid_particle
                        & (cumdef >= history_h170_cumulative_deflection_edges[b])
                        & (cumdef < history_h170_cumulative_deflection_edges[b + 1])
                    )
                    history_h170_cumulative_deflection_hist[b] += _sym_hist(cm)[17]
                for ri in range(len(history_r_edges_m) - 1):
                    for zi in range(len(history_z_edges_m) - 1):
                        fm = (
                            valid_particle
                            & (first_r >= history_r_edges_m[ri])
                            & (first_r < history_r_edges_m[ri + 1])
                            & (arrays['first_particle_z'] >= history_z_edges_m[zi])
                            & (arrays['first_particle_z'] < history_z_edges_m[zi + 1])
                        )
                        lm = (
                            valid_particle
                            & (last_r >= history_r_edges_m[ri])
                            & (last_r < history_r_edges_m[ri + 1])
                            & (arrays['last_particle_z'] >= history_z_edges_m[zi])
                            & (arrays['last_particle_z'] < history_z_edges_m[zi + 1])
                        )
                        history_h170_first_spatial[ri, zi] += _sym_hist(fm)[17]
                        history_h170_last_spatial[ri, zi] += _sym_hist(lm)[17]
                # provenance of the single largest particle turn in each detected history.
                max_type = ints['max_deflection_event_type']
                max_order = ints['max_deflection_event_order']
                max_bin = ints['max_deflection_event_bin']
                max_macro = ints['max_deflection_event_macro_bounces']
                max_micro = ints['max_deflection_event_terminal_micro_order']
                max_ld = np.asarray(
                    arrays['max_deflection_event_internal_path_over_diameter'],
                    dtype=np.float64,
                )
                max_inc = np.asarray(
                    arrays['max_deflection_event_incidence_deg'], dtype=np.float64
                )
                max_exit = np.asarray(
                    arrays['max_deflection_event_exit_deg'], dtype=np.float64
                )
                for code in causal_event_codes:
                    deflection_max_event_type_scores[code] += _sym_hist(
                        valid_particle & (max_type == code)
                    )
                for oi in range(5):
                    om = (max_order == (oi + 1)) if oi < 4 else (max_order >= 5)
                    deflection_max_event_order_scores[oi] += _sym_hist(
                        valid_particle & om
                    )
                for bi in range(npart):
                    deflection_max_event_bin_scores[bi] += _sym_hist(
                        valid_particle & (max_bin == bi)
                    )
                # maximum-turn internal-path diagnostics.
                for mbi in range(5):
                    mm = (max_macro == mbi) if mbi < 4 else (max_macro >= 4)
                    internal_path_max_macro_bounce_scores[mbi] += _sym_hist(
                        valid_particle & mm
                    )
                    for code in causal_event_codes:
                        internal_path_h170_type_by_macro[code, mbi] += _sym_hist(
                            valid_particle & mm & (max_type == code)
                        )[17]
                for moi in range(6):
                    qm = (max_micro == moi) if moi < 5 else (max_micro >= 5)
                    internal_path_max_terminal_micro_order_scores[moi] += _sym_hist(
                        valid_particle & qm
                    )
                internal_path_max_length_ratio_sums += _sym_hist(
                    valid_particle, weights=max_ld
                )
                internal_path_max_macro_sums += _sym_hist(
                    valid_particle, weights=np.asarray(max_macro, dtype=np.float64)
                )
                internal_path_max_micro_sums += _sym_hist(
                    valid_particle, weights=np.asarray(max_micro, dtype=np.float64)
                )
                for ldi in range(len(internal_path_h170_length_ratio_edges) - 1):
                    lm = (
                        valid_particle
                        & (max_ld >= internal_path_h170_length_ratio_edges[ldi])
                        & (max_ld < internal_path_h170_length_ratio_edges[ldi + 1])
                    )
                    internal_path_h170_length_ratio_hist[ldi] += _sym_hist(lm)[17]
                # maximum-turn particle size / branch / interface-angle diagnostics.
                inc_defined = (
                    valid_particle
                    & np.isfinite(max_inc)
                    & (max_inc >= 0.0)
                    & (max_inc <= 90.0001)
                )
                exit_defined = (
                    valid_particle
                    & np.isfinite(max_exit)
                    & (max_exit >= 0.0)
                    & (max_exit <= 90.0001)
                )
                interface_incidence_sums += _sym_hist(inc_defined, weights=max_inc)
                interface_incidence_defined_scores += _sym_hist(inc_defined)
                interface_exit_sums += _sym_hist(exit_defined, weights=max_exit)
                interface_exit_defined_scores += _sym_hist(exit_defined)
                inc_class = np.zeros(n, dtype=np.int32)
                exit_class = np.zeros(n, dtype=np.int32)
                for ai in range(len(interface_angle_edges) - 1):
                    im = (
                        inc_defined
                        & (max_inc >= interface_angle_edges[ai])
                        & (max_inc < interface_angle_edges[ai + 1])
                    )
                    inc_class[im] = ai + 1
                    em = (
                        exit_defined
                        & (max_exit >= interface_angle_edges[ai])
                        & (max_exit < interface_angle_edges[ai + 1])
                    )
                    exit_class[em] = ai + 1
                for ac in range(7):
                    im = valid_particle & (inc_class == ac)
                    em = valid_particle & (exit_class == ac)
                    interface_h170_incidence_hist[ac] += _sym_hist(im)[17]
                    interface_h170_exit_hist[ac] += _sym_hist(em)[17]
                    for code in causal_event_codes:
                        interface_h170_type_by_incidence[code, ac] += _sym_hist(
                            im & (max_type == code)
                        )[17]
                        interface_h170_type_by_exit[code, ac] += _sym_hist(
                            em & (max_type == code)
                        )[17]
                    for bi in range(npart):
                        interface_h170_psd_by_incidence[bi, ac] += _sym_hist(
                            im & (max_bin == bi)
                        )[17]
                        interface_h170_psd_by_exit[bi, ac] += _sym_hist(
                            em & (max_bin == bi)
                        )[17]
                for code in causal_event_codes:
                    for bi in range(npart):
                        interface_h170_type_by_psd[code, bi] += _sym_hist(
                            valid_particle & (max_type == code) & (max_bin == bi)
                        )[17]
                max_r = np.sqrt(
                    np.maximum(
                        arrays['max_deflection_event_x'] ** 2
                        + arrays['max_deflection_event_y'] ** 2,
                        0.0,
                    )
                )
                for ri in range(len(history_r_edges_m) - 1):
                    for zi in range(len(history_z_edges_m) - 1):
                        mm = (
                            valid_particle
                            & (max_r >= history_r_edges_m[ri])
                            & (max_r < history_r_edges_m[ri + 1])
                            & (
                                arrays['max_deflection_event_z']
                                >= history_z_edges_m[zi]
                            )
                            & (
                                arrays['max_deflection_event_z']
                                < history_z_edges_m[zi + 1]
                            )
                        )
                        deflection_h170_max_spatial[ri, zi] += _sym_hist(mm)[17]
                # Compact H170 angle histograms by final-event type for causal interpretation.
                for b in range(len(causal_h170_angle_edges) - 1):
                    lo = causal_h170_angle_edges[b]
                    hi = causal_h170_angle_edges[b + 1]
                    am = lav & (last_angle >= lo) & (last_angle < hi)
                    h = _sym_hist(am)
                    causal_h170_last_angle_hist[0, b] += h[17]
                    for code in causal_event_codes:
                        h = _sym_hist(am & (last_type == code))
                        causal_h170_last_angle_hist[code, b] += h[17]
                    fh = _sym_hist(fav & (first_angle >= lo) & (first_angle < hi))
                    causal_h170_first_angle_hist[b] += fh[17]
                pfcode = np.asarray(ints['production_failure_code'], dtype=np.int32)
                pfmask = pfcode != 0
                if np.any(pfmask):
                    bc = np.bincount(
                        pfcode[pfmask].astype(np.int64),
                        minlength=len(production_failure_code_counts),
                    )
                    production_failure_code_counts[: len(bc)] += bc[
                        : len(production_failure_code_counts)
                    ]
                    for sub, count in zip(
                        *np.unique(
                            np.asarray(
                                ints['production_failure_subcode'], dtype=np.int32
                            )[pfmask],
                            return_counts=True,
                        )
                    ):
                        production_failure_subcode_counts[int(sub)] = (
                            production_failure_subcode_counts.get(int(sub), 0)
                            + int(count)
                        )
                    room = production_failure_sample_limit - len(
                        production_failure_samples
                    )
                    if room > 0:
                        loc = np.flatnonzero(pfmask)[:room]
                        for jj in loc:
                            code = int(pfcode[jj])
                            sub = int(ints['production_failure_subcode'][jj])
                            production_failure_samples.append(
                                {
                                    'ray_id': int(global_start + jj),
                                    'failure_code': code,
                                    'failure_name': PRODUCTION_FAILURE_NAMES.get(
                                        code, f'CODE_{code}'
                                    ),
                                    'failure_subcode': sub,
                                    'failure_subcode_name': PRODUCTION_FAILURE_SUBCODE_NAMES.get(
                                        sub, f'SUBCODE_{sub}'
                                    ),
                                    'trace_iteration': int(
                                        ints['production_failure_trace_iteration'][jj]
                                    ),
                                    'macro_bounces': int(
                                        ints['production_failure_macro_bounces'][jj]
                                    ),
                                    'micro_order': int(
                                        ints['production_failure_micro_order'][jj]
                                    ),
                                    'medium_code': int(
                                        ints['production_failure_medium'][jj]
                                    ),
                                    'medium_name': PRODUCTION_MEDIUM_NAMES.get(
                                        int(ints['production_failure_medium'][jj]),
                                        'unknown',
                                    ),
                                    'event_channel_code': int(
                                        ints['production_failure_event_channel'][jj]
                                    ),
                                    'event_channel_name': PRODUCTION_EVENT_CHANNEL_NAMES.get(
                                        int(
                                            ints['production_failure_event_channel'][jj]
                                        ),
                                        'unknown',
                                    ),
                                    'x': float(arrays['failure_x'][jj]),
                                    'y': float(arrays['failure_y'][jj]),
                                    'z': float(arrays['failure_z'][jj]),
                                    'vx': float(arrays['failure_vx'][jj]),
                                    'vy': float(arrays['failure_vy'][jj]),
                                    'vz': float(arrays['failure_vz'][jj]),
                                    'previous_event_type': int(
                                        ints['particle_diag_previous_event'][jj]
                                    ),
                                    'previous_particle_event': bool(
                                        int(ints['particle_diag_bin'][jj]) >= 0
                                    ),
                                    'particle_bin': int(ints['particle_diag_bin'][jj]),
                                    'particle_radius_m': float(
                                        arrays['particle_diag_radius'][jj]
                                    ),
                                    'particle_diameter_m': 2.0
                                    * float(arrays['particle_diag_radius'][jj]),
                                    'sampled_impact_parameter_rho': float(
                                        arrays['particle_diag_rho'][jj]
                                    ),
                                    'sampled_phi_rad': float(
                                        arrays['particle_diag_phi'][jj]
                                    ),
                                    'particle_center_x': float(
                                        arrays['particle_diag_cx'][jj]
                                    ),
                                    'particle_center_y': float(
                                        arrays['particle_diag_cy'][jj]
                                    ),
                                    'particle_center_z': float(
                                        arrays['particle_diag_cz'][jj]
                                    ),
                                    'particle_entry_x': float(
                                        arrays['particle_diag_entry_x'][jj]
                                    ),
                                    'particle_entry_y': float(
                                        arrays['particle_diag_entry_y'][jj]
                                    ),
                                    'particle_entry_z': float(
                                        arrays['particle_diag_entry_z'][jj]
                                    ),
                                    'particle_exit_x': float(
                                        arrays['particle_diag_exit_x'][jj]
                                    ),
                                    'particle_exit_y': float(
                                        arrays['particle_diag_exit_y'][jj]
                                    ),
                                    'particle_exit_z': float(
                                        arrays['particle_diag_exit_z'][jj]
                                    ),
                                    'pre_particle_x': float(
                                        arrays['particle_diag_pre_x'][jj]
                                    ),
                                    'pre_particle_y': float(
                                        arrays['particle_diag_pre_y'][jj]
                                    ),
                                    'pre_particle_z': float(
                                        arrays['particle_diag_pre_z'][jj]
                                    ),
                                    'pre_particle_vx': float(
                                        arrays['particle_diag_pre_vx'][jj]
                                    ),
                                    'pre_particle_vy': float(
                                        arrays['particle_diag_pre_vy'][jj]
                                    ),
                                    'pre_particle_vz': float(
                                        arrays['particle_diag_pre_vz'][jj]
                                    ),
                                    'post_particle_x': float(
                                        arrays['particle_diag_post_x'][jj]
                                    ),
                                    'post_particle_y': float(
                                        arrays['particle_diag_post_y'][jj]
                                    ),
                                    'post_particle_z': float(
                                        arrays['particle_diag_post_z'][jj]
                                    ),
                                    'post_particle_vx': float(
                                        arrays['particle_diag_post_vx'][jj]
                                    ),
                                    'post_particle_vy': float(
                                        arrays['particle_diag_post_vy'][jj]
                                    ),
                                    'post_particle_vz': float(
                                        arrays['particle_diag_post_vz'][jj]
                                    ),
                                    'particle_center_radial_clearance_m': float(
                                        arrays['particle_diag_center_radial_clearance'][
                                            jj
                                        ]
                                    ),
                                    'particle_finite_wall_clearance_m': float(
                                        arrays['particle_diag_finite_wall_clearance'][
                                            jj
                                        ]
                                    ),
                                    'particle_bottom_clearance_m': float(
                                        arrays['particle_diag_bottom_clearance'][jj]
                                    ),
                                    'particle_top_clearance_m': float(
                                        arrays['particle_diag_top_clearance'][jj]
                                    ),
                                    'particle_overlap_class': int(
                                        ints['particle_diag_overlap_class'][jj]
                                    ),
                                    'particle_proposal_accepted': int(
                                        ints['particle_diag_proposal_accepted'][jj]
                                    ),
                                }
                            )
                generic_status4 = int(np.count_nonzero(ints['status'] == 4))
                specific_failures = int(np.count_nonzero(pfmask))
                if generic_status4 != specific_failures:
                    raise RuntimeError(
                        f'production failure-code closure mismatch in chunk: status4={generic_status4}, specific={specific_failures}'
                    )
                valid_air = (
                    (ints['status'] == 1)
                    & np.isfinite(arrays['air_x'])
                    & np.isfinite(arrays['air_vx'])
                )
                entered = ints['entered_water'] > 0
                ic = ints['interaction_count'][entered]
                acrylic_trapped_tir_total += int(np.count_nonzero(ints['status'] == 8))
                valid_air_total += int(valid_air.sum())
                numerical_failure_total += specific_failures
                entered_total += int(entered.sum())
                srcok = ints['status'] != 3
                source_accept_total += int(np.count_nonzero(srcok))
                source_direct_accept_total += int(
                    np.count_nonzero(srcok & (ints['source_bore_reflections'] == 0))
                )
                source_reflected_accept_total += int(
                    np.count_nonzero(srcok & (ints['source_bore_reflections'] > 0))
                )
                source_bore_interactions_total += int(
                    ints['source_bore_interactions'].sum()
                )
                source_bore_reflections_total += int(
                    ints['source_bore_reflections'].sum()
                )
                top_exit_total += int(np.count_nonzero(ints['status'] == 6))
                bottom_exit_total += int(np.count_nonzero(ints['status'] == 7))
                axial_total += int(
                    np.count_nonzero((ints['status'] == 6) | (ints['status'] == 7))
                )
                if ic.size:
                    sum_ic += float(ic.sum())
                    sum_gic += float(ints['geometric_interactions'][entered].sum())
                    sum_wic += float(ints['wave_interactions'][entered].sum())
                    sum_wcoh += float(ints['coherent_wave_interactions'][entered].sum())
                    sum_wmorph += float(
                        ints['morphology_wave_interactions'][entered].sum()
                    )
                    ballistic_total += int(np.count_nonzero(ic == 0))
                    bc = np.bincount(ic.astype(np.int64))
                    if bc.size > ic_hist.size:
                        ic_hist = np.pad(ic_hist, (0, bc.size - ic_hist.size))
                    ic_hist[: bc.size] += bc
                    pr = ints['particle_reflections'][entered]
                    per = ints['particle_entry_reflections'][entered]
                    pir = ints['particle_internal_reflections'][entered]
                    particle_absorbed_total += int(
                        np.count_nonzero(ints['particle_absorptions'][entered] > 0)
                    )
                    cir = ints['cell_inner_reflections'][entered]
                    cor = ints['cell_outer_reflections'][entered]
                    ct = ints['cell_tir'][entered]
                    sum_pr += float(pr.sum())
                    rays_entry_pr += int(np.count_nonzero(per > 0))
                    rays_internal_pr += int(np.count_nonzero(pir > 0))
                    sum_cir += float(cir.sum())
                    sum_cor += float(cor.sum())
                    rays_ctir += int(np.count_nonzero(ct > 0))
                    sum_top_ref += float(ints['top_reflections'][entered].sum())
                    rays_top_tir += int(np.count_nonzero(ints['top_tir'][entered] > 0))
                    sum_bottom_inner += float(
                        ints['bottom_inner_reflections'][entered].sum()
                    )
                    sum_bottom_outer += float(
                        ints['bottom_outer_reflections'][entered].sum()
                    )
                    rays_bottom_tir += int(
                        np.count_nonzero(ints['bottom_tir'][entered] > 0)
                    )
                if collect_rays:
                    ray_chunks.append(
                        {
                            **arrays,
                            **ints,
                            'native_detector_index': nid_h,
                            'mirror_detector_index': mid_h,
                            'native_detector_bore_reflections': cp.asnumpy(nbc),
                            'mirror_detector_bore_reflections': cp.asnumpy(mbc),
                            'native_ring_face_reflections': cp.asnumpy(nfc),
                            'mirror_ring_face_reflections': cp.asnumpy(mfc),
                        }
                    )
                del (
                    outf,
                    outi,
                    af,
                    ai,
                    hnat,
                    hmir,
                    hnd,
                    hmd,
                    hns,
                    hms,
                    hnf,
                    hmf,
                    pdnat,
                    pdmir,
                    iangn,
                    iangm,
                    lnat,
                    lmir,
                    nid,
                    mid,
                    nbc,
                    mbc,
                    nfc,
                    mfc,
                )
            total_rays += batchN
            native_total += batch_native
            mirror_total += batch_mirror
            native_direct_total += batch_native_direct
            mirror_direct_total += batch_mirror_direct
            native_stray_total += batch_native_stray
            mirror_stray_total += batch_mirror_stray
            native_face_total += batch_native_face
            mirror_face_total += batch_mirror_face
            photodiode_native_total += batch_photodiode_native
            photodiode_mirror_total += batch_photodiode_mirror
            incidence_native_sum_total += batch_incidence_native_sum
            incidence_mirror_sum_total += batch_incidence_mirror_sum
            legacy_native_total += batch_legacy_native
            legacy_mirror_total += batch_legacy_mirror
            batch_hardware_sym = (
                0.5 * (batch_native.astype(float) + batch_mirror.astype(float))
                if self.symmetry_average_exact
                else batch_native.astype(float)
            )
            batch_photodiode_sym = (
                0.5 * (batch_photodiode_native + batch_photodiode_mirror)
                if self.symmetry_average_exact
                else batch_photodiode_native.copy()
            )
            batch_sym = (
                batch_hardware_sym
                if self.photodiode_response_model == 'ideal_hard_aperture'
                else batch_photodiode_sym
            )
            batch_legacy_sym = (
                0.5
                * (
                    batch_legacy_native.astype(float)
                    + batch_legacy_mirror.astype(float)
                )
                if self.symmetry_average_exact
                else batch_legacy_native.astype(float)
            )
            sym_total += batch_sym
            legacy_sym_total += batch_legacy_sym
            batch_scores.append(batch_sym.copy())
            hardware_batch_scores.append(batch_hardware_sym.copy())
            photodiode_batch_scores.append(batch_photodiode_sym.copy())
            legacy_batch_scores.append(batch_legacy_sym.copy())
            den = float(sym_total.sum())
            cum = sym_total / den if den > 0 else np.zeros_like(sym_total)
            l1 = (
                float(np.sum(np.abs(cum - prev_cum)))
                if prev_cum is not None
                else float('nan')
            )
            if np.isfinite(l1):
                recent_l1.append(l1)
            prev_cum = cum.copy()
            rear_min = float(np.min(sym_total[15:18]))
            batch_rows.append(
                {
                    'batch': len(batch_rows) + 1,
                    'batch_rays': batchN,
                    'cumulative_rays': total_rays,
                    'batch_native_hits': int(batch_native.sum()),
                    'batch_mirror_hits': int(batch_mirror.sum()),
                    'batch_hardware_candidate_score': float(batch_hardware_sym.sum()),
                    'batch_photodiode_weighted_score': float(
                        batch_photodiode_sym.sum()
                    ),
                    'batch_selected_score': float(batch_sym.sum()),
                    'cumulative_selected_score': float(sym_total.sum()),
                    'cumulative_H150_score': float(sym_total[15]),
                    'cumulative_H160_score': float(sym_total[16]),
                    'cumulative_H170_score': float(sym_total[17]),
                    'cumulative_min_rear_score_H150_H170': rear_min,
                    'batch_legacy_channel_score': float(batch_legacy_sym.sum()),
                    'cumulative_legacy_channel_score': float(legacy_sym_total.sum()),
                    'cumulative_l1_change': l1,
                }
            )
            if adaptive and total_rays >= minN and sym_total.sum() >= target:
                rear_ok = (rear_target is None) or (rear_min >= rear_target)
                h170_ok = (h170_target is None) or (float(sym_total[17]) >= h170_target)
                stable = True
                if stability_l1_tolerance is not None:
                    wdw = max(1, int(stability_window))
                    stable = len(recent_l1) >= wdw and all(
                        q <= float(stability_l1_tolerance) for q in recent_l1[-wdw:]
                    )
                if rear_ok and h170_ok and stable:
                    bits = ['target_detector_score']
                    if rear_target is not None:
                        bits.append('rear_channels')
                    if h170_target is not None:
                        bits.append('H170')
                    if stability_l1_tolerance is not None:
                        bits.append('stability')
                    stop_reason = '_and_'.join(bits)
                    break
        if adaptive and total_rays >= maxN:
            rear_min = float(np.min(sym_total[15:18]))
            if sym_total.sum() < target:
                stop_reason = 'max_rays_before_target'
            elif rear_target is not None and rear_min < rear_target:
                stop_reason = 'max_rays_before_rear_channel_target'
            elif h170_target is not None and float(sym_total[17]) < h170_target:
                stop_reason = 'max_rays_before_H170_target'
            elif stability_l1_tolerance is not None and stop_reason.startswith('fixed'):
                stop_reason = 'max_rays_before_stability'
            elif stop_reason.startswith('fixed'):
                stop_reason = 'max_rays'
        hardware_sym_total = (
            0.5 * (native_total.astype(float) + mirror_total.astype(float))
            if self.symmetry_average_exact
            else native_total.astype(float)
        )
        photodiode_sym_total = (
            0.5 * (photodiode_native_total + photodiode_mirror_total)
            if self.symmetry_average_exact
            else photodiode_native_total.copy()
        )
        nsum = float(native_total.sum())
        msum = float(mirror_total.sum())
        hardware_ssum = float(hardware_sym_total.sum())
        nn = native_total.astype(float) / nsum if nsum > 0 else np.zeros(18)
        nm = mirror_total.astype(float) / msum if msum > 0 else np.zeros(18)
        nh = hardware_sym_total / hardware_ssum if hardware_ssum > 0 else np.zeros(18)
        pnsum = float(photodiode_native_total.sum())
        pmsum = float(photodiode_mirror_total.sum())
        pssum = float(photodiode_sym_total.sum())
        pnn = photodiode_native_total / pnsum if pnsum > 0 else np.zeros(18)
        pnm = photodiode_mirror_total / pmsum if pmsum > 0 else np.zeros(18)
        pns = photodiode_sym_total / pssum if pssum > 0 else np.zeros(18)
        selected_native_sum = (
            nsum if self.photodiode_response_model == 'ideal_hard_aperture' else pnsum
        )
        selected_mirror_sum = (
            msum if self.photodiode_response_model == 'ideal_hard_aperture' else pmsum
        )
        ssum = float(sym_total.sum())
        ns = sym_total / ssum if ssum > 0 else np.zeros(18)
        direct_sym = (
            0.5
            * (native_direct_total.astype(float) + mirror_direct_total.astype(float))
            if self.symmetry_average_exact
            else native_direct_total.astype(float)
        )
        stray_sym = (
            0.5 * (native_stray_total.astype(float) + mirror_stray_total.astype(float))
            if self.symmetry_average_exact
            else native_stray_total.astype(float)
        )
        det_stray_frac = np.divide(
            stray_sym,
            hardware_sym_total,
            out=np.zeros(18, dtype=float),
            where=hardware_sym_total > 0,
        )
        face_sym = (
            0.5 * (native_face_total.astype(float) + mirror_face_total.astype(float))
            if self.symmetry_average_exact
            else native_face_total.astype(float)
        )
        det_face_frac = np.divide(
            face_sym,
            hardware_sym_total,
            out=np.zeros(18, dtype=float),
            where=hardware_sym_total > 0,
        )
        photodiode_relative = np.divide(
            photodiode_sym_total,
            hardware_sym_total,
            out=np.zeros(18, dtype=float),
            where=hardware_sym_total > 0,
        )
        incidence_sym_sum = (
            0.5 * (incidence_native_sum_total + incidence_mirror_sum_total)
            if self.symmetry_average_exact
            else incidence_native_sum_total.copy()
        )
        mean_incidence = np.divide(
            incidence_sym_sum,
            hardware_sym_total,
            out=np.full(18, np.nan, dtype=float),
            where=hardware_sym_total > 0,
        )
        lnsum = float(legacy_native_total.sum())
        lmsum = float(legacy_mirror_total.sum())
        lssum = float(legacy_sym_total.sum())
        lnn = legacy_native_total.astype(float) / lnsum if lnsum > 0 else np.zeros(18)
        lnm = legacy_mirror_total.astype(float) / lmsum if lmsum > 0 else np.zeros(18)
        lns = legacy_sym_total / lssum if lssum > 0 else np.zeros(18)
        se = _jackknife_normalized_se(batch_scores)
        hardware_se = _jackknife_normalized_se(hardware_batch_scores)
        photodiode_se = _jackknife_normalized_se(photodiode_batch_scores)
        legacy_se = _jackknife_normalized_se(legacy_batch_scores)
        denom_enter = max(entered_total, 1)
        # Detector-path mechanism fractions are normalized to the hard-aperture
        # symmetry score in each channel.  They diagnose the current particle
        # physics; they are not correction factors or fitted detector gains.
        path_diag = {'hardware_symmetry_scores': hardware_sym_total.copy()}
        denp = np.asarray(hardware_sym_total, dtype=np.float64)
        for pk, pv in path_scores.items():
            path_diag[pk + '_scores'] = pv.copy()
            path_diag[pk + '_fraction'] = np.divide(
                pv, denp, out=np.zeros_like(pv), where=denp > 0
            )
        path_diag['mean_particle_interactions_by_channel'] = np.divide(
            path_interaction_sums, denp, out=np.full(18, np.nan), where=denp > 0
        )
        path_diag['mean_geometric_interactions_by_channel'] = np.divide(
            path_geometric_interaction_sums,
            denp,
            out=np.full(18, np.nan),
            where=denp > 0,
        )
        path_diag['mean_wave_interactions_by_channel'] = np.divide(
            path_wave_interaction_sums, denp, out=np.full(18, np.nan), where=denp > 0
        )
        path_diag['mean_coherent_wave_interactions_by_channel'] = np.divide(
            path_coherent_wave_interaction_sums,
            denp,
            out=np.full(18, np.nan),
            where=denp > 0,
        )
        path_diag['mean_morphology_wave_interactions_by_channel'] = np.divide(
            path_morphology_wave_interaction_sums,
            denp,
            out=np.full(18, np.nan),
            where=denp > 0,
        )
        path_diag['mean_entry_reflections_by_channel'] = np.divide(
            path_entry_reflection_sums, denp, out=np.full(18, np.nan), where=denp > 0
        )
        path_diag['mean_internal_reflections_by_channel'] = np.divide(
            path_internal_reflection_sums, denp, out=np.full(18, np.nan), where=denp > 0
        )
        # Causal final-event and path-length outputs. Fractions use all detected
        # hardware candidates in each channel; conditional means use valid particle histories.
        for code in causal_event_codes:
            name = PARTICLE_EVENT_TYPE_NAMES[code]
            path_diag[f'last_event_{name}_scores'] = causal_last_event_scores[
                code
            ].copy()
            path_diag[f'last_event_{name}_fraction'] = _safe_fraction(
                causal_last_event_scores[code], denp
            )
            path_diag[f'first_event_{name}_scores'] = causal_first_event_scores[
                code
            ].copy()
            path_diag[f'first_event_{name}_fraction'] = _safe_fraction(
                causal_first_event_scores[code], denp
            )
            path_diag[f'last_event_{name}_mean_deflection_deg_by_channel'] = np.divide(
                causal_last_event_angle_sums[code],
                causal_last_event_angle_counts[code],
                out=np.full(18, np.nan),
                where=causal_last_event_angle_counts[code] > 0,
            )
        path_diag['first_event_backscatter_scores'] = (
            causal_first_backscatter_scores.copy()
        )
        path_diag['first_event_backscatter_fraction'] = _safe_fraction(
            causal_first_backscatter_scores, denp
        )
        path_diag['first_event_source_hemisphere_scores'] = (
            causal_first_source_hemi_scores.copy()
        )
        path_diag['first_event_source_hemisphere_fraction'] = _safe_fraction(
            causal_first_source_hemi_scores, denp
        )
        path_diag['mean_first_particle_deflection_deg_by_channel'] = np.divide(
            causal_first_angle_sums,
            causal_first_angle_counts,
            out=np.full(18, np.nan),
            where=causal_first_angle_counts > 0,
        )
        path_diag['mean_last_particle_deflection_deg_by_channel'] = np.divide(
            causal_last_angle_sums,
            causal_last_angle_counts,
            out=np.full(18, np.nan),
            where=causal_last_angle_counts > 0,
        )
        path_diag['mean_total_internal_particle_path_m_by_channel'] = np.divide(
            causal_internal_path_sums,
            causal_path_counts,
            out=np.full(18, np.nan),
            where=causal_path_counts > 0,
        )
        path_diag['mean_total_internal_particle_opl_m_by_channel'] = np.divide(
            causal_internal_opl_sums,
            causal_path_counts,
            out=np.full(18, np.nan),
            where=causal_path_counts > 0,
        )
        path_diag['mean_water_path_after_last_particle_event_m_by_channel'] = np.divide(
            causal_post_last_water_path_sums,
            causal_path_counts,
            out=np.full(18, np.nan),
            where=causal_path_counts > 0,
        )
        path_diag['mean_last_particle_event_order_by_channel'] = np.divide(
            causal_last_order_sums,
            causal_path_counts,
            out=np.full(18, np.nan),
            where=causal_path_counts > 0,
        )
        path_diag['causal_last_event_partition_closure_error_scores'] = path_scores[
            'any_particle'
        ] - sum(causal_last_event_scores.values())
        path_diag['causal_first_event_partition_closure_error_scores'] = path_scores[
            'any_particle'
        ] - sum(causal_first_event_scores.values())
        path_diag['H170_last_event_angle_bin_edges_deg'] = (
            causal_h170_angle_edges.copy()
        )
        path_diag['H170_last_event_angle_histogram_scores_by_event_code'] = (
            causal_h170_last_angle_hist.copy()
        )
        path_diag['H170_first_event_angle_histogram_scores'] = (
            causal_h170_first_angle_hist.copy()
        )
        # Exact partition-closure diagnostics.  Zero means the mutually exclusive bins
        # recover the corresponding parent score exactly.
        path_diag['interaction_order_partition_closure_error_scores'] = path_scores[
            'any_particle'
        ] - (
            path_scores['single_interaction']
            + path_scores['interaction_order_2']
            + path_scores['interaction_order_3']
            + path_scores['interaction_order_4plus']
        )
        path_diag['multiple_branch_partition_closure_error_scores'] = path_scores[
            'multiple_interaction'
        ] - (
            path_scores['mixed_geometric_wave']
            + path_scores['geometric_only_multiple']
            + path_scores['wave_only_multiple']
        )
        path_diag['single_branch_partition_closure_error_scores'] = path_scores[
            'single_interaction'
        ] - (
            path_scores['single_external_entry_reflection_escape']
            + path_scores['single_internal_reflection_escape']
            + path_scores['single_surface_transmission_escape']
            + path_scores['single_coherent_wave_escape']
        )
        path_diag['exact_particle_order_labels'] = ['0', '1', '2', '3', '4', '5+']
        path_diag['exact_particle_order_scores'] = exact_particle_order_scores.copy()
        path_diag['exact_geometric_order_labels'] = ['0', '1', '2', '3', '4', '5+']
        path_diag['exact_geometric_order_scores'] = exact_geometric_order_scores.copy()
        path_diag['exact_particle_order_partition_closure_error_scores'] = (
            denp - np.sum(exact_particle_order_scores, axis=0)
        )
        path_diag['exact_geometric_order_partition_closure_error_scores'] = (
            denp - np.sum(exact_geometric_order_scores, axis=0)
        )
        path_diag['first_particle_bin_scores'] = first_particle_bin_scores.copy()
        path_diag['last_particle_bin_scores'] = last_particle_bin_scores.copy()
        path_diag['first_geometric_bin_scores'] = first_geometric_bin_scores.copy()
        path_diag['last_geometric_bin_scores'] = last_geometric_bin_scores.copy()
        path_diag['first_wave_bin_scores'] = first_wave_bin_scores.copy()
        path_diag['last_wave_bin_scores'] = last_wave_bin_scores.copy()
        path_diag['first_particle_bin_partition_closure_error_scores'] = path_scores[
            'any_particle'
        ] - np.sum(first_particle_bin_scores, axis=0)
        path_diag['last_particle_bin_partition_closure_error_scores'] = path_scores[
            'any_particle'
        ] - np.sum(last_particle_bin_scores, axis=0)
        path_diag['wall_path_scores'] = {
            k: v.copy() for k, v in wall_path_scores.items()
        }
        path_diag['wall_path_partition_closure_error_scores'] = denp - sum(
            wall_path_scores.values()
        )
        path_diag['particle_after_acrylic_return_scores'] = (
            particle_after_acrylic_return_scores.copy()
        )
        path_diag['acrylic_reflection_order_labels'] = ['0', '1', '2', '3', '4+']
        path_diag['acrylic_reflection_order_scores'] = (
            acrylic_reflection_order_scores.copy()
        )
        path_diag['acrylic_reflection_order_partition_closure_error_scores'] = (
            denp - np.sum(acrylic_reflection_order_scores, axis=0)
        )
        # angular-diffusion / sequence / spatial diagnostics.
        for _mk in history_mean_metric_names:
            path_diag['mean_' + _mk + '_by_channel'] = np.divide(
                history_metric_sums[_mk],
                history_metric_counts[_mk],
                out=np.full(18, np.nan),
                where=history_metric_counts[_mk] > 0,
            )
        path_diag['deflection_history_class_scores'] = {
            k: v.copy() for k, v in history_deflection_history_scores.items()
        }
        path_diag['deflection_history_class_partition_closure_error_scores'] = (
            denp - sum(history_deflection_history_scores.values())
        )
        path_diag['branch_sequence_class_scores'] = {
            k: v.copy() for k, v in history_branch_sequence_scores.items()
        }
        path_diag['branch_sequence_class_partition_closure_error_scores'] = denp - sum(
            history_branch_sequence_scores.values()
        )
        path_diag['branch_transition_order_labels'] = ['0', '1', '2', '3', '4', '5+']
        path_diag['branch_transition_order_scores'] = (
            history_branch_transition_scores.copy()
        )
        path_diag['branch_transition_order_partition_closure_error_scores'] = (
            path_scores['any_particle']
            - np.sum(history_branch_transition_scores, axis=0)
        )
        path_diag['event_type_transition_order_labels'] = [
            '0',
            '1',
            '2',
            '3',
            '4',
            '5+',
        ]
        path_diag['event_type_transition_order_scores'] = (
            history_event_transition_scores.copy()
        )
        path_diag['event_type_transition_order_partition_closure_error_scores'] = (
            path_scores['any_particle']
            - np.sum(history_event_transition_scores, axis=0)
        )
        path_diag['H170_cumulative_particle_deflection_bin_edges_deg'] = (
            history_h170_cumulative_deflection_edges.copy()
        )
        path_diag['H170_cumulative_particle_deflection_histogram_scores'] = (
            history_h170_cumulative_deflection_hist.copy()
        )
        path_diag['H170_spatial_radial_bin_edges_m'] = history_r_edges_m.copy()
        path_diag['H170_spatial_axial_bin_edges_m'] = history_z_edges_m.copy()
        path_diag['H170_first_particle_spatial_score_map'] = (
            history_h170_first_spatial.copy()
        )
        path_diag['H170_last_particle_spatial_score_map'] = (
            history_h170_last_spatial.copy()
        )
        _h170_particle_score = float(path_scores['any_particle'][17])
        path_diag['H170_cumulative_deflection_histogram_closure_error_score'] = (
            _h170_particle_score
            - float(np.sum(history_h170_cumulative_deflection_hist))
        )
        path_diag['H170_first_particle_spatial_map_closure_error_score'] = (
            _h170_particle_score - float(np.sum(history_h170_first_spatial))
        )
        path_diag['H170_last_particle_spatial_map_closure_error_score'] = (
            _h170_particle_score - float(np.sum(history_h170_last_spatial))
        )
        # maximum-deflection-event provenance.
        path_diag['max_deflection_event_type_scores'] = {
            PARTICLE_EVENT_TYPE_NAMES[code]: v.copy()
            for code, v in deflection_max_event_type_scores.items()
        }
        path_diag['max_deflection_event_type_partition_closure_error_scores'] = (
            path_scores['any_particle'] - sum(deflection_max_event_type_scores.values())
        )
        path_diag['max_deflection_event_order_labels'] = ['1', '2', '3', '4', '5+']
        path_diag['max_deflection_event_order_scores'] = (
            deflection_max_event_order_scores.copy()
        )
        path_diag['max_deflection_event_order_partition_closure_error_scores'] = (
            path_scores['any_particle']
            - np.sum(deflection_max_event_order_scores, axis=0)
        )
        path_diag['max_deflection_event_bin_scores'] = (
            deflection_max_event_bin_scores.copy()
        )
        path_diag['max_deflection_event_bin_partition_closure_error_scores'] = (
            path_scores['any_particle']
            - np.sum(deflection_max_event_bin_scores, axis=0)
        )
        path_diag['H170_max_deflection_event_spatial_score_map'] = (
            deflection_h170_max_spatial.copy()
        )
        path_diag['H170_max_deflection_event_spatial_map_closure_error_score'] = (
            _h170_particle_score - float(np.sum(deflection_h170_max_spatial))
        )
        # maximum-turn finite-particle internal-path drill-down.
        path_diag['max_deflection_event_macro_bounce_labels'] = [
            '0',
            '1',
            '2',
            '3',
            '4+',
        ]
        path_diag['max_deflection_event_macro_bounce_scores'] = (
            internal_path_max_macro_bounce_scores.copy()
        )
        path_diag[
            'max_deflection_event_macro_bounce_partition_closure_error_scores'
        ] = path_scores['any_particle'] - np.sum(
            internal_path_max_macro_bounce_scores, axis=0
        )
        path_diag['max_deflection_event_terminal_micro_order_labels'] = [
            '0',
            '1',
            '2',
            '3',
            '4',
            '5+',
        ]
        path_diag['max_deflection_event_terminal_micro_order_scores'] = (
            internal_path_max_terminal_micro_order_scores.copy()
        )
        path_diag[
            'max_deflection_event_terminal_micro_order_partition_closure_error_scores'
        ] = path_scores['any_particle'] - np.sum(
            internal_path_max_terminal_micro_order_scores, axis=0
        )
        _particle_score_denominator = np.asarray(
            path_scores['any_particle'], dtype=np.float64
        )
        path_diag[
            'mean_max_deflection_event_internal_path_over_diameter_by_channel'
        ] = np.divide(
            internal_path_max_length_ratio_sums,
            _particle_score_denominator,
            out=np.full(18, np.nan),
            where=_particle_score_denominator > 0,
        )
        path_diag['mean_max_deflection_event_macro_bounces_by_channel'] = np.divide(
            internal_path_max_macro_sums,
            _particle_score_denominator,
            out=np.full(18, np.nan),
            where=_particle_score_denominator > 0,
        )
        path_diag['mean_max_deflection_event_terminal_micro_order_by_channel'] = (
            np.divide(
                internal_path_max_micro_sums,
                _particle_score_denominator,
                out=np.full(18, np.nan),
                where=_particle_score_denominator > 0,
            )
        )
        path_diag['H170_max_deflection_internal_path_over_diameter_edges'] = (
            internal_path_h170_length_ratio_edges.copy()
        )
        path_diag['H170_max_deflection_internal_path_over_diameter_scores'] = (
            internal_path_h170_length_ratio_hist.copy()
        )
        path_diag[
            'H170_max_deflection_internal_path_over_diameter_histogram_closure_error_score'
        ] = _h170_particle_score - float(np.sum(internal_path_h170_length_ratio_hist))
        path_diag['H170_max_deflection_event_type_by_macro_bounce_scores'] = (
            internal_path_h170_type_by_macro.copy()
        )
        path_diag[
            'H170_max_deflection_event_type_by_macro_bounce_closure_error_score'
        ] = _h170_particle_score - float(np.sum(internal_path_h170_type_by_macro))
        # maximum-turn size/branch/interface-angle correlations.
        path_diag['max_deflection_interface_angle_class_labels'] = list(
            interface_angle_labels
        )
        path_diag['max_deflection_interface_angle_edges_deg'] = (
            interface_angle_edges.copy()
        )
        path_diag['H170_max_deflection_incidence_angle_scores'] = (
            interface_h170_incidence_hist.copy()
        )
        path_diag['H170_max_deflection_exit_angle_scores'] = (
            interface_h170_exit_hist.copy()
        )
        path_diag['H170_max_deflection_event_type_by_psd_bin_scores'] = (
            interface_h170_type_by_psd.copy()
        )
        path_diag['H170_max_deflection_event_type_by_incidence_angle_scores'] = (
            interface_h170_type_by_incidence.copy()
        )
        path_diag['H170_max_deflection_event_type_by_exit_angle_scores'] = (
            interface_h170_type_by_exit.copy()
        )
        path_diag['H170_max_deflection_psd_bin_by_incidence_angle_scores'] = (
            interface_h170_psd_by_incidence.copy()
        )
        path_diag['H170_max_deflection_psd_bin_by_exit_angle_scores'] = (
            interface_h170_psd_by_exit.copy()
        )
        path_diag['mean_max_deflection_event_incidence_deg_by_channel'] = np.divide(
            interface_incidence_sums,
            interface_incidence_defined_scores,
            out=np.full(18, np.nan),
            where=interface_incidence_defined_scores > 0,
        )
        path_diag['mean_max_deflection_event_exit_deg_by_channel'] = np.divide(
            interface_exit_sums,
            interface_exit_defined_scores,
            out=np.full(18, np.nan),
            where=interface_exit_defined_scores > 0,
        )
        path_diag['max_deflection_event_incidence_defined_scores'] = (
            interface_incidence_defined_scores.copy()
        )
        path_diag['max_deflection_event_exit_defined_scores'] = (
            interface_exit_defined_scores.copy()
        )
        path_diag[
            'H170_max_deflection_incidence_angle_histogram_closure_error_score'
        ] = _h170_particle_score - float(np.sum(interface_h170_incidence_hist))
        path_diag['H170_max_deflection_exit_angle_histogram_closure_error_score'] = (
            _h170_particle_score - float(np.sum(interface_h170_exit_hist))
        )
        path_diag['H170_max_deflection_event_type_by_psd_bin_closure_error_score'] = (
            _h170_particle_score - float(np.sum(interface_h170_type_by_psd))
        )
        path_diag[
            'H170_max_deflection_event_type_by_incidence_angle_closure_error_score'
        ] = _h170_particle_score - float(np.sum(interface_h170_type_by_incidence))
        path_diag[
            'H170_max_deflection_event_type_by_exit_angle_closure_error_score'
        ] = _h170_particle_score - float(np.sum(interface_h170_type_by_exit))
        path_diag[
            'H170_max_deflection_psd_bin_by_incidence_angle_closure_error_score'
        ] = _h170_particle_score - float(np.sum(interface_h170_psd_by_incidence))
        path_diag['H170_max_deflection_psd_bin_by_exit_angle_closure_error_score'] = (
            _h170_particle_score - float(np.sum(interface_h170_psd_by_exit))
        )
        # Diagnostic bookkeeping is fail-closed: every mutually exclusive score partition must close.
        for _name in (
            'exact_particle_order_partition_closure_error_scores',
            'exact_geometric_order_partition_closure_error_scores',
            'first_particle_bin_partition_closure_error_scores',
            'last_particle_bin_partition_closure_error_scores',
            'wall_path_partition_closure_error_scores',
            'acrylic_reflection_order_partition_closure_error_scores',
            'deflection_history_class_partition_closure_error_scores',
            'branch_sequence_class_partition_closure_error_scores',
            'branch_transition_order_partition_closure_error_scores',
            'event_type_transition_order_partition_closure_error_scores',
            'max_deflection_event_type_partition_closure_error_scores',
            'max_deflection_event_order_partition_closure_error_scores',
            'max_deflection_event_bin_partition_closure_error_scores',
            'max_deflection_event_macro_bounce_partition_closure_error_scores',
            'max_deflection_event_terminal_micro_order_partition_closure_error_scores',
        ):
            _err = np.asarray(path_diag[_name], dtype=np.float64)
            if np.any(np.abs(_err) > 1.0e-9):
                raise RuntimeError(
                    f'detector-score decomposition closure failure {_name}: max_abs={float(np.max(np.abs(_err)))}'
                )
        for _name in (
            'H170_cumulative_deflection_histogram_closure_error_score',
            'H170_first_particle_spatial_map_closure_error_score',
            'H170_last_particle_spatial_map_closure_error_score',
            'H170_max_deflection_event_spatial_map_closure_error_score',
            'H170_max_deflection_internal_path_over_diameter_histogram_closure_error_score',
            'H170_max_deflection_event_type_by_macro_bounce_closure_error_score',
            'H170_max_deflection_incidence_angle_histogram_closure_error_score',
            'H170_max_deflection_exit_angle_histogram_closure_error_score',
            'H170_max_deflection_event_type_by_psd_bin_closure_error_score',
            'H170_max_deflection_event_type_by_incidence_angle_closure_error_score',
            'H170_max_deflection_event_type_by_exit_angle_closure_error_score',
            'H170_max_deflection_psd_bin_by_incidence_angle_closure_error_score',
            'H170_max_deflection_psd_bin_by_exit_angle_closure_error_score',
        ):
            if abs(float(path_diag[_name])) > 1.0e-9:
                raise RuntimeError(
                    f'H170 spatial/angular diagnostic closure failure {_name}: error={float(path_diag[_name])}'
                )
        ray_data = None
        if collect_rays and ray_chunks:
            keys = ray_chunks[0].keys()
            ray_data = {k: np.concatenate([q[k] for q in ray_chunks]) for k in keys}
            ni = ray_data['native_detector_index']
            mi = ray_data['mirror_detector_index']
            ray_data['native_detector_deg'] = np.where(
                ni >= 0, DETECTOR_ANGLES_DEG[np.clip(ni, 0, 17)], np.nan
            )
            ray_data['mirror_detector_deg'] = np.where(
                mi >= 0, DETECTOR_ANGLES_DEG[np.clip(mi, 0, 17)], np.nan
            )
        hmxy = cp.asnumpy(hxy).reshape(vis, vis) if vis > 0 else None
        hmzy = cp.asnumpy(hzy).reshape(vis, vis) if vis > 0 else None
        light_diag = None
        if light_enabled:
            seg_total = float(cp.asnumpy(light_segment_total)[0])
            dep_total = float(cp.asnumpy(light_deposited_total)[0])
            grid_total = cp.asnumpy(light_total).reshape(light_nz, light_ny, light_nx)
            grid_post = cp.asnumpy(light_post).reshape(light_nz, light_ny, light_nx)
            grid_event_weight = cp.asnumpy(light_event_weight).reshape(
                light_nz, light_ny, light_nx
            )
            grid_event_count = cp.asnumpy(light_event_count).reshape(
                light_nz, light_ny, light_nx
            )
            tc = cp.asnumpy(light_tcount)[:light_rep]
            to = cp.asnumpy(light_toverflow)[:light_rep]
            th = [
                cp.asnumpy(q)[: light_rep * light_max_segments].reshape(
                    light_rep, light_max_segments
                )
                for q in light_traj_float
            ]
            event_grid_total = int(np.sum(grid_event_count, dtype=np.uint64))
            event_total = int(cp.asnumpy(light_event_total)[0])
            nonfinite_count = sum(
                int(np.count_nonzero(~np.isfinite(a)))
                for a in (grid_total, grid_post, grid_event_weight)
            )
            negative_count = sum(
                int(np.count_nonzero(a < 0))
                for a in (grid_total, grid_post, grid_event_weight)
            )
            outside_record = cp.asnumpy(light_event_outside_count)
            outside_example = None
            if int(outside_record[6]):
                xyz = outside_record[1:4].view(np.float32).astype(float)
                outside_example = {
                    'global_ray_id': int(outside_record[4]),
                    'x_m': float(xyz[0]),
                    'y_m': float(xyz[1]),
                    'z_m': float(xyz[2]),
                    'coordinate_tolerance_m': float(
                        outside_record[5:6].view(np.float32)[0]
                    ),
                    'x_bounds_m': [-float(self.rin), float(self.rin)],
                    'y_bounds_m': [-float(self.rin), float(self.rin)],
                    'z_bounds_m': [float(self.zmin), float(light_zmax)],
                }
            light_diag = {
                'track_length_m': grid_total,
                'post_scatter_track_length_m': grid_post,
                'scatter_event_weight': grid_event_weight,
                'scatter_event_count': grid_event_count,
                'nx': light_nx,
                'ny': light_ny,
                'nz': light_nz,
                'x_min_m': -self.rin,
                'x_max_m': self.rin,
                'y_min_m': -self.rin,
                'y_max_m': self.rin,
                'z_min_m': self.zmin,
                'z_max_m': light_zmax,
                'diagnostic_ray_limit': light_max_rays,
                'diagnostic_rays_processed': max(
                    0, min(total_rays, light_max_rays - ray_id_start)
                ),
                'representative_ray_limit': light_rep,
                'max_segments_per_ray': light_max_segments,
                'segment_count': int(cp.asnumpy(light_segment_count)[0]),
                'scatter_event_total': event_total,
                'deposited_scatter_event_total': event_grid_total,
                'scatter_event_closure_error': event_grid_total - event_total,
                'error_count': int(cp.asnumpy(light_error_count)[0]),
                'hard_error_count': int(cp.asnumpy(light_error_count)[0]),
                'boundary_adjustment_count': int(
                    cp.asnumpy(light_boundary_adjustment_count)[0]
                ),
                'event_outside_count': int(outside_record[0]),
                'first_outside_event': outside_example,
                'nonfinite_count': nonfinite_count,
                'negative_value_count': negative_count,
                'diagnostic_memory_bytes': int(light_bytes),
                'diagnostic_active_wall_time_s': float(
                    time.perf_counter() - light_diag_t0
                ),
                'segment_length_total_m': seg_total,
                'deposited_length_total_m': dep_total,
                'deposition_closure_relative_error': abs(dep_total - seg_total)
                / max(abs(seg_total), 1e-30),
                'grid_closure_relative_error': abs(
                    float(np.sum(grid_total, dtype=np.float64)) - seg_total
                )
                / max(abs(seg_total), 1e-30),
                'trajectory_count': tc,
                'trajectory_overflow': to,
                'trajectory_x0_m': th[0],
                'trajectory_y0_m': th[1],
                'trajectory_z0_m': th[2],
                'trajectory_x1_m': th[3],
                'trajectory_y1_m': th[4],
                'trajectory_z1_m': th[5],
                'trajectory_particle_order': cp.asnumpy(light_torder)[
                    : light_rep * light_max_segments
                ].reshape(light_rep, light_max_segments),
                'trajectory_event_code': cp.asnumpy(light_tevent)[
                    : light_rep * light_max_segments
                ].reshape(light_rep, light_max_segments),
                'selection_method': 'first global CUDA ray IDs; deterministic and independent of transport RNG',
                'normalization': 'raw metres of water-path track length or raw accepted scatter-event counts; normalized views divide by diagnostic_rays_processed',
                'ray_statistical_weight': 1.0,
            }
        guard_record = cp.asnumpy(light_event_outside_count)
        guard_example = None
        if int(guard_record[27]):
            guard_values = guard_record[11:25].view(np.float32).astype(float)
            guard_example = dict(
                zip(
                    (
                        'pre_x_m',
                        'pre_y_m',
                        'pre_z_m',
                        'vx',
                        'vy',
                        'vz',
                        'proposed_travel_m',
                        'side_distance_m',
                        'top_distance_m',
                        'bottom_distance_m',
                        'boundary_distance_m',
                        'result_x_m',
                        'result_y_m',
                        'result_z_m',
                    ),
                    map(float, guard_values),
                )
            )
            guard_example.update(
                global_ray_id=int(guard_record[8]),
                iteration=int(guard_record[9]),
                particle_selected=bool(guard_record[10]),
            )
        if ph is not None:
            ph['water_advance_guard_failure_count'] = int(guard_record[7])
            ph['first_water_advance_guard_failure'] = guard_example
        if light_diag is not None:
            light_diag['water_advance_guard_failure_count'] = int(guard_record[7])
            light_diag['first_water_advance_guard_failure'] = guard_example
        vg = vortex_surface_geometry(
            self.free_surface_model,
            self.zmean,
            self.rin,
            self.vortex_delta_h,
            self.vortex_core_radius,
            self.gravity,
        )
        pf_counts = {
            PRODUCTION_FAILURE_NAMES.get(i, f'CODE_{i}'): int(v)
            for i, v in enumerate(production_failure_code_counts)
            if i != 0 and int(v) != 0
        }
        pf_sub_counts = {
            str(k): int(v) for k, v in sorted(production_failure_subcode_counts.items())
        }
        production_watchdog_count = sum(
            int(production_failure_code_counts[i])
            for i in PRODUCTION_WATCHDOG_CODES
            if i < len(production_failure_code_counts)
        )
        production_geometry_failure_count = sum(
            int(production_failure_code_counts[i])
            for i in PRODUCTION_GEOMETRY_FAILURE_CODES
            if i < len(production_failure_code_counts)
        )
        production_nonfinite_count = sum(
            int(production_failure_code_counts[i])
            for i in PRODUCTION_NONFINITE_CODES
            if i < len(production_failure_code_counts)
        )
        production_failure_diagnostics = {
            'code_counts': pf_counts,
            'subcode_counts': pf_sub_counts,
            'samples': production_failure_samples,
            'sample_limit': production_failure_sample_limit,
            'count_closure_ok': int(production_failure_code_counts[1:].sum())
            == int(numerical_failure_total),
            'particle_proposal_count': int(particle_proposal_total),
            'particle_accessible_count': int(particle_accessible_total),
            'particle_null_count': int(particle_null_total),
            'acrylic_trapped_tir_count': int(acrylic_trapped_tir_total),
            'acrylic_trapped_tir_fraction': float(
                acrylic_trapped_tir_total / max(total_rays, 1)
            ),
        }
        result = ForwardResult(
            detector_angles_deg=DETECTOR_ANGLES_DEG.copy(),
            raw_hits=native_total.copy(),
            normalized_response=ns,
            native_exact_hits=native_total.copy(),
            mirror_exact_hits=mirror_total.copy(),
            symmetry_exact_scores=sym_total.copy(),
            normalized_native_exact_response=nn,
            normalized_mirror_exact_response=nm,
            normalized_symmetry_exact_response=ns,
            normalized_response_jackknife_se=se,
            hardware_native_hits=native_total.copy(),
            hardware_mirror_hits=mirror_total.copy(),
            hardware_symmetry_scores=hardware_sym_total.copy(),
            normalized_hardware_native_response=nn,
            normalized_hardware_mirror_response=nm,
            normalized_hardware_symmetry_response=nh,
            hardware_normalized_jackknife_se=hardware_se,
            legacy_native_channel_counts=legacy_native_total.copy(),
            legacy_mirror_channel_counts=legacy_mirror_total.copy(),
            legacy_symmetry_channel_scores=legacy_sym_total.copy(),
            normalized_legacy_native_response=lnn,
            normalized_legacy_mirror_response=lnm,
            normalized_legacy_symmetry_response=lns,
            legacy_normalized_jackknife_se=legacy_se,
            legacy_acceptance_deg=self.legacy_acceptance_deg,
            valid_exit_count=valid_air_total,
            numerical_failure_count=numerical_failure_total,
            numerical_failure_fraction=numerical_failure_total / max(total_rays, 1),
            production_failure_count=numerical_failure_total,
            production_failure_code_counts=pf_counts,
            production_failure_subcode_counts=pf_sub_counts,
            production_watchdog_count=production_watchdog_count,
            production_geometry_failure_count=production_geometry_failure_count,
            production_nonfinite_count=production_nonfinite_count,
            production_failure_diagnostics=production_failure_diagnostics,
            n_rays=total_rays,
            requested_n_rays=requested,
            stopped_adaptively=bool(adaptive and total_rays < maxN),
            stop_reason=stop_reason,
            particle_event_model=self.particle_event_model,
            particle_surface_rms_slope_deg=self.particle_surface_rms_slope_deg,
            particle_surface_beckmann_alpha=self.particle_surface_beckmann_alpha,
            wave_surface_rms_height_nm=self.wave_surface_rms_height_m * 1e9,
            wave_spheroid_aspect_ratio=self.wave_spheroid_aspect_ratio,
            wave_spheroid_orientation_model=self.wave_spheroid_orientation_model,
            wave_spheroid_orientation_kappa=self.wave_spheroid_orientation_kappa,
            aggregation_model=ph['aggregation_model'],
            aggregation_collision_exposure=ph['aggregation_collision_exposure'],
            aggregation_number_reduction_fraction=ph[
                'aggregation_number_reduction_fraction'
            ],
            aggregation_mass_conservation_error=ph[
                'aggregation_mass_conservation_error'
            ],
            aggregation_projected_area_ratio=ph['aggregation_projected_area_ratio'],
            aggregation_largest_bin_mass_fraction=ph[
                'aggregation_largest_bin_mass_fraction'
            ],
            aggregation_overflow_event_fraction=ph[
                'aggregation_overflow_event_fraction'
            ],
            aggregation_steps=ph['aggregation_steps'],
            population_number_reduction_fraction=ph[
                'population_number_reduction_fraction'
            ],
            fragmentation_breakup_exposure=ph['fragmentation_breakup_exposure'],
            fragmentation_reference_diameter_m=ph['fragmentation_reference_diameter_m'],
            fragmentation_size_exponent=ph['fragmentation_size_exponent'],
            fragmentation_underflow_event_fraction=ph[
                'fragmentation_underflow_event_fraction'
            ],
            aggregation_events_per_initial_particle=ph[
                'aggregation_events_per_initial_particle'
            ],
            fragmentation_events_per_initial_particle=ph[
                'fragmentation_events_per_initial_particle'
            ],
            wave_coherent_fraction_nominal_cuda=ph[
                'wave_coherent_fraction_nominal_cuda'
            ],
            mean_coherent_wave_interactions=(
                sum_wcoh / denom_enter if entered_total else float('nan')
            ),
            mean_morphology_wave_interactions=(
                sum_wmorph / denom_enter if entered_total else float('nan')
            ),
            morphology_fraction_of_wave_interactions=(
                sum_wmorph / sum_wic if sum_wic > 0 else 0.0
            ),
            mu_geom_per_m=ph['mu_geom'],
            mu_wave_per_m=ph['mu_wave'],
            mu_event_per_m=ph['mu_event'],
            mu_sca_per_m=ph['mu_sca'],
            mu_ext_per_m=ph['mu_ext'],
            geometric_optical_depth_diameter=ph['mu_geom'] * (2 * self.rin),
            optical_depth_diameter=ph['mu_event'] * (2 * self.rin),
            mean_interactions=sum_ic / denom_enter if entered_total else float('nan'),
            mean_geometric_interactions=(
                sum_gic / denom_enter if entered_total else float('nan')
            ),
            mean_wave_interactions=(
                sum_wic / denom_enter if entered_total else float('nan')
            ),
            median_interactions=_histogram_median(ic_hist),
            ballistic_fraction=(
                ballistic_total / denom_enter if entered_total else float('nan')
            ),
            mean_fresnel_reflections=(
                sum_pr / denom_enter if entered_total else float('nan')
            ),
            entry_reflection_fraction=(
                rays_entry_pr / denom_enter if entered_total else float('nan')
            ),
            internal_reflection_fraction=(
                rays_internal_pr / denom_enter if entered_total else float('nan')
            ),
            particle_absorption_fraction=(
                particle_absorbed_total / denom_enter if entered_total else float('nan')
            ),
            particle_absorbed_ray_count=particle_absorbed_total,
            mean_cell_inner_reflections=(
                sum_cir / denom_enter if entered_total else float('nan')
            ),
            mean_cell_outer_reflections=(
                sum_cor / denom_enter if entered_total else float('nan')
            ),
            cell_tir_fraction=(
                rays_ctir / denom_enter if entered_total else float('nan')
            ),
            mean_top_surface_reflections=(
                sum_top_ref / denom_enter if entered_total else float('nan')
            ),
            top_surface_tir_fraction=(
                rays_top_tir / denom_enter if entered_total else float('nan')
            ),
            mean_top_surface_interactions=(
                (sum_top_ref + top_exit_total) / denom_enter
                if entered_total
                else float('nan')
            ),
            free_surface_model=self.free_surface_model,
            vortex_wall_center_height_difference_mm=self.vortex_delta_h * 1e3,
            vortex_core_radius_mm=(
                self.vortex_core_radius * 1e3
                if self.free_surface_model == 'localized_scully_vortex'
                else float('nan')
            ),
            vortex_asymptotic_height_parameter_mm=vg['h_inf'] * 1e3,
            vortex_area_mean_uncorrected_rise_mm=vg['h_bar'] * 1e3,
            vortex_equivalent_core_rpm=vg['equivalent_core_rpm'],
            vortex_equivalent_rigid_body_rpm=vg['equivalent_rigid_body_rpm'],
            vortex_center_surface_z_mm=vg['center_z'] * 1e3,
            vortex_wall_surface_z_mm=vg['wall_z'] * 1e3,
            sediment_transport_model=self.sediment_transport_model,
            sediment_eddy_diffusivity_m2_s=self.sediment_eddy_diffusivity,
            sediment_majorant_geom_per_m=ph['sediment_majorant_geom_per_m'],
            sediment_majorant_wave_per_m=ph['sediment_majorant_wave_per_m'],
            sediment_majorant_total_per_m=ph['sediment_majorant_total_per_m'],
            sediment_max_bin_density_scale=(
                float(np.max(ph['sediment_max_density_scale']))
                if len(ph['sediment_max_density_scale'])
                else 1.0
            ),
            meridional_circulation_ratio=self.meridional_circulation_ratio,
            meridional_peak_speed_m_s=self.meridional_peak_speed,
            meridional_grid_nr=int(ph['meridional_grid_nr']),
            meridional_grid_nz=int(ph['meridional_grid_nz']),
            meridional_solver_iterations=int(ph['meridional_solver_iterations']),
            meridional_solver_max_residual=float(ph['meridional_solver_max_residual']),
            mean_bottom_inner_reflections=(
                sum_bottom_inner / denom_enter if entered_total else float('nan')
            ),
            mean_bottom_outer_reflections=(
                sum_bottom_outer / denom_enter if entered_total else float('nan')
            ),
            bottom_tir_fraction=(
                rays_bottom_tir / denom_enter if entered_total else float('nan')
            ),
            top_exit_fraction=top_exit_total / max(total_rays, 1),
            bottom_exit_fraction=bottom_exit_total / max(total_rays, 1),
            source_bore_surface_model=self.source_bore_surface_model,
            detector_bore_surface_model=self.detector_bore_surface_model,
            bore_refractive_index=self.bore_refractive_index,
            ring_inner_face_surface_model=self.ring_inner_face_surface_model,
            ring_inner_face_refractive_index=self.ring_inner_face_refractive_index,
            max_ring_face_bounces=self.max_ring_face_bounces,
            source_collimator_acceptance_fraction=source_accept_total
            / max(total_rays, 1),
            source_direct_acceptance_fraction=source_direct_accept_total
            / max(total_rays, 1),
            source_reflected_acceptance_fraction=source_reflected_accept_total
            / max(total_rays, 1),
            source_bore_interactions_per_ray=source_bore_interactions_total
            / max(total_rays, 1),
            source_bore_reflections_per_ray=source_bore_reflections_total
            / max(total_rays, 1),
            hardware_native_direct_hits=native_direct_total.copy(),
            hardware_mirror_direct_hits=mirror_direct_total.copy(),
            hardware_symmetry_direct_scores=direct_sym.copy(),
            hardware_native_stray_hits=native_stray_total.copy(),
            hardware_mirror_stray_hits=mirror_stray_total.copy(),
            hardware_symmetry_stray_scores=stray_sym.copy(),
            detector_stray_fraction_by_channel=det_stray_frac.copy(),
            detector_stray_detection_fraction=float(stray_sym.sum())
            / max(total_rays, 1),
            hardware_native_ring_face_stray_hits=native_face_total.copy(),
            hardware_mirror_ring_face_stray_hits=mirror_face_total.copy(),
            hardware_symmetry_ring_face_stray_scores=face_sym.copy(),
            detector_ring_face_stray_fraction_by_channel=det_face_frac.copy(),
            detector_ring_face_stray_detection_fraction=float(face_sym.sum())
            / max(total_rays, 1),
            photodiode_response_model=self.photodiode_response_model,
            sfh213_half_angle_deg=10.0,
            sfh213_directional_fit_exponent=3.169925001442312,
            photodiode_native_weighted_scores=photodiode_native_total.copy(),
            photodiode_mirror_weighted_scores=photodiode_mirror_total.copy(),
            photodiode_symmetry_weighted_scores=photodiode_sym_total.copy(),
            normalized_photodiode_native_response=pnn,
            normalized_photodiode_mirror_response=pnm,
            normalized_photodiode_symmetry_response=pns,
            photodiode_relative_response_by_channel=photodiode_relative.copy(),
            detector_mean_incidence_angle_deg_by_channel=mean_incidence.copy(),
            hardware_candidate_detection_fraction=hardware_ssum / max(total_rays, 1),
            entered_water_fraction=entered_total / max(total_rays, 1),
            air_exit_fraction=valid_air_total / max(total_rays, 1),
            axial_loss_fraction=axial_total / max(total_rays, 1),
            detector_detection_fraction=ssum / max(total_rays, 1),
            native_detector_detection_fraction=selected_native_sum / max(total_rays, 1),
            mirror_detector_detection_fraction=selected_mirror_sum / max(total_rays, 1),
            legacy_channel_score_per_ray=lssum / max(total_rays, 1),
            symmetry_variance_reduction=self.symmetry_average_exact,
            batch_diagnostics=batch_rows,
            particle_diagnostics=ph,
            detector_particle_path_diagnostics=path_diag,
            ray_data=ray_data,
            heatmap_xy=hmxy,
            heatmap_zy=hmzy,
        )

        result.light_transport_diagnostics = light_diag
        return result


# ============================================================================
# STANDALONE RUNTIME / REPORTING / PLOTTING
# ============================================================================

# Constructor template. _build_model applies current USER CONFIGURATION values.
MODEL_DEFAULT_KWARGS = dict(
    n_water=1.33,
    n_particle=1.59,
    n_acrylic=1.4906,
    n_air=1.0,
    tube_inner_radius_m=0.0465,
    tube_outer_radius_m=0.0500,
    tube_length_m=0.491,
    water_height_m=0.142,
    sensor_height_above_bottom_m=0.093,
    sensor_ring_inner_radius_m=0.0505,
    sensor_ring_outer_radius_m=0.0655,
    counterbore_depth_m=0.008,
    through_bore_diameter_m=0.004,
    counterbore_diameter_m=0.0087,
    source_launch_radius_m=0.0655,
    legacy_detector_acceptance_deg=6.5,
    source_beam_sigma_m=1e-5,
    source_bore_surface_model='absorbing',
    detector_bore_surface_model='absorbing',
    bore_refractive_index=1.53,
    max_bore_bounces=16,
    ring_inner_face_surface_model='absorbing',
    ring_inner_face_refractive_index=1.53,
    max_ring_face_bounces=16,
    photodiode_response_model='ideal_hard_aperture',
    alpha1=0.45,
    alpha2=5.0,
    source_angular_model='beta_radiance_3d',
    density_kg_per_m3=2600.0,
    bottom_disc_thickness_m=0.003,
    bottom_external_refractive_index=1.0,
    max_axial_bounces=64,
    max_internal_bounces=4095,
    max_cell_bounces=64,
    symmetry_average_exact=True,
    particle_event_model='hybrid_mie_exact_rate',
    hybrid_wave_scale=1.0,
    mie_phase_grid_size=2049,
    particle_surface_rms_slope_deg=25.0,
    wave_surface_rms_height_m=100e-9,
    wave_spheroid_aspect_ratio=1.0,
    wave_spheroid_orientation_model='isotropic',
    wave_spheroid_orientation_kappa=0.0,
    free_surface_model='localized_scully_vortex',
    vortex_wall_center_height_difference_m=0.050,
    vortex_core_radius_m=0.020,
    sediment_transport_model='schiller_naumann_drift_diffusion',
    sediment_eddy_diffusivity_m2_s=1e-3,
    meridional_circulation_ratio=0.0,
    meridional_grid_nr=48,
    meridional_grid_nz=96,
    meridional_solver_max_iterations=8000,
    meridional_solver_check_interval=250,
    meridional_solver_tolerance=1e-6,
    meridional_solver_sor=1.75,
    water_density_kg_per_m3=997.0,
    water_dynamic_viscosity_pa_s=8.9e-4,
    gravity_m_s2=9.81,
    sediment_transport_quadrature_points=4096,
    sediment_radial_lookup_points=2049,
    aggregation_model='none',
    aggregation_collision_exposure=0.0,
    fragmentation_breakup_exposure=0.0,
    fragmentation_reference_diameter_m=100e-6,
    fragmentation_size_exponent=1.0,
    aggregation_step_safety=0.2,
    aggregation_max_steps=200000,
)


# ============================================================================
# GEOMETRY / STEADY-STATE SEDIMENT ADAPTERS
# ============================================================================

# Supplied comparison settings; not the current user configuration or proof
# of absolute-amplitude validation. Match tests check settings only.
COMPARISON_GEOMETRY = {
    'sample_cell_diameter_m': 0.093,
    'sample_cell_height_m': 0.491,
    'water_level_m': 0.142,
    'sensor_ring_height_m': 0.093,
    'acrylic_wall_thickness_m': 0.0035,
    'sensor_ring_inner_clearance_m': 0.0005,
    'sensor_ring_radial_width_m': 0.0150,
    'counterbore_depth_m': 0.008,
    'through_bore_diameter_m': 0.004,
    'counterbore_diameter_m': 0.0087,
    'bottom_disc_thickness_m': 0.003,
}

COMPARISON_SEDIMENT = {
    'sediment_distribution': 'steady_state',
    'sediment_transport_model': 'schiller_naumann_drift_diffusion',
    'sediment_eddy_diffusivity_m2_s': 1.0e-3,
    'sediment_particle_density_kg_m3': 2600.0,
    'water_density_kg_m3': 997.0,
    'water_dynamic_viscosity_pa_s': 8.9e-4,
    'gravity_m_s2': 9.81,
    'meridional_circulation_ratio': 0.0,
    'enable_vortex': True,
    'vortex_model': 'localized_scully_vortex',
    'vortex_wall_center_height_difference_m': 0.050,
    'vortex_core_radius_m': 0.020,
    'sediment_transport_quadrature_points': 4096,
    'sediment_radial_lookup_points': 2049,
    'meridional_grid_nr': 48,
    'meridional_grid_nz': 96,
    'meridional_solver_max_iterations': 8000,
    'meridional_solver_check_interval': 250,
    'meridional_solver_tolerance': 1.0e-6,
    'meridional_solver_sor': 1.75,
}


def _isclose(a, b, rtol=0.0, atol=1e-12):
    return bool(np.isclose(float(a), float(b), rtol=rtol, atol=atol))


def _resolved_geometry():
    rin = 0.5 * float(SAMPLE_CELL_DIAMETER_M)
    rout = rin + float(ACRYLIC_WALL_THICKNESS_M)
    ring_in = rout + float(SENSOR_RING_INNER_CLEARANCE_M)
    ring_out = ring_in + float(SENSOR_RING_RADIAL_WIDTH_M)
    source_launch = (
        ring_out if SOURCE_LAUNCH_RADIUS_M is None else float(SOURCE_LAUNCH_RADIUS_M)
    )
    return {
        'sample_cell_diameter_m': float(SAMPLE_CELL_DIAMETER_M),
        'sample_cell_radius_m': rin,
        'sample_cell_height_m': float(SAMPLE_CELL_HEIGHT_M),
        'water_level_m': float(WATER_LEVEL_M),
        'water_depth_m': float(WATER_LEVEL_M),
        'sensor_ring_height_m': float(SENSOR_RING_HEIGHT_M),
        'acrylic_wall_thickness_m': float(ACRYLIC_WALL_THICKNESS_M),
        'tube_outer_radius_m': rout,
        'sensor_ring_inner_clearance_m': float(SENSOR_RING_INNER_CLEARANCE_M),
        'sensor_ring_inner_radius_m': ring_in,
        'sensor_ring_radial_width_m': float(SENSOR_RING_RADIAL_WIDTH_M),
        'sensor_ring_outer_radius_m': ring_out,
        'source_launch_radius_m': source_launch,
        'counterbore_depth_m': float(COUNTERBORE_DEPTH_M),
        'through_bore_diameter_m': float(THROUGH_BORE_DIAMETER_M),
        'counterbore_diameter_m': float(COUNTERBORE_DIAMETER_M),
        'bottom_disc_thickness_m': float(BOTTOM_DISC_THICKNESS_M),
        'height_reference': 'metres above the internal bottom/water-base plane; production CUDA uses sensor plane z=0 internally',
    }


def _resolved_sediment_config():
    distribution = str(SEDIMENT_DISTRIBUTION).strip().lower()
    steady = distribution == 'steady_state'
    model = str(SEDIMENT_TRANSPORT_MODEL).strip().lower() if steady else 'uniform'
    surface_model = str(VORTEX_MODEL).strip().lower() if bool(ENABLE_VORTEX) else 'flat'
    delta_h = (
        float(VORTEX_WALL_CENTER_HEIGHT_DIFFERENCE_M) if bool(ENABLE_VORTEX) else 0.0
    )
    return {
        'sediment_distribution': distribution,
        'steady_state_sediment_enabled': steady,
        'sediment_transport_model': model,
        'configured_steady_state_transport_model': str(SEDIMENT_TRANSPORT_MODEL)
        .strip()
        .lower(),
        'sediment_eddy_diffusivity_m2_s': float(SEDIMENT_EDDY_DIFFUSIVITY_M2_S),
        'sediment_particle_density_kg_m3': float(SEDIMENT_PARTICLE_DENSITY_KG_M3),
        'water_density_kg_m3': float(WATER_DENSITY_KG_M3),
        'water_dynamic_viscosity_pa_s': float(WATER_DYNAMIC_VISCOSITY_PA_S),
        'gravity_m_s2': float(GRAVITY_M_S2),
        'meridional_circulation_ratio': float(MERIDIONAL_CIRCULATION_RATIO),
        'enable_vortex': bool(ENABLE_VORTEX),
        'free_surface_model': surface_model,
        'vortex_model': str(VORTEX_MODEL).strip().lower(),
        'vortex_wall_center_height_difference_m': delta_h,
        'configured_vortex_wall_center_height_difference_m': float(
            VORTEX_WALL_CENTER_HEIGHT_DIFFERENCE_M
        ),
        'vortex_core_radius_m': float(VORTEX_CORE_RADIUS_M),
        'sediment_transport_quadrature_points': int(
            SEDIMENT_TRANSPORT_QUADRATURE_POINTS
        ),
        'sediment_radial_lookup_points': int(SEDIMENT_RADIAL_LOOKUP_POINTS),
        'meridional_grid_nr': int(MERIDIONAL_GRID_NR),
        'meridional_grid_nz': int(MERIDIONAL_GRID_NZ),
        'meridional_solver_max_iterations': int(MERIDIONAL_SOLVER_MAX_ITERATIONS),
        'meridional_solver_check_interval': int(MERIDIONAL_SOLVER_CHECK_INTERVAL),
        'meridional_solver_tolerance': float(MERIDIONAL_SOLVER_TOLERANCE),
        'meridional_solver_sor': float(MERIDIONAL_SOLVER_SOR),
    }


def _resolved_light_transport_config():
    enabled = bool(SAVE_LIGHT_TRANSPORT_VISUALISATIONS or SAVE_LIGHT_TRANSPORT_DATA)
    return {
        'enabled': enabled,
        'save_visualisations': bool(SAVE_LIGHT_TRANSPORT_VISUALISATIONS),
        'save_data': bool(SAVE_LIGHT_TRANSPORT_DATA),
        'max_rays': int(LIGHT_TRANSPORT_DIAGNOSTIC_MAX_RAYS),
        'nx': int(LIGHT_TRANSPORT_GRID_NX),
        'ny': int(LIGHT_TRANSPORT_GRID_NY),
        'nz': int(LIGHT_TRANSPORT_GRID_NZ),
        'representative_rays': int(LIGHT_TRANSPORT_REPRESENTATIVE_RAYS),
        'max_segments_per_ray': int(LIGHT_TRANSPORT_MAX_SEGMENTS_PER_RAY),
        'vertical_slice_thickness_m': float(LIGHT_TRANSPORT_VERTICAL_SLICE_THICKNESS_M),
        'sensor_slice_thickness_m': float(LIGHT_TRANSPORT_SENSOR_SLICE_THICKNESS_M),
        'log_floor': float(LIGHT_TRANSPORT_LOG_FLOOR),
        'display_low_percentile': float(LIGHT_TRANSPORT_DISPLAY_LOW_PERCENTILE),
        'display_high_percentile': float(LIGHT_TRANSPORT_DISPLAY_HIGH_PERCENTILE),
        'display_3d_min_percentile': float(LIGHT_TRANSPORT_3D_MIN_DISPLAY_PERCENTILE),
        'display_3d_max_voxels': int(LIGHT_TRANSPORT_3D_MAX_DISPLAY_VOXELS),
        'display_3d_alpha_min': float(LIGHT_TRANSPORT_3D_ALPHA_MIN),
        'display_3d_alpha_max': float(LIGHT_TRANSPORT_3D_ALPHA_MAX),
        'display_3d_marker_size_min': float(LIGHT_TRANSPORT_3D_MARKER_SIZE_MIN),
        'display_3d_marker_size_max': float(LIGHT_TRANSPORT_3D_MARKER_SIZE_MAX),
        'display_3d_surface_margin_m': float(LIGHT_TRANSPORT_3D_SURFACE_MARGIN_M),
    }


def _matches_comparison_geometry(g=None):
    g = _resolved_geometry() if g is None else g
    for k, v in COMPARISON_GEOMETRY.items():
        if not _isclose(g[k], v):
            return False
    return bool(
        SOURCE_LAUNCH_RADIUS_M is None or _isclose(g['source_launch_radius_m'], 0.0655)
    )


def _matches_comparison_sediment(d=None):
    d = _resolved_sediment_config() if d is None else d
    checks = {
        'sediment_distribution': d['sediment_distribution']
        == COMPARISON_SEDIMENT['sediment_distribution'],
        'sediment_transport_model': d['sediment_transport_model']
        == COMPARISON_SEDIMENT['sediment_transport_model'],
        'enable_vortex': d['enable_vortex'] == COMPARISON_SEDIMENT['enable_vortex'],
        'vortex_model': d['vortex_model'] == COMPARISON_SEDIMENT['vortex_model'],
    }
    if not all(checks.values()):
        return False
    for k in (
        'sediment_eddy_diffusivity_m2_s',
        'sediment_particle_density_kg_m3',
        'water_density_kg_m3',
        'water_dynamic_viscosity_pa_s',
        'gravity_m_s2',
        'meridional_circulation_ratio',
        'vortex_wall_center_height_difference_m',
        'vortex_core_radius_m',
        'sediment_transport_quadrature_points',
        'sediment_radial_lookup_points',
        'meridional_grid_nr',
        'meridional_grid_nz',
        'meridional_solver_max_iterations',
        'meridional_solver_check_interval',
        'meridional_solver_tolerance',
        'meridional_solver_sor',
    ):
        if not _isclose(d[k], COMPARISON_SEDIMENT[k]):
            return False
    return True


def _validate_user_configuration():
    g = _resolved_geometry()
    d = _resolved_sediment_config()
    ld = _resolved_light_transport_config()
    if not (
        np.isfinite(g['sample_cell_diameter_m']) and g['sample_cell_diameter_m'] > 0
    ):
        raise ValueError('SAMPLE_CELL_DIAMETER_M must be finite and > 0')
    if not (np.isfinite(g['sample_cell_height_m']) and g['sample_cell_height_m'] > 0):
        raise ValueError('SAMPLE_CELL_HEIGHT_M must be finite and > 0')
    if not (
        np.isfinite(g['water_level_m'])
        and 0 < g['water_level_m'] <= g['sample_cell_height_m']
    ):
        raise ValueError(
            'WATER_LEVEL_M must satisfy 0 < WATER_LEVEL_M <= SAMPLE_CELL_HEIGHT_M'
        )
    if not (
        np.isfinite(g['sensor_ring_height_m'])
        and 0 < g['sensor_ring_height_m'] < g['sample_cell_height_m']
    ):
        raise ValueError(
            'SENSOR_RING_HEIGHT_M must lie strictly between the internal bottom and wall top'
        )
    if g['sensor_ring_height_m'] >= g['water_level_m']:
        raise ValueError(
            'detector geometry requires SENSOR_RING_HEIGHT_M < WATER_LEVEL_M (detector plane submerged)'
        )
    for k in (
        'acrylic_wall_thickness_m',
        'sensor_ring_inner_clearance_m',
        'sensor_ring_radial_width_m',
        'counterbore_depth_m',
        'through_bore_diameter_m',
        'counterbore_diameter_m',
        'bottom_disc_thickness_m',
    ):
        if not (np.isfinite(g[k]) and g[k] > 0):
            raise ValueError(f'{k} must be finite and > 0')
    if not (g['through_bore_diameter_m'] < g['counterbore_diameter_m']):
        raise ValueError(
            'THROUGH_BORE_DIAMETER_M must be smaller than COUNTERBORE_DIAMETER_M'
        )
    # Matches the production constructor's radial topology requirement.
    if not (
        0
        < g['sample_cell_radius_m']
        < g['tube_outer_radius_m']
        < g['sensor_ring_inner_radius_m']
        < g['sensor_ring_outer_radius_m'] - g['counterbore_depth_m']
        <= g['source_launch_radius_m']
        <= g['sensor_ring_outer_radius_m']
    ):
        raise ValueError(
            'resolved sample-cell / ring / source radial geometry is inconsistent'
        )
    if str(SEDIMENT_DISTRIBUTION).strip().lower() not in {'uniform', 'steady_state'}:
        raise ValueError('SEDIMENT_DISTRIBUTION must be "uniform" or "steady_state"')
    if str(SEDIMENT_TRANSPORT_MODEL).strip().lower() not in {
        'stokes_drift_diffusion',
        'schiller_naumann_drift_diffusion',
        'schiller_naumann_meridional_advection_diffusion',
    }:
        raise ValueError(
            'SEDIMENT_TRANSPORT_MODEL is invalid; use one of the documented steady-state models'
        )
    if d['vortex_model'] not in {'localized_scully_vortex', 'parabolic_vortex'}:
        raise ValueError(
            "VORTEX_MODEL must be 'localized_scully_vortex' or 'parabolic_vortex'"
        )
    if d[
        'sediment_transport_model'
    ] == 'schiller_naumann_meridional_advection_diffusion' and (
        not d['enable_vortex'] or d['vortex_wall_center_height_difference_m'] <= 0.0
    ):
        raise ValueError(
            'meridional advection-diffusion requires an enabled non-zero supported vortex'
        )
    if not (
        np.isfinite(d['sediment_eddy_diffusivity_m2_s'])
        and d['sediment_eddy_diffusivity_m2_s'] > 0
    ):
        raise ValueError('SEDIMENT_EDDY_DIFFUSIVITY_M2_S must be finite and > 0')
    if not (
        np.isfinite(d['sediment_particle_density_kg_m3'])
        and d['sediment_particle_density_kg_m3'] > d['water_density_kg_m3'] > 0
    ):
        raise ValueError('sediment particle density must exceed positive water density')
    if not (
        np.isfinite(d['water_dynamic_viscosity_pa_s'])
        and d['water_dynamic_viscosity_pa_s'] > 0
    ):
        raise ValueError('WATER_DYNAMIC_VISCOSITY_PA_S must be finite and > 0')
    if not (np.isfinite(d['gravity_m_s2']) and d['gravity_m_s2'] > 0):
        raise ValueError('GRAVITY_M_S2 must be finite and > 0')
    if d['free_surface_model'] == 'localized_scully_vortex' and not (
        np.isfinite(d['vortex_core_radius_m']) and d['vortex_core_radius_m'] > 0
    ):
        raise ValueError(
            'VORTEX_CORE_RADIUS_M must be finite and > 0 for localized_scully_vortex'
        )
    if not (
        np.isfinite(float(VORTEX_WALL_CENTER_HEIGHT_DIFFERENCE_M))
        and float(VORTEX_WALL_CENTER_HEIGHT_DIFFERENCE_M) >= 0
    ):
        raise ValueError(
            'VORTEX_WALL_CENTER_HEIGHT_DIFFERENCE_M must be finite and >= 0'
        )
    if not (0.0 <= d['meridional_circulation_ratio'] <= 0.5):
        raise ValueError('MERIDIONAL_CIRCULATION_RATIO must lie in [0, 0.5]')
    if int(SEDIMENT_OUTPUT_NR) < 8 or int(SEDIMENT_OUTPUT_NZ) < 8:
        raise ValueError('SEDIMENT_OUTPUT_NR and SEDIMENT_OUTPUT_NZ must each be >= 8')
    if int(SEDIMENT_PROFILE_HEIGHT_POINTS) < 16:
        raise ValueError('SEDIMENT_PROFILE_HEIGHT_POINTS must be >= 16')
    if not SEDIMENT_PROFILE_RADII_FRACTION:
        raise ValueError('SEDIMENT_PROFILE_RADII_FRACTION must not be empty')
    if any(
        (not np.isfinite(float(x))) or float(x) < 0.0 or float(x) > 1.0
        for x in SEDIMENT_PROFILE_RADII_FRACTION
    ):
        raise ValueError('SEDIMENT_PROFILE_RADII_FRACTION values must lie in [0, 1]')
    if not PSD_PROFILE_HEIGHT_FRACTIONS:
        raise ValueError('PSD_PROFILE_HEIGHT_FRACTIONS must not be empty')
    if any(
        (not np.isfinite(float(x))) or float(x) < 0.0 or float(x) > 1.0
        for x in PSD_PROFILE_HEIGHT_FRACTIONS
    ):
        raise ValueError('PSD_PROFILE_HEIGHT_FRACTIONS values must lie in [0, 1]')
    if int(VORTEX_OUTPUT_NR) < 16:
        raise ValueError('VORTEX_OUTPUT_NR must be >= 16')
    # Validate the free surface against the physical wall before CuPy/CUDA is touched.
    zmin = -g['sensor_ring_height_m']
    zmean = g['water_level_m'] - g['sensor_ring_height_m']
    zwall_top = g['sample_cell_height_m'] - g['sensor_ring_height_m']
    vg = vortex_surface_geometry(
        d['free_surface_model'],
        zmean,
        g['sample_cell_radius_m'],
        d['vortex_wall_center_height_difference_m'],
        d['vortex_core_radius_m'],
        d['gravity_m_s2'],
    )
    if not (float(vg['wall_z']) < zwall_top):
        raise ValueError(
            'configured vortex/free surface reaches or exceeds the physical sample-cell wall top'
        )
    if float(vg['center_z']) <= zmin:
        raise ValueError(
            'configured vortex depression reaches or passes the internal bottom'
        )
    if ld['enabled']:
        if ld['max_rays'] < 1:
            raise ValueError('LIGHT_TRANSPORT_DIAGNOSTIC_MAX_RAYS must be >= 1')
        if min(ld['nx'], ld['ny'], ld['nz']) < 4:
            raise ValueError('LIGHT_TRANSPORT_GRID_NX/NY/NZ must each be >= 4')
        if ld['representative_rays'] < 0 or ld['representative_rays'] > ld['max_rays']:
            raise ValueError(
                'LIGHT_TRANSPORT_REPRESENTATIVE_RAYS must lie in [0, diagnostic max rays]'
            )
        if ld['max_segments_per_ray'] < 1:
            raise ValueError('LIGHT_TRANSPORT_MAX_SEGMENTS_PER_RAY must be >= 1')
        if not (
            np.isfinite(ld['vertical_slice_thickness_m'])
            and ld['vertical_slice_thickness_m'] > 0
            and np.isfinite(ld['sensor_slice_thickness_m'])
            and ld['sensor_slice_thickness_m'] > 0
        ):
            raise ValueError('light-transport slice thicknesses must be finite and > 0')
        if not (np.isfinite(ld['log_floor']) and ld['log_floor'] > 0):
            raise ValueError('LIGHT_TRANSPORT_LOG_FLOOR must be finite and > 0')
        if not (
            np.isfinite(ld['display_low_percentile'])
            and np.isfinite(ld['display_high_percentile'])
            and 0.0
            <= ld['display_low_percentile']
            < ld['display_high_percentile']
            <= 100.0
        ):
            raise ValueError(
                'LIGHT_TRANSPORT_DISPLAY percentiles must satisfy 0 <= low < high <= 100'
            )
        if not (
            np.isfinite(ld['display_3d_min_percentile'])
            and 0.0 <= ld['display_3d_min_percentile'] <= 100.0
        ):
            raise ValueError(
                'LIGHT_TRANSPORT_3D_MIN_DISPLAY_PERCENTILE must lie in [0, 100]'
            )
        if ld['display_3d_max_voxels'] < 1 or ld['display_3d_max_voxels'] > 2_000_000:
            raise ValueError(
                'LIGHT_TRANSPORT_3D_MAX_DISPLAY_VOXELS must lie in [1, 2000000]'
            )
        if not (
            np.isfinite(ld['display_3d_alpha_min'])
            and np.isfinite(ld['display_3d_alpha_max'])
            and 0.0 <= ld['display_3d_alpha_min'] <= ld['display_3d_alpha_max'] <= 1.0
        ):
            raise ValueError(
                'LIGHT_TRANSPORT_3D_ALPHA_MIN/MAX must satisfy 0 <= min <= max <= 1'
            )
        if not (
            np.isfinite(ld['display_3d_marker_size_min'])
            and np.isfinite(ld['display_3d_marker_size_max'])
            and 0.0
            < ld['display_3d_marker_size_min']
            <= ld['display_3d_marker_size_max']
            <= 1000.0
        ):
            raise ValueError(
                'LIGHT_TRANSPORT_3D_MARKER_SIZE_MIN/MAX must satisfy 0 < min <= max <= 1000'
            )
        if not (
            np.isfinite(ld['display_3d_surface_margin_m'])
            and 0.0
            <= ld['display_3d_surface_margin_m']
            <= 0.25 * g['sample_cell_height_m']
        ):
            raise ValueError(
                'LIGHT_TRANSPORT_3D_SURFACE_MARGIN_M must be finite and lie in [0, 0.25*cell height]'
            )
        requested_bytes = (
            ld['nx'] * ld['ny'] * ld['nz'] * 16
            + max(ld['representative_rays'] * ld['max_segments_per_ray'], 1) * 32
            + max(ld['representative_rays'], 1) * 8
        )
        if requested_bytes > 1_000_000_000:
            raise ValueError(
                f'light diagnostic allocation is too large ({requested_bytes/1e9:.3g} GB); reduce grid dimensions, representative rays, or segments per ray'
            )
    return g, d


class ReportingClaritasForwardModel(ClaritasForwardModel):
    """Production model with host copies of sediment fields for reporting.

    The superclass performs every production precompute and CUDA transport step.
    This override does not change any device input or random sampling; it only
    copies already-generated sediment fields back to the host after precompute.
    """

    def _prepare_gpu_particle_physics(self, *args, **kwargs):
        prep = super()._prepare_gpu_particle_physics(*args, **kwargs)
        cp = self.cp
        dev = prep.get('device', {})
        host = prep.get('host', {})
        try:
            host['_standalone_sediment_radial_log'] = cp.asnumpy(dev['sed_radial_log'])
            host['_standalone_sediment_grid'] = cp.asnumpy(dev['sed_grid'])
            host['_standalone_sediment_grid_nr'] = int(dev['sed_grid_nr'])
            host['_standalone_sediment_grid_nz'] = int(dev['sed_grid_nz'])
            host['_standalone_sediment_grid_zmax_m'] = float(dev['sed_grid_zmax'])
        except Exception:
            # Reporting data must never modify or mask a production transport failure.
            host['_standalone_sediment_radial_log'] = np.array([], dtype=np.float64)
            host['_standalone_sediment_grid'] = np.array([], dtype=np.float64)
        return prep


def _free_surface_z_sensor_frame(r_m, g, d):
    r = np.asarray(r_m, dtype=np.float64)
    R = float(g['sample_cell_radius_m'])
    zmean = float(g['water_level_m'] - g['sensor_ring_height_m'])
    dh = float(d['vortex_wall_center_height_difference_m'])
    a = float(d['vortex_core_radius_m'])
    model = d['free_surface_model']
    if model == 'flat' or dh <= 0.0:
        return np.full_like(r, zmean, dtype=np.float64)
    if model == 'parabolic_vortex':
        return zmean + dh * (r * r / (R * R) - 0.5)
    vg = vortex_surface_geometry(
        'localized_scully_vortex', zmean, R, dh, a, d['gravity_m_s2']
    )
    return zmean - float(vg['h_bar']) + float(vg['h_inf']) * r * r / (a * a + r * r)


def _particle_state_arrays(ph):
    """Return production per-bin arrays needed for reporting-only local diagnostics."""
    required = ('number_density_by_bin', 'particle_mass_kg', 'diameter_m')
    if not ph or any(k not in ph for k in required):
        return None
    a = {
        k: np.asarray(ph.get(k, []), dtype=np.float64)
        for k in (
            'number_density_by_bin',
            'particle_mass_kg',
            'diameter_m',
            'geometric_cross_section_m2',
            'scattering_cross_section_m2',
            'extinction_cross_section_m2',
            'wave_cross_section_m2',
            'event_cross_section_m2',
            'source_weight',
            'sediment_vertical_inverse_length_B_per_m',
        )
    }
    n = len(a['number_density_by_bin'])
    if n == 0 or len(a['particle_mass_kg']) != n or len(a['diameter_m']) != n:
        return None
    for k in (
        'geometric_cross_section_m2',
        'scattering_cross_section_m2',
        'extinction_cross_section_m2',
        'wave_cross_section_m2',
        'event_cross_section_m2',
        'source_weight',
        'sediment_vertical_inverse_length_B_per_m',
    ):
        if len(a[k]) != n:
            a[k] = np.zeros(n, dtype=np.float64)
    a['bulk_mass_by_bin_kg_m3'] = a['number_density_by_bin'] * a['particle_mass_kg']
    a['bulk_total_number_density_m3'] = float(np.sum(a['number_density_by_bin']))
    a['bulk_mass_concentration_kg_m3'] = float(np.sum(a['bulk_mass_by_bin_kg_m3']))
    return a


def _local_bin_density_scale(ph, g, d, r_m, height_above_bottom_m):
    """Host equivalent of CUDA ``sediment_local_scale`` for reporting only."""
    a = _particle_state_arrays(ph)
    if a is None:
        return np.array([], dtype=np.float64)
    nbin = len(a['number_density_by_bin'])
    R = float(g['sample_cell_radius_m'])
    r = float(np.clip(float(r_m), 0.0, R))
    h = max(float(height_above_bottom_m), 0.0)
    zmin = -float(g['sensor_ring_height_m'])
    z = zmin + h
    zsurf = float(_free_surface_z_sensor_frame(np.array([r]), g, d)[0])
    if z > zsurf + 1e-12:
        return np.full(nbin, np.nan, dtype=np.float64)
    model = d['sediment_transport_model']
    if model == 'uniform':
        return np.ones(nbin, dtype=np.float64)

    if model == 'schiller_naumann_meridional_advection_diffusion':
        nr = int(ph.get('_standalone_sediment_grid_nr', 0))
        nz = int(ph.get('_standalone_sediment_grid_nz', 0))
        zgridmax = float(ph.get('_standalone_sediment_grid_zmax_m', zmin))
        grid = np.asarray(ph.get('_standalone_sediment_grid', []), dtype=np.float64)
        if nr < 2 or nz < 2 or not (zgridmax > zmin) or grid.size != nbin * nr * nz:
            raise RuntimeError(
                'reporting requires the host copy of the production meridional sediment grid'
            )
        grid = grid.reshape(nbin, nr, nz)
        dr = R / nr
        dz = (zgridmax - zmin) / nz
        ur = np.clip(r / dr - 0.5, 0.0, nr - 1.0)
        uz = np.clip((z - zmin) / dz - 0.5, 0.0, nz - 1.0)
        i = int(np.floor(ur))
        k = int(np.floor(uz))
        tr = ur - i
        tz = uz - k
        if i >= nr - 1:
            i, tr = nr - 2, 1.0
        if k >= nz - 1:
            k, tz = nz - 2, 1.0
        q00 = grid[:, i, k]
        q10 = grid[:, i + 1, k]
        q01 = grid[:, i, k + 1]
        q11 = grid[:, i + 1, k + 1]
        q0 = (1.0 - tr) * q00 + tr * q10
        q1 = (1.0 - tr) * q01 + tr * q11
        return np.maximum((1.0 - tz) * q0 + tz * q1, 0.0)

    A = np.asarray(
        ph.get('sediment_radial_log_coefficient_A', np.zeros(nbin)), dtype=np.float64
    )
    B = np.asarray(
        ph.get('sediment_vertical_inverse_length_B_per_m', np.zeros(nbin)),
        dtype=np.float64,
    )
    norm = np.asarray(
        ph.get('sediment_normalization_scaled', np.ones(nbin)), dtype=np.float64
    )
    if len(A) != nbin or len(B) != nbin or len(norm) != nbin:
        raise RuntimeError('incomplete production sediment arrays for reporting')
    radial_log = np.zeros(nbin, dtype=np.float64)
    if model == 'stokes_drift_diffusion':
        rr2 = r * r
        R2 = R * R
        if d['free_surface_model'] == 'parabolic_vortex':
            radial_log = A * (rr2 - R2)
        else:
            core = float(d['vortex_core_radius_m'])
            a2 = core * core
            f = rr2 / (a2 + rr2)
            fR = R2 / (a2 + R2)
            radial_log = A * (f - fR)
    else:
        flat = np.asarray(
            ph.get('_standalone_sediment_radial_log', []), dtype=np.float64
        )
        nlookup = int(d['sediment_radial_lookup_points'])
        if flat.size != nbin * nlookup:
            raise RuntimeError(
                'reporting requires the host copy of the production radial sediment table'
            )
        table = flat.reshape(nbin, nlookup)
        u = np.clip(r / R * (nlookup - 1), 0.0, nlookup - 1.0)
        j = min(int(np.floor(u)), nlookup - 2)
        tr = u - j
        radial_log = (1.0 - tr) * table[:, j] + tr * table[:, j + 1]
        radial_log = np.minimum(radial_log, 0.0)
    scale = np.zeros(nbin, dtype=np.float64)
    good = norm > 0.0
    scale[good] = np.exp(radial_log[good] - B[good] * h) / norm[good]
    return scale


def _weighted_median(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    good = np.isfinite(values) & np.isfinite(weights) & (weights >= 0)
    if not np.any(good) or float(np.sum(weights[good])) <= 0:
        return float('nan')
    v = values[good]
    w = weights[good]
    order = np.argsort(v)
    v = v[order]
    w = w[order]
    c = np.cumsum(w) / np.sum(w)
    return float(v[min(int(np.searchsorted(c, 0.5, side='left')), len(v) - 1)])


def _local_sediment_state(ph, g, d, r_m, height_above_bottom_m):
    a = _particle_state_arrays(ph)
    if a is None:
        return None
    scale = _local_bin_density_scale(ph, g, d, r_m, height_above_bottom_m)
    if scale.size == 0 or np.any(~np.isfinite(scale)):
        return None
    nloc = a['number_density_by_bin'] * scale
    mloc = nloc * a['particle_mass_kg']
    ntot = float(np.sum(nloc))
    mtot = float(np.sum(mloc))
    bulk_n = a['bulk_total_number_density_m3']
    bulk_m = a['bulk_mass_concentration_kg_m3']
    numfrac = nloc / ntot if ntot > 0 else np.zeros_like(nloc)
    massfrac = mloc / mtot if mtot > 0 else np.zeros_like(mloc)
    bulk_numfrac = (
        a['number_density_by_bin'] / bulk_n if bulk_n > 0 else np.zeros_like(nloc)
    )
    bulk_massfrac = (
        a['bulk_mass_by_bin_kg_m3'] / bulk_m if bulk_m > 0 else np.zeros_like(mloc)
    )
    d_um = a['diameter_m'] * 1e6
    mu_sca = float(np.sum(nloc * a['scattering_cross_section_m2']))
    mu_ext = float(np.sum(nloc * a['extinction_cross_section_m2']))
    mu_abs = max(mu_ext - mu_sca, 0.0)
    return {
        'scale': scale,
        'number_density_by_bin': nloc,
        'mass_concentration_by_bin_kg_m3': mloc,
        'total_number_density_m3': ntot,
        'total_mass_concentration_kg_m3': mtot,
        'relative_number_density_to_bulk': (
            ntot / bulk_n if bulk_n > 0 else float('nan')
        ),
        'relative_mass_concentration_to_bulk': (
            mtot / bulk_m if bulk_m > 0 else float('nan')
        ),
        'local_number_fraction': numfrac,
        'local_mass_fraction': massfrac,
        'bulk_number_fraction': bulk_numfrac,
        'bulk_mass_fraction': bulk_massfrac,
        'number_weighted_mean_diameter_um': (
            float(np.sum(numfrac * d_um)) if ntot > 0 else float('nan')
        ),
        'mass_weighted_mean_diameter_um': (
            float(np.sum(massfrac * d_um)) if mtot > 0 else float('nan')
        ),
        'number_median_diameter_um': _weighted_median(d_um, nloc),
        'mu_sca_per_m': mu_sca,
        'mu_ext_per_m': mu_ext,
        'mu_abs_per_m': mu_abs,
        'single_scattering_albedo': mu_sca / mu_ext if mu_ext > 0 else float('nan'),
        'diameter_um': d_um,
    }


def _profile_radii(g):
    fracs = sorted(
        set(float(np.clip(x, 0.0, 1.0)) for x in SEDIMENT_PROFILE_RADII_FRACTION)
    )
    return [(f, f * float(g['sample_cell_radius_m'])) for f in fracs]


def _sensor_radii(g):
    # Detector/source hardware lies outside the water bore, so 0.95R is the
    # physically relevant near-wall liquid location rather than the ring radius.
    fracs = sorted(
        set([0.0, 0.5, 0.9, 0.95] + [float(x) for x in SEDIMENT_PROFILE_RADII_FRACTION])
    )
    return [(f, f * float(g['sample_cell_radius_m'])) for f in fracs if 0.0 <= f <= 1.0]


def _vertical_profile_rows(material, concentration, wavelength_nm, ph, g, d):
    rows = []
    for rf, r in _profile_radii(g):
        local_surface = float(
            _free_surface_z_sensor_frame(np.array([r]), g, d)[0]
            + g['sensor_ring_height_m']
        )
        heights = np.linspace(0.0, local_surface, int(SEDIMENT_PROFILE_HEIGHT_POINTS))
        for h in heights:
            st = _local_sediment_state(ph, g, d, r, h)
            if st is None:
                continue
            rows.append(
                {
                    'material': material,
                    'concentration_g_L': float(concentration),
                    'wavelength_nm': float(wavelength_nm),
                    'radius_fraction': rf,
                    'radius_m': r,
                    'height_above_bottom_m': float(h),
                    'height_fraction': (
                        float(h / local_surface) if local_surface > 0 else 0.0
                    ),
                    'local_water_column_height_m': local_surface,
                    'total_mass_concentration_kg_m3': st[
                        'total_mass_concentration_kg_m3'
                    ],
                    'total_number_density_m3': st['total_number_density_m3'],
                    'relative_concentration': st['relative_mass_concentration_to_bulk'],
                }
            )
    return rows


def _radial_profile_rows(material, concentration, wavelength_nm, ph, g, d):
    R = float(g['sample_cell_radius_m'])
    rvals = np.linspace(0.0, 0.995 * R, int(SEDIMENT_OUTPUT_NR))
    heights = [
        0.05 * g['water_level_m'],
        g['sensor_ring_height_m'],
        0.50 * g['water_level_m'],
        0.95 * g['water_level_m'],
    ]
    rows = []
    for h in sorted(set(float(x) for x in heights)):
        for r in rvals:
            st = _local_sediment_state(ph, g, d, r, h)
            if st is None:
                continue
            rows.append(
                {
                    'material': material,
                    'concentration_g_L': float(concentration),
                    'wavelength_nm': float(wavelength_nm),
                    'height_above_bottom_m': h,
                    'height_fraction_of_mean_water_level': h / g['water_level_m'],
                    'radius_m': float(r),
                    'radius_fraction': float(r / R),
                    'total_mass_concentration_kg_m3': st[
                        'total_mass_concentration_kg_m3'
                    ],
                    'total_number_density_m3': st['total_number_density_m3'],
                    'relative_concentration': st['relative_mass_concentration_to_bulk'],
                }
            )
    return rows


def _psd_vs_height_rows(material, concentration, wavelength_nm, ph, g, d):
    a = _particle_state_arrays(ph)
    if a is None:
        return []
    rows = []
    for rf, r in _profile_radii(g):
        local_surface = float(
            _free_surface_z_sensor_frame(np.array([r]), g, d)[0]
            + g['sensor_ring_height_m']
        )
        for hf in sorted(set(float(x) for x in PSD_PROFILE_HEIGHT_FRACTIONS)):
            h = hf * local_surface
            st = _local_sediment_state(ph, g, d, r, h)
            if st is None:
                continue
            for i, diam in enumerate(st['diameter_um']):
                bn = float(st['bulk_number_fraction'][i])
                bm = float(st['bulk_mass_fraction'][i])
                ln = float(st['local_number_fraction'][i])
                lm = float(st['local_mass_fraction'][i])
                rows.append(
                    {
                        'material': material,
                        'concentration_g_L': float(concentration),
                        'wavelength_nm': float(wavelength_nm),
                        'radius_fraction': rf,
                        'radius_m': r,
                        'height_above_bottom_m': h,
                        'height_fraction': hf,
                        'particle_bin': i,
                        'diameter_um': float(diam),
                        'bulk_psd_mass_fraction': bm,
                        'local_mass_fraction': lm,
                        'bulk_psd_number_fraction': bn,
                        'local_number_fraction': ln,
                        'local_number_abundance_ratio_to_bulk_psd': (
                            ln / bn if bn > 0 else None
                        ),
                        'local_mass_abundance_ratio_to_bulk_psd': (
                            lm / bm if bm > 0 else None
                        ),
                        'local_number_density_m3': float(
                            st['number_density_by_bin'][i]
                        ),
                        'local_mass_concentration_kg_m3': float(
                            st['mass_concentration_by_bin_kg_m3'][i]
                        ),
                    }
                )
    return rows


def _sensor_ring_rows(material, concentration, wavelength_nm, ph, g, d):
    a = _particle_state_arrays(ph)
    if a is None:
        return [], []
    h = float(g['sensor_ring_height_m'])
    bulk_n = a['bulk_total_number_density_m3']
    bulk_m = a['bulk_mass_concentration_kg_m3']
    d_um = a['diameter_m'] * 1e6
    bulk_numfrac = (
        a['number_density_by_bin'] / bulk_n if bulk_n > 0 else np.zeros_like(d_um)
    )
    bulk_massfrac = (
        a['bulk_mass_by_bin_kg_m3'] / bulk_m if bulk_m > 0 else np.zeros_like(d_um)
    )
    bulk_nmean = float(np.sum(bulk_numfrac * d_um)) if bulk_n > 0 else float('nan')
    bulk_mmean = float(np.sum(bulk_massfrac * d_um)) if bulk_m > 0 else float('nan')
    summary = []
    optical = []
    for rf, r in _sensor_radii(g):
        st = _local_sediment_state(ph, g, d, r, h)
        if st is None:
            continue
        common = {
            'material': material,
            'concentration_g_L': float(concentration),
            'wavelength_nm': float(wavelength_nm),
            'sensor_ring_height_m': h,
            'radius_m': r,
            'radius_fraction': rf,
            'total_mass_concentration_kg_m3': st['total_mass_concentration_kg_m3'],
            'total_number_density_m3': st['total_number_density_m3'],
            'relative_concentration': st['relative_mass_concentration_to_bulk'],
            'local_number_weighted_mean_diameter_um': st[
                'number_weighted_mean_diameter_um'
            ],
            'local_mass_weighted_mean_diameter_um': st[
                'mass_weighted_mean_diameter_um'
            ],
            'local_median_diameter_um': st['number_median_diameter_um'],
            'bulk_number_weighted_mean_diameter_um': bulk_nmean,
            'bulk_mass_weighted_mean_diameter_um': bulk_mmean,
            'local_to_bulk_number_mean_diameter_ratio': (
                st['number_weighted_mean_diameter_um'] / bulk_nmean
                if bulk_nmean > 0
                else None
            ),
            'local_to_bulk_mass_mean_diameter_ratio': (
                st['mass_weighted_mean_diameter_um'] / bulk_mmean
                if bulk_mmean > 0
                else None
            ),
        }
        summary.append(common)
        optical.append(
            {
                **common,
                'local_scattering_coefficient_per_m': st['mu_sca_per_m'],
                'local_extinction_coefficient_per_m': st['mu_ext_per_m'],
                'local_absorption_coefficient_per_m': st['mu_abs_per_m'],
                'local_single_scattering_albedo': st['single_scattering_albedo'],
            }
        )
    return summary, optical


def _sediment_distribution_rows(ph, g, d):
    """Sample the actual production steady spatial field on an r-z reporting grid."""
    a = _particle_state_arrays(ph)
    if a is None:
        return []
    R = float(g['sample_cell_radius_m'])
    nr = int(SEDIMENT_OUTPUT_NR)
    nz = int(SEDIMENT_OUTPUT_NZ)
    rnodes = (np.arange(nr, dtype=float) + 0.5) * R / nr
    ztop = float(
        np.max(_free_surface_z_sensor_frame(rnodes, g, d) + g['sensor_ring_height_m'])
    )
    hnodes = (np.arange(nz, dtype=float) + 0.5) * ztop / nz
    rows = []
    for r in rnodes:
        for h in hnodes:
            st = _local_sediment_state(ph, g, d, r, h)
            if st is None:
                continue
            rows.append(
                {
                    'r_m': float(r),
                    'radius_fraction': float(r / R),
                    'height_above_bottom_m': float(h),
                    'height_fraction_of_mean_water_level': float(
                        h / g['water_level_m']
                    ),
                    'number_density_m3': st['total_number_density_m3'],
                    'mass_concentration_kg_m3': st['total_mass_concentration_kg_m3'],
                    'relative_number_density_to_bulk': st[
                        'relative_number_density_to_bulk'
                    ],
                    'relative_mass_concentration_to_bulk': st[
                        'relative_mass_concentration_to_bulk'
                    ],
                }
            )
    return rows


def _sensor_ring_psd_rows(material, concentration, wavelength_nm, ph, g, d):
    a = _particle_state_arrays(ph)
    if a is None:
        return []
    rows = []
    h = float(g['sensor_ring_height_m'])
    for rf, r in _sensor_radii(g):
        st = _local_sediment_state(ph, g, d, r, h)
        if st is None:
            continue
        for i, diam in enumerate(st['diameter_um']):
            rows.append(
                {
                    'material': material,
                    'concentration_g_L': float(concentration),
                    'wavelength_nm': float(wavelength_nm),
                    'sensor_ring_height_m': h,
                    'radius_m': r,
                    'radius_fraction': rf,
                    'particle_bin': i,
                    'diameter_um': float(diam),
                    'bulk_psd_mass_fraction': float(st['bulk_mass_fraction'][i]),
                    'local_mass_fraction': float(st['local_mass_fraction'][i]),
                    'bulk_psd_number_fraction': float(st['bulk_number_fraction'][i]),
                    'local_number_fraction': float(st['local_number_fraction'][i]),
                    'local_number_density_m3': float(st['number_density_by_bin'][i]),
                    'local_mass_concentration_kg_m3': float(
                        st['mass_concentration_by_bin_kg_m3'][i]
                    ),
                }
            )
    return rows


def _sediment_summary_metrics(ph, g, d, distribution_rows, sensor_rows):
    metrics = {
        'sediment_bottom_top_ratio': None,
        'sediment_center_wall_ratio': None,
        'sediment_concentration_cv': None,
        'sensor_ring_to_bulk_concentration_ratio': None,
        'sensor_ring_mean_diameter_to_bulk_mean_diameter_ratio': None,
    }
    if not distribution_rows:
        return metrics
    rr = np.array([q['r_m'] for q in distribution_rows], dtype=float)
    rel = np.array(
        [q['relative_mass_concentration_to_bulk'] for q in distribution_rows],
        dtype=float,
    )
    good = np.isfinite(rel)
    w = np.maximum(rr, 1e-12)
    if np.any(good):
        wm = np.sum(w[good] * rel[good]) / np.sum(w[good])
        metrics['sediment_concentration_cv'] = (
            float(
                np.sqrt(np.sum(w[good] * (rel[good] - wm) ** 2) / np.sum(w[good])) / wm
            )
            if wm > 0
            else None
        )
    # Physically comparable bottom/top diagnostic: same normalized vertical position
    # at every radius, avoiding the curved Scully free-surface bias.
    vals_bottom = []
    vals_top = []
    weights = []
    for rf, r in _profile_radii(g):
        surf = float(
            _free_surface_z_sensor_frame(np.array([r]), g, d)[0]
            + g['sensor_ring_height_m']
        )
        sb = _local_sediment_state(ph, g, d, r, 0.01 * surf)
        st = _local_sediment_state(ph, g, d, r, 0.99 * surf)
        if sb is not None and st is not None:
            vals_bottom.append(sb['total_mass_concentration_kg_m3'])
            vals_top.append(st['total_mass_concentration_kg_m3'])
            weights.append(max(r, 0.01 * g['sample_cell_radius_m']))
    if vals_bottom and np.average(vals_top, weights=weights) > 0:
        metrics['sediment_bottom_top_ratio'] = float(
            np.average(vals_bottom, weights=weights)
            / np.average(vals_top, weights=weights)
        )
    if sensor_rows:
        byrf = {round(float(q['radius_fraction']), 6): q for q in sensor_rows}
        if (
            0.0 in byrf
            and 0.95 in byrf
            and float(byrf[0.95]['total_mass_concentration_kg_m3']) > 0
        ):
            metrics['sediment_center_wall_ratio'] = float(
                byrf[0.0]['total_mass_concentration_kg_m3']
                / byrf[0.95]['total_mass_concentration_kg_m3']
            )
        sw = np.array(
            [
                max(float(q['radius_m']), 0.01 * g['sample_cell_radius_m'])
                for q in sensor_rows
            ]
        )
        sr = np.array([float(q['relative_concentration']) for q in sensor_rows])
        metrics['sensor_ring_to_bulk_concentration_ratio'] = float(
            np.average(sr, weights=sw)
        )
        ratios = np.array(
            [float(q['local_to_bulk_mass_mean_diameter_ratio']) for q in sensor_rows]
        )
        metrics['sensor_ring_mean_diameter_to_bulk_mean_diameter_ratio'] = float(
            np.average(ratios, weights=sw)
        )
    return metrics


def _write_sediment_diagnostics(
    out: Path, material, concentration, wavelength_nm, ph, g, d, detailed=True
):
    result = {
        'distribution_rows': [],
        'vertical_rows': [],
        'radial_rows': [],
        'psd_height_rows': [],
        'sensor_rows': [],
        'sensor_psd_rows': [],
        'optical_rows': [],
        'metrics': {},
    }
    if not EXPORT_SEDIMENT_FIELDS and not EXPORT_LOCAL_PSD_PROFILES:
        return result
    distribution = _sediment_distribution_rows(ph, g, d)
    sensor_rows, optical_rows = _sensor_ring_rows(
        material, concentration, wavelength_nm, ph, g, d
    )
    sensor_psd_rows = (
        _sensor_ring_psd_rows(material, concentration, wavelength_nm, ph, g, d)
        if EXPORT_LOCAL_PSD_PROFILES
        else []
    )
    result['distribution_rows'] = distribution
    result['sensor_rows'] = sensor_rows
    result['sensor_psd_rows'] = sensor_psd_rows
    result['optical_rows'] = optical_rows
    result['metrics'] = _sediment_summary_metrics(ph, g, d, distribution, sensor_rows)
    if SAVE_CSV and sensor_rows:
        _save_csv(
            out / 'sensor_ring_sediment_summary.csv', sensor_rows[0].keys(), sensor_rows
        )
    if SAVE_CSV and optical_rows:
        _save_csv(
            out / 'sensor_ring_optical_environment.csv',
            optical_rows[0].keys(),
            optical_rows,
        )
    if SAVE_CSV and sensor_psd_rows:
        _save_csv(
            out / 'sensor_ring_local_psd.csv',
            sensor_psd_rows[0].keys(),
            sensor_psd_rows,
        )
    if not detailed:
        return result

    vertical = _vertical_profile_rows(material, concentration, wavelength_nm, ph, g, d)
    radial = _radial_profile_rows(material, concentration, wavelength_nm, ph, g, d)
    psd_rows = (
        _psd_vs_height_rows(material, concentration, wavelength_nm, ph, g, d)
        if EXPORT_LOCAL_PSD_PROFILES
        else []
    )
    result['vertical_rows'] = vertical
    result['radial_rows'] = radial
    result['psd_height_rows'] = psd_rows
    if SAVE_CSV:
        if distribution:
            _save_csv(
                out / 'sediment_distribution.csv', distribution[0].keys(), distribution
            )
            _save_csv(
                out / 'sediment_distribution_rz.csv',
                distribution[0].keys(),
                distribution,
            )
        if vertical:
            _save_csv(
                out / 'sediment_vertical_profile.csv', vertical[0].keys(), vertical
            )
        if radial:
            _save_csv(out / 'sediment_radial_profile.csv', radial[0].keys(), radial)
        if psd_rows:
            _save_csv(out / 'sediment_psd_vs_height.csv', psd_rows[0].keys(), psd_rows)

    if not (SAVE_PLOTS or SHOW_PLOTS):
        return result
    plt = _get_pyplot()
    if SAVE_SEDIMENT_DIAGNOSTIC_PLOTS and distribution:
        rr = np.array([q['r_m'] for q in distribution]) * 1e3
        hh = np.array([q['height_above_bottom_m'] for q in distribution]) * 1e3
        rel = np.array(
            [q['relative_mass_concentration_to_bulk'] for q in distribution],
            dtype=float,
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(rr, hh, c=rel, s=10)
        fig.colorbar(sc, ax=ax, label='Mass concentration / bulk mean')
        ax.axhline(
            g['sensor_ring_height_m'] * 1e3,
            linestyle='--',
            linewidth=1,
            label='Sensor ring height',
        )
        ax.set_xlabel('Radius r (mm)')
        ax.set_ylabel('Height above internal bottom (mm)')
        ax.set_title('Steady-state sediment distribution')
        ax.legend()
        _plot_save(fig, out / 'sediment_distribution_rz.png')
    if SAVE_SEDIMENT_DIAGNOSTIC_PLOTS and vertical:
        fig, ax = plt.subplots(figsize=(7, 5))
        for rf, _ in _profile_radii(g):
            q = [x for x in vertical if abs(x['radius_fraction'] - rf) < 1e-12]
            ax.plot(
                [x['relative_concentration'] for x in q],
                [x['height_above_bottom_m'] * 1e3 for x in q],
                label=f'r/R={rf:g}',
            )
        ax.axhline(
            g['sensor_ring_height_m'] * 1e3,
            linestyle='--',
            linewidth=1,
            label='Sensor ring',
        )
        ax.set_xlabel('Mass concentration / bulk mean')
        ax.set_ylabel('Height above internal bottom (mm)')
        ax.set_title('Steady-state vertical concentration profiles')
        ax.grid(True, alpha=0.25)
        ax.legend()
        _plot_save(fig, out / 'sediment_vertical_profile.png')
    if SAVE_SEDIMENT_DIAGNOSTIC_PLOTS and radial:
        fig, ax = plt.subplots(figsize=(8, 5))
        for h in sorted(set(float(x['height_above_bottom_m']) for x in radial)):
            q = sorted(
                [x for x in radial if abs(x['height_above_bottom_m'] - h) < 1e-12],
                key=lambda x: x['radius_m'],
            )
            ax.plot(
                [x['radius_m'] * 1e3 for x in q],
                [x['relative_concentration'] for x in q],
                label=f'{h*1e3:.1f} mm',
            )
        ax.set_xlabel('Radius r (mm)')
        ax.set_ylabel('Mass concentration / bulk mean')
        ax.set_title('Steady-state radial concentration profiles')
        ax.grid(True, alpha=0.25)
        ax.legend(title='Height')
        _plot_save(fig, out / 'sediment_radial_profile.png')

    if SAVE_PSD_DIAGNOSTIC_PLOTS and psd_rows:
        # Height/diameter heatmap at centreline (or the smallest configured radius).
        rf = min(set(float(x['radius_fraction']) for x in psd_rows), key=abs)
        q = [x for x in psd_rows if abs(x['radius_fraction'] - rf) < 1e-12]
        heights = sorted(set(float(x['height_fraction']) for x in q))
        diam = sorted(set(float(x['diameter_um']) for x in q))
        mat = np.full((len(heights), len(diam)), np.nan)
        hi = {v: i for i, v in enumerate(heights)}
        di = {v: i for i, v in enumerate(diam)}
        for x in q:
            mat[hi[float(x['height_fraction'])], di[float(x['diameter_um'])]] = float(
                x['local_number_abundance_ratio_to_bulk_psd']
            )
        fig, ax = plt.subplots(figsize=(9, 5))
        pc = ax.pcolormesh(
            np.array(diam), np.array(heights) * 100.0, mat, shading='auto'
        )
        ax.set_xscale('log')
        fig.colorbar(pc, ax=ax, label='Local number fraction / bulk number fraction')
        ax.set_xlabel('Particle diameter (µm)')
        ax.set_ylabel('Height through local water column (%)')
        ax.set_title(f'PSD segregation vs height (r/R={rf:g})')
        _plot_save(fig, out / 'sediment_psd_vs_height.png')

        fig, ax = plt.subplots(figsize=(8, 5))
        bulk = None
        for hf in sorted(set(float(x['height_fraction']) for x in q)):
            z = sorted(
                [x for x in q if abs(x['height_fraction'] - hf) < 1e-12],
                key=lambda x: x['diameter_um'],
            )
            if bulk is None:
                bulk = (
                    [x['diameter_um'] for x in z],
                    [x['bulk_psd_mass_fraction'] for x in z],
                )
            ax.semilogx(
                [x['diameter_um'] for x in z],
                [x['local_mass_fraction'] for x in z],
                marker='o',
                markersize=2,
                label=f'{hf*100:g}%',
            )
        if bulk is not None:
            ax.semilogx(bulk[0], bulk[1], linestyle='--', linewidth=2, label='Bulk PSD')
        ax.set_xlabel('Particle diameter (µm)')
        ax.set_ylabel('Local mass fraction')
        ax.set_title(f'Local PSD at selected heights (r/R={rf:g})')
        ax.grid(True, alpha=0.25)
        ax.legend(title='Height')
        _plot_save(fig, out / 'local_psd_profiles.png')

        # Local PSD at the actual sensor-ring height.
        a = _particle_state_arrays(ph)
        fig, ax = plt.subplots(figsize=(8, 5))
        if a is not None:
            bulkm = (
                a['bulk_mass_by_bin_kg_m3'] / a['bulk_mass_concentration_kg_m3']
                if a['bulk_mass_concentration_kg_m3'] > 0
                else np.zeros_like(a['diameter_m'])
            )
            ax.semilogx(
                a['diameter_m'] * 1e6,
                bulkm,
                linestyle='--',
                linewidth=2,
                label='Bulk PSD',
            )
            for rf, r in _sensor_radii(g):
                st = _local_sediment_state(ph, g, d, r, g['sensor_ring_height_m'])
                if st is not None:
                    ax.semilogx(
                        st['diameter_um'],
                        st['local_mass_fraction'],
                        marker='o',
                        markersize=2,
                        label=f'r/R={rf:g}',
                    )
            ax.set_xlabel('Particle diameter (µm)')
            ax.set_ylabel('Mass fraction')
            ax.set_title('Local PSD at sensor-ring height')
            ax.grid(True, alpha=0.25)
            ax.legend()
            _plot_save(fig, out / 'sensor_ring_local_psd.png')

            # Representative per-bin vertical segregation at centreline.
            idx = np.unique(
                np.round(np.linspace(0, len(a['diameter_m']) - 1, 5)).astype(int)
            )
            r = 0.0
            surf = float(
                _free_surface_z_sensor_frame(np.array([r]), g, d)[0]
                + g['sensor_ring_height_m']
            )
            hs = np.linspace(0, surf, int(SEDIMENT_PROFILE_HEIGHT_POINTS))
            fig, ax = plt.subplots(figsize=(8, 5))
            for i in idx:
                vals = []
                for h in hs:
                    sc = _local_bin_density_scale(ph, g, d, r, h)
                    vals.append(sc[i] if len(sc) > i else np.nan)
                ax.plot(
                    np.array(vals), hs * 1e3, label=f'{a["diameter_m"][i]*1e6:.3g} µm'
                )
            ax.axhline(
                g['sensor_ring_height_m'] * 1e3,
                linestyle='--',
                linewidth=1,
                label='Sensor ring',
            )
            ax.set_xlabel('Local bin density / bulk-bin density')
            ax.set_ylabel('Height above internal bottom (mm)')
            ax.set_title('Selected particle-size vertical profiles (centreline)')
            ax.grid(True, alpha=0.25)
            ax.legend()
            _plot_save(fig, out / 'selected_particle_vertical_profiles.png')
    return result


def _vortex_profile_rows(g, d):
    if not d['enable_vortex'] or d['free_surface_model'] not in {
        'localized_scully_vortex',
        'parabolic_vortex',
    }:
        return []
    R = float(g['sample_cell_radius_m'])
    a = float(d['vortex_core_radius_m'])
    dh = float(d['vortex_wall_center_height_difference_m'])
    zmean = float(g['water_level_m'] - g['sensor_ring_height_m'])
    grav = float(d['gravity_m_s2'])
    model = d['free_surface_model']
    vg = vortex_surface_geometry(model, zmean, R, dh, a, grav)
    omega = float(vg['angular_velocity_rad_s'])
    r = np.linspace(0.0, R, int(VORTEX_OUTPUT_NR))
    if model == 'parabolic_vortex':
        vtheta = omega * r
        omega_local = np.full_like(r, omega)
    else:
        vtheta = omega * r / (1.0 + (r / a) ** 2)
        omega_local = omega / (1.0 + (r / a) ** 2)
    rows = []
    for ri, vi, oi in zip(r, vtheta, omega_local):
        zs = float(_free_surface_z_sensor_frame(np.array([ri]), g, d)[0])
        rows.append(
            {
                'vortex_model': model,
                'r_m': float(ri),
                'free_surface_height_above_bottom_m': float(
                    zs + g['sensor_ring_height_m']
                ),
                'tangential_velocity_m_s': float(vi),
                'angular_velocity_rad_s': float(oi),
                'equivalent_rpm': float(oi * 60.0 / (2.0 * math.pi)),
                'omega_core_rad_s': (
                    float(omega) if model == 'localized_scully_vortex' else float('nan')
                ),
            }
        )
    return rows


def _write_vortex_outputs(out: Path, g, d):
    if not SAVE_VORTEX_PROFILE:
        return []
    rows = _vortex_profile_rows(g, d)
    if not rows:
        return []
    _save_csv(out / 'vortex_field.csv', rows[0].keys(), rows)
    if SAVE_PLOTS or SHOW_PLOTS:
        plt = _get_pyplot()
        r = np.array([q['r_m'] for q in rows]) * 1e3
        v = np.array([q['tangential_velocity_m_s'] for q in rows])
        label = (
            'rigid-body parabolic'
            if d['free_surface_model'] == 'parabolic_vortex'
            else 'localized Scully'
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(r, v)
        ax.set_xlabel('Radius r (mm)')
        ax.set_ylabel('Tangential velocity (m/s)')
        ax.set_title(f'{label} vortex velocity profile')
        ax.grid(True, alpha=0.25)
        _plot_save(fig, out / 'vortex_velocity_profile.png')
        h = np.array([q['free_surface_height_above_bottom_m'] for q in rows]) * 1e3
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(r, h)
        ax.set_xlabel('Radius r (mm)')
        ax.set_ylabel('Free-surface height above bottom (mm)')
        ax.set_title(f'{label} free-surface profile')
        ax.grid(True, alpha=0.25)
        _plot_save(fig, out / 'vortex_free_surface_profile.png')
    return rows


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _write_json(path: Path, data):
    path.write_text(
        json.dumps(_json_safe(data), indent=2, sort_keys=True), encoding='utf-8'
    )


def _format_number_for_path(x: float) -> str:
    x = float(x)
    if x.is_integer():
        return str(int(x))
    return (f"{x:g}").replace('.', 'p').replace('-', 'm')


def _session_root() -> Path:
    base = Path(OUTPUT_DIR).expanduser().resolve()
    if CREATE_TIMESTAMPED_SESSION:
        base = base / datetime.now().strftime('%Y%m%d_%H%M%S')
    base.mkdir(parents=True, exist_ok=True)
    return base


def _case_output_dir(
    session_root: Path, material: str, wavelength_nm: float, concentration_g_l: float
) -> Path:
    p = (
        session_root
        / material.lower()
        / f"{_format_number_for_path(wavelength_nm)}nm"
        / f"{_format_number_for_path(concentration_g_l)}gL"
    )
    if not p.exists():
        p.mkdir(parents=True)
        return p
    # Never silently destroy an existing run.
    i = 2
    while True:
        q = p.with_name(p.name + f"_run{i:02d}")
        if not q.exists():
            q.mkdir(parents=True)
            return q
        i += 1


def _validate_material(material: str) -> str:
    m = str(material).strip().lower()
    if m not in {'loess', 'kaolin'}:
        raise ValueError("material must be 'loess' or 'kaolin'")
    return m


def _optical_profile_override(material: str):
    """Resolve the built-in particle n(d) and configured imaginary index.

    None selects the material's stored exploratory k(d) profile. Scalars and
    per-bin arrays override it; zero gives nonabsorbing particles.
    """
    material = _validate_material(material)
    n, _k_source = get_material_optical_constants(material, None, 1.59)
    n = np.asarray(n, dtype=np.float64)
    if PARTICLE_N_REAL_OVERRIDE is not None:
        q = np.asarray(PARTICLE_N_REAL_OVERRIDE, dtype=np.float64)
        if q.ndim == 0:
            n = np.full_like(n, float(q))
        elif q.shape == n.shape:
            n = q.copy()
        else:
            raise ValueError(
                f"PARTICLE_N_REAL_OVERRIDE must be scalar or contain {n.size} values for {material}"
            )
    if PARTICLE_K_IMAG_OVERRIDE is None:
        # Resolve per material so combined loess/kaolin sweeps use matching bins.
        k = np.asarray(_k_source, dtype=np.float64).copy()
    else:
        kq = np.asarray(PARTICLE_K_IMAG_OVERRIDE, dtype=np.float64)
        if kq.ndim == 0:
            k = np.full_like(n, float(kq))
        elif kq.shape == n.shape:
            k = kq.copy()
        else:
            raise ValueError(
                f"PARTICLE_K_IMAG_OVERRIDE must be scalar or contain {n.size} values for {material}"
            )
    if not np.all(np.isfinite(n)) or not np.all(np.isfinite(k)):
        raise ValueError('particle n and k must be finite')
    if np.any(n <= 0) or np.any(k < 0):
        raise ValueError('particle n must be >0 and k must be >=0')
    return {material: {'n_real': n.tolist(), 'k_imag': k.tolist()}}, n, k


def _build_model(material: str, wavelength_nm: float):
    profile, nvals, kvals = _optical_profile_override(material)
    g, d = _validate_user_configuration()
    kw = dict(MODEL_DEFAULT_KWARGS)
    kw.update(
        n_water=float(WATER_REFRACTIVE_INDEX),
        n_acrylic=float(ACRYLIC_REFRACTIVE_INDEX),
        n_air=float(AIR_REFRACTIVE_INDEX),
        particle_wavelength_m=float(wavelength_nm) * 1e-9,
        particle_optical_constants_by_material=profile,
        particle_surface_rms_slope_deg=float(PARTICLE_SURFACE_RMS_SLOPE_DEG),
        wave_surface_rms_height_m=float(WAVE_SURFACE_RMS_HEIGHT_NM) * 1e-9,
        # Sample-cell geometry.  All are native constructor parameters.
        tube_inner_radius_m=g['sample_cell_radius_m'],
        tube_outer_radius_m=g['tube_outer_radius_m'],
        tube_length_m=g['sample_cell_height_m'],
        water_height_m=g['water_level_m'],
        sensor_height_above_bottom_m=g['sensor_ring_height_m'],
        sensor_ring_inner_radius_m=g['sensor_ring_inner_radius_m'],
        sensor_ring_outer_radius_m=g['sensor_ring_outer_radius_m'],
        source_launch_radius_m=g['source_launch_radius_m'],
        counterbore_depth_m=g['counterbore_depth_m'],
        through_bore_diameter_m=g['through_bore_diameter_m'],
        counterbore_diameter_m=g['counterbore_diameter_m'],
        bottom_disc_thickness_m=g['bottom_disc_thickness_m'],
        # Native steady sediment / vortex controls.
        density_kg_per_m3=d['sediment_particle_density_kg_m3'],
        sediment_transport_model=d['sediment_transport_model'],
        sediment_eddy_diffusivity_m2_s=d['sediment_eddy_diffusivity_m2_s'],
        water_density_kg_per_m3=d['water_density_kg_m3'],
        water_dynamic_viscosity_pa_s=d['water_dynamic_viscosity_pa_s'],
        gravity_m_s2=d['gravity_m_s2'],
        free_surface_model=d['free_surface_model'],
        vortex_wall_center_height_difference_m=d[
            'vortex_wall_center_height_difference_m'
        ],
        vortex_core_radius_m=d['vortex_core_radius_m'],
        meridional_circulation_ratio=d['meridional_circulation_ratio'],
        sediment_transport_quadrature_points=d['sediment_transport_quadrature_points'],
        sediment_radial_lookup_points=d['sediment_radial_lookup_points'],
        meridional_grid_nr=d['meridional_grid_nr'],
        meridional_grid_nz=d['meridional_grid_nz'],
        meridional_solver_max_iterations=d['meridional_solver_max_iterations'],
        meridional_solver_check_interval=d['meridional_solver_check_interval'],
        meridional_solver_tolerance=d['meridional_solver_tolerance'],
        meridional_solver_sor=d['meridional_solver_sor'],
        chunk_size=int(CHUNK_SIZE),
        statistics_batch_rays=int(STATISTICS_BATCH_RAYS),
    )
    return ReportingClaritasForwardModel(**kw), kw, nvals, kvals


def _require_cuda_and_select_device():
    try:
        import cupy as cp
    except Exception as exc:
        raise RuntimeError(
            'CUDA/CuPy is required for CLARITAS production simulation.'
        ) from exc
    try:
        count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError(
            'CuPy is installed but no usable CUDA device/runtime was found.'
        ) from exc
    if not (0 <= int(GPU_DEVICE) < count):
        raise RuntimeError(
            f'GPU_DEVICE={GPU_DEVICE} is invalid; CUDA reports {count} device(s).'
        )
    cp.cuda.Device(int(GPU_DEVICE)).use()
    props = cp.cuda.runtime.getDeviceProperties(int(GPU_DEVICE))
    name = props.get('name', 'CUDA GPU') if isinstance(props, dict) else 'CUDA GPU'
    if isinstance(name, bytes):
        name = name.decode(errors='replace')
    return cp, str(name)


def _wavelength_assumption_summary(material: str, wavelength_nm: float, nvals, kvals):
    return {
        'wavelength_nm': float(wavelength_nm),
        'material': material,
        'water_refractive_index': float(WATER_REFRACTIVE_INDEX),
        'acrylic_refractive_index': float(ACRYLIC_REFRACTIVE_INDEX),
        'air_refractive_index': float(AIR_REFRACTIVE_INDEX),
        'particle_n_min': float(np.min(nvals)),
        'particle_n_max': float(np.max(nvals)),
        'particle_k_min': float(np.min(kvals)),
        'particle_k_max': float(np.max(kvals)),
        'particle_index_basis': 'fixed size-dependent n(d); no spectral dispersion table',
        'particle_k_basis': (
            'stored exploratory per-bin k(d); fixed across wavelengths'
            if PARTICLE_K_IMAG_OVERRIDE is None
            else 'configured scalar or per-bin k override; fixed across wavelengths'
        ),
        'medium_dispersion_basis': 'water/acrylic indices are user-configurable constants; no automatic spectral dispersion',
        'multiwavelength_status': 'wavelength-explicit physics recomputed; refractive-index dispersion assumed fixed unless overridden',
    }


def _print_wavelength_summary(info):
    print('  Wavelength physics:')
    print(f"    wavelength: {info['wavelength_nm']:g} nm")
    print(
        f"    water n:    {info['water_refractive_index']:.8g} (constant unless edited)"
    )
    print(
        f"    acrylic n:  {info['acrylic_refractive_index']:.8g} (constant unless edited)"
    )
    print(
        f"    particle n: {info['particle_n_min']:.8g} .. {info['particle_n_max']:.8g} (size-dependent n(d))"
    )
    print(
        f"    particle k: {info['particle_k_min']:.8g} .. {info['particle_k_max']:.8g}"
    )
    if abs(float(info['wavelength_nm']) - 622.0) > 1e-12:
        print(
            '    NOTE: has no refractive-index dispersion tables; n(d), water n and acrylic n are being reused across wavelength.'
        )


def _save_csv(path: Path, fieldnames, rows):
    if not SAVE_CSV:
        return
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: _json_safe(row.get(k)) for k in fieldnames})


def _get_pyplot():
    import matplotlib

    if not SHOW_PLOTS:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    return plt


def _plot_save(fig, path: Path):
    fig.tight_layout()
    if SAVE_PLOTS:
        fig.savefig(path, dpi=180)
    if SHOW_PLOTS:
        fig.show()
    else:
        plt = _get_pyplot()
        plt.close(fig)


def _plot_save_light(fig, path: Path):
    fig.savefig(
        path,
        dpi=LIGHT_TRANSPORT_PLOT_DPI,
        bbox_inches='tight',
        pad_inches=0.12,
        facecolor=fig.get_facecolor(),
    )
    if SHOW_PLOTS:
        fig.show()
    else:
        _get_pyplot().close(fig)


def _light_transport_cross_sections(diag, cfg):
    """Build source-axis and sensor-plane fields from the CUDA 3-D history grid."""
    nx, ny, nz = (int(diag[k]) for k in ('nx', 'ny', 'nz'))
    x = np.linspace(diag['x_min_m'], diag['x_max_m'], nx, endpoint=False) + (
        diag['x_max_m'] - diag['x_min_m']
    ) / (2 * nx)
    y = np.linspace(diag['y_min_m'], diag['y_max_m'], ny, endpoint=False) + (
        diag['y_max_m'] - diag['y_min_m']
    ) / (2 * ny)
    z = np.linspace(diag['z_min_m'], diag['z_max_m'], nz, endpoint=False) + (
        diag['z_max_m'] - diag['z_min_m']
    ) / (2 * nz)
    dx = (diag['x_max_m'] - diag['x_min_m']) / nx
    dy = (diag['y_max_m'] - diag['y_min_m']) / ny
    dz = (diag['z_max_m'] - diag['z_min_m']) / nz
    rays = max(int(diag['diagnostic_rays_processed']), 1)
    vol = dx * dy * dz
    xi = np.flatnonzero(np.abs(x) <= 0.5 * cfg['vertical_slice_thickness_m'])
    zi = np.flatnonzero(np.abs(z) <= 0.5 * cfg['sensor_slice_thickness_m'])
    if not len(xi):
        xi = np.array([int(np.argmin(np.abs(x)))])
    if not len(zi):
        zi = np.array([int(np.argmin(np.abs(z)))])
    total = np.asarray(diag['track_length_m'], dtype=float)
    post = np.asarray(diag['post_scatter_track_length_m'], dtype=float)
    events = np.asarray(diag['scatter_event_count'], dtype=float)
    vr = total[:, :, xi].sum(2)
    vp = post[:, :, xi].sum(2)
    ve = events[:, :, xi].sum(2)
    hr = total[zi].sum(0)
    hp = post[zi].sum(0)
    he = events[zi].sum(0)
    return {
        'x': x,
        'y': y,
        'z': z,
        'xi': xi,
        'zi': zi,
        'dx': dx,
        'dy': dy,
        'dz': dz,
        'voxel_volume': vol,
        'vr': vr,
        'vp': vp,
        've': ve,
        'hr': hr,
        'hp': hp,
        'he': he,
        'vd': vr / (rays * vol * len(xi)),
        'vpd': vp / (rays * vol * len(xi)),
        'ved': ve / (rays * vol * len(xi)),
        'hd': hr / (rays * vol * len(zi)),
        'hpd': hp / (rays * vol * len(zi)),
        'hed': he / (rays * vol * len(zi)),
    }


def _robust_light_log_scale(values, water_mask, cfg):
    """Return a deterministic, display-only log10 scale for in-water density."""
    a = np.asarray(values, dtype=np.float64)
    mask = np.asarray(water_mask, dtype=bool)
    positive = a[mask & np.isfinite(a) & (a > 0.0)]
    low = float(
        cfg.get('display_low_percentile', LIGHT_TRANSPORT_DISPLAY_LOW_PERCENTILE)
    )
    high = float(
        cfg.get('display_high_percentile', LIGHT_TRANSPORT_DISPLAY_HIGH_PERCENTILE)
    )
    floor = float(cfg.get('log_floor', LIGHT_TRANSPORT_LOG_FLOOR))
    if not len(positive):
        base = float(np.log10(floor))
        return base, base + 1.0, 'empty_fallback', 0
    logv = np.log10(positive)
    vmin = float(np.percentile(logv, low))
    vmax = float(np.percentile(logv, high))
    mode = 'percentile'
    if not (np.isfinite(vmin) and np.isfinite(vmax)):
        base = float(np.log10(floor))
        vmin, vmax, mode = base, base + 1.0, 'nonfinite_fallback'
    elif vmax - vmin < 1.0e-9:
        centre = 0.5 * (vmin + vmax)
        vmin, vmax, mode = centre - 0.5, centre + 0.5, 'constant_fallback'
    return vmin, vmax, mode, int(len(positive))


def _deterministic_spatial_cloud_selection(candidate_indices, max_points):
    """Select a fixed-size, whole-volume subset in deterministic Morton order."""
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    limit = int(max_points)
    count = int(len(candidates))
    if count <= limit:
        return np.array(candidates, copy=True)
    if candidates.ndim != 2 or candidates.shape[1] != 3 or limit < 1:
        raise ValueError(
            'cloud candidates must have shape (N,3) and max_points must be >= 1'
        )

    def spread(v):
        v = np.asarray(v, dtype=np.uint64) & np.uint64(0x1FFFFF)
        v = (v | (v << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
        v = (v | (v << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
        v = (v | (v << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
        v = (v | (v << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
        return (v | (v << np.uint64(2))) & np.uint64(0x1249249249249249)

    z, y, x = candidates.T
    morton = spread(x) | (spread(y) << np.uint64(1)) | (spread(z) << np.uint64(2))
    order = np.argsort(morton, kind='stable')
    positions = ((2 * np.arange(limit, dtype=np.int64) + 1) * count) // (2 * limit)
    return candidates[order[positions]]


def _light_transport_display_state(diag, q, g, d, cfg):
    """Build display masks/normalization without modifying any raw diagnostic array."""
    R = float(g['sample_cell_radius_m'])
    sensor_h = float(g['sensor_ring_height_m'])
    X, Y = np.meshgrid(q['x'], q['y'], indexing='xy')
    radial = np.sqrt(X * X + Y * Y)
    surface_h = _free_surface_z_sensor_frame(radial, g, d) + sensor_h
    height = q['z'][:, None, None] + sensor_h
    volume_mask = (
        (radial[None, :, :] <= R + 1.0e-12)
        & (height >= -1.0e-12)
        & (height <= surface_h[None, :, :] + 1.0e-12)
    )
    rays = max(int(diag['diagnostic_rays_processed']), 1)
    volume_density = np.asarray(diag['track_length_m'], dtype=np.float64) / (
        rays * q['voxel_volume']
    )
    vertical_surface_h = _free_surface_z_sensor_frame(np.abs(q['y']), g, d) + sensor_h
    vertical_height = q['z'][:, None] + sensor_h
    vertical_mask = (
        (np.abs(q['y'])[None, :] <= R + 1.0e-12)
        & (vertical_height >= -1.0e-12)
        & (vertical_height <= vertical_surface_h[None, :] + 1.0e-12)
    )
    horizontal_surface_z = _free_surface_z_sensor_frame(radial, g, d)
    horizontal_mask = (radial <= R + 1.0e-12) & (horizontal_surface_z >= -1.0e-12)
    vmin, vmax, scale_mode, npositive = _robust_light_log_scale(
        volume_density, volume_mask, cfg
    )
    rr = np.linspace(0.0, R, 1025)
    surface_max_h = float(np.max(_free_surface_z_sensor_frame(rr, g, d) + sensor_h))
    margin = float(
        cfg.get('display_3d_surface_margin_m', LIGHT_TRANSPORT_3D_SURFACE_MARGIN_M)
    )
    plot_zmax = min(float(g['sample_cell_height_m']), surface_max_h + margin)
    p3d = float(
        cfg.get('display_3d_min_percentile', LIGHT_TRANSPORT_3D_MIN_DISPLAY_PERCENTILE)
    )
    cloud_valid = volume_mask & np.isfinite(volume_density) & (volume_density > 0.0)
    cloud_log = (
        np.log10(volume_density[cloud_valid])
        if np.any(cloud_valid)
        else np.empty(0, dtype=np.float64)
    )
    threshold = float(np.percentile(cloud_log, p3d)) if len(cloud_log) else None
    cloud_mask = (
        cloud_valid
        if threshold is None
        else cloud_valid
        & (
            np.log10(
                np.maximum(
                    volume_density,
                    float(cfg.get('log_floor', LIGHT_TRANSPORT_LOG_FLOOR)),
                )
            )
            >= threshold
        )
    )
    candidates = np.argwhere(cloud_mask)
    candidate_count = int(len(candidates))
    max_points = int(
        cfg.get('display_3d_max_voxels', LIGHT_TRANSPORT_3D_MAX_DISPLAY_VOXELS)
    )
    cloud_indices = _deterministic_spatial_cloud_selection(candidates, max_points)
    if len(cloud_indices):
        cloud_values = volume_density[tuple(cloud_indices.T)]
        cloud_norm = np.clip((np.log10(cloud_values) - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        cloud_values = np.empty(0, dtype=np.float64)
        cloud_norm = np.empty(0, dtype=np.float64)
    return {
        'volume_density': volume_density,
        'volume_water_mask': volume_mask,
        'vertical_water_mask': vertical_mask,
        'horizontal_water_mask': horizontal_mask,
        'display_log10_vmin': vmin,
        'display_log10_vmax': vmax,
        'display_scale_mode': scale_mode,
        'display_positive_voxel_count': npositive,
        'surface_max_height_m': surface_max_h,
        'plot_3d_z_min_m': 0.0,
        'plot_3d_z_max_m': plot_zmax,
        'plot_3d_surface_margin_m': margin,
        'cloud_threshold_log10': threshold,
        'cloud_candidate_count': candidate_count,
        'cloud_max_display_voxels': max_points,
        'cloud_selection_method': 'deterministic Morton-order whole-volume stratification',
        'cloud_indices': cloud_indices,
        'cloud_values': cloud_values,
        'cloud_normalized': cloud_norm,
        'cloud_display_count': int(len(cloud_indices)),
    }


def _masked_light_log_field(field, water_mask, cfg):
    """Log-transform only for display; outside-water values render white."""
    a = np.asarray(field, dtype=np.float64)
    floor = float(cfg.get('log_floor', LIGHT_TRANSPORT_LOG_FLOOR))
    loga = np.full(a.shape, np.log10(floor), dtype=np.float64)
    good = np.isfinite(a) & (a > 0.0)
    loga[good] = np.log10(a[good])
    return np.ma.array(
        loga, mask=(~np.asarray(water_mask, dtype=bool)) | (~np.isfinite(a))
    )


def _save_csv_unconditional(path, fieldnames, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for row in rows:
            w.writerow({k: _json_safe(row.get(k)) for k in fieldnames})


class CaseIntegrityError(RuntimeError):
    """Explicit operational stop; case reports have already been persisted."""


def _resolved_failure_policy():
    if DIAGNOSTIC_POLICY not in ('report', 'strict'):
        raise ValueError('DIAGNOSTIC_POLICY must be report or strict')
    limits = {
        'max_failed_ray_fraction': MAX_FAILED_RAY_FRACTION,
        'max_unexplained_event_fraction': MAX_UNEXPLAINED_EVENT_FRACTION,
        'max_diagnostic_closure_error': MAX_DIAGNOSTIC_CLOSURE_ERROR,
        'strict_closure_error': STRICT_DIAGNOSTIC_CLOSURE_ERROR,
    }
    for key, value in limits.items():
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f'{key} must be finite and nonnegative, or None')
    return dict(
        limits,
        mode=DIAGNOSTIC_POLICY,
        abort_on_systemic_diagnostics=bool(ABORT_ON_SYSTEMIC_DIAGNOSTICS),
    )


def _production_integrity(launched, failed, diagnostics, policy):
    if launched <= 0 or failed < 0 or failed > launched:
        raise ValueError('corrupted production ray counts')
    if not diagnostics.get('count_closure_ok', True):
        raise ValueError('production failure categories do not reconcile')
    fraction = failed / launched
    limit = policy['max_failed_ray_fraction']
    fatal = limit is not None and fraction > limit
    return {
        'launched_rays': int(launched),
        'failed_rays': int(failed),
        'failed_ray_fraction': fraction,
        'normalization_rays': int(launched),
        'failure_categories': diagnostics.get('code_counts', {}),
        'failure_subcategories': diagnostics.get('subcode_counts', {}),
        'failure_samples': diagnostics.get('samples', []),
        'limitations': (
            [
                'failed rays terminate without completion; valid histories and prefixes remain; no successful-ray renormalization'
            ]
            if failed
            else []
        ),
        'fatal': bool(fatal),
        'status': 'systemic_failure' if fatal else ('limited' if failed else 'ok'),
    }


def _enforce_case_integrity(production, diagnostic):
    if production['fatal'] or diagnostic.get('fatal', False):
        raise CaseIntegrityError(
            'case exceeded configured integrity policy; inspect persisted integrity reports and run_summary.json'
        )


def _diagnostic_integrity(diag, policy):
    shape = (int(diag['nz']), int(diag['ny']), int(diag['nx']))
    if min(shape) <= 0:
        raise ValueError('invalid diagnostic grid dimensions')
    for axis in ('x', 'y', 'z'):
        lo, hi = float(diag[axis + '_min_m']), float(diag[axis + '_max_m'])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError('corrupted diagnostic grid bounds')
    if int(diag['diagnostic_rays_processed']) <= 0:
        raise ValueError('invalid diagnostic launched-ray denominator')
    field_validity = {}
    for key in (
        'track_length_m',
        'post_scatter_track_length_m',
        'scatter_event_weight',
        'scatter_event_count',
    ):
        array = np.asarray(diag[key])
        if array.shape != shape:
            raise ValueError(
                f'corrupted diagnostic shape: {key}: {array.shape}, expected {shape}'
            )
        field_validity[key] = bool(np.all(np.isfinite(array)) and np.all(array >= 0))
    counts = np.asarray(diag['trajectory_count'])
    nseg = int(diag['max_segments_per_ray'])
    if counts.ndim != 1 or np.any(counts < 0) or np.any(counts > nseg):
        raise ValueError('corrupted trajectory counts')
    if np.asarray(diag['trajectory_overflow']).shape != counts.shape:
        raise ValueError('corrupted trajectory overflow shape')
    for key in (
        'trajectory_x0_m',
        'trajectory_y0_m',
        'trajectory_z0_m',
        'trajectory_x1_m',
        'trajectory_y1_m',
        'trajectory_z1_m',
        'trajectory_particle_order',
        'trajectory_event_code',
    ):
        a = np.asarray(diag[key])
        if a.shape != (len(counts), nseg):
            raise ValueError(f'corrupted trajectory shape: {key}')
        field_validity[key] = all(
            np.all(np.isfinite(a[i, : int(n)])) for i, n in enumerate(counts)
        )
    observed = int(diag['scatter_event_total'])
    deposited = int(diag['deposited_scatter_event_total'])
    rejected = int(diag.get('event_outside_count', 0))
    if min(observed, deposited, rejected) < 0:
        raise ValueError('corrupted diagnostic event counts')
    discrepancy = deposited + rejected - observed
    grid_event_discrepancy = (
        int(np.sum(diag['scatter_event_count'], dtype=np.uint64)) - deposited
        if field_validity['scatter_event_count']
        else None
    )
    unexplained_fraction = max(
        abs(discrepancy), abs(grid_event_discrepancy or 0)
    ) / max(observed, 1)
    closure = float(diag['deposition_closure_relative_error'])
    grid_closure = float(diag['grid_closure_relative_error'])
    issues = []
    if rejected:
        issues.append('rejected diagnostic events explicitly accounted')
    if discrepancy or grid_event_discrepancy:
        issues.append('unexplained diagnostic event discrepancy')
    if int(diag.get('hard_error_count', diag.get('error_count', 0))):
        issues.append('diagnostic traversal errors')
    if int(diag.get('water_advance_guard_failure_count', 0)):
        issues.append('water guard terminated rays')
    unsafe = (
        not all(field_validity.values())
        or int(diag.get('nonfinite_count', 0)) > 0
        or int(diag.get('negative_value_count', 0)) > 0
    )
    if unsafe:
        issues.append('unsafe diagnostic fields; derived outputs skipped')
    strict_limit = policy['strict_closure_error']
    bad_closure = (
        not np.isfinite(closure)
        or not np.isfinite(grid_closure)
        or (strict_limit is not None and max(closure, grid_closure) > strict_limit)
    )
    if bad_closure:
        issues.append('diagnostic track-length closure exceeds strict limit')
    systemic = unsafe or not np.isfinite(closure) or not np.isfinite(grid_closure)
    for value, limit in (
        (unexplained_fraction, policy['max_unexplained_event_fraction']),
        (max(closure, grid_closure), policy['max_diagnostic_closure_error']),
    ):
        systemic = systemic or (limit is not None and value > limit)
    return {
        'policy': policy,
        'diagnostic_status': 'limited' if issues else 'ok',
        'limitations': (
            [
                'failed-ray history prefixes remain in diagnostics; failed rays stay in launched denominators'
            ]
            if int(diag.get('water_advance_guard_failure_count', 0))
            else []
        ),
        'issues': issues,
        'field_validity': field_validity,
        'derived_outputs_safe': not unsafe,
        'observed_diagnostic_events': observed,
        'deposited_diagnostic_events': deposited,
        'rejected_diagnostic_events': rejected,
        'reconciled_event_discrepancy': discrepancy,
        'event_accounting_reconciled': discrepancy == 0 and grid_event_discrepancy == 0,
        'deposited_grid_event_discrepancy': grid_event_discrepancy,
        'deposited_only_event_discrepancy': deposited - observed,
        'unexplained_event_fraction': unexplained_fraction,
        'diagnostic_clipped_track_length_m': diag['segment_length_total_m'],
        'physical_full_path_length_m': None,
        'path_length_scope': 'segment totals are clipped to diagnostic grid; full physical path is not collected',
        'systemic_diagnostic_issue': bool(systemic),
        'fatal': bool(
            (policy['mode'] == 'strict' and issues)
            or (policy['abort_on_systemic_diagnostics'] and systemic)
        ),
        'skipped_outputs': (
            []
            if not unsafe
            else [
                'diagnostic cross-section CSVs, volume product, trajectories and diagnostic plots: unsafe fields'
            ]
        ),
        'first_outside_event': diag.get('first_outside_event'),
        'first_water_advance_guard_failure': diag.get(
            'first_water_advance_guard_failure'
        ),
    }


def _write_light_transport_outputs(out: Path, diag, g, d, cfg, context=None):
    if not diag:
        return {'diagnostic_status': 'disabled', 'fatal': False, 'skipped_outputs': []}
    integrity = _diagnostic_integrity(diag, _resolved_failure_policy())
    _write_json(out / 'light_transport_integrity_report.json', integrity)
    if integrity['derived_outputs_safe']:
        summary = _write_light_transport_outputs_checked(out, diag, g, d, cfg, context)
    else:
        summary = {
            key: value
            for key, value in diag.items()
            if not isinstance(value, np.ndarray)
        }
        if cfg['save_data']:
            safe = {
                key: diag[key]
                for key, valid in integrity['field_validity'].items()
                if valid
                and key
                in (
                    'track_length_m',
                    'post_scatter_track_length_m',
                    'scatter_event_weight',
                    'scatter_event_count',
                )
            }
            if safe:
                np.savez_compressed(
                    out / 'light_transport_safe_raw_fields.npz',
                    **safe,
                    diagnostic_rays_processed=diag['diagnostic_rays_processed'],
                    ray_statistical_weight=diag['ray_statistical_weight'],
                    grid_bounds_m=np.array(
                        [
                            [diag[k + '_min_m'], diag[k + '_max_m']]
                            for k in ('x', 'y', 'z')
                        ]
                    ),
                )
    summary.update(integrity)
    _write_json(out / 'light_transport_summary.json', summary)
    return summary


def _write_light_transport_outputs_checked(out: Path, diag, g, d, cfg, context=None):
    if not diag:
        return {}
    q = _light_transport_cross_sections(diag, cfg)
    display = _light_transport_display_state(diag, q, g, d, cfg)
    rays = max(int(diag['diagnostic_rays_processed']), 1)
    context = context or {}
    subtitle = (
        f"{context.get('material','validation')} | {context.get('concentration_g_L','?')} g/L | {context.get('wavelength_nm','?')} nm\n"
        f"{d['free_surface_model']} | production {context.get('production_rays','?')} rays; diagnostic {rays} rays"
    )
    names = {
        0: 'propagation_or_boundary',
        1: 'geometric_external_reflection',
        2: 'geometric_transmission',
        3: 'geometric_internal_reflection',
        4: 'coherent_mie',
        5: 'morphology_external_reflection',
        6: 'morphology_transmission',
        7: 'morphology_internal_reflection',
        8: 'null_collision',
        9: 'particle_absorption',
        10: 'particle_transport_failure',
    }
    counts = np.asarray(diag['trajectory_count'], dtype=int)
    overflow = np.asarray(diag['trajectory_overflow'], dtype=int)
    tr = []
    for ray, n in enumerate(counts):
        for j in range(int(n)):
            code = int(diag['trajectory_event_code'][ray, j])
            tr.append(
                {
                    'ray_id': ray,
                    'segment_index': j,
                    'x0_m': diag['trajectory_x0_m'][ray, j],
                    'y0_m': diag['trajectory_y0_m'][ray, j],
                    'z0_m_sensor_frame': diag['trajectory_z0_m'][ray, j],
                    'x1_m': diag['trajectory_x1_m'][ray, j],
                    'y1_m': diag['trajectory_y1_m'][ray, j],
                    'z1_m_sensor_frame': diag['trajectory_z1_m'][ray, j],
                    'start_height_above_bottom_m': diag['trajectory_z0_m'][ray, j]
                    + g['sensor_ring_height_m'],
                    'end_height_above_bottom_m': diag['trajectory_z1_m'][ray, j]
                    + g['sensor_ring_height_m'],
                    'segment_length_m': float(
                        np.sqrt(
                            (
                                diag['trajectory_x1_m'][ray, j]
                                - diag['trajectory_x0_m'][ray, j]
                            )
                            ** 2
                            + (
                                diag['trajectory_y1_m'][ray, j]
                                - diag['trajectory_y0_m'][ray, j]
                            )
                            ** 2
                            + (
                                diag['trajectory_z1_m'][ray, j]
                                - diag['trajectory_z0_m'][ray, j]
                            )
                            ** 2
                        )
                    ),
                    'ray_weight': float(diag['ray_statistical_weight']),
                    'particle_order_before_segment': int(
                        diag['trajectory_particle_order'][ray, j]
                    ),
                    'segment_after_particle_scatter': bool(
                        diag['trajectory_particle_order'][ray, j] > 0
                    ),
                    'terminal_event_code': code,
                    'terminal_event_name': names.get(code, 'unknown'),
                    'ray_truncated': bool(overflow[ray]),
                }
            )
    if cfg['save_data']:
        vr = [
            {
                'y_m': y,
                'height_above_bottom_m': z + g['sensor_ring_height_m'],
                'z_m_sensor_frame': z,
                'track_length_m': q['vr'][iz, iy],
                'post_scatter_track_length_m': q['vp'][iz, iy],
                'scatter_event_count': int(q['ve'][iz, iy]),
                'track_length_density_per_ray_m2': q['vd'][iz, iy],
                'post_scatter_density_per_ray_m2': q['vpd'][iz, iy],
                'scatter_event_density_per_ray_m3': q['ved'][iz, iy],
            }
            for iz, z in enumerate(q['z'])
            for iy, y in enumerate(q['y'])
        ]
        hr = [
            {
                'x_m': x,
                'y_m': y,
                'track_length_m': q['hr'][iy, ix],
                'post_scatter_track_length_m': q['hp'][iy, ix],
                'scatter_event_count': int(q['he'][iy, ix]),
                'track_length_density_per_ray_m2': q['hd'][iy, ix],
                'post_scatter_density_per_ray_m2': q['hpd'][iy, ix],
                'scatter_event_density_per_ray_m3': q['hed'][iy, ix],
            }
            for iy, y in enumerate(q['y'])
            for ix, x in enumerate(q['x'])
        ]
        _save_csv_unconditional(
            out / 'light_transport_vertical_cross_section.csv', vr[0].keys(), vr
        )
        _save_csv_unconditional(
            out / 'light_transport_sensor_plane_cross_section.csv', hr[0].keys(), hr
        )
        fields = (
            tr[0].keys()
            if tr
            else (
                'ray_id',
                'segment_index',
                'x0_m',
                'y0_m',
                'z0_m_sensor_frame',
                'x1_m',
                'y1_m',
                'z1_m_sensor_frame',
                'start_height_above_bottom_m',
                'end_height_above_bottom_m',
                'segment_length_m',
                'ray_weight',
                'particle_order_before_segment',
                'segment_after_particle_scatter',
                'terminal_event_code',
                'terminal_event_name',
                'ray_truncated',
            )
        )
        _save_csv_unconditional(
            out / 'light_transport_representative_trajectories.csv', fields, tr
        )
        np.savez_compressed(
            out / 'light_transport_volume.npz',
            total_track_length_weighted_m=diag['track_length_m'],
            post_scatter_track_length_weighted_m=diag['post_scatter_track_length_m'],
            particle_scatter_event_weight=diag['scatter_event_weight'],
            particle_scatter_event_count=diag['scatter_event_count'],
            x_edges_m=np.linspace(
                diag['x_min_m'], diag['x_max_m'], int(diag['nx']) + 1
            ),
            y_edges_m=np.linspace(
                diag['y_min_m'], diag['y_max_m'], int(diag['ny']) + 1
            ),
            z_edges_above_bottom_m=np.linspace(
                diag['z_min_m'] + g['sensor_ring_height_m'],
                diag['z_max_m'] + g['sensor_ring_height_m'],
                int(diag['nz']) + 1,
            ),
            x_centres_m=q['x'],
            y_centres_m=q['y'],
            z_centres_above_bottom_m=q['z'] + g['sensor_ring_height_m'],
            diagnostic_rays_processed=np.asarray(rays),
            ray_statistical_weight=np.asarray(diag['ray_statistical_weight']),
        )
    summary = {
        'diagnostics_enabled': True,
        'provenance': 'actual CUDA production ray histories in water; no synthetic rays',
        'diagnostic_rays_processed': int(diag['diagnostic_rays_processed']),
        'diagnostic_ray_limit': int(diag['diagnostic_ray_limit']),
        'grid_shape_zyx': [int(diag['nz']), int(diag['ny']), int(diag['nx'])],
        'grid_bounds_m': {
            'x': [diag['x_min_m'], diag['x_max_m']],
            'y': [diag['y_min_m'], diag['y_max_m']],
            'z_sensor_frame': [diag['z_min_m'], diag['z_max_m']],
        },
        'voxel_size_m': [q['dx'], q['dy'], q['dz']],
        'raw_quantity': 'water track length (m) and accepted scatter-event count',
        'normalized_quantity': 'track-length density per diagnostic ray (m^-2 ray^-1); event density per diagnostic ray (m^-3 ray^-1)',
        'diagnostic_memory_bytes': int(diag['diagnostic_memory_bytes']),
        'diagnostic_active_wall_time_s': float(diag['diagnostic_active_wall_time_s']),
        'segment_count': int(diag['segment_count']),
        'scatter_event_total': int(diag['scatter_event_total']),
        'deposited_scatter_event_total': int(diag['deposited_scatter_event_total']),
        'scatter_event_closure_error': int(diag['scatter_event_closure_error']),
        'segment_length_total_m': diag['segment_length_total_m'],
        'deposited_length_total_m': diag['deposited_length_total_m'],
        'deposition_closure_relative_error': diag['deposition_closure_relative_error'],
        'grid_closure_relative_error': diag['grid_closure_relative_error'],
        'water_advance_guard_failure_count': int(
            diag.get('water_advance_guard_failure_count', 0)
        ),
        'first_water_advance_guard_failure': diag.get(
            'first_water_advance_guard_failure'
        ),
        'diagnostic_error_count': int(diag['error_count']),
        'diagnostic_hard_error_count': int(
            diag.get('hard_error_count', diag['error_count'])
        ),
        'boundary_adjustment_count': int(diag.get('boundary_adjustment_count', 0)),
        'event_outside_count': int(diag.get('event_outside_count', 0)),
        'nonfinite_count': int(diag['nonfinite_count']),
        'negative_value_count': int(diag['negative_value_count']),
        'representative_rays_requested': int(diag['representative_ray_limit']),
        'representative_rays_with_segments': int(np.count_nonzero(counts)),
        'representative_trajectory_segments': int(np.sum(counts)),
        'representative_trajectory_truncated_rays': int(np.count_nonzero(overflow)),
        'max_segments_per_ray': int(diag['max_segments_per_ray']),
        'representative_ray_selection': diag['selection_method'],
        'vertical_plane_definition': 'y-z plane derived from the production source at (x=0,y=-SOURCE_R,z=0) and nominal +y direction; slab average about x=0',
        'vertical_slice_thickness_requested_m': cfg['vertical_slice_thickness_m'],
        'vertical_slice_thickness_actual_m': len(q['xi']) * q['dx'],
        'horizontal_plane_definition': 'sensor plane z=0; slab average in production sensor-frame z',
        'sensor_slice_thickness_requested_m': cfg['sensor_slice_thickness_m'],
        'sensor_slice_thickness_actual_m': len(q['zi']) * q['dz'],
        'display_colour_scheme': 'strict greyscale; lower track-length density is darker and higher density is lighter',
        'display_density_to_brightness_direction': 'increasing density maps monotonically from black/dark grey to light grey/white',
        'display_normalization': 'shared robust log10 percentile scale from finite positive in-water 3-D track-length density',
        'display_low_percentile': float(
            cfg.get('display_low_percentile', LIGHT_TRANSPORT_DISPLAY_LOW_PERCENTILE)
        ),
        'display_high_percentile': float(
            cfg.get('display_high_percentile', LIGHT_TRANSPORT_DISPLAY_HIGH_PERCENTILE)
        ),
        'display_log10_vmin': display['display_log10_vmin'],
        'display_log10_vmax': display['display_log10_vmax'],
        'vertical_display_log10_vmin': display['display_log10_vmin'],
        'vertical_display_log10_vmax': display['display_log10_vmax'],
        'sensor_plane_display_log10_vmin': display['display_log10_vmin'],
        'sensor_plane_display_log10_vmax': display['display_log10_vmax'],
        'display_scale_mode': display['display_scale_mode'],
        'display_positive_voxel_count': display['display_positive_voxel_count'],
        'display_scale_shared_by_2d_sections': True,
        'outside_water_rendering': 'white/transparent',
        'detector_geometry_drawn': False,
        'scatter_event_contours_drawn': False,
        'isometric_track_length_display_percentile': float(
            cfg.get(
                'display_3d_min_percentile', LIGHT_TRANSPORT_3D_MIN_DISPLAY_PERCENTILE
            )
        ),
        'isometric_requested_min_display_percentile': float(
            cfg.get(
                'display_3d_min_percentile', LIGHT_TRANSPORT_3D_MIN_DISPLAY_PERCENTILE
            )
        ),
        'isometric_actual_min_display_percentile': (
            float(
                cfg.get(
                    'display_3d_min_percentile',
                    LIGHT_TRANSPORT_3D_MIN_DISPLAY_PERCENTILE,
                )
            )
            if display['cloud_threshold_log10'] is not None
            else None
        ),
        'isometric_cloud_threshold_log10': display['cloud_threshold_log10'],
        'isometric_cloud_candidate_voxels': display['cloud_candidate_count'],
        'isometric_eligible_voxel_count': display['cloud_candidate_count'],
        'isometric_cloud_displayed_voxels': display['cloud_display_count'],
        'isometric_max_display_voxels': display['cloud_max_display_voxels'],
        'isometric_cloud_selection_method': display['cloud_selection_method'],
        'isometric_alpha_min': float(
            cfg.get('display_3d_alpha_min', LIGHT_TRANSPORT_3D_ALPHA_MIN)
        ),
        'isometric_alpha_max': float(
            cfg.get('display_3d_alpha_max', LIGHT_TRANSPORT_3D_ALPHA_MAX)
        ),
        'isometric_marker_size_min': float(
            cfg.get('display_3d_marker_size_min', LIGHT_TRANSPORT_3D_MARKER_SIZE_MIN)
        ),
        'isometric_marker_size_max': float(
            cfg.get('display_3d_marker_size_max', LIGHT_TRANSPORT_3D_MARKER_SIZE_MAX)
        ),
        'isometric_plot_z_min_m': display['plot_3d_z_min_m'],
        'isometric_plot_z_max_m': display['plot_3d_z_max_m'],
        'maximum_water_surface_height_above_bottom_m': display['surface_max_height_m'],
        'isometric_surface_margin_m': display['plot_3d_surface_margin_m'],
        'display_threshold_affects_raw_data': False,
        'source_callouts_outside_data_area': True,
        'transport_invariance_design': 'diagnostic writes consume no RNG and do not influence transport',
        'trajectory_event_codes': {str(k): v for k, v in names.items()},
    }
    if cfg['save_visualisations']:
        plt = _get_pyplot()
        R = float(g['sample_cell_radius_m'])
        sensor_h = float(g['sensor_ring_height_m'])
        vmin = float(display['display_log10_vmin'])
        vmax = float(display['display_log10_vmax'])
        cmap = plt.get_cmap('gray').with_extremes(
            bad=(1.0, 1.0, 1.0, 1.0), under=(0.0, 0.0, 0.0, 1.0)
        )
        # Vertical source-axis section: only the water domain and neutral reference geometry.
        vfield = _masked_light_log_field(q['vd'], display['vertical_water_mask'], cfg)
        fig, ax = plt.subplots(figsize=(9.5, 7))
        fig.subplots_adjust(left=0.31, right=0.84, bottom=0.12, top=0.79)
        im = ax.imshow(
            vfield,
            origin='lower',
            aspect='auto',
            extent=[
                diag['y_min_m'] * 1e3,
                diag['y_max_m'] * 1e3,
                (diag['z_min_m'] + sensor_h) * 1e3,
                (diag['z_max_m'] + sensor_h) * 1e3,
            ],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation='bilinear',
            interpolation_stage='rgba',
        )
        surf = _free_surface_z_sensor_frame(np.abs(q['y']), g, d) + sensor_h
        ax.plot(q['y'] * 1e3, surf * 1e3, color='0.2', lw=1.3, label='water surface')
        ax.axhline(sensor_h * 1e3, color='0.45', ls='--', lw=0.9, label='sensor plane')
        ax.axvline(-R * 1e3, color='0.25', lw=0.8)
        ax.axvline(R * 1e3, color='0.25', lw=0.8)
        ax.set_xlim(-1.03 * R * 1e3, 1.03 * R * 1e3)
        ax.set_ylim(0.0, display['plot_3d_z_max_m'] * 1e3)
        ax.plot(
            [-R * 1e3],
            [sensor_h * 1e3],
            marker='>',
            ms=6,
            markerfacecolor='white',
            markeredgecolor='black',
            markeredgewidth=0.8,
        )
        zfrac = np.clip(
            sensor_h / max(float(display['plot_3d_z_max_m']), 1.0e-12), 0.05, 0.95
        )
        source_note = ax.annotate(
            'source entry',
            xy=(-R * 1e3, sensor_h * 1e3),
            xycoords='data',
            xytext=(-0.31, zfrac),
            textcoords='axes fraction',
            ha='right',
            va='center',
            annotation_clip=False,
            color='black',
            fontsize=8,
            bbox={
                'boxstyle': 'round,pad=.2',
                'facecolor': 'white',
                'edgecolor': '0.35',
                'alpha': 0.98,
            },
            arrowprops={
                'arrowstyle': '->',
                'color': 'black',
                'lw': 0.8,
                'shrinkA': 2,
                'shrinkB': 2,
            },
        )
        source_note.set_gid('source-entry-vertical')
        ax.set(
            xlabel='Source-axis y (mm)',
            ylabel='Height above internal bottom (mm)',
            title='CUDA light-history density: vertical source-axis section\n'
            + subtitle,
        )
        ax.title.set_fontsize(11)
        fig.colorbar(
            im, ax=ax, label='log10 track-length density (m$^{-2}$ ray$^{-1}$)'
        )
        ax.legend(loc='lower right', fontsize=8, framealpha=0.85)
        _plot_save_light(fig, out / 'light_transport_vertical_cross_section.png')
        # Horizontal sensor-plane section: no detector rings, event contours or coloured overlays.
        hfield = _masked_light_log_field(q['hd'], display['horizontal_water_mask'], cfg)
        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        fig.subplots_adjust(left=0.12, right=0.82, bottom=0.24, top=0.79)
        im = ax.imshow(
            hfield,
            origin='lower',
            extent=[
                diag['x_min_m'] * 1e3,
                diag['x_max_m'] * 1e3,
                diag['y_min_m'] * 1e3,
                diag['y_max_m'] * 1e3,
            ],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            interpolation='bilinear',
            interpolation_stage='rgba',
        )
        from matplotlib.patches import Circle

        ax.add_patch(Circle((0, 0), R * 1e3, fill=False, color='0.25', lw=1.0))
        ax.set_xlim(-1.03 * R * 1e3, 1.03 * R * 1e3)
        ax.set_ylim(-1.03 * R * 1e3, 1.03 * R * 1e3)
        ax.set_aspect('equal')
        ax.plot(
            [0],
            [-R * 1e3],
            marker='^',
            ms=6,
            markerfacecolor='white',
            markeredgecolor='black',
            markeredgewidth=0.8,
        )
        source_note = ax.annotate(
            'source entry',
            xy=(0, -R * 1e3),
            xycoords='data',
            xytext=(0.5, -0.20),
            textcoords='axes fraction',
            ha='center',
            va='top',
            annotation_clip=False,
            color='black',
            fontsize=8,
            bbox={
                'boxstyle': 'round,pad=.2',
                'facecolor': 'white',
                'edgecolor': '0.35',
                'alpha': 0.98,
            },
            arrowprops={
                'arrowstyle': '->',
                'color': 'black',
                'lw': 0.8,
                'shrinkA': 2,
                'shrinkB': 2,
            },
        )
        source_note.set_gid('source-entry-horizontal')
        ax.set(
            xlabel='x (mm)',
            ylabel='Source-axis y (mm)',
            title='CUDA light-history density: sensor-plane section\n' + subtitle,
        )
        ax.title.set_fontsize(11)
        fig.colorbar(
            im, ax=ax, label='log10 track-length density (m$^{-2}$ ray$^{-1}$)'
        )
        _plot_save_light(fig, out / 'light_transport_sensor_plane_cross_section.png')
        # Isometric density cloud: crop to the water column and use deterministic greyscale opacity.
        fig = plt.figure(figsize=(10.5, 8), facecolor='white')
        fig.subplots_adjust(left=0.17, right=0.82, bottom=0.10, top=0.82)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor((0.09, 0.09, 0.09))
        ang = np.linspace(0, 2 * np.pi, 128)
        pane = (0.11, 0.11, 0.11, 1.0)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color(pane)
            axis._axinfo['grid']['color'] = (0.42, 0.42, 0.42, 0.55)
            axis.label.set_color('0.85')
            axis.set_tick_params(colors='0.85')
            axis.line.set_color('0.65')
        wall_h = float(_free_surface_z_sensor_frame(np.array([R]), g, d)[0] + sensor_h)
        rr = np.linspace(0, R, 128)
        aa = np.linspace(0, 2 * np.pi, 256)
        RR, AA = np.meshgrid(rr, aa)
        ZZ = _free_surface_z_sensor_frame(RR, g, d) + sensor_h
        ax.plot_surface(
            RR * np.cos(AA) * 1e3,
            RR * np.sin(AA) * 1e3,
            ZZ * 1e3,
            color='0.72',
            alpha=0.07,
            linewidth=0,
            shade=False,
            rcount=256,
            ccount=128,
            antialiased=True,
        )
        vid = display['cloud_indices']
        normed = display['cloud_normalized']
        if len(vid):
            amin = float(cfg.get('display_3d_alpha_min', LIGHT_TRANSPORT_3D_ALPHA_MIN))
            amax = float(cfg.get('display_3d_alpha_max', LIGHT_TRANSPORT_3D_ALPHA_MAX))
            smin = float(
                cfg.get(
                    'display_3d_marker_size_min', LIGHT_TRANSPORT_3D_MARKER_SIZE_MIN
                )
            )
            smax = float(
                cfg.get(
                    'display_3d_marker_size_max', LIGHT_TRANSPORT_3D_MARKER_SIZE_MAX
                )
            )
            grey = 0.06 + 0.94 * np.power(normed, 0.90)
            alpha = amin + (amax - amin) * np.power(normed, 1.10)
            rgba = np.column_stack((grey, grey, grey, alpha))
            sizes = smin + (smax - smin) * np.power(normed, 1.10)
            ax.scatter(
                q['x'][vid[:, 2]] * 1e3,
                q['y'][vid[:, 1]] * 1e3,
                (q['z'][vid[:, 0]] + sensor_h) * 1e3,
                s=sizes,
                c=rgba,
                marker='.',
                depthshade=False,
                edgecolors='none',
                linewidths=0,
                rasterized=True,
            )
        ax.scatter(
            [0],
            [-R * 1e3],
            [sensor_h * 1e3],
            marker='*',
            s=60,
            c='white',
            edgecolors='black',
            linewidths=0.7,
            depthshade=False,
        )
        lim = 1.03 * R
        zmax = float(display['plot_3d_z_max_m'])
        ax.set_xlim(-lim * 1e3, lim * 1e3)
        ax.set_ylim(-lim * 1e3, lim * 1e3)
        ax.set_zlim(0.0, zmax * 1e3)
        ax.title.set_fontsize(11)
        ax.set_box_aspect((2 * lim, 2 * lim, zmax))
        ax.view_init(elev=30, azim=0)
        ax.set_axis_off()
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        sm = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
        sm.set_array([])
        fig.colorbar(
            sm,
            ax=ax,
            pad=0.08,
            shrink=0.68,
            label='log10 track-length density (m$^{-2}$ ray$^{-1}$)',
        )
        from mpl_toolkits.mplot3d import proj3d

        fig.canvas.draw()
        sx, sy, _ = proj3d.proj_transform(0.0, -R * 1e3, sensor_h * 1e3, ax.get_proj())
        _plot_save_light(fig, out / 'light_transport_3d_isometric.png')
    _write_json(out / 'light_transport_summary.json', summary)
    return summary


def _nearest_detector_index(angle_deg: float) -> int:
    return int(np.argmin(np.abs(DETECTOR_ANGLES_DEG - float(angle_deg))))


def _write_detector_outputs(
    out: Path, material: str, concentration: float, wavelength_nm: float, seed: int, res
):
    score = np.asarray(res.hardware_symmetry_scores, dtype=float)
    norm = np.asarray(res.normalized_hardware_symmetry_response, dtype=float)
    jk = np.asarray(res.hardware_normalized_jackknife_se, dtype=float)
    native = np.asarray(res.hardware_native_hits, dtype=np.int64)
    mirror = np.asarray(res.hardware_mirror_hits, dtype=np.int64)
    expected = (len(DETECTOR_ANGLES_DEG),)
    if any(a.shape != expected for a in (score, norm, jk, native, mirror)):
        raise ValueError('corrupted production detector array shape')
    if any(
        not np.all(np.isfinite(a)) or np.any(a < 0)
        for a in (score, norm, native, mirror)
    ):
        raise CaseIntegrityError('nonfinite or negative production detector field')
    rows = []
    for i, angle in enumerate(DETECTOR_ANGLES_DEG):
        rows.append(
            {
                'material': material,
                'concentration_g_L': concentration,
                'wavelength_nm': wavelength_nm,
                'seed': seed,
                'n_rays': int(res.n_rays),
                'angle_deg': float(angle),
                'absolute_score': float(score[i]),
                'normalized_score': float(norm[i]),
                'native_hit_count': int(native[i]),
                'mirror_hit_count': int(mirror[i]),
                'absolute_score_sqrtN_se_approx': float(math.sqrt(max(score[i], 0.0))),
                'normalized_score_jackknife_se': (
                    float(jk[i]) if np.isfinite(jk[i]) else None
                ),
            }
        )
    _save_csv(out / 'detector_response.csv', rows[0].keys(), rows)
    return rows


def _write_particle_optics(
    out: Path, material: str, concentration: float, wavelength_nm: float, ph
):
    keys = [
        'diameter_m',
        'source_weight',
        'number_density_by_bin',
        'n_real',
        'k_imag',
        'size_parameter',
        'qext',
        'qsca',
        'qabs',
        'qback',
        'asymmetry_g',
        'single_scattering_albedo',
        'geometric_cross_section_m2',
        'scattering_cross_section_m2',
        'extinction_cross_section_m2',
        'event_cross_section_m2',
        'mu_geom_by_bin',
        'mu_sca_by_bin',
        'mu_ext_by_bin',
        'mu_wave_by_bin',
        'mu_event_by_bin',
        'sediment_settling_velocity_m_s',
        'sediment_vertical_inverse_length_B_per_m',
        'sediment_max_density_scale',
    ]
    n = len(np.asarray(ph['diameter_m']))
    rows = []
    for i in range(n):
        r = {
            'material': material,
            'concentration_g_L': concentration,
            'wavelength_nm': wavelength_nm,
            'bin_index': i,
            'diameter_um': float(np.asarray(ph['diameter_m'])[i] * 1e6),
        }
        for k in keys:
            if k == 'diameter_m' or k not in ph:
                continue
            arr = np.asarray(ph[k])
            if arr.shape == (n,):
                r[k] = float(arr[i])
        rows.append(r)
    if rows:
        _save_csv(out / 'particle_optical_properties.csv', rows[0].keys(), rows)
    return rows


def _phase_function_outputs(out: Path, model, material: str, concentration: float, ph):
    theta, cdf = model.phase_table_diagnostics(material, concentration)
    theta = np.asarray(theta, dtype=float)
    cdf = np.asarray(cdf, dtype=float)
    weights = np.asarray(
        ph.get('wave_event_weights', ph.get('source_weight')), dtype=float
    )
    if weights.ndim != 1 or weights.size != cdf.shape[0] or float(weights.sum()) <= 0:
        weights = np.ones(cdf.shape[0], dtype=float)
    weights = weights / weights.sum()
    mean_cdf = np.sum(cdf * weights[:, None], axis=0)
    pdf = np.gradient(mean_cdf, theta, edge_order=1)
    pdf = np.maximum(pdf, 0.0)
    phase_rows = [
        {
            'angle_deg': float(np.degrees(theta[i])),
            'event_weighted_mie_cdf': float(mean_cdf[i]),
            'event_weighted_mie_pdf_per_rad': float(pdf[i]),
        }
        for i in range(theta.size)
    ]
    if SAVE_PHASE_FUNCTION_CSV:
        _save_csv(out / 'phase_function.csv', phase_rows[0].keys(), phase_rows)
    return np.degrees(theta), pdf


def _plot_case(
    out: Path,
    material: str,
    concentration: float,
    wavelength_nm: float,
    detector_rows,
    particle_rows,
    phase_angle,
    phase_pdf,
    summary,
):
    if not SAVE_PLOTS and not SHOW_PLOTS:
        return
    plt = _get_pyplot()
    angles = np.array([r['angle_deg'] for r in detector_rows], dtype=float)
    scores = np.array([r['absolute_score'] for r in detector_rows], dtype=float)
    norm = np.array([r['normalized_score'] for r in detector_rows], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(angles, scores, marker='o')
    ax.set_xlabel('Detector angle (°)')
    ax.set_ylabel('Absolute detector score')
    ax.set_title(f'{material.capitalize()} {concentration:g} g/L, {wavelength_nm:g} nm')
    ax.grid(True, alpha=0.25)
    _plot_save(fig, out / 'detector_response_absolute.png')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(angles, norm, marker='o')
    ax.set_xlabel('Detector angle (°)')
    ax.set_ylabel('Normalized detector score')
    ax.set_title(f'{material.capitalize()} {concentration:g} g/L, {wavelength_nm:g} nm')
    ax.grid(True, alpha=0.25)
    _plot_save(fig, out / 'detector_response_normalized.png')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(angles, np.maximum(scores, 1e-12), marker='o')
    ax.set_xlabel('Detector angle (°)')
    ax.set_ylabel('Absolute detector score (log scale)')
    ax.set_title(f'{material.capitalize()} {concentration:g} g/L, {wavelength_nm:g} nm')
    ax.grid(True, alpha=0.25)
    _plot_save(fig, out / 'detector_response_log.png')

    if particle_rows:
        d = np.array([r['diameter_um'] for r in particle_rows], dtype=float)
        w = np.array(
            [r.get('source_weight', np.nan) for r in particle_rows], dtype=float
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogx(d, w, marker='o')
        ax.set_xlabel('Particle diameter (µm)')
        ax.set_ylabel('Mass-fraction source weight')
        ax.set_title(f'{material.capitalize()} particle size distribution')
        ax.grid(True, alpha=0.25)
        _plot_save(fig, out / 'particle_size_distribution.png')

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(phase_angle, np.maximum(phase_pdf, 1e-20))
    ax.set_xlabel('Scattering angle (°)')
    ax.set_ylabel('Event-weighted Mie probability density (rad⁻¹)')
    ax.set_title(f'{material.capitalize()} Mie phase diagnostic, {wavelength_nm:g} nm')
    ax.grid(True, alpha=0.25)
    _plot_save(fig, out / 'phase_function.png')

    coeff_names = ['mu_geom_per_m', 'mu_wave_per_m', 'mu_sca_per_m', 'mu_ext_per_m']
    coeff_vals = [float(summary.get(k, 0.0)) for k in coeff_names]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(['geom event', 'wave event', 'scattering', 'extinction'], coeff_vals)
    ax.set_ylabel('Coefficient (m⁻¹)')
    ax.set_title('Optical coefficients')
    ax.grid(True, axis='y', alpha=0.25)
    _plot_save(fig, out / 'optical_coefficients.png')


def _summary_dict(
    material: str,
    concentration: float,
    wavelength_nm: float,
    seed: int,
    gpu_name: str,
    runtime_s: float,
    res,
    ph,
    wl_info,
    out: Path,
):
    total_nd = float(
        np.sum(np.asarray(ph.get('number_density_by_bin', []), dtype=float))
    )
    mu_sca = float(res.mu_sca_per_m)
    mu_ext = float(res.mu_ext_per_m)
    mu_abs = max(mu_ext - mu_sca, 0.0)
    albedo = mu_sca / mu_ext if mu_ext > 0 else None
    g = _resolved_geometry()
    d = _resolved_sediment_config()
    zmean = g['water_level_m'] - g['sensor_ring_height_m']
    vg = vortex_surface_geometry(
        d['free_surface_model'],
        zmean,
        g['sample_cell_radius_m'],
        d['vortex_wall_center_height_difference_m'],
        d['vortex_core_radius_m'],
        d['gravity_m_s2'],
    )
    return {
        'claritas_source_sha256': _current_source_sha256(),
        'configuration_valid': True,
        'material': material,
        'concentration_g_L': float(concentration),
        'wavelength_nm': float(wavelength_nm),
        'n_rays': int(res.n_rays),
        'requested_n_rays': int(res.requested_n_rays),
        'seed': int(seed),
        'runtime_s': float(runtime_s),
        'gpu_device_index': int(GPU_DEVICE),
        'gpu_name': gpu_name,
        'output_directory': str(out),
        'stop_reason': res.stop_reason,
        'production_failure_count': int(res.production_failure_count),
        'entered_water_fraction': float(res.entered_water_fraction),
        'air_exit_fraction': float(res.air_exit_fraction),
        'axial_loss_fraction': float(res.axial_loss_fraction),
        'detector_detection_fraction': float(res.detector_detection_fraction),
        'mean_particle_interactions': float(res.mean_interactions),
        'mean_geometric_interactions': float(res.mean_geometric_interactions),
        'mean_wave_interactions': float(res.mean_wave_interactions),
        'total_particle_number_density_m3': total_nd,
        'mu_geom_per_m': float(res.mu_geom_per_m),
        'mu_wave_per_m': float(res.mu_wave_per_m),
        'mu_event_per_m': float(res.mu_event_per_m),
        'mu_sca_per_m': mu_sca,
        'mu_ext_per_m': mu_ext,
        'mu_abs_per_m': mu_abs,
        'single_scattering_albedo': albedo,
        'optical_depth_diameter': float(res.optical_depth_diameter),
        'geometric_optical_depth_diameter': float(res.geometric_optical_depth_diameter),
        'sediment_distribution': d['sediment_distribution'],
        'steady_state_sediment_enabled': d['steady_state_sediment_enabled'],
        'sediment_transport_model': res.sediment_transport_model,
        'sediment_eddy_diffusivity_m2_s': float(res.sediment_eddy_diffusivity_m2_s),
        'sediment_particle_density_kg_m3': d['sediment_particle_density_kg_m3'],
        'water_density_kg_m3': d['water_density_kg_m3'],
        'water_dynamic_viscosity_pa_s': d['water_dynamic_viscosity_pa_s'],
        'gravity_m_s2': d['gravity_m_s2'],
        'vortex_enabled': d['enable_vortex'],
        'free_surface_model': d['free_surface_model'],
        'vortex_wall_center_height_difference_m': d[
            'vortex_wall_center_height_difference_m'
        ],
        'vortex_core_radius_m': (
            d['vortex_core_radius_m']
            if d['free_surface_model'] == 'localized_scully_vortex'
            else None
        ),
        'vortex_equivalent_core_rpm': (
            float(vg['equivalent_core_rpm'])
            if np.isfinite(vg['equivalent_core_rpm'])
            else None
        ),
        'vortex_equivalent_rigid_body_rpm': (
            float(vg['equivalent_rigid_body_rpm'])
            if np.isfinite(vg['equivalent_rigid_body_rpm'])
            else None
        ),
        'vortex_center_surface_height_above_bottom_m': float(
            vg['center_z'] + g['sensor_ring_height_m']
        ),
        'vortex_wall_surface_height_above_bottom_m': float(
            vg['wall_z'] + g['sensor_ring_height_m']
        ),
        'meridional_circulation_ratio': d['meridional_circulation_ratio'],
        'sample_cell_diameter_m': g['sample_cell_diameter_m'],
        'sample_cell_height_m': g['sample_cell_height_m'],
        'water_level_m': g['water_level_m'],
        'sensor_ring_height_m': g['sensor_ring_height_m'],
        'validated_geometry_baseline': _matches_comparison_geometry(g),
        'validated_sediment_baseline': _matches_comparison_sediment(d),
        'comparison_configuration_match': bool(
            _matches_comparison_geometry(g) and _matches_comparison_sediment(d)
        ),
        'parabolic_transport_validation_pass': (
            bool(
                int(res.production_failure_count) == 0
                and np.isfinite(float(res.sediment_majorant_total_per_m))
            )
            if d['free_surface_model'] == 'parabolic_vortex'
            else None
        ),
        'particle_event_model': res.particle_event_model,
        'particle_surface_rms_slope_deg': float(res.particle_surface_rms_slope_deg),
        'wave_surface_rms_height_nm': float(res.wave_surface_rms_height_nm),
        'wavelength_assumptions': wl_info,
    }


def run_one_case(
    session_root: Path,
    material: str,
    concentration: float,
    wavelength_nm: float,
    seed: int,
    gpu_name: str,
    detailed_sediment=True,
):
    material = _validate_material(material)
    concentration = float(concentration)
    wavelength_nm = float(wavelength_nm)
    if concentration < 0 or not np.isfinite(concentration):
        raise ValueError('concentration must be finite and >=0')
    if wavelength_nm <= 0 or not np.isfinite(wavelength_nm):
        raise ValueError('wavelength must be finite and >0')
    out = _case_output_dir(session_root, material, wavelength_nm, concentration)
    model, model_kw, nvals, kvals = _build_model(material, wavelength_nm)
    wl_info = _wavelength_assumption_summary(material, wavelength_nm, nvals, kvals)
    _print_wavelength_summary(wl_info)

    g = _resolved_geometry()
    d = _resolved_sediment_config()
    light_cfg = _resolved_light_transport_config()
    config = {
        'claritas_source_sha256': _current_source_sha256(),
        'material': material,
        'concentration_g_L': concentration,
        'bulk_concentration_definition': 'total sediment loading integrated over the actual water volume; not local sensor/bottom/top concentration',
        'wavelength_nm': wavelength_nm,
        'n_rays': int(N_RAYS),
        'seed': int(seed),
        'gpu_device': int(GPU_DEVICE),
        'summary_angles_deg': list(SUMMARY_ANGLES_DEG),
        'sediment_profile_radii_fraction': list(SEDIMENT_PROFILE_RADII_FRACTION),
        'psd_profile_height_fractions': list(PSD_PROFILE_HEIGHT_FRACTIONS),
        'model_kwargs': model_kw,
        'geometry': g,
        'sediment': d,
        'light_transport_diagnostics': light_cfg,
        'failure_policy': _resolved_failure_policy(),
        'validated_geometry_baseline': _matches_comparison_geometry(g),
        'validated_sediment_baseline': _matches_comparison_sediment(d),
        'wavelength_assumptions': wl_info,
    }
    _write_json(out / 'run_config.json', config)

    t0 = time.perf_counter()
    res = model.simulate(
        material,
        concentration,
        n_rays=int(N_RAYS),
        seed=int(seed),
        collect_rays=False,
        heatmap_size=0,
        light_transport_config=light_cfg,
    )
    runtime_s = time.perf_counter() - t0
    ph = res.particle_diagnostics or {}
    production_integrity = _production_integrity(
        int(res.n_rays),
        int(res.production_failure_count),
        res.production_failure_diagnostics,
        _resolved_failure_policy(),
    )
    _write_json(out / 'production_integrity_report.json', production_integrity)
    _write_json(
        out / 'water_boundary_report.json',
        {
            'guard_failure_count': ph.get('water_advance_guard_failure_count', 0),
            'first_failure': ph.get('first_water_advance_guard_failure'),
            'production_failure_samples': res.production_failure_diagnostics.get(
                'samples', []
            ),
        },
    )
    detector_rows = _write_detector_outputs(
        out, material, concentration, wavelength_nm, seed, res
    )
    particle_rows = _write_particle_optics(
        out, material, concentration, wavelength_nm, ph
    )
    phase_angle, phase_pdf = _phase_function_outputs(
        out, model, material, concentration, ph
    )
    sed = _write_sediment_diagnostics(
        out,
        material,
        concentration,
        wavelength_nm,
        ph,
        g,
        d,
        detailed=detailed_sediment,
    )
    vortex_rows = _write_vortex_outputs(out, g, d)
    light_summary = _write_light_transport_outputs(
        out,
        res.light_transport_diagnostics,
        g,
        d,
        light_cfg,
        {
            'material': material,
            'concentration_g_L': concentration,
            'wavelength_nm': wavelength_nm,
            'production_rays': int(res.n_rays),
        },
    )
    summary = _summary_dict(
        material,
        concentration,
        wavelength_nm,
        seed,
        gpu_name,
        runtime_s,
        res,
        ph,
        wl_info,
        out,
    )
    summary['production_integrity'] = production_integrity
    summary['light_transport_diagnostics'] = light_summary
    summary['light_transport_diagnostic_rays'] = int(
        light_summary.get('diagnostic_rays_processed', 0)
    )
    summary['light_transport_deposition_closure_relative_error'] = light_summary.get(
        'deposition_closure_relative_error'
    )
    summary['light_transport_trajectory_truncated_rays'] = int(
        light_summary.get('representative_trajectory_truncated_rays', 0)
    )
    summary.update(
        {
            'light_transport_visualisation_enabled': bool(
                light_cfg['save_visualisations']
            ),
            'light_transport_diagnostic_ray_limit': int(light_cfg['max_rays']),
            'light_transport_diagnostic_rays_used': int(
                light_summary.get('diagnostic_rays_processed', 0)
            ),
            'light_transport_grid_nx': int(light_cfg['nx']),
            'light_transport_grid_ny': int(light_cfg['ny']),
            'light_transport_grid_nz': int(light_cfg['nz']),
            'light_transport_memory_bytes': light_summary.get(
                'diagnostic_memory_bytes'
            ),
            'light_transport_segment_count': light_summary.get('segment_count'),
            'light_transport_total_in_water_segment_length_m': light_summary.get(
                'segment_length_total_m'
            ),
            'light_transport_deposited_track_length_m': light_summary.get(
                'deposited_length_total_m'
            ),
            'light_transport_track_length_closure_relative_error': light_summary.get(
                'deposition_closure_relative_error'
            ),
            'light_transport_particle_event_count': light_summary.get(
                'scatter_event_total'
            ),
            'light_transport_deposited_particle_event_count': light_summary.get(
                'deposited_scatter_event_total'
            ),
            'light_transport_particle_event_closure_error': light_summary.get(
                'scatter_event_closure_error'
            ),
            'light_transport_trajectory_count': light_summary.get(
                'representative_rays_with_segments'
            ),
            'light_transport_trajectory_overflow_count': light_summary.get(
                'representative_trajectory_truncated_rays'
            ),
            'light_transport_voxel_error_count': light_summary.get(
                'diagnostic_error_count'
            ),
            'light_transport_boundary_adjustment_count': light_summary.get(
                'boundary_adjustment_count'
            ),
            'light_transport_event_outside_count': light_summary.get(
                'event_outside_count'
            ),
            'light_transport_nonfinite_count': light_summary.get('nonfinite_count'),
            'light_transport_negative_value_count': light_summary.get(
                'negative_value_count'
            ),
            'light_transport_runtime_s': light_summary.get(
                'diagnostic_active_wall_time_s'
            ),
            'light_transport_vertical_slice_thickness_m': light_summary.get(
                'vertical_slice_thickness_actual_m'
            ),
            'light_transport_sensor_plane_slice_thickness_m': light_summary.get(
                'sensor_slice_thickness_actual_m'
            ),
        }
    )
    summary.update(sed.get('metrics', {}))
    summary['sediment_distribution_rows_written'] = (
        int(len(sed.get('distribution_rows', []))) if detailed_sediment else 0
    )
    summary['sensor_ring_sediment_rows_written'] = int(len(sed.get('sensor_rows', [])))
    summary['sensor_ring_local_psd_rows_written'] = int(
        len(sed.get('sensor_psd_rows', []))
    )
    summary['sensor_ring_optical_environment_rows_written'] = int(
        len(sed.get('optical_rows', []))
    )
    summary['vortex_profile_rows_written'] = int(len(vortex_rows))
    _write_json(out / 'run_summary.json', summary)
    _save_csv(out / 'simulation_summary.csv', summary.keys(), [summary])
    _plot_case(
        out,
        material,
        concentration,
        wavelength_nm,
        detector_rows,
        particle_rows,
        phase_angle,
        phase_pdf,
        summary,
    )

    print(
        f"  Complete: {runtime_s:.1f} s; detection fraction={res.detector_detection_fraction:.6g}; failures={res.production_failure_count}"
    )
    print(
        f"  Rays: launched={res.n_rays:,}; failed={res.production_failure_count:,} ({production_integrity['failed_ray_fraction']:.3g}); categories={production_integrity['failure_categories']}"
    )
    print(
        f"  Diagnostics: {light_summary['diagnostic_status']}; limited/skipped={light_summary.get('skipped_outputs',[])}"
    )
    print(f"  Output: {out}")
    _enforce_case_integrity(production_integrity, light_summary)
    return (
        summary,
        detector_rows,
        sed.get('sensor_rows', []),
        sed.get('optical_rows', []),
    )


def _aggregate_sweep_plots(
    session_root: Path, summaries, detector_rows, sensor_rows=None, optical_rows=None
):
    if not summaries:
        return
    if SAVE_CSV:
        _save_csv(session_root / 'sweep_summary.csv', summaries[0].keys(), summaries)
        _save_csv(
            session_root / 'sweep_detector_response.csv',
            detector_rows[0].keys(),
            detector_rows,
        )
        if len({s['wavelength_nm'] for s in summaries}) > 1:
            _save_csv(
                session_root / 'wavelength_sweep_summary.csv',
                summaries[0].keys(),
                summaries,
            )
            _save_csv(
                session_root / 'wavelength_sweep_detector_response.csv',
                detector_rows[0].keys(),
                detector_rows,
            )
        if len({s['concentration_g_L'] for s in summaries}) > 1:
            _save_csv(
                session_root / 'concentration_sweep_summary.csv',
                summaries[0].keys(),
                summaries,
            )
            _save_csv(
                session_root / 'concentration_sweep_detector_response.csv',
                detector_rows[0].keys(),
                detector_rows,
            )
        sediment_keys = [
            k
            for k in summaries[0].keys()
            if k.startswith('sediment_')
            or k.startswith('sensor_ring_')
            or k
            in {
                'material',
                'concentration_g_L',
                'wavelength_nm',
                'sediment_distribution',
                'sediment_transport_model',
                'sediment_eddy_diffusivity_m2_s',
                'vortex_enabled',
            }
        ]
        sediment_summary = [{k: s.get(k) for k in sediment_keys} for s in summaries]
        if sediment_summary:
            _save_csv(
                session_root / 'sweep_sediment_summary.csv',
                sediment_summary[0].keys(),
                sediment_summary,
            )
        if sensor_rows:
            _save_csv(
                session_root / 'sweep_sensor_ring_sediment_summary.csv',
                sensor_rows[0].keys(),
                sensor_rows,
            )
        if optical_rows:
            _save_csv(
                session_root / 'sweep_sensor_ring_optical_environment.csv',
                optical_rows[0].keys(),
                optical_rows,
            )
            if len({s['wavelength_nm'] for s in summaries}) > 1:
                _save_csv(
                    session_root / 'sensor_ring_optical_environment_by_wavelength.csv',
                    optical_rows[0].keys(),
                    optical_rows,
                )

    selected = []
    for row in detector_rows:
        for target in SUMMARY_ANGLES_DEG:
            if abs(float(row['angle_deg']) - float(target)) < 1e-9:
                selected.append(dict(row))
                break
    if selected and SAVE_CSV:
        _save_csv(
            session_root / 'selected_angles_summary.csv', selected[0].keys(), selected
        )

    if not SAVE_PLOTS and not SHOW_PLOTS:
        return
    plt = _get_pyplot()
    materials = sorted({s['material'] for s in summaries})
    concs = sorted({float(s['concentration_g_L']) for s in summaries})
    waves = sorted({float(s['wavelength_nm']) for s in summaries})

    # Detector response by wavelength for each material/concentration.
    if len(waves) > 1:
        for material in materials:
            for conc in concs:
                q = [
                    r
                    for r in detector_rows
                    if r['material'] == material
                    and float(r['concentration_g_L']) == conc
                ]
                if not q:
                    continue
                fig, ax = plt.subplots(figsize=(8, 5))
                for wav in waves:
                    z = sorted(
                        [r for r in q if float(r['wavelength_nm']) == wav],
                        key=lambda r: r['angle_deg'],
                    )
                    if z:
                        ax.plot(
                            [r['angle_deg'] for r in z],
                            [r['normalized_score'] for r in z],
                            marker='o',
                            label=f'{wav:g} nm',
                        )
                ax.set_xlabel('Detector angle (°)')
                ax.set_ylabel('Normalized detector score')
                ax.set_title(
                    f'{material.capitalize()} {conc:g} g/L: detector response by wavelength'
                )
                ax.legend()
                ax.grid(True, alpha=0.25)
                _plot_save(
                    fig,
                    session_root
                    / f'{material}_{_format_number_for_path(conc)}gL_detector_response_by_wavelength.png',
                )

                fig, ax = plt.subplots(figsize=(8, 5))
                for angle in SUMMARY_ANGLES_DEG:
                    z = sorted(
                        [
                            r
                            for r in q
                            if abs(float(r['angle_deg']) - float(angle)) < 1e-9
                        ],
                        key=lambda r: r['wavelength_nm'],
                    )
                    if z:
                        ax.plot(
                            [r['wavelength_nm'] for r in z],
                            [r['absolute_score'] for r in z],
                            marker='o',
                            label=f'{angle:g}°',
                        )
                ax.set_xlabel('Wavelength (nm)')
                ax.set_ylabel('Absolute detector score')
                ax.set_title(
                    f'{material.capitalize()} {conc:g} g/L: selected angles vs wavelength'
                )
                ax.legend()
                ax.grid(True, alpha=0.25)
                _plot_save(
                    fig,
                    session_root
                    / f'{material}_{_format_number_for_path(conc)}gL_selected_angles_vs_wavelength.png',
                )

                z = sorted(
                    [r for r in q if abs(float(r['angle_deg']) - 170.0) < 1e-9],
                    key=lambda r: r['wavelength_nm'],
                )
                if z:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(
                        [r['wavelength_nm'] for r in z],
                        [r['absolute_score'] for r in z],
                        marker='o',
                    )
                    ax.set_xlabel('Wavelength (nm)')
                    ax.set_ylabel('H170 absolute score')
                    ax.set_title(
                        f'{material.capitalize()} {conc:g} g/L: H170 vs wavelength'
                    )
                    ax.grid(True, alpha=0.25)
                    _plot_save(
                        fig,
                        session_root
                        / f'{material}_{_format_number_for_path(conc)}gL_H170_vs_wavelength.png',
                    )

    # Detector response / selected angles by concentration for each material/wavelength.
    if len(concs) > 1:
        for material in materials:
            for wav in waves:
                q = [
                    r
                    for r in detector_rows
                    if r['material'] == material and float(r['wavelength_nm']) == wav
                ]
                if not q:
                    continue
                fig, ax = plt.subplots(figsize=(8, 5))
                for conc in concs:
                    z = sorted(
                        [r for r in q if float(r['concentration_g_L']) == conc],
                        key=lambda r: r['angle_deg'],
                    )
                    if z:
                        ax.plot(
                            [r['angle_deg'] for r in z],
                            [r['normalized_score'] for r in z],
                            marker='o',
                            label=f'{conc:g} g/L',
                        )
                ax.set_xlabel('Detector angle (°)')
                ax.set_ylabel('Normalized detector score')
                ax.set_title(
                    f'{material.capitalize()} {wav:g} nm: detector response by concentration'
                )
                ax.legend()
                ax.grid(True, alpha=0.25)
                _plot_save(
                    fig,
                    session_root
                    / f'{material}_{_format_number_for_path(wav)}nm_detector_response_by_concentration.png',
                )

                fig, ax = plt.subplots(figsize=(8, 5))
                for angle in SUMMARY_ANGLES_DEG:
                    z = sorted(
                        [
                            r
                            for r in q
                            if abs(float(r['angle_deg']) - float(angle)) < 1e-9
                        ],
                        key=lambda r: r['concentration_g_L'],
                    )
                    if z:
                        ax.plot(
                            [r['concentration_g_L'] for r in z],
                            [r['absolute_score'] for r in z],
                            marker='o',
                            label=f'{angle:g}°',
                        )
                ax.set_xlabel('Concentration (g/L)')
                ax.set_ylabel('Absolute detector score')
                ax.set_title(
                    f'{material.capitalize()} {wav:g} nm: selected angles vs concentration'
                )
                ax.legend()
                ax.grid(True, alpha=0.25)
                _plot_save(
                    fig,
                    session_root
                    / f'{material}_{_format_number_for_path(wav)}nm_selected_angles_vs_concentration.png',
                )


def main():
    policy = _resolved_failure_policy()
    g, d = _validate_user_configuration()  # host-side validation before CUDA is touched
    geometry_baseline = _matches_comparison_geometry(g)
    sediment_baseline = _matches_comparison_sediment(d)
    cp, gpu_name = _require_cuda_and_select_device()
    session_root = _session_root()
    print('CLARITAS standalone')
    print(f'  GPU: {gpu_name} [device {GPU_DEVICE}]')
    print(f'  Rays/case: {int(N_RAYS):,}')
    print(
        f"  Diagnostic policy: {policy['mode']}; failed-ray stop fraction: {policy['max_failed_ray_fraction']}"
    )
    light_cfg = _resolved_light_transport_config()
    if light_cfg['enabled']:
        print(
            f"  CUDA light-history diagnostics: first {min(int(N_RAYS),light_cfg['max_rays']):,} rays; grid {light_cfg['nx']}x{light_cfg['ny']}x{light_cfg['nz']}"
        )
    print(
        f"  Cell: ID {g['sample_cell_diameter_m']*1e3:.3f} mm; height {g['sample_cell_height_m']*1e3:.3f} mm; water {g['water_level_m']*1e3:.3f} mm"
    )
    print(
        f"  Sensor ring height: {g['sensor_ring_height_m']*1e3:.3f} mm above internal bottom"
    )
    if d['sediment_distribution'] == 'uniform':
        print('  Sediment distribution: uniform')
    else:
        print(
            f"  Sediment distribution: steady state ({d['sediment_transport_model']})"
        )
        print(
            f"  Eddy diffusivity D_t: {d['sediment_eddy_diffusivity_m2_s']:.6g} m^2/s"
        )
    print(
        f"  Prescribed vortex field: {'enabled' if d['enable_vortex'] else 'disabled'} ({d['free_surface_model']})"
    )
    vg = vortex_surface_geometry(
        d['free_surface_model'],
        g['water_level_m'] - g['sensor_ring_height_m'],
        g['sample_cell_radius_m'],
        d['vortex_wall_center_height_difference_m'],
        d['vortex_core_radius_m'],
        d['gravity_m_s2'],
    )
    if d['enable_vortex'] and d['free_surface_model'] == 'parabolic_vortex':
        print(
            f"  Parabolic rigid-body angular speed: {vg['angular_velocity_rad_s']:.6g} rad/s ({vg['equivalent_rigid_body_rpm']:.6g} rpm)"
        )
    elif d['enable_vortex'] and d['free_surface_model'] == 'localized_scully_vortex':
        print(
            f"  Localized-Scully core angular speed: {vg['angular_velocity_rad_s']:.6g} rad/s ({vg['equivalent_core_rpm']:.6g} rpm)"
        )
    print('  Configuration valid: True')
    print(f'  Supplied comparison geometry match: {geometry_baseline}')
    print(f'  Supplied comparison sediment match: {sediment_baseline}')
    print(
        f'  Supplied comparison configuration match: {bool(geometry_baseline and sediment_baseline)}'
    )
    if not geometry_baseline:
        print('  NOTE: Geometry differs from the supplied comparison apparatus.')
    if not sediment_baseline:
        print(
            '  NOTE: Sediment transport/vortex configuration differs from the supplied comparison configuration.'
        )
    print(
        '  NOTE: steady-state sediment transport is spatial, not a transient time-domain settling simulation.'
    )
    print(f'  Output session: {session_root}')

    if RUN_SWEEP:
        cases = list(itertools.product(MATERIALS, CONCENTRATIONS_G_L, WAVELENGTHS_NM))
    else:
        cases = [(MATERIAL, CONCENTRATION_G_L, WAVELENGTH_NM)]

    summaries = []
    detector_all = []
    sensor_all = []
    optical_all = []
    detailed_materials = set()
    for idx, (material, conc, wav) in enumerate(cases, start=1):
        seed = int(RANDOM_SEED if USE_SAME_SEED_FOR_SWEEP else RANDOM_SEED + idx - 1)
        detailed = (
            (not RUN_SWEEP)
            or bool(DETAILED_DIAGNOSTICS_PER_SWEEP_CASE)
            or (str(material).lower() not in detailed_materials)
        )
        print(
            f'[{idx}/{len(cases)}] {material} {float(conc):g} g/L {float(wav):g} nm; seed={seed}'
        )
        try:
            summary, detector_rows, sensor_rows, optical_rows = run_one_case(
                session_root,
                material,
                float(conc),
                float(wav),
                seed,
                gpu_name,
                detailed_sediment=detailed,
            )
        except Exception as exc:
            # Report and immediately re-raise, never hide programming or CUDA errors.
            import traceback

            failure = {
                'case_index': idx,
                'material': material,
                'concentration_g_L': float(conc),
                'wavelength_nm': float(wav),
                'seed': seed,
                'exception_type': type(exc).__name__,
                'message': str(exc),
                'traceback': traceback.format_exc(),
                'completed_cases': len(summaries),
            }
            _write_json(session_root / 'case_failure_report.json', failure)
            _write_json(
                _case_output_dir(session_root, material, float(wav), float(conc))
                / 'case_failure_report.json',
                failure,
            )
            _write_json(
                session_root / 'sweep_progress.json',
                {'completed_cases': summaries, 'fatal_case': failure},
            )
            raise
        summaries.append(summary)
        detector_all.extend(detector_rows)
        sensor_all.extend(sensor_rows)
        optical_all.extend(optical_rows)
        _write_json(
            session_root / 'sweep_progress.json',
            {'completed_cases': summaries, 'fatal_case': None},
        )
        if detailed:
            detailed_materials.add(str(material).lower())

    if len(cases) > 1:
        _aggregate_sweep_plots(
            session_root, summaries, detector_all, sensor_all, optical_all
        )
    print('CLARITAS standalone complete.')
    print(f'Results: {session_root}')


if __name__ == '__main__':
    main()
