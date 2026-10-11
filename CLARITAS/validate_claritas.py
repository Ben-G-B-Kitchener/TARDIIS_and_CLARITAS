"""CLARITAS host/static checks and optional CUDA integration checks.

Run with --cuda-full on an NVIDIA/CuPy system to include device validation.
A skipped check is not a GPU pass.
"""

from __future__ import annotations
import argparse
import ast
import inspect
import json
import math
import re
import sys
import tempfile
import traceback
import warnings
from pathlib import Path
import numpy as np
import CLARITAS as c


class Gate:

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def check(self, name, fn):
        try:
            fn()
        except Exception as exc:
            self.failed += 1
            print(f'FAIL  {name}: {type(exc).__name__}: {exc}')
            if ARGS.verbose:
                traceback.print_exc()
        else:
            self.passed += 1
            print(f'PASS  {name}')

    def skip(self, name, reason):
        self.skipped += 1
        print(f'SKIP  {name}: {reason}')


def close(actual, expected, rtol=1e-11, atol=1e-13):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)


def test_source_identity():
    assert c.SAMPLE_CELL_DIAMETER_M == 0.098
    assert c.MERIDIONAL_CIRCULATION_RATIO == 0.5
    assert c.PARTICLE_K_IMAG_OVERRIDE == 0.0
    digest = c._current_source_sha256()
    assert len(digest) == 64 and all((q in '0123456789abcdef' for q in digest))


def test_parabolic_geometry():
    zmean, R, D, g = (0.049, 0.0465, 0.05, 9.81)
    v = c.vortex_surface_geometry('parabolic_vortex', zmean, R, D, -1.0, g)
    close(v['center_z'], zmean - D / 2)
    close(v['wall_z'], zmean + D / 2)
    close(v['angular_velocity_rad_s'] ** 2, 2 * g * D / R**2)
    close(v['vtheta_peak_m_s'], v['angular_velocity_rad_s'] * R)
    close(v['equivalent_rigid_body_rpm'], 203.40092931205768)
    r = (np.arange(200000, dtype=np.float64) + 0.5) * R / 200000
    z = zmean + D * (r * r / R**2 - 0.5)
    dzdr = 2 * D * r / R**2
    close(dzdr[[0, -1]], [D / (200000 * R), 2 * D * (1 - 0.5 / 200000) / R])
    close(v['angular_velocity_rad_s'] * r, np.sqrt(2 * g * D / R**2) * r)
    close(v['angular_velocity_rad_s'] ** 2 * r, 2 * g * D / R**2 * r)
    assert np.all(np.isfinite(z)) and np.all(np.isfinite(dzdr))
    mean = 2 / R**2 * np.sum(r * z) * (R / 200000)
    close(mean, zmean, rtol=0.0, atol=2e-12)
    flat = c.vortex_surface_geometry('parabolic_vortex', zmean, R, 0.0, -1.0, g)
    close(
        [flat['center_z'], flat['wall_z'], flat['angular_velocity_rad_s']],
        [zmean, zmean, 0.0],
    )


def test_parabolic_stokes_analytic():
    d = np.array([1e-06, 1e-05, 0.0001])
    R, D, g, mu, rho_p, Dt = (0.0465, 0.05, 9.81, 0.00089, 2600.0, 0.001)
    q = c.sediment_transport_coefficients_host(
        d,
        0.049,
        -0.093,
        R,
        D,
        -1.0,
        Dt,
        particle_density_kg_m3=rho_p,
        water_dynamic_viscosity_pa_s=mu,
        gravity_m_s2=g,
        quadrature_points=2048,
        radial_lookup_points=257,
        transport_model='stokes_drift_diffusion',
        free_surface_model='parabolic_vortex',
    )
    tau = rho_p * d * d / (18 * mu)
    omega2 = 2 * g * D / R**2
    A = tau * omega2 / (2 * Dt)
    close(q['radial_log_coefficient'], A)
    close(q['radial_center_log_factor'], -A * R**2)
    close(q['radial_log_lookup'][:, -1], 0.0, atol=0.0)
    assert np.all(np.diff(q['radial_log_lookup'], axis=1) >= -1e-14)


def _independent_volume_mean(q, model, transport, n=32768):
    R, zmean, zmin, D, a = (0.0465, 0.049, -0.093, 0.05, 0.02)
    r = (np.arange(n, dtype=np.float64) + 0.5) * R / n
    if model == 'parabolic_vortex':
        zs = zmean + D * (r * r / R**2 - 0.5)
    else:
        vg = c.vortex_surface_geometry(model, zmean, R, D, a)
        zs = zmean - vg['h_bar'] + vg['h_inf'] * r * r / (a * a + r * r)
    H = zs - zmin
    if transport == 'stokes_drift_diffusion':
        A = q['radial_log_coefficient'][:, None]
        if model == 'parabolic_vortex':
            log_r = A * (r[None, :] ** 2 - R**2)
        else:
            f = r * r / (a * a + r * r)
            fR = R * R / (a * a + R * R)
            log_r = A * (f[None, :] - fR)
    else:
        rn = q['radial_lookup_r_m']
        log_r = np.vstack([np.interp(r, rn, row) for row in q['radial_log_lookup']])
    B = q['vertical_inverse_length_m'][:, None]
    iz = np.where(np.abs(B) < 1e-14, H[None, :], -np.expm1(-B * H[None, :]) / B)
    avg = (
        2
        / (R * R * (zmean - zmin))
        * np.sum(r[None, :] * np.exp(log_r) * iz, axis=1)
        * (R / n)
    )
    return avg / q['normalization_scaled']


def test_parabolic_schiller_and_normalization():
    d = np.array([1e-06, 1e-05, 0.0001])
    for transport in ('stokes_drift_diffusion', 'schiller_naumann_drift_diffusion'):
        q = c.sediment_transport_coefficients_host(
            d,
            0.049,
            -0.093,
            0.0465,
            0.05,
            -1.0,
            0.001,
            quadrature_points=4096,
            radial_lookup_points=513,
            transport_model=transport,
            free_surface_model='parabolic_vortex',
        )
        assert np.all(np.isfinite(q['normalization_scaled']))
        assert np.all(q['normalization_scaled'] > 0)
        assert np.all(q['radial_log_lookup'] <= 1e-14)
        assert np.all(q['max_density_scale'] >= 1.0)
        close(
            _independent_volume_mean(q, 'parabolic_vortex', transport),
            1.0,
            rtol=0.0003,
            atol=3e-06,
        )
        for r, z in (
            (0.0, -0.093),
            (0.95 * 0.0465, -0.093),
            (0.0, 0.0),
            (0.95 * 0.0465, 0.0),
        ):
            log_r = np.array(
                [
                    np.interp(r, q['radial_lookup_r_m'], row)
                    for row in q['radial_log_lookup']
                ]
            )
            scales = (
                np.exp(log_r - q['vertical_inverse_length_m'] * (z + 0.093))
                / q['normalization_scaled']
            )
            assert np.all(np.isfinite(scales)) and np.all(scales >= 0.0)
            assert np.all(scales <= q['max_density_scale'] * (1 + 2e-13))
    q = c.sediment_transport_coefficients_host(
        d,
        0.049,
        -0.093,
        0.0465,
        0.05,
        -1.0,
        0.001,
        transport_model='schiller_naumann_drift_diffusion',
        free_surface_model='parabolic_vortex',
    )
    w = q['radial_max_slip_velocity_m_s']
    re = 997.0 * d * w / 0.00089
    tau = 2600.0 * d * d / (18 * 0.00089)
    drive = tau * (2 * 9.81 * 0.05 / 0.0465**2) * 0.0465
    close(w * (1 + 0.15 * re**0.687), drive, rtol=3e-12, atol=1e-15)


def test_scully_reference():
    d = np.array([1e-06, 1e-05, 0.0001])
    vg = c.vortex_surface_geometry('localized_scully_vortex', 0.049, 0.0465, 0.05, 0.02)
    close(
        [
            vg['center_z'],
            vg['wall_z'],
            vg['angular_velocity_rad_s'],
            vg['equivalent_core_rpm'],
        ],
        [
            0.010106395724732792,
            0.060106395724732795,
            53.90912787903789,
            514.7942507833191,
        ],
    )
    qs = c.sediment_transport_coefficients_host(
        d,
        0.049,
        -0.093,
        0.0465,
        0.05,
        0.02,
        0.001,
        quadrature_points=512,
        radial_lookup_points=129,
        transport_model='stokes_drift_diffusion',
    )
    close(
        qs['radial_log_coefficient'],
        [9.43333905e-05, 0.00943333905, 0.943333905],
        rtol=5e-09,
    )
    close(
        qs['radial_center_log_factor'],
        [-7.96067416e-05, -0.00796067416, -0.796067416],
        rtol=5e-09,
    )
    q = c.sediment_transport_coefficients_host(
        d,
        0.049,
        -0.093,
        0.0465,
        0.05,
        0.02,
        0.001,
        quadrature_points=512,
        radial_lookup_points=129,
        transport_model='schiller_naumann_drift_diffusion',
    )
    close(
        q['settling_velocity_m_s'],
        [9.81600492e-07, 9.8024982e-05, 0.00856690624],
        rtol=5e-09,
    )
    close(
        q['radial_log_coefficient'],
        [9.43249639e-05, 0.0094106463, 0.767242433],
        rtol=5e-09,
    )
    close(q['normalization_scaled'], [0.99991352, 0.9914242, 0.50525183], rtol=2e-08)
    close(
        q['radial_center_log_factor'],
        [-7.95996304e-05, -0.00794152404, -0.647466075],
        rtol=5e-09,
    )
    close(
        q['radial_log_lookup'][2, [0, 32, 64, 96, 128]],
        [
            -0.6474660749782518,
            -0.4573880863269777,
            -0.21780269501287686,
            -0.07669369764286141,
            0.0,
        ],
        rtol=5e-12,
    )
    logw = np.array(
        [
            np.interp(0.95 * 0.0465, q['radial_lookup_r_m'], row)
            for row in q['radial_log_lookup']
        ]
    )
    wall_bottom = np.exp(logw) / q['normalization_scaled']
    close(
        wall_bottom,
        [1.0000851691924935, 1.008516884202098, 1.9556481703857034],
        rtol=5e-12,
    )
    close(q['omega_rad_s'], 53.90912787903789)
    r = np.array([0.0, 0.01, 0.03, 0.0465])
    z = np.array([-0.05, 0.0, 0.02, 0.05])
    ur, uz, _ = c.meridional_velocity_host(
        r, z, 0.049, -0.093, 0.0465, 0.05, 0.02, 0.05
    )
    close(ur, [0.0, -0.00263629, -0.00217236, 0.0], rtol=2e-06, atol=2e-09)
    close(uz, [-0.02621261, -0.01772844, 0.00121733, 0.0], rtol=2e-06, atol=2e-09)


def test_parabolic_meridional_host():
    R, zmean, zmin, D, chi = (0.0465, 0.049, -0.093, 0.05, 0.05)
    r = np.linspace(0.0, R, 64)
    bottom = np.full_like(r, zmin)
    surface = zmean + D * (r * r / R**2 - 0.5)
    ur_b, uz_b, meta = c.meridional_velocity_host(
        r, bottom, zmean, zmin, R, D, -1.0, chi, free_surface_model='parabolic_vortex'
    )
    ur_t, uz_t, _ = c.meridional_velocity_host(
        r, surface, zmean, zmin, R, D, -1.0, chi, free_surface_model='parabolic_vortex'
    )
    close(uz_b, 0.0, atol=2e-15)
    close(ur_b[[0, -1]], 0.0, atol=2e-15)
    close(uz_b[-1], 0.0, atol=2e-15)
    hp = 2 * D * r / R**2
    close(uz_t - hp * ur_t, 0.0, atol=2e-13)
    ur0, uz0, _ = c.meridional_velocity_host(
        r,
        0.5 * (bottom + surface),
        zmean,
        zmin,
        R,
        D,
        -1.0,
        0.0,
        free_surface_model='parabolic_vortex',
    )
    close(ur0, 0.0, atol=0.0)
    close(uz0, 0.0, atol=0.0)
    close(meta['vtheta_peak_m_s'], math.sqrt(2 * 9.81 * D / R**2) * R)
    rr, zz, h = (0.021, 0.0, 1e-06)

    def uv(rv, zv):
        return c.meridional_velocity_host(
            np.array([rv]),
            np.array([zv]),
            zmean,
            zmin,
            R,
            D,
            -1.0,
            chi,
            free_surface_model='parabolic_vortex',
        )[:2]

    urp = uv(rr + h, zz)[0][0]
    urm = uv(rr - h, zz)[0][0]
    ur0 = uv(rr, zz)[0][0]
    uzp = uv(rr, zz + h)[1][0]
    uzm = uv(rr, zz - h)[1][0]
    div = (urp - urm) / (2 * h) + ur0 / rr + (uzp - uzm) / (2 * h)
    assert abs(div) < 2e-07, div


def test_configuration_matrix_and_outputs():
    names = (
        'SEDIMENT_DISTRIBUTION',
        'SEDIMENT_TRANSPORT_MODEL',
        'ENABLE_VORTEX',
        'VORTEX_MODEL',
        'VORTEX_WALL_CENTER_HEIGHT_DIFFERENCE_M',
        'VORTEX_CORE_RADIUS_M',
    )
    old = {k: getattr(c, k) for k in names}
    try:
        cases = [
            (
                'uniform',
                'schiller_naumann_drift_diffusion',
                False,
                'localized_scully_vortex',
            ),
            (
                'uniform',
                'schiller_naumann_drift_diffusion',
                True,
                'localized_scully_vortex',
            ),
            ('uniform', 'schiller_naumann_drift_diffusion', True, 'parabolic_vortex'),
            (
                'steady_state',
                'stokes_drift_diffusion',
                False,
                'localized_scully_vortex',
            ),
            ('steady_state', 'stokes_drift_diffusion', True, 'parabolic_vortex'),
            (
                'steady_state',
                'schiller_naumann_drift_diffusion',
                True,
                'parabolic_vortex',
            ),
            (
                'steady_state',
                'schiller_naumann_meridional_advection_diffusion',
                True,
                'parabolic_vortex',
            ),
            (
                'steady_state',
                'schiller_naumann_meridional_advection_diffusion',
                True,
                'localized_scully_vortex',
            ),
        ]
        for dist, transport, enabled, vortex in cases:
            c.SEDIMENT_DISTRIBUTION = dist
            c.SEDIMENT_TRANSPORT_MODEL = transport
            c.ENABLE_VORTEX = enabled
            c.VORTEX_MODEL = vortex
            c.VORTEX_WALL_CENTER_HEIGHT_DIFFERENCE_M = 0.05
            c.VORTEX_CORE_RADIUS_M = 0.02
            g, d = c._validate_user_configuration()
            assert d['free_surface_model'] == (vortex if enabled else 'flat')
        c.SEDIMENT_DISTRIBUTION = 'steady_state'
        c.SEDIMENT_TRANSPORT_MODEL = 'schiller_naumann_drift_diffusion'
        c.ENABLE_VORTEX = True
        c.VORTEX_MODEL = 'parabolic_vortex'
        c.VORTEX_CORE_RADIUS_M = -1.0
        g, d = c._validate_user_configuration()
        rows = c._vortex_profile_rows(g, d)
        assert rows and rows[0]['vortex_model'] == 'parabolic_vortex'
        assert {
            'angular_velocity_rad_s',
            'equivalent_rpm',
            'tangential_velocity_m_s',
        } <= set(rows[0])
        assert not c._matches_comparison_sediment(d)
    finally:
        for k, v in old.items():
            setattr(c, k, v)


def test_fail_closed_configuration():
    names = (
        'ENABLE_VORTEX',
        'VORTEX_MODEL',
        'VORTEX_WALL_CENTER_HEIGHT_DIFFERENCE_M',
        'SEDIMENT_EDDY_DIFFUSIVITY_M2_S',
        'WATER_DENSITY_KG_M3',
        'WATER_DYNAMIC_VISCOSITY_PA_S',
        'GRAVITY_M_S2',
        'MERIDIONAL_CIRCULATION_RATIO',
        'SAVE_LIGHT_TRANSPORT_VISUALISATIONS',
        'SAVE_LIGHT_TRANSPORT_DATA',
        'LIGHT_TRANSPORT_DIAGNOSTIC_MAX_RAYS',
        'LIGHT_TRANSPORT_GRID_NX',
        'LIGHT_TRANSPORT_REPRESENTATIVE_RAYS',
        'LIGHT_TRANSPORT_MAX_SEGMENTS_PER_RAY',
        'LIGHT_TRANSPORT_VERTICAL_SLICE_THICKNESS_M',
        'LIGHT_TRANSPORT_LOG_FLOOR',
        'LIGHT_TRANSPORT_DISPLAY_LOW_PERCENTILE',
        'LIGHT_TRANSPORT_DISPLAY_HIGH_PERCENTILE',
        'LIGHT_TRANSPORT_3D_MIN_DISPLAY_PERCENTILE',
        'LIGHT_TRANSPORT_3D_MAX_DISPLAY_VOXELS',
        'LIGHT_TRANSPORT_3D_ALPHA_MIN',
        'LIGHT_TRANSPORT_3D_ALPHA_MAX',
        'LIGHT_TRANSPORT_3D_MARKER_SIZE_MIN',
        'LIGHT_TRANSPORT_3D_MARKER_SIZE_MAX',
        'LIGHT_TRANSPORT_3D_SURFACE_MARGIN_M',
    )
    old = {k: getattr(c, k) for k in names}
    try:
        for model, depth in (
            ('unknown', 0.05),
            ('parabolic_vortex', -0.01),
            ('parabolic_vortex', float('nan')),
        ):
            c.ENABLE_VORTEX = True
            c.VORTEX_MODEL = model
            c.VORTEX_WALL_CENTER_HEIGHT_DIFFERENCE_M = depth
            try:
                c._validate_user_configuration()
            except ValueError:
                pass
            else:
                raise AssertionError(
                    f'invalid configuration accepted: {model}, {depth}'
                )
        c.VORTEX_MODEL = 'parabolic_vortex'
        c.VORTEX_WALL_CENTER_HEIGHT_DIFFERENCE_M = 0.05
        for key, bad in (
            ('SEDIMENT_EDDY_DIFFUSIVITY_M2_S', 0.0),
            ('WATER_DENSITY_KG_M3', 2700.0),
            ('WATER_DYNAMIC_VISCOSITY_PA_S', 0.0),
            ('GRAVITY_M_S2', float('inf')),
            ('MERIDIONAL_CIRCULATION_RATIO', 0.6),
        ):
            prior = getattr(c, key)
            setattr(c, key, bad)
            try:
                c._validate_user_configuration()
            except ValueError:
                pass
            else:
                raise AssertionError(f'invalid {key} accepted: {bad}')
            finally:
                setattr(c, key, prior)
        for radius in (0.0, float('nan')):
            try:
                c.vortex_surface_geometry('parabolic_vortex', 0.05, radius, 0.05, -1.0)
            except ValueError:
                pass
            else:
                raise AssertionError(f'invalid radius accepted: {radius}')
        try:
            c.vortex_surface_geometry('parabolic_vortex', 0.05, 1e-300, 1e300, -1.0)
        except (ValueError, OverflowError):
            pass
        else:
            raise AssertionError('non-finite derived angular velocity was accepted')
        c.SAVE_LIGHT_TRANSPORT_VISUALISATIONS = True
        c.SAVE_LIGHT_TRANSPORT_DATA = True
        for key, bad in (
            ('LIGHT_TRANSPORT_DIAGNOSTIC_MAX_RAYS', 0),
            ('LIGHT_TRANSPORT_GRID_NX', 3),
            ('LIGHT_TRANSPORT_GRID_NX', 1000000),
            (
                'LIGHT_TRANSPORT_REPRESENTATIVE_RAYS',
                c.LIGHT_TRANSPORT_DIAGNOSTIC_MAX_RAYS + 1,
            ),
            ('LIGHT_TRANSPORT_MAX_SEGMENTS_PER_RAY', 0),
            ('LIGHT_TRANSPORT_VERTICAL_SLICE_THICKNESS_M', 0.0),
            ('LIGHT_TRANSPORT_LOG_FLOOR', float('nan')),
            ('LIGHT_TRANSPORT_DISPLAY_LOW_PERCENTILE', 100.0),
            ('LIGHT_TRANSPORT_DISPLAY_HIGH_PERCENTILE', 101.0),
            ('LIGHT_TRANSPORT_3D_MIN_DISPLAY_PERCENTILE', -1.0),
            ('LIGHT_TRANSPORT_3D_MAX_DISPLAY_VOXELS', 0),
            ('LIGHT_TRANSPORT_3D_ALPHA_MIN', -0.1),
            ('LIGHT_TRANSPORT_3D_ALPHA_MAX', 1.1),
            ('LIGHT_TRANSPORT_3D_MARKER_SIZE_MIN', 0.0),
            ('LIGHT_TRANSPORT_3D_MARKER_SIZE_MAX', 1001.0),
            ('LIGHT_TRANSPORT_3D_SURFACE_MARGIN_M', -0.001),
        ):
            prior = getattr(c, key)
            setattr(c, key, bad)
            try:
                c._validate_user_configuration()
            except ValueError:
                pass
            else:
                raise AssertionError(f'invalid light diagnostic {key} accepted: {bad}')
            finally:
                setattr(c, key, prior)
    finally:
        for k, v in old.items():
            setattr(c, k, v)


def test_cuda_source_static():
    src = c.CUDA_SRC
    assert 'FREE_SURFACE_MODEL==1' in src
    assert 'omega2*r' in src
    assert 'meridional_surface_geometry_cuda' in src
    assert 'if(VORTEX_DELTA_H>0.0 && FREE_SURFACE_MODEL!=2)' not in src
    assert 'sediment_local_scale(TRANSPORT_MODEL,FREE_SURFACE_MODEL' in src
    assert 'light_diag_record_segment' in src and 'light_diag_record_event' in src
    assert 'LIGHT_DIAG_ENABLED' in src
    assert 'boundary_adjustment_count' in src and 'event_outside_count' in src
    assert 'u>=1.0f-completion_tol' in src
    assert (
        'rnd_uniform'
        not in src[
            src.index('__device__ void light_diag_record_segment') : src.index(
                '__global__ void build_theta_grid_kernel'
            )
        ]
    )


def test_light_transport_host_products():
    nz, ny, nx = (12, 10, 8)
    shape = (nz, ny, nx)
    total = np.zeros(shape, dtype=np.float32)
    total[2:10, 4:6, 3:5] = 0.01
    post = 0.25 * total
    events = np.zeros(shape, dtype=np.uint32)
    events[6, 5, 4] = 3
    nrep, nseg = (2, 4)
    traj = lambda: np.zeros((nrep, nseg), dtype=np.float32)
    diag = {
        'track_length_m': total,
        'post_scatter_track_length_m': post,
        'scatter_event_weight': events.astype(np.float32),
        'scatter_event_count': events,
        'nx': nx,
        'ny': ny,
        'nz': nz,
        'x_min_m': -0.0465,
        'x_max_m': 0.0465,
        'y_min_m': -0.0465,
        'y_max_m': 0.0465,
        'z_min_m': -0.093,
        'z_max_m': 0.074,
        'diagnostic_ray_limit': 32,
        'diagnostic_rays_processed': 32,
        'representative_ray_limit': nrep,
        'max_segments_per_ray': nseg,
        'segment_count': 2,
        'scatter_event_total': 3,
        'deposited_scatter_event_total': 3,
        'scatter_event_closure_error': 0,
        'error_count': 0,
        'hard_error_count': 0,
        'boundary_adjustment_count': 1214,
        'event_outside_count': 0,
        'nonfinite_count': 0,
        'negative_value_count': 0,
        'diagnostic_memory_bytes': int(total.nbytes * 4),
        'diagnostic_active_wall_time_s': 0.01,
        'segment_length_total_m': float(total.sum()),
        'deposited_length_total_m': float(total.sum()),
        'deposition_closure_relative_error': 5.594129053221538e-10,
        'grid_closure_relative_error': 0.0,
        'trajectory_count': np.array([1, 1]),
        'trajectory_overflow': np.zeros(nrep, dtype=int),
        'trajectory_x0_m': traj(),
        'trajectory_y0_m': traj(),
        'trajectory_z0_m': traj(),
        'trajectory_x1_m': traj(),
        'trajectory_y1_m': traj(),
        'trajectory_z1_m': traj(),
        'trajectory_particle_order': np.zeros((nrep, nseg), dtype=int),
        'trajectory_event_code': np.zeros((nrep, nseg), dtype=int),
        'selection_method': 'validation',
        'ray_statistical_weight': 1.0,
    }
    diag['trajectory_y0_m'][:, 0] = -0.04
    diag['trajectory_y1_m'][:, 0] = 0.04
    cfg = {
        'save_data': True,
        'save_visualisations': True,
        'vertical_slice_thickness_m': 0.02,
        'sensor_slice_thickness_m': 0.03,
        'log_floor': 1e-12,
        'display_low_percentile': 2.0,
        'display_high_percentile': 99.5,
        'display_3d_min_percentile': 5.0,
        'display_3d_max_voxels': 100000,
        'display_3d_alpha_min': 0.06,
        'display_3d_alpha_max': 0.5,
        'display_3d_marker_size_min': 2.0,
        'display_3d_marker_size_max': 12.0,
        'display_3d_surface_margin_m': 0.005,
    }
    g = c._resolved_geometry()
    d = c._resolved_sediment_config()
    raw_before = {
        k: np.array(diag[k], copy=True)
        for k in (
            'track_length_m',
            'post_scatter_track_length_m',
            'scatter_event_weight',
            'scatter_event_count',
        )
    }
    with tempfile.TemporaryDirectory(prefix='claritas_light_') as td:
        out = Path(td) / 'rendered'
        out.mkdir()
        captured_callouts = {}
        original_plot_save = c._plot_save_light

        def inspect_plot_save(fig, path):
            from mpl_toolkits.mplot3d import proj3d

            fig.canvas.draw()
            ax = fig.axes[0]
            notes = [
                a
                for a in ax.texts
                if str(a.get_gid() or '').startswith('source-entry-')
            ]
            if path.name == 'light_transport_3d_isometric.png' and (not notes):
                original_plot_save(fig, path)
                return
            assert len(notes) == 1, path
            note = notes[0]
            position = note.get_position()
            assert not (0.0 <= position[0] <= 1.0 and 0.0 <= position[1] <= 1.0), path
            box = note.get_window_extent(fig.canvas.get_renderer())
            assert (
                box.x0 >= -1.0
                and box.y0 >= -1.0
                and (box.x1 <= fig.bbox.width + 1.0)
                and (box.y1 <= fig.bbox.height + 1.0)
            ), path
            if path.name == 'light_transport_vertical_cross_section.png':
                close(
                    note.xy,
                    [
                        -g['sample_cell_radius_m'] * 1000.0,
                        g['sensor_ring_height_m'] * 1000.0,
                    ],
                    rtol=0,
                    atol=1e-10,
                )
            elif path.name == 'light_transport_sensor_plane_cross_section.png':
                close(
                    note.xy,
                    [0.0, -g['sample_cell_radius_m'] * 1000.0],
                    rtol=0,
                    atol=1e-10,
                )
            else:
                px, py, _ = proj3d.proj_transform(
                    0.0,
                    -g['sample_cell_radius_m'] * 1000.0,
                    g['sensor_ring_height_m'] * 1000.0,
                    ax.get_proj(),
                )
                close(note.xy, [px, py], rtol=0, atol=1e-10)
            if ax.images:
                im = ax.images[0]
                low = np.mean(im.get_cmap()(im.norm(im.norm.vmin))[:3])
                high = np.mean(im.get_cmap()(im.norm(im.norm.vmax))[:3])
                assert low < high, path
                assert fig.axes[-1].get_ylim()[0] < fig.axes[-1].get_ylim()[1], path
            captured_callouts[path.name] = position
            original_plot_save(fig, path)

        c._plot_save_light = inspect_plot_save
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                summary = c._write_light_transport_outputs(out, diag, g, d, cfg)
        finally:
            c._plot_save_light = original_plot_save
        required = (
            'light_transport_vertical_cross_section.png',
            'light_transport_sensor_plane_cross_section.png',
            'light_transport_3d_isometric.png',
            'light_transport_vertical_cross_section.csv',
            'light_transport_sensor_plane_cross_section.csv',
            'light_transport_volume.npz',
            'light_transport_representative_trajectories.csv',
            'light_transport_summary.json',
        )
        assert all(
            (
                (out / name).is_file() and (out / name).stat().st_size > 0
                for name in required
            )
        )
        assert (
            summary['diagnostic_rays_processed'] == 32
            and summary['diagnostic_error_count'] == 0
        )
        assert (
            summary['boundary_adjustment_count'] == 1214
            and summary['diagnostic_hard_error_count'] == 0
        )
        assert (
            summary['detector_geometry_drawn'] is False
            and summary['scatter_event_contours_drawn'] is False
        )
        assert summary['display_colour_scheme'].startswith('strict greyscale')
        assert summary['display_density_to_brightness_direction'].startswith(
            'increasing density'
        )
        assert (
            summary['source_callouts_outside_data_area'] is True
            and len(captured_callouts) == 2
        )
        assert (
            summary['vertical_display_log10_vmin']
            == summary['sensor_plane_display_log10_vmin']
            == summary['display_log10_vmin']
        )
        assert (
            summary['vertical_display_log10_vmax']
            == summary['sensor_plane_display_log10_vmax']
            == summary['display_log10_vmax']
        )
        assert (
            np.isfinite(summary['display_log10_vmin'])
            and summary['display_log10_vmin'] < summary['display_log10_vmax']
        )
        expected_zmax = min(
            g['sample_cell_height_m'],
            summary['maximum_water_surface_height_above_bottom_m']
            + cfg['display_3d_surface_margin_m'],
        )
        close(summary['isometric_plot_z_max_m'], expected_zmax, rtol=0, atol=1e-12)
        assert (
            summary['isometric_plot_z_max_m'] < g['sample_cell_height_m']
            and summary['isometric_cloud_displayed_voxels'] > 0
        )
        q = c._light_transport_cross_sections(diag, cfg)
        state = c._light_transport_display_state(diag, q, g, d, cfg)
        state_again = c._light_transport_display_state(diag, q, g, d, cfg)
        assert np.array_equal(state['cloud_indices'], state_again['cloud_indices'])
        assert summary['isometric_cloud_displayed_voxels'] == min(
            summary['isometric_eligible_voxel_count'],
            summary['isometric_max_display_voxels'],
        )
        assert summary['isometric_max_display_voxels'] == cfg['display_3d_max_voxels']
        assert (
            summary['isometric_requested_min_display_percentile']
            == cfg['display_3d_min_percentile']
        )
        assert (
            summary['isometric_alpha_min'] == cfg['display_3d_alpha_min']
            and summary['isometric_alpha_max'] == cfg['display_3d_alpha_max']
        )
        assert (
            summary['isometric_marker_size_min'] == cfg['display_3d_marker_size_min']
            and summary['isometric_marker_size_max']
            == cfg['display_3d_marker_size_max']
        )
        assert not state['horizontal_water_mask'][0, 0]
        assert bool(
            c._masked_light_log_field(
                q['hd'], state['horizontal_water_mask'], cfg
            ).mask[0, 0]
        )
        for name in (
            'light_transport_vertical_cross_section.png',
            'light_transport_sensor_plane_cross_section.png',
            'light_transport_3d_isometric.png',
        ):
            rgba = c._get_pyplot().imread(out / name)
            rgb = np.asarray(rgba[..., :3], dtype=float)
            assert float(np.max(np.ptp(rgb, axis=2))) <= 1.0 / 255.0 + 1e-07, name
        for key, before in raw_before.items():
            assert np.array_equal(np.asarray(diag[key]), before), key
        with np.load(out / 'light_transport_volume.npz') as vol:
            close(vol['total_track_length_weighted_m'], total, rtol=0, atol=0)
        raw_only = Path(td) / 'raw_only'
        raw_only.mkdir()
        c._write_light_transport_outputs(
            raw_only, diag, g, d, dict(cfg, save_visualisations=False)
        )
        for name in (
            'light_transport_vertical_cross_section.csv',
            'light_transport_sensor_plane_cross_section.csv',
            'light_transport_representative_trajectories.csv',
        ):
            assert (out / name).read_bytes() == (raw_only / name).read_bytes(), name
        with np.load(out / 'light_transport_volume.npz') as a, np.load(
            raw_only / 'light_transport_volume.npz'
        ) as b:
            assert a.files == b.files
            for key in a.files:
                assert np.array_equal(a[key], b[key]), key
        bad = dict(diag)
        bad.update(error_count=1, hard_error_count=1, boundary_adjustment_count=0)
        report = c._write_light_transport_outputs(
            out, bad, g, d, dict(cfg, save_data=False, save_visualisations=False)
        )
        assert report['diagnostic_status'] == 'limited' and (not report['fatal'])
        original_policy = c.DIAGNOSTIC_POLICY
        try:
            c.DIAGNOSTIC_POLICY = 'strict'
            strict = c._write_light_transport_outputs(
                out, bad, g, d, dict(cfg, save_data=False, save_visualisations=False)
            )
            assert strict['fatal']
        finally:
            c.DIAGNOSTIC_POLICY = original_policy
        bad = dict(diag)
        bad.update(
            event_outside_count=1, scatter_event_closure_error=-1, scatter_event_total=4
        )
        report = c._write_light_transport_outputs(
            out, bad, g, d, dict(cfg, save_data=False, save_visualisations=False)
        )
        assert (
            report['event_accounting_reconciled']
            and report['deposited_only_event_discrepancy'] == -1
        )


def test_run_case_light_config_binding():
    src = inspect.getsource(c.run_one_case)
    assignment = src.index('light_cfg = _resolved_light_transport_config()')
    first_use = min(
        src.index("'light_transport_diagnostics': light_cfg"),
        src.index('light_transport_config=light_cfg'),
    )
    assert assignment < first_use


def test_warning_free_diagnostic_span_arithmetic():
    src = ''.join(inspect.getsource(c.ClaritasForwardModel.simulate).split())
    assert (
        "particle_event_max_r_m'].astype(np.float64)-arrays['particle_event_min_r_m'].astype(np.float64)"
        in src
    )
    assert (
        "particle_event_max_z_m'].astype(np.float64)-arrays['particle_event_min_z_m'].astype(np.float64)"
        in src
    )
    hi = np.asarray([-2e38], dtype=np.float32)
    lo = np.asarray([2e38], dtype=np.float32)
    with np.errstate(over='raise', invalid='raise'):
        span = hi.astype(np.float64) - lo.astype(np.float64)
    assert np.isfinite(span[0]) and span[0] < 0


def test_light_display_scaling_fallbacks():
    cfg = c._resolved_light_transport_config()
    mask = np.ones((2, 3), dtype=bool)
    cases = (
        (np.zeros((2, 3)), 'empty_fallback'),
        (np.ones((2, 3)) * 4.0, 'constant_fallback'),
        (np.asarray([[0, 0, 0], [0, 2, 0]], dtype=float), 'constant_fallback'),
        (np.asarray([[1, 2, 4], [8, 16, 32]], dtype=float), 'percentile'),
    )
    for values, expected_mode in cases:
        with np.errstate(all='raise'):
            vmin, vmax, mode, count = c._robust_light_log_scale(values, mask, cfg)
        assert np.isfinite(vmin) and np.isfinite(vmax) and (vmin < vmax)
        assert mode == expected_mode and count == int(np.count_nonzero(values > 0))


def test_light_display_semantics_and_selection():
    plt = c._get_pyplot()
    cmap = plt.get_cmap('gray').with_extremes(
        bad=(1.0, 1.0, 1.0, 1.0), under=(0.0, 0.0, 0.0, 1.0)
    )
    low = float(np.mean(cmap(0.0)[:3]))
    high = float(np.mean(cmap(1.0)[:3]))
    assert low < high and low <= 0.01 and (high >= 0.99)
    zz, yy, xx = np.indices((20, 18, 16))
    candidates = np.column_stack((zz.ravel(), yy.ravel(), xx.ravel()))
    first = c._deterministic_spatial_cloud_selection(candidates, 500)
    second = c._deterministic_spatial_cloud_selection(candidates, 500)
    assert len(first) == 500 and np.array_equal(first, second)
    assert all(
        (np.ptp(first[:, axis]) >= limit - 2 for axis, limit in enumerate((20, 18, 16)))
    )
    assert len(c._deterministic_spatial_cloud_selection(candidates[:123], 500)) == 123
    src = inspect.getsource(c._deterministic_spatial_cloud_selection)
    assert 'random' not in src.lower() and 'morton' in src.lower()


def test_light_plotting_source_static():
    src = ''.join(inspect.getsource(c._write_light_transport_outputs_checked).split())
    assert "get_cmap('gray').with_extremes(" in src
    assert 'bad=(1.0,1.0,1.0,1.0)' in src and 'under=(0.0,0.0,0.0,1.0)' in src
    assert '.set_bad(' not in src and '.set_under(' not in src
    assert '.contour(' not in src and 'DETECTOR_ANGLES_DEG' not in src
    assert (
        'sensor_ring_inner_radius_m' not in src
        and 'sensor_ring_outer_radius_m' not in src
    )
    for colour in (
        "'inferno'",
        "'lime'",
        "'cyan'",
        "'orange'",
        "'deepskyblue'",
        "'red'",
    ):
        assert colour not in src
    assert 'set_box_aspect((2*lim,2*lim,zmax))' in src
    assert "display['plot_3d_z_max_m']" in src
    assert (
        'annotation_clip=False' in src
        and src.count("source_note.set_gid('source-entry-") == 2
    )
    assert "ax.text(0,-R*1e3,sensor_h*1e3,'sourceentry'" not in src
    assert (
        'LIGHT_TRANSPORT_3D_ALPHA_MIN' in src
        and 'LIGHT_TRANSPORT_3D_MARKER_SIZE_MAX' in src
    )


def test_production_cuda_launch_arity():
    match = re.search(
        '__global__ void trace_kernel\\((.*?)\\)\\s*\\{', c.CUDA_SRC, re.S
    )
    assert match is not None
    formal = len([q for q in match.group(1).split(',') if q.strip()])
    tree = ast.parse(Path(c.__file__).read_text(encoding='utf-8'))
    launch_counts = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and (node.func.attr == 'kernel')
            and node.args
            and isinstance(node.args[-1], ast.Tuple)
        ):
            tup = node.args[-1]
            names = [
                e.value.id
                for e in tup.elts
                if isinstance(e, ast.Starred) and isinstance(e.value, ast.Name)
            ]
            if names == ['light_traj_float', 'outf', 'outi']:
                launch_counts.append(
                    sum((not isinstance(e, ast.Starred) for e in tup.elts))
                    + 6
                    + 71
                    + 58
                )
    assert launch_counts == [formal], (launch_counts, formal)


def _cuda_model():
    return c.ClaritasForwardModel(
        free_surface_model='parabolic_vortex',
        vortex_wall_center_height_difference_m=0.05,
        vortex_core_radius_m=0.02,
        sediment_transport_model='schiller_naumann_drift_diffusion',
        meridional_circulation_ratio=0.05,
        particle_event_model='geometric',
        chunk_size=2048,
        statistics_batch_rays=2048,
    )


def _cuda_precompute(model, transport):
    cp = model.cp
    d = np.array([1e-06, 1e-05, 0.0001])
    n = len(d)
    nr = 129
    arr = lambda: cp.empty(n, dtype=cp.float64)
    out = {
        k: arr()
        for k in (
            'tau',
            'ws',
            'sre',
            'sdrag',
            'A',
            'B',
            'norm',
            'mx',
            'rmax',
            'rre',
            'rcenter',
            'sc',
            'sw',
            'bc',
            'bw',
        )
    }
    rlog = cp.empty(n * nr, dtype=cp.float64)
    rwr = cp.empty_like(rlog)
    major = cp.empty(4, dtype=cp.float64)
    err = cp.zeros(1, dtype=cp.int32)
    tg = cp.asarray([1.0, 2.0, 3.0])
    tw = cp.asarray([0.5, 0.25, 0.125])
    code = c.ClaritasForwardModel.SEDIMENT_TRANSPORT_MODEL_CODES[transport]
    model.kernels['sediment_transport_precompute_kernel'](
        (1,),
        (1,),
        (
            cp.asarray(d),
            np.int32(n),
            np.int32(code),
            np.float64(2600.0),
            np.float64(997.0),
            np.float64(0.00089),
            np.float64(9.81),
            np.float64(0.001),
            np.float64(0.0465),
            np.float64(-0.093),
            np.float64(0.049),
            np.int32(1),
            np.float64(0.05),
            np.float64(0.02),
            np.int32(2048),
            np.int32(nr),
            tg,
            tw,
            out['tau'],
            out['ws'],
            out['sre'],
            out['sdrag'],
            out['A'],
            out['B'],
            out['norm'],
            out['mx'],
            out['rmax'],
            out['rre'],
            out['rcenter'],
            rlog,
            rwr,
            out['sc'],
            out['sw'],
            out['bc'],
            out['bw'],
            major,
            err,
        ),
    )
    cp.cuda.Stream.null.synchronize()
    assert int(cp.asnumpy(err)[0]) == 0
    h = c.sediment_transport_coefficients_host(
        d,
        0.049,
        -0.093,
        0.0465,
        0.05,
        0.02,
        0.001,
        quadrature_points=2048,
        radial_lookup_points=nr,
        transport_model=transport,
        free_surface_model='parabolic_vortex',
    )
    pairs = (
        ('tau', 'response_time_s'),
        ('ws', 'settling_velocity_m_s'),
        ('sre', 'settling_reynolds'),
        ('sdrag', 'settling_drag_factor'),
        ('A', 'radial_log_coefficient'),
        ('B', 'vertical_inverse_length_m'),
        ('norm', 'normalization_scaled'),
        ('mx', 'max_density_scale'),
        ('rmax', 'radial_max_slip_velocity_m_s'),
        ('rre', 'radial_max_reynolds'),
        ('rcenter', 'radial_center_log_factor'),
    )
    for dk, hk in pairs:
        close(cp.asnumpy(out[dk]), h[hk], rtol=3e-11, atol=3e-13)
    close(
        cp.asnumpy(rlog).reshape(n, nr), h['radial_log_lookup'], rtol=3e-11, atol=3e-13
    )
    assert np.all(np.isfinite(cp.asnumpy(major)))
    return (h, out, rlog, rwr, major)


def test_cuda_compile_and_parity():
    m = _cuda_model()
    cp = m.cp
    rays = cp.asarray(
        [[0.0, 0.0, -0.05, 0.0, 0.0, 1.0], [0.02, 0.0, -0.05, 0.0, 0.0, 1.0]],
        dtype=cp.float32,
    )
    n = len(rays)
    outs = [cp.empty(n, dtype=cp.float32) for _ in range(6)]
    m.kernels['free_surface_geometry_test_kernel'](
        (1,),
        (32,),
        (
            rays,
            np.int32(n),
            np.float32(0.049),
            np.float32(0.0465),
            np.int32(1),
            np.float32(0.05),
            np.float32(0.02),
            *outs,
        ),
    )
    cp.cuda.Stream.null.synchronize()
    zs = cp.asnumpy(outs[0])
    expected = 0.049 + 0.05 * (np.array([0.0, 0.02]) ** 2 / 0.0465**2 - 0.5)
    close(zs, expected, rtol=3e-06, atol=3e-07)
    normals = np.column_stack([cp.asnumpy(q) for q in outs[3:]])
    slopes = 2 * 0.05 * np.array([0.0, 0.02]) / 0.0465**2
    expected_normals = np.column_stack([slopes, np.zeros(2), -np.ones(2)])
    expected_normals /= np.linalg.norm(expected_normals, axis=1)[:, None]
    close(normals, expected_normals, rtol=4e-06, atol=3e-07)
    for transport in (
        'stokes_drift_diffusion',
        'schiller_naumann_drift_diffusion',
        'schiller_naumann_meridional_advection_diffusion',
    ):
        h, out, rlog, rwr, major = _cuda_precompute(m, transport)
        xyz = np.array(
            [
                [0.0, 0.0, -0.093],
                [0.95 * 0.0465, 0.0, -0.093],
                [0.0, 0.0, 0.0],
                [0.95 * 0.0465, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        for pidx in range(3):
            dev = cp.empty(len(xyz), dtype=cp.float32)
            model_code = c.ClaritasForwardModel.SEDIMENT_TRANSPORT_MODEL_CODES[
                transport
            ]
            field_code = 2 if model_code == 3 else model_code
            dummy_grid = cp.ones(8, dtype=cp.float64)
            m.kernels['sediment_field_test_kernel'](
                (1,),
                (32,),
                (
                    cp.asarray(xyz),
                    np.int32(len(xyz)),
                    np.int32(pidx),
                    np.int32(field_code),
                    np.int32(1),
                    out['A'],
                    out['B'],
                    out['norm'],
                    rlog,
                    np.int32(129),
                    dummy_grid,
                    np.int32(2),
                    np.int32(2),
                    np.float32(0.074),
                    np.float32(0.0465),
                    np.float32(-0.093),
                    np.float32(0.02),
                    dev,
                ),
            )
            cp.cuda.Stream.null.synchronize()
            log_r = np.array(
                [
                    np.interp(
                        float(q[0]),
                        h['radial_lookup_r_m'],
                        h['radial_log_lookup'][pidx],
                    )
                    for q in xyz
                ]
            )
            expected_scale = (
                np.exp(
                    log_r - h['vertical_inverse_length_m'][pidx] * (xyz[:, 2] + 0.093)
                )
                / h['normalization_scaled'][pidx]
            )
            close(cp.asnumpy(dev), expected_scale, rtol=2e-05, atol=2e-07)
            assert np.max(cp.asnumpy(dev)) <= h['max_density_scale'][pidx] * (1 + 3e-06)
    rz = np.array(
        [[0.0, -0.02], [0.01, 0.0], [0.03, 0.02], [0.0465, 0.04]], dtype=np.float32
    )
    urd = cp.empty(len(rz), dtype=cp.float32)
    uzd = cp.empty_like(urd)
    peak = 0.05 * math.sqrt(2 * 9.81 * 0.05 / 0.0465**2) * 0.0465
    m.kernels['meridional_velocity_test_kernel'](
        (1,),
        (32,),
        (
            cp.asarray(rz),
            np.int32(len(rz)),
            np.float32(-0.093),
            np.float32(0.049),
            np.float32(0.0465),
            np.int32(1),
            np.float32(0.05),
            np.float32(0.02),
            np.float32(peak),
            urd,
            uzd,
        ),
    )
    urh, uzh, _ = c.meridional_velocity_host(
        rz[:, 0],
        rz[:, 1],
        0.049,
        -0.093,
        0.0465,
        0.05,
        0.02,
        0.05,
        free_surface_model='parabolic_vortex',
    )
    close(cp.asnumpy(urd), urh, rtol=3e-05, atol=2e-07)
    close(cp.asnumpy(uzd), uzh, rtol=3e-05, atol=2e-07)
    nr, nz, ztop = (16, 32, 0.074)
    active = cp.empty(nr * nz, dtype=cp.uint8)
    m.kernels['meridional_active_mask_kernel'](
        (2,),
        (256,),
        (
            np.int32(nr),
            np.int32(nz),
            np.float64(0.0465),
            np.float64(-0.093),
            np.float64(ztop),
            np.float64(0.049),
            np.int32(1),
            np.float64(0.05),
            np.float64(0.02),
            active,
        ),
    )
    cp.cuda.Stream.null.synchronize()
    rr = (np.arange(nr) + 0.5) * 0.0465 / nr
    zz = -0.093 + (np.arange(nz) + 0.5) * (ztop + 0.093) / nz
    expected_active = zz[None, :] < 0.049 + 0.05 * (rr[:, None] ** 2 / 0.0465**2 - 0.5)
    assert np.array_equal(
        cp.asnumpy(active).reshape(nr, nz).astype(bool), expected_active
    )


def test_cuda_light_diagnostic_kernels():
    m = _cuda_model()
    cp = m.cp
    nx, ny, nz = (12, 10, 14)
    R = 0.0465
    zmin = -0.093
    zmax = 0.074
    inv = 1 / math.sqrt(3)
    seg = np.asarray(
        [
            [0, 0, -0.08, 0, 0, 1, 0.12],
            [-0.04, 0, 0, 1, 0, 0, 0.08],
            [-0.03, -0.03, -0.08, inv, inv, inv, 0.08],
            [0, 0, -0.02, 1, 0, 0, 0.03],
            [-0.06, 0, 0, 1, 0, 0, 0.03],
            [0, 0, 0, 1, 0, 0, R],
            [R, 0, 0, -1, 0, 0, 2 * R],
            [0, 0, zmin, 0, 0, 1, zmax - zmin],
            [0, 0, 0, 1, 0, 0, R + 1e-07],
        ],
        dtype=np.float32,
    )
    total = cp.zeros(nx * ny * nz, dtype=cp.float32)
    post = cp.zeros_like(total)
    st = cp.zeros(1, dtype=cp.float64)
    dt = cp.zeros(1, dtype=cp.float64)
    sc = cp.zeros(1, dtype=cp.uint64)
    err = cp.zeros(1, dtype=cp.uint32)
    adj = cp.zeros(1, dtype=cp.uint32)
    m.kernels['light_diag_segment_test_kernel'](
        (1,),
        (32,),
        (
            cp.asarray(seg),
            np.int32(len(seg)),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            np.float32(R),
            np.float32(zmin),
            np.float32(zmax),
            total,
            post,
            st,
            dt,
            sc,
            err,
            adj,
        ),
    )
    cp.cuda.Stream.null.synchronize()
    a = cp.asnumpy(total)
    p = cp.asnumpy(post)
    assert int(cp.asnumpy(err)[0]) == 0 and int(cp.asnumpy(sc)[0]) == len(seg)
    assert (
        np.all(np.isfinite(a))
        and np.all(a >= 0)
        and np.all(p >= 0)
        and np.all(p <= a + 1e-08)
    )
    expected_length = (
        float(seg[:4, 6].sum(dtype=np.float64)) + 0.0165 + R + 2 * R + (zmax - zmin) + R
    )
    close(float(cp.asnumpy(st)[0]), expected_length, rtol=2e-07, atol=2e-08)
    close(float(cp.asnumpy(dt)[0]), float(cp.asnumpy(st)[0]), rtol=2e-06, atol=2e-07)
    close(
        float(a.sum(dtype=np.float64)), float(cp.asnumpy(st)[0]), rtol=2e-06, atol=2e-07
    )
    xyz = np.asarray(
        [
            [0, 0, 0],
            [0.02, -0.01, -0.04],
            [-0.03, 0.02, 0.05],
            [R + 5e-08, 0, 0],
            [0.2, 0, 0],
        ],
        dtype=np.float32,
    )
    ew = cp.zeros(nx * ny * nz, dtype=cp.float32)
    ec = cp.zeros(nx * ny * nz, dtype=cp.uint32)
    et = cp.zeros(1, dtype=cp.uint64)
    ee = cp.zeros(1, dtype=cp.uint32)
    eo = cp.zeros(32, dtype=cp.uint32)
    adj.fill(0)
    m.kernels['light_diag_event_test_kernel'](
        (1,),
        (32,),
        (
            cp.asarray(xyz),
            np.int32(len(xyz)),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
            np.float32(R),
            np.float32(zmin),
            np.float32(zmax),
            ew,
            ec,
            et,
            ee,
            eo,
            adj,
        ),
    )
    cp.cuda.Stream.null.synchronize()
    assert (
        int(cp.asnumpy(et)[0]) == 5
        and int(cp.asnumpy(ec).sum()) == 4
        and (int(cp.asnumpy(ee)[0]) == 1)
        and (int(cp.asnumpy(eo)[0]) == 1)
    )
    assert int(cp.asnumpy(adj)[0]) >= 1
    close(cp.asnumpy(ew), cp.asnumpy(ec), rtol=0, atol=0)


def test_cuda_meridional_solver():
    model = c.ReportingClaritasForwardModel(
        free_surface_model='parabolic_vortex',
        vortex_wall_center_height_difference_m=0.05,
        vortex_core_radius_m=0.02,
        sediment_transport_model='schiller_naumann_meridional_advection_diffusion',
        meridional_circulation_ratio=0.05,
        meridional_grid_nr=16,
        meridional_grid_nz=32,
        meridional_solver_max_iterations=3000,
        meridional_solver_check_interval=100,
        meridional_solver_tolerance=1e-05,
        meridional_solver_sor=1.5,
        particle_event_model='geometric',
        chunk_size=2048,
        statistics_batch_rays=2048,
    )
    prep = model._prepare_gpu_particle_physics('loess', 0.5)
    host, dev = (prep['host'], prep['device'])
    assert host['meridional_solver_iterations'] >= 1000
    assert host['meridional_solver_max_residual'] <= host['meridional_solver_tolerance']
    residual = np.asarray(host['meridional_solver_residual_by_bin'])
    assert np.all(np.isfinite(residual))
    nr, nz = (dev['sed_grid_nr'], dev['sed_grid_nz'])
    grid = model.cp.asnumpy(dev['sed_grid']).reshape(prep['n'], nr, nz)
    r = (np.arange(nr) + 0.5) * model.rin / nr
    ztop = float(dev['sed_grid_zmax'])
    z = model.zmin + (np.arange(nz) + 0.5) * (ztop - model.zmin) / nz
    surface = model.zmean + model.vortex_delta_h * (r * r / model.rin**2 - 0.5)
    active = z[None, :] < surface[:, None]
    dr = model.rin / nr
    dz = (ztop - model.zmin) / nz
    rw = np.arange(nr) * dr
    re = rw + dr
    cell_volume = 0.5 * (re * re - rw * rw)[:, None] * dz
    mass = np.sum(grid * active[None, :, :] * cell_volume[None, :, :], axis=(1, 2))
    target = 0.5 * model.rin**2 * (model.zmean - model.zmin)
    close(mass, target, rtol=4e-12, atol=2e-16)
    active_values = grid[:, active]
    assert np.all(np.isfinite(active_values)) and np.all(active_values >= 0.0)
    close(
        np.max(active_values, axis=1),
        host['sediment_max_density_scale'],
        rtol=2e-12,
        atol=2e-14,
    )
    close(
        np.min(active_values, axis=1),
        host['sediment_min_density_scale'],
        rtol=2e-12,
        atol=2e-14,
    )
    assert np.isfinite(host['sediment_majorant_total_per_m'])
    assert host['sediment_majorant_total_per_m'] >= 0.0


def test_cuda_full_ray():
    names = (
        'ENABLE_VORTEX',
        'VORTEX_MODEL',
        'VORTEX_WALL_CENTER_HEIGHT_DIFFERENCE_M',
        'VORTEX_CORE_RADIUS_M',
        'SEDIMENT_DISTRIBUTION',
        'SEDIMENT_TRANSPORT_MODEL',
        'SAVE_VORTEX_PROFILE',
        'SAVE_CSV',
        'SAVE_PLOTS',
        'SHOW_PLOTS',
    )
    old = {k: getattr(c, k) for k in names}
    geometries = {}
    try:
        c.ENABLE_VORTEX = True
        c.VORTEX_WALL_CENTER_HEIGHT_DIFFERENCE_M = 0.05
        c.VORTEX_CORE_RADIUS_M = 0.02
        c.SEDIMENT_DISTRIBUTION = 'steady_state'
        c.SEDIMENT_TRANSPORT_MODEL = 'schiller_naumann_drift_diffusion'
        c.SAVE_VORTEX_PROFILE = True
        c.SAVE_CSV = True
        c.SAVE_PLOTS = True
        c.SHOW_PLOTS = False
        with tempfile.TemporaryDirectory(prefix='claritas_validate_') as td:
            root = Path(td)
            for vortex in ('parabolic_vortex', 'localized_scully_vortex'):
                c.VORTEX_MODEL = vortex
                model = c.ReportingClaritasForwardModel(
                    free_surface_model=vortex,
                    vortex_wall_center_height_difference_m=0.05,
                    vortex_core_radius_m=0.02,
                    sediment_transport_model='schiller_naumann_drift_diffusion',
                    meridional_circulation_ratio=0.0,
                    particle_event_model='hybrid_mie_exact_rate',
                    hybrid_wave_scale=1.0,
                    chunk_size=2048,
                    statistics_batch_rays=2048,
                    max_internal_bounces=4095,
                )
                a = model.simulate('loess', 0.5, n_rays=2048, seed=24630001)
                light_cfg = {
                    'enabled': True,
                    'max_rays': 2048,
                    'nx': 24,
                    'ny': 24,
                    'nz': 32,
                    'representative_rays': 16,
                    'max_segments_per_ray': 128,
                }
                b = model.simulate(
                    'loess',
                    0.5,
                    n_rays=2048,
                    seed=24630001,
                    light_transport_config=light_cfg,
                )
                assert a.free_surface_model == vortex and b.free_surface_model == vortex
                assert (
                    a.production_failure_count == 0 and b.production_failure_count == 0
                )
                close(
                    a.symmetry_exact_scores, b.symmetry_exact_scores, atol=0.0, rtol=0.0
                )
                assert np.all(np.isfinite(a.symmetry_exact_scores))
                ld = b.light_transport_diagnostics
                assert (
                    ld
                    and ld['diagnostic_rays_processed'] == 2048
                    and (ld['segment_count'] > 0)
                )
                assert (
                    ld['hard_error_count'] == 0
                    and ld['event_outside_count'] == 0
                    and (ld['deposition_closure_relative_error'] < 2e-06)
                )
                assert (
                    np.sum(ld['track_length_m']) > 0
                    and np.sum(ld['scatter_event_count']) == ld['scatter_event_total']
                )
                out = root / vortex
                out.mkdir()
                g, d = c._validate_user_configuration()
                rows = c._write_vortex_outputs(out, g, d)
                assert rows and all(
                    (
                        (out / q).is_file()
                        for q in (
                            'vortex_field.csv',
                            'vortex_velocity_profile.png',
                            'vortex_free_surface_profile.png',
                        )
                    )
                )
                output_cfg = dict(
                    light_cfg,
                    save_data=True,
                    save_visualisations=True,
                    vertical_slice_thickness_m=0.006,
                    sensor_slice_thickness_m=0.006,
                    log_floor=1e-12,
                )
                ls = c._write_light_transport_outputs(
                    out,
                    ld,
                    g,
                    d,
                    output_cfg,
                    {
                        'material': 'loess',
                        'concentration_g_L': 0.5,
                        'wavelength_nm': 622,
                        'production_rays': 2048,
                    },
                )
                required = (
                    'light_transport_vertical_cross_section.png',
                    'light_transport_sensor_plane_cross_section.png',
                    'light_transport_3d_isometric.png',
                    'light_transport_vertical_cross_section.csv',
                    'light_transport_sensor_plane_cross_section.csv',
                    'light_transport_volume.npz',
                    'light_transport_representative_trajectories.csv',
                    'light_transport_summary.json',
                )
                assert all(
                    (
                        (out / name).is_file() and (out / name).stat().st_size > 0
                        for name in required
                    )
                )
                assert (
                    ls['diagnostic_error_count'] == 0
                    and ls['scatter_event_closure_error'] == 0
                )
                nvals, kvals = c.get_material_optical_constants('loess')
                wl = c._wavelength_assumption_summary('loess', 622.0, nvals, kvals)
                summary = c._summary_dict(
                    'loess',
                    0.5,
                    622.0,
                    24630001,
                    'validation GPU',
                    0.0,
                    a,
                    a.particle_diagnostics or {},
                    wl,
                    out,
                )
                c._write_json(out / 'run_summary.json', summary)
                loaded = json.loads(
                    (out / 'run_summary.json').read_text(encoding='utf-8')
                )
                assert loaded['free_surface_model'] == vortex
                assert loaded['production_failure_count'] == 0
                assert loaded['configuration_valid'] is True
                geometries[vortex] = (
                    loaded['vortex_center_surface_height_above_bottom_m'],
                    loaded['vortex_wall_surface_height_above_bottom_m'],
                )
            assert (
                geometries['parabolic_vortex'] != geometries['localized_scully_vortex']
            )
    finally:
        for k, v in old.items():
            setattr(c, k, v)


def cuda_available():
    try:
        import cupy as cp

        return (cp.cuda.runtime.getDeviceCount() > 0, cp)
    except Exception:
        return (False, None)


def main():
    gate = Gate()
    for name, fn in (
        ('configuration and source identity', test_source_identity),
        ('parabolic free-surface geometry', test_parabolic_geometry),
        ('parabolic Stokes analytic transport', test_parabolic_stokes_analytic),
        (
            'parabolic Schiller-Naumann and normalization',
            test_parabolic_schiller_and_normalization,
        ),
        ('localized-Scully reference values', test_scully_reference),
        ('parabolic meridional host field', test_parabolic_meridional_host),
        (
            'supported configuration matrix and outputs',
            test_configuration_matrix_and_outputs,
        ),
        ('invalid configuration fail-closed', test_fail_closed_configuration),
        ('light-history host products', test_light_transport_host_products),
        ('run-case light configuration binding', test_run_case_light_config_binding),
        (
            'warning-free diagnostic span arithmetic',
            test_warning_free_diagnostic_span_arithmetic,
        ),
        ('light-display scale fallbacks', test_light_display_scaling_fallbacks),
        (
            'light-display direction and deterministic 3-D selection',
            test_light_display_semantics_and_selection,
        ),
        (
            'greyscale water-domain plotting integration',
            test_light_plotting_source_static,
        ),
        ('production CUDA launch arity', test_production_cuda_launch_arity),
        ('CUDA source static integration', test_cuda_source_static),
    ):
        gate.check(name, fn)
    if ARGS.cuda or ARGS.cuda_full:
        available, cp = cuda_available()
        if available:
            gpu_name = cp.cuda.runtime.getDeviceProperties(0).get('name', 'CUDA GPU')
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode(errors='replace')
            print(f'CUDA device: {gpu_name}')
            gate.check('CUDA compile and host parity', test_cuda_compile_and_parity)
            gate.check(
                'CUDA light-history deposition and event kernels',
                test_cuda_light_diagnostic_kernels,
            )
            if ARGS.cuda_full:
                gate.check(
                    'CUDA parabolic meridional solver', test_cuda_meridional_solver
                )
                gate.check('CUDA deterministic small-ray case', test_cuda_full_ray)
            else:
                gate.skip('CUDA parabolic meridional solver', 'use --cuda-full')
                gate.skip('CUDA deterministic small-ray case', 'use --cuda-full')
        else:
            gate.skip('CUDA compile and host parity', 'CuPy/CUDA device unavailable')
            gate.skip(
                'CUDA light-history deposition and event kernels',
                'CuPy/CUDA device unavailable',
            )
            gate.skip(
                'CUDA parabolic meridional solver', 'CuPy/CUDA device unavailable'
            )
            gate.skip(
                'CUDA deterministic small-ray case', 'CuPy/CUDA device unavailable'
            )
    else:
        gate.skip('CUDA compile and host parity', 'use --cuda on a CUDA-capable system')
        gate.skip(
            'CUDA light-history deposition and event kernels',
            'use --cuda on a CUDA-capable system',
        )
        gate.skip(
            'CUDA parabolic meridional solver',
            'use --cuda-full on a CUDA-capable system',
        )
        gate.skip(
            'CUDA deterministic small-ray case',
            'use --cuda-full on a CUDA-capable system',
        )
    print(f'RESULT: passed={gate.passed} failed={gate.failed} skipped={gate.skipped}')
    return 1 if gate.failed else 0


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument(
        '--cuda',
        action='store_true',
        help='compile CUDA and run host/device parity tests',
    )
    p.add_argument(
        '--cuda-full',
        action='store_true',
        help='also run a deterministic small production ray case',
    )
    p.add_argument('--verbose', action='store_true')
    ARGS = p.parse_args()
    raise SystemExit(main())
