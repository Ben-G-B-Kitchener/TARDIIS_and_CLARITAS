#!/usr/bin/env python3
"""Small CPU policy/export/sweep tests; no CUDA execution."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import CLARITAS as c


def fixture():
    a = np.ones((2, 2, 2), dtype=np.float32)
    d = dict(
        nx=2,
        ny=2,
        nz=2,
        track_length_m=a,
        post_scatter_track_length_m=a.copy(),
        scatter_event_weight=a.copy(),
        scatter_event_count=a.astype(np.uint32),
        scatter_event_total=9,
        deposited_scatter_event_total=8,
        event_outside_count=1,
        scatter_event_closure_error=-1,
        deposition_closure_relative_error=1e-12,
        grid_closure_relative_error=1e-12,
        segment_length_total_m=8.0,
        deposited_length_total_m=8.0,
        trajectory_count=np.zeros(0, dtype=int),
        trajectory_overflow=np.zeros(0, dtype=int),
        max_segments_per_ray=4,
        nonfinite_count=0,
        negative_value_count=0,
        hard_error_count=1,
        water_advance_guard_failure_count=1,
        diagnostic_rays_processed=100,
        ray_statistical_weight=1.0,
    )
    for k in (
        'trajectory_x0_m',
        'trajectory_y0_m',
        'trajectory_z0_m',
        'trajectory_x1_m',
        'trajectory_y1_m',
        'trajectory_z1_m',
        'trajectory_particle_order',
        'trajectory_event_code',
    ):
        d[k] = np.zeros((0, 4))
    for k in ('x', 'y', 'z'):
        d[k + '_min_m'] = -0.049
        d[k + '_max_m'] = 0.049
    return d


class PolicyTests(unittest.TestCase):
    def test_one_failure_keeps_launched_denominator(self):
        p = c._production_integrity(
            100_000_000,
            1,
            {'count_closure_ok': True, 'code_counts': {'water_guard': 1}},
            c._resolved_failure_policy(),
        )
        c._enforce_case_integrity(
            p, c._diagnostic_integrity(fixture(), c._resolved_failure_policy())
        )
        self.assertFalse(p['fatal'])
        self.assertEqual(p['normalization_rays'], 100_000_000)
        self.assertEqual(p['failed_ray_fraction'], 1e-8)
        # The actual cross-section exporter uses all diagnostic launched histories.
        d = fixture()
        q = c._light_transport_cross_sections(
            d, {'vertical_slice_thickness_m': 0.049, 'sensor_slice_thickness_m': 0.049}
        )
        expected = q['vr'] / (100 * q['dx'] * q['dy'] * q['dz'] * len(q['xi']))
        np.testing.assert_allclose(q['vd'], expected)

    def test_rejected_events_reconcile(self):
        r = c._diagnostic_integrity(fixture(), c._resolved_failure_policy())
        self.assertEqual(r['reconciled_event_discrepancy'], 0)
        self.assertEqual(r['deposited_only_event_discrepancy'], -1)
        self.assertFalse(r['fatal'])
        self.assertEqual(r['diagnostic_status'], 'limited')

    def test_unexplained_discrepancy_visible(self):
        d = fixture()
        d['event_outside_count'] = 0
        r = c._diagnostic_integrity(d, c._resolved_failure_policy())
        self.assertEqual(r['reconciled_event_discrepancy'], -1)
        self.assertFalse(r['event_accounting_reconciled'])
        self.assertFalse(r['fatal'])
        p = c._resolved_failure_policy()
        p['abort_on_systemic_diagnostics'] = True
        self.assertTrue(c._diagnostic_integrity(d, p)['fatal'])

    def test_strict_reports_then_stops(self):
        p = c._resolved_failure_policy()
        p['mode'] = 'strict'
        self.assertTrue(c._diagnostic_integrity(fixture(), p)['fatal'])
        with self.assertRaises(c.CaseIntegrityError):
            c._enforce_case_integrity(
                {'fatal': False}, c._diagnostic_integrity(fixture(), p)
            )
        d = fixture()
        d.update(
            scatter_event_total=8,
            event_outside_count=0,
            hard_error_count=0,
            water_advance_guard_failure_count=0,
        )
        self.assertFalse(c._diagnostic_integrity(d, p)['fatal'])
        d['grid_closure_relative_error'] = 0.01
        self.assertTrue(c._diagnostic_integrity(d, p)['fatal'])

    def test_unsafe_fields_skip_derived_export(self):
        d = fixture()
        d['track_length_m'][0, 0, 0] = np.nan
        with tempfile.TemporaryDirectory() as td, patch.object(
            c,
            '_write_light_transport_outputs_checked',
            side_effect=AssertionError('unsafe renderer called'),
        ):
            r = c._write_light_transport_outputs(
                Path(td), d, {}, {}, {'save_data': True}
            )
            self.assertFalse(r['fatal'])
            self.assertTrue(r['skipped_outputs'])
            self.assertTrue((Path(td) / 'light_transport_summary.json').exists())
            with np.load(Path(td) / 'light_transport_safe_raw_fields.npz') as raw:
                self.assertNotIn('track_length_m', raw.files)
                self.assertIn('scatter_event_count', raw.files)

    def test_production_detector_nonfinite_and_shape_fatal(self):
        from types import SimpleNamespace

        a = np.zeros(len(c.DETECTOR_ANGLES_DEG))
        res = SimpleNamespace(
            hardware_symmetry_scores=a.copy(),
            normalized_hardware_symmetry_response=a.copy(),
            hardware_normalized_jackknife_se=a.copy(),
            hardware_native_hits=a.copy(),
            hardware_mirror_hits=a.copy(),
        )
        res.hardware_symmetry_scores[0] = np.nan
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(c.CaseIntegrityError):
                c._write_detector_outputs(Path(td), 'loess', 0.5, 622, 1, res)
            res.hardware_symmetry_scores = np.zeros(1)
            with self.assertRaises(ValueError):
                c._write_detector_outputs(Path(td), 'loess', 0.5, 622, 1, res)

    def test_shapes_and_counts_unrecoverable(self):
        d = fixture()
        d['track_length_m'] = np.zeros((1,))
        with self.assertRaises(ValueError):
            c._diagnostic_integrity(d, c._resolved_failure_policy())
        with self.assertRaises(ValueError):
            c._production_integrity(
                10, 1, {'count_closure_ok': False}, c._resolved_failure_policy()
            )
        with patch.object(c, 'DIAGNOSTIC_POLICY', 'invalid'):
            with self.assertRaises(ValueError):
                c._resolved_failure_policy()

    def test_systemic_ray_limit(self):
        p = c._resolved_failure_policy()
        self.assertTrue(c._production_integrity(100, 2, {}, p)['fatal'])
        p['max_failed_ray_fraction'] = None
        self.assertFalse(c._production_integrity(100, 2, {}, p)['fatal'])

    def sweep(self, fatal=False):
        calls = []

        def run(root, mat, conc, wav, seed, gpu_name, **kw):
            calls.append(conc)
            if fatal and len(calls) == 2:
                raise RuntimeError('simulated CUDA failure')
            integrity = c._diagnostic_integrity(fixture(), c._resolved_failure_policy())
            return (
                {
                    'material': mat,
                    'concentration_g_L': conc,
                    'wavelength_nm': wav,
                    'light_transport_diagnostics': integrity,
                },
                [],
                [],
                [],
            )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with patch.multiple(
                c,
                RUN_SWEEP=True,
                MATERIALS=['loess'],
                CONCENTRATIONS_G_L=[0.5, 2, 4],
                WAVELENGTHS_NM=[622],
            ), patch.object(
                c, '_require_cuda_and_select_device', return_value=(None, 'CPU mocked')
            ), patch.object(
                c, '_session_root', return_value=root
            ), patch.object(
                c, 'run_one_case', side_effect=run
            ), patch.object(
                c, '_aggregate_sweep_plots'
            ):
                if fatal:
                    with self.assertRaisesRegex(RuntimeError, 'simulated CUDA failure'):
                        c.main()
                    import json

                    progress = json.loads((root / 'sweep_progress.json').read_text())
                    self.assertEqual(len(progress['completed_cases']), 1)
                    self.assertTrue((root / 'case_failure_report.json').exists())
                else:
                    c.main()
        return calls

    def test_recoverable_diagnostic_continues_sweep(self):
        self.assertEqual(self.sweep(), [0.5, 2, 4])

    def test_unrecoverable_stops_and_preserves_previous_case(self):
        self.assertEqual(self.sweep(True), [0.5, 2])


if __name__ == '__main__':
    unittest.main()
