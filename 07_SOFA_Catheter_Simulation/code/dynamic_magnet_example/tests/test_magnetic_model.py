"""Numerical checks that do not require SOFA."""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import mag_parameters, magnetic_model  # noqa: E402


class MagneticModelTests(unittest.TestCase):
    def test_analytic_gradient_matches_field_finite_difference(self):
        magnets = mag_parameters.build("Set3")
        point = np.array([0.041, -0.052, -0.117])
        _, analytic = magnetic_model.cuboid_field_and_gradient(
            point, magnets.external_dimensions, magnets.remanence_z
        )
        numerical = np.zeros((3, 3))
        step = 2e-7
        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = step
            plus, _ = magnetic_model.cuboid_field_and_gradient(
                point + delta, magnets.external_dimensions,
                magnets.remanence_z,
            )
            minus, _ = magnetic_model.cuboid_field_and_gradient(
                point - delta, magnets.external_dimensions,
                magnets.remanence_z,
            )
            numerical[:, axis] = (plus - minus) / (2.0 * step)
        np.testing.assert_allclose(analytic, numerical, rtol=2e-6, atol=2e-8)

    def test_undeformed_sofa_axis_matches_matlab_moment(self):
        magnets = mag_parameters.build("Set3")
        s = np.sqrt(0.5)
        moment = magnetic_model.catheter_moment_world(
            [0.0, -s, 0.0, s], magnets
        )
        np.testing.assert_allclose(
            moment, magnets.m_c_matlab_local, rtol=0.0, atol=1e-14
        )

    def test_coaxial_default_force_points_toward_external_magnet(self):
        magnets = mag_parameters.build("Set3")
        s = np.sqrt(0.5)
        state = magnetic_model.magnetic_wrench(
            [0.0, 0.0, 0.105], [0.0, -s, 0.0, s],
            [0.0, 0.0, 0.25], [1.0, 0.0, 0.0, 0.0], magnets,
        )
        self.assertGreater(state.force[2], 0.0)
        np.testing.assert_allclose(state.force[:2], 0.0, atol=1e-12)
        np.testing.assert_allclose(state.torque, 0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
