"""Parameter, physical-time path and notebook checks, independent of SOFA."""

from __future__ import annotations

import json
import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils import mag_parameters, parameters, trajectory  # noqa: E402


class DynamicInputTests(unittest.TestCase):
    def test_matlab_default_recipe(self):
        p = parameters.build()
        self.assertAlmostEqual(p.E, 1.888567124751886e6)
        self.assertAlmostEqual(p.nu, 0.422429111252843)
        np.testing.assert_array_equal(p.gravity, np.zeros(3))
        self.assertEqual(p.n_elem, 50)
        self.assertEqual(p.rayleigh_mass, 0)
        self.assertEqual(p.rayleigh_stiffness, 0.1)

    def test_alternate_recipes_explicit(self):
        p = parameters.build(recipe="Trial3")
        self.assertEqual(p.E, 1.05e6)
        self.assertEqual(p.rayleigh_stiffness, 0.01)
        np.testing.assert_array_equal(p.gravity, [9.81, 0, 0])
        p = parameters.build(recipe="StaticMagnet")
        self.assertEqual(p.E, 1.05e6)
        self.assertEqual(p.nu, 0.45)
        np.testing.assert_array_equal(p.gravity, [0, 0, 0])

    def test_substeps_preserve_physical_horizon(self):
        p = parameters.build(steps=20, dt=0.005, substeps=4)
        self.assertEqual(len(p.time), 21)
        self.assertAlmostEqual(p.time[-1], 0.1)
        self.assertAlmostEqual(p.integration_dt, 0.00125)

    def test_invalid_numerics_rejected(self):
        for values in ({"dt": 0}, {"substeps": 0}, {"n_elem": 0},
                       {"steps": 2.5}, {"rayleigh_mass": -1}, {"E": float("nan")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                parameters.build(**values)

    def test_line_duration_less_than_second_reaches_amplitude(self):
        p, _ = trajectory.build("Line", 10, amp=0.1, dt=0.005)
        self.assertAlmostEqual(p[-1, 0], 0.1)

    def test_spiral_uses_physical_not_pseudo_time(self):
        p, _ = trajectory.build("Spiral", 50, freq=0.2, amp=0.1, dt=0.01)
        t = 0.5
        np.testing.assert_allclose(p[-1], [0.1*np.cos(4*np.pi*0.2*t),
                                          0.1*np.sin(4*np.pi*0.2*t), 0.155])

    def test_all_paths_finite_and_quaternions_normalized(self):
        for shape in trajectory.SHAPES:
            with self.subTest(shape=shape):
                p, q = trajectory.build(shape, 50, external_quaternion_wxyz=[0.7071, 0.7071, 0, 0])
                self.assertEqual(p.shape, (51, 3))
                self.assertTrue(np.all(np.isfinite(p)))
                np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1)

    def test_default_step_and_orientation_from_matlab(self):
        p, q = trajectory.build("Step", 3)
        np.testing.assert_allclose(p[0], [0.1, 0.1, 0.155])
        np.testing.assert_array_equal(q[0], [0, 1, 0, 0])

    def test_slerp_shortest_arc_and_antipodal_quaternions(self):
        p, q = trajectory.interpolate_pose([0, 0, 0], [1, 0, 0, 0],
                                           [2, 0, 0], [-1, 0, 0, 0], 0.5)
        np.testing.assert_allclose(p, [1, 0, 0])
        np.testing.assert_allclose(q, [1, 0, 0, 0])
        _, q = trajectory.interpolate_pose([0, 0, 0], [1, 0, 0, 0],
                                           [0, 0, 0], [0, 1, 0, 0], 0.5)
        np.testing.assert_allclose(q, [np.sqrt(0.5), np.sqrt(0.5), 0, 0])

    def test_tip_magnet_cylinder_mass_and_inertia(self):
        m = mag_parameters.build()
        self.assertAlmostEqual(m.mass_c, 0.0011)
        inertia = m.inertia_c_sofa_local
        self.assertAlmostEqual(inertia[0, 0], m.mass_c*m.r_c**2/2)
        self.assertAlmostEqual(inertia[1, 1], m.mass_c*(3*m.r_c**2+m.l_c**2)/12)
        self.assertGreater(np.linalg.eigvalsh(inertia).min(), 0)

    def test_notebook_three_clean_code_cells(self):
        notebook = json.loads((ROOT / "mode.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(len(notebook["cells"]), 3)
        for cell in notebook["cells"]:
            self.assertEqual(cell["cell_type"], "code")
            self.assertEqual(cell["outputs"], [])
            source = "".join(line for line in cell["source"] if not line.startswith("%"))
            compile(source, "mode.ipynb", "exec")


if __name__ == "__main__":
    unittest.main()
