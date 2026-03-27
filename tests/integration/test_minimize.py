"""
Integration tests for LAMMPS minimization — requires real LAMMPS.

These tests verify that Lammps.minimize() produces physically sensible
output when run against the Zr vacancy example system.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
ZR_DIR = os.path.join(_REPO_ROOT, "examples/main-functionality/Zr")


def _real_lammps_available():
    try:
        import lammps as _lm
        return not isinstance(_lm, MagicMock)
    except ImportError:
        return False


skip_no_lammps = pytest.mark.skipif(
    not _real_lammps_available(),
    reason="LAMMPS Python bindings not installed",
)


@skip_no_lammps
class TestMinimizeZr(unittest.TestCase):
    """Minimize the Zr vacancy state with real LAMMPS and check the result."""

    @classmethod
    def setUpClass(cls):
        original_cwd = os.getcwd()
        os.chdir(ZR_DIR)
        try:
            from HopDec.State import read
            from HopDec.Input import getParams
            from HopDec.Lammps import Lammps

            cls.params = getParams()
            cls.state = read("V1.dat")
            lmp = Lammps(cls.params)
            cls.max_move = lmp.minimize(cls.state, verbose=False)
        finally:
            os.chdir(original_cwd)

    def test_energy_is_negative(self):
        """A minimized Zr crystal must have negative cohesive energy."""
        self.assertLess(self.state.totalEnergy, 0.0)

    def test_energy_reasonable_magnitude(self):
        """Total energy per atom for Zr ~ -6 eV; rough sanity check."""
        epa = self.state.totalEnergy / self.state.NAtoms
        self.assertGreater(epa, -15.0)
        self.assertLess(epa, -1.0)

    def test_max_move_is_nonnegative(self):
        """Maximum atomic displacement during minimization must be ≥ 0."""
        self.assertGreaterEqual(self.max_move, 0.0)

    def test_max_move_below_threshold(self):
        """Displacements must be below the configured maxMoveMin limit."""
        self.assertLessEqual(self.max_move, self.params.maxMoveMin)

    def test_positions_unchanged_count(self):
        """Minimization must not change the number of atoms."""
        self.assertEqual(len(self.state.pos) // 3, self.state.NAtoms)

    def test_forces_not_stored_but_energy_set(self):
        """After minimization, totalEnergy must be a finite float."""
        self.assertTrue(np.isfinite(self.state.totalEnergy))


if __name__ == "__main__":
    unittest.main()
