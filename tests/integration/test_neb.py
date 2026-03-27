"""
Integration tests for NEB barrier calculation — requires real LAMMPS.

These tests run a full NEB calculation between the two CuNi double-vacancy
states and verify that the returned barriers are physically meaningful.
"""

import os
import unittest

import numpy as np
import pytest

from tests.integration.conftest import NEB_DIR, skip_no_lammps


@skip_no_lammps
class TestNEBCuNi(unittest.TestCase):
    """Run NEB on the CuNi DV1→DV2 pair and inspect the barriers."""

    @classmethod
    def setUpClass(cls):
        original_cwd = os.getcwd()
        os.chdir(NEB_DIR)
        try:
            from HopDec.State import read
            from HopDec.Input import getParams
            import HopDec.NEB as NEB

            params = getParams()
            # Use loose tolerances so the test finishes quickly
            params.NEBForceTolerance = 0.5
            params.NEBMaxIterations = 50
            params.verbose = 0

            initial = read("DV1.dat")
            final = read("DV2.dat")

            # Minimize both endpoints first
            from HopDec.Lammps import Lammps
            lmp = Lammps(params)
            lmp.minimize(initial, verbose=False)
            lmp.minimize(final, verbose=False)

            cls.conn = NEB.main(initial, final, params)
        finally:
            os.chdir(original_cwd)

    def test_connection_has_transitions(self):
        """NEB must find at least one transition."""
        self.assertGreater(len(self.conn), 0)

    def test_forward_barrier_positive(self):
        """Forward barrier must be > 0 eV (two distinct minima)."""
        barrier = self.conn.transitions[0].forwardBarrier
        self.assertGreater(barrier, 0.0)

    def test_reverse_barrier_positive(self):
        """Reverse barrier must be > 0 eV."""
        barrier = self.conn.transitions[0].reverseBarrier
        self.assertGreater(barrier, 0.0)

    def test_barrier_below_10_eV(self):
        """Physical defect-migration barriers in metals are typically < 5 eV."""
        barrier = self.conn.transitions[0].forwardBarrier
        self.assertLess(barrier, 10.0)

    def test_dE_energy_conservation(self):
        """dE = finalEnergy − initialEnergy.  forward − reverse ≈ dE."""
        t = self.conn.transitions[0]
        self.assertAlmostEqual(
            t.forwardBarrier - t.reverseBarrier,
            t.dE,
            delta=0.05,   # loose: NEB converged to coarse tolerance
        )

    def test_initial_and_final_states_linked(self):
        """The transition must connect the states we passed in."""
        t = self.conn.transitions[0]
        self.assertIsNotNone(t.initialState)
        self.assertIsNotNone(t.finalState)


if __name__ == "__main__":
    unittest.main()
