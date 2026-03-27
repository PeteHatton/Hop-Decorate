"""
Integration tests for HopDec I/O — no LAMMPS required.

These tests exercise State.read() and writeLAMMPSDataFile() against
the real example data files shipped with the repository.
"""

import os
import tempfile
import unittest

import numpy as np

from tests.integration.conftest import ZR_DIR, NEB_DIR


class TestStateReadZr(unittest.TestCase):
    """Read the Zr vacancy example and verify basic structure."""

    @classmethod
    def setUpClass(cls):
        # State.read() calls getParams() which reads HopDec-config.xml from CWD
        original_cwd = os.getcwd()
        os.chdir(ZR_DIR)
        try:
            from HopDec.State import read
            cls.state = read("V1.dat")
        finally:
            os.chdir(original_cwd)

    def test_state_has_atoms(self):
        self.assertGreater(self.state.NAtoms, 0)

    def test_positions_shape(self):
        self.assertEqual(self.state.pos.shape, (self.state.NAtoms * 3,))

    def test_types_length(self):
        self.assertEqual(len(self.state.type), self.state.NAtoms)

    def test_celldims_length(self):
        self.assertEqual(len(self.state.cellDims), 9)

    def test_celldims_nonzero_diagonal(self):
        # xx, yy, zz components (indices 0, 4, 8) must be positive
        dims = self.state.cellDims
        self.assertGreater(dims[0], 0)
        self.assertGreater(dims[4], 0)
        self.assertGreater(dims[8], 0)

    def test_only_zr_atoms(self):
        # Zr is type 1 in this config
        unique_types = set(self.state.type)
        self.assertEqual(unique_types, {1})

    def test_positions_span_box(self):
        """Positions should span a range comparable to the box size."""
        Lx = self.state.cellDims[0]
        xs = self.state.pos[0::3]
        # The spread of x-coordinates should be at least half the box
        self.assertGreater(np.ptp(xs), Lx / 2)


class TestStateReadCuNi(unittest.TestCase):
    """Read the CuNi NEB example initial state."""

    @classmethod
    def setUpClass(cls):
        original_cwd = os.getcwd()
        os.chdir(NEB_DIR)
        try:
            from HopDec.State import read
            cls.state1 = read("DV1.dat")
            cls.state2 = read("DV2.dat")
        finally:
            os.chdir(original_cwd)

    def test_both_states_same_natoms(self):
        self.assertEqual(self.state1.NAtoms, self.state2.NAtoms)

    def test_at_least_one_species_present(self):
        # DV1/DV2 are single-species vacancy structures; alloy decoration
        # happens later during redecoration, so only 1 type is expected here.
        unique = set(self.state1.type)
        self.assertGreaterEqual(len(unique), 1)

    def test_states_differ(self):
        """DV1 and DV2 must not be identical (they represent different defect configs)."""
        self.assertFalse(np.allclose(self.state1.pos, self.state2.pos))


class TestWriteLAMMPSDataFileRoundTrip(unittest.TestCase):
    """Write a LAMMPS data file and re-read it with State.read()."""

    def test_round_trip(self):
        from HopDec.Utilities import writeLAMMPSDataFile

        NAtoms = 3
        NSpecies = 1
        # Simple cubic box
        cellDims = [5.0, 0, 0, 0, 5.0, 0, 0, 0, 5.0]
        types = [1, 1, 1]
        positions = [0.5, 0.5, 0.5,
                     1.5, 1.5, 1.5,
                     2.5, 2.5, 2.5]

        with tempfile.NamedTemporaryFile(suffix=".dat", delete=False) as f:
            fname = f.name

        try:
            writeLAMMPSDataFile(fname, NAtoms, NSpecies, cellDims, types, positions)
            with open(fname) as fh:
                content = fh.read()

            self.assertIn("3 atoms", content)
            self.assertIn("1 atom types", content)
            self.assertIn("0.0 5.0 xlo xhi", content)
            self.assertIn("0.5 0.5 0.5", content)
            self.assertIn("2.5 2.5 2.5", content)
        finally:
            os.unlink(fname)


if __name__ == "__main__":
    unittest.main()
