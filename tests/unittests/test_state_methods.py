"""
Tests for State class methods.

Covers: __init__, copy, atomPos, atomSeparation, setCellDims,
        getIndicesFromPositions, getMinSepList, volume, calcTemperature.

All tests use only in-memory State objects — no LAMMPS required.
"""

import unittest
import numpy as np

from HopDec.State import State


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(n=3):
    """Return a simple orthogonal State with n atoms on a 10 Å grid."""
    s = State(n)
    s.cellDims = np.array([10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0])
    for i in range(n):
        s.pos[3 * i]     = float(i + 1)
        s.pos[3 * i + 1] = float(i + 1)
        s.pos[3 * i + 2] = float(i + 1)
    s.type[:] = 1
    return s


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------

class TestStateInit(unittest.TestCase):

    def setUp(self):
        self.s = State(4)

    def test_NAtoms(self):
        self.assertEqual(self.s.NAtoms, 4)

    def test_pos_shape(self):
        self.assertEqual(len(self.s.pos), 12)

    def test_type_shape(self):
        self.assertEqual(len(self.s.type), 4)

    def test_cellDims_default_diagonal(self):
        """Default cell is a 100 Å cube (diagonal of the 9-element matrix)."""
        self.assertEqual(self.s.cellDims[0], 100.0)
        self.assertEqual(self.s.cellDims[4], 100.0)
        self.assertEqual(self.s.cellDims[8], 100.0)

    def test_cellDims_off_diagonal_zero(self):
        off = [1, 2, 3, 5, 6, 7]
        for i in off:
            self.assertEqual(self.s.cellDims[i], 0.0)

    def test_totalEnergy_none(self):
        self.assertIsNone(self.s.totalEnergy)

    def test_labels_empty(self):
        self.assertEqual(self.s.canLabel, "")
        self.assertEqual(self.s.nonCanLabel, "")

    def test_time_zero(self):
        self.assertEqual(self.s.time, 0)

    def test_defectLists_empty(self):
        self.assertEqual(len(self.s.defectIndices), 0)
        self.assertEqual(len(self.s.defectPositions), 0)
        self.assertEqual(len(self.s.defectTypes), 0)


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------

class TestStateCopy(unittest.TestCase):

    def setUp(self):
        self.orig = _make_state(3)
        self.orig.totalEnergy = -42.0
        self.orig.canLabel = "abc"
        self.orig.nonCanLabel = "xyz"
        self.copy = self.orig.copy()

    def test_copy_is_different_object(self):
        self.assertIsNot(self.orig, self.copy)

    def test_NAtoms_preserved(self):
        self.assertEqual(self.copy.NAtoms, self.orig.NAtoms)

    def test_pos_values_match(self):
        np.testing.assert_array_equal(self.copy.pos, self.orig.pos)

    def test_pos_independent(self):
        """Mutating the copy must not affect the original."""
        self.copy.pos[0] = 999.0
        self.assertNotEqual(self.orig.pos[0], 999.0)

    def test_cellDims_independent(self):
        self.copy.cellDims[0] = 999.0
        self.assertNotEqual(self.orig.cellDims[0], 999.0)

    def test_type_independent(self):
        self.copy.type[0] = 99
        self.assertNotEqual(self.orig.type[0], 99)

    def test_totalEnergy_preserved(self):
        self.assertEqual(self.copy.totalEnergy, -42.0)

    def test_labels_preserved(self):
        self.assertEqual(self.copy.canLabel, "abc")
        self.assertEqual(self.copy.nonCanLabel, "xyz")


# ---------------------------------------------------------------------------
# atomPos
# ---------------------------------------------------------------------------

class TestAtomPos(unittest.TestCase):

    def setUp(self):
        self.s = _make_state(3)

    def test_returns_correct_position(self):
        # Atom 0 should be at (1, 1, 1)
        np.testing.assert_array_equal(self.s.atomPos(0), [1.0, 1.0, 1.0])

    def test_returns_correct_position_atom2(self):
        np.testing.assert_array_equal(self.s.atomPos(2), [3.0, 3.0, 3.0])

    def test_returns_view_mutates_state(self):
        """atomPos returns a view — changes propagate back to state.pos."""
        self.s.atomPos(1)[0] = 999.0
        self.assertEqual(self.s.pos[3], 999.0)

    def test_out_of_range_raises_IndexError(self):
        with self.assertRaises(IndexError):
            self.s.atomPos(3)


# ---------------------------------------------------------------------------
# atomSeparation
# ---------------------------------------------------------------------------

class TestAtomSeparation(unittest.TestCase):

    def setUp(self):
        self.s = State(2)
        self.s.cellDims = np.array([100.0, 0, 0, 0, 100.0, 0, 0, 0, 100.0])
        # Atom 0 at origin, atom 1 at (3, 4, 0) → separation = 5
        self.s.pos[:] = [0.0, 0.0, 0.0, 3.0, 4.0, 0.0]

    def test_known_separation(self):
        self.assertAlmostEqual(self.s.atomSeparation(0, 1), 5.0, places=8)

    def test_self_separation_is_zero(self):
        self.assertAlmostEqual(self.s.atomSeparation(0, 0), 0.0, places=8)

    def test_symmetric(self):
        self.assertAlmostEqual(
            self.s.atomSeparation(0, 1),
            self.s.atomSeparation(1, 0),
            places=8,
        )

    def test_out_of_range_raises_IndexError(self):
        with self.assertRaises(IndexError):
            self.s.atomSeparation(0, 5)


# ---------------------------------------------------------------------------
# setCellDims
# ---------------------------------------------------------------------------

class TestSetCellDims(unittest.TestCase):

    def test_sets_correct_dims(self):
        s = State(1)
        new_dims = [5.0, 0, 0, 0, 6.0, 0, 0, 0, 7.0]
        s.setCellDims(new_dims)
        self.assertAlmostEqual(s.cellDims[0], 5.0)
        self.assertAlmostEqual(s.cellDims[4], 6.0)
        self.assertAlmostEqual(s.cellDims[8], 7.0)

    def test_wrong_length_is_ignored(self):
        s = State(1)
        original = s.cellDims.copy()
        s.setCellDims([1.0, 2.0, 3.0])  # wrong length
        np.testing.assert_array_equal(s.cellDims, original)

    def test_triclinic_dims(self):
        s = State(1)
        triclinic = [5.0, 1.0, 0.5, 0.0, 6.0, 0.3, 0.0, 0.0, 7.0]
        s.setCellDims(triclinic)
        np.testing.assert_array_almost_equal(s.cellDims, triclinic)


# ---------------------------------------------------------------------------
# volume
# ---------------------------------------------------------------------------

class TestVolume(unittest.TestCase):

    def test_cubic_volume(self):
        s = State(1)
        s.cellDims = np.array([5.0, 0, 0, 0, 5.0, 0, 0, 0, 5.0])
        self.assertAlmostEqual(s.volume(), 125.0)

    def test_rectangular_volume(self):
        s = State(1)
        s.cellDims = np.array([2.0, 0, 0, 0, 3.0, 0, 0, 0, 4.0])
        self.assertAlmostEqual(s.volume(), 24.0)

    def test_default_volume(self):
        s = State(1)
        self.assertAlmostEqual(s.volume(), 100.0 ** 3)


# ---------------------------------------------------------------------------
# getIndicesFromPositions
# ---------------------------------------------------------------------------

class TestGetIndicesFromPositions(unittest.TestCase):

    def setUp(self):
        self.s = _make_state(3)   # atoms at (1,1,1), (2,2,2), (3,3,3)

    def test_finds_exact_match(self):
        result = self.s.getIndicesFromPositions([[1.0, 1.0, 1.0]])
        self.assertIn(0, result)

    def test_finds_within_tolerance(self):
        result = self.s.getIndicesFromPositions([[2.05, 2.0, 2.0]], maxSep=0.1)
        self.assertIn(1, result)

    def test_no_match_returns_empty(self):
        result = self.s.getIndicesFromPositions([[9.0, 9.0, 9.0]], maxSep=0.1)
        self.assertEqual(len(result), 0)

    def test_empty_input_returns_empty(self):
        result = self.s.getIndicesFromPositions([])
        self.assertEqual(len(result), 0)

    def test_negative_maxSep_raises(self):
        with self.assertRaises(ValueError):
            self.s.getIndicesFromPositions([[1.0, 1.0, 1.0]], maxSep=-0.1)

    def test_zero_maxSep_raises(self):
        with self.assertRaises(ValueError):
            self.s.getIndicesFromPositions([[1.0, 1.0, 1.0]], maxSep=0.0)


# ---------------------------------------------------------------------------
# getMinSepList
# ---------------------------------------------------------------------------

class TestGetMinSepList(unittest.TestCase):

    def setUp(self):
        self.s = _make_state(3)   # atoms at (1,1,1), (2,2,2), (3,3,3)

    def test_finds_closest_atom(self):
        query = [2.1, 2.0, 2.0]
        atomList = np.array([0, 1, 2])
        minSep, minIdx = self.s.getMinSepList(query, atomList, maxSpacing=10.0)
        self.assertIsNotNone(minSep)
        self.assertEqual(minIdx, 1)   # atom index 1 in atomList is closest

    def test_empty_list_returns_none(self):
        minSep, minIdx = self.s.getMinSepList([1.0, 1.0, 1.0], np.array([]), maxSpacing=10.0)
        self.assertIsNone(minSep)
        self.assertIsNone(minIdx)

    def test_atomListSize_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            self.s.getMinSepList([1.0, 1.0, 1.0], np.array([0, 1]), atomListSize=5)

    def test_atomListSize_negative_raises(self):
        with self.assertRaises(ValueError):
            self.s.getMinSepList([1.0, 1.0, 1.0], np.array([0, 1]), atomListSize=-1)

    def test_nothing_within_maxSpacing_returns_none(self):
        query = [50.0, 50.0, 50.0]
        atomList = np.array([0, 1, 2])
        minSep, minIdx = self.s.getMinSepList(query, atomList, maxSpacing=0.5)
        self.assertIsNone(minSep)
        self.assertIsNone(minIdx)


# ---------------------------------------------------------------------------
# calcTemperature
# ---------------------------------------------------------------------------

class TestCalcTemperature(unittest.TestCase):

    def test_known_temperature(self):
        """
        T = 2 * KE_sum / (3 * kB * N)
        With N=2, KE=[0.1, 0.1], kB=8.6173324e-5:
        KE_sum = 0.2
        T = 0.4 / (3 * 8.6173324e-5 * 2) ≈ 774.4 K
        """
        from HopDec.Constants import boltzmann
        s = State(2)
        s.KE = np.array([0.1, 0.1])
        T_expected = 2 * 0.2 / (3 * boltzmann * 2)
        self.assertAlmostEqual(s.calcTemperature(), T_expected, places=4)

    def test_zero_KE_gives_zero_temperature(self):
        s = State(3)
        s.KE = np.zeros(3)
        self.assertAlmostEqual(s.calcTemperature(), 0.0, places=8)

    def test_temperature_positive_for_positive_KE(self):
        s = State(4)
        s.KE = np.ones(4) * 0.05
        self.assertGreater(s.calcTemperature(), 0.0)


if __name__ == "__main__":
    unittest.main()
