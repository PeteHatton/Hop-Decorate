"""
Tests for HopDec.Transitions — Transition and Connection classes.

No LAMMPS required; State objects are constructed directly in memory.
"""

import os
import math
import tempfile
import unittest
import numpy as np

from HopDec.State import State
from HopDec.Transitions import Transition, Connection
from HopDec.Constants import boltzmann


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_state(label="A"):
    s = State(2)
    s.cellDims = np.array([10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0])
    s.pos = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
    s.type = np.array([1, 2], dtype=np.int32)
    s.canLabel = label
    s.nonCanLabel = label
    s.defectPositions = np.array([])
    s.defectTypes = np.array([])
    s.defectIndices = np.array([])
    s.nDefects = 0
    return s


# ---------------------------------------------------------------------------
# Transition.__init__
# ---------------------------------------------------------------------------

class TestTransitionInit(unittest.TestCase):

    def setUp(self):
        self.s1 = _make_state("A")
        self.s2 = _make_state("B")
        self.t = Transition(self.s1, self.s2)

    def test_initial_state(self):
        self.assertIs(self.t.initialState, self.s1)

    def test_final_state(self):
        self.assertIs(self.t.finalState, self.s2)

    def test_saddle_state_none(self):
        self.assertIsNone(self.t.saddleState)

    def test_barriers_zero(self):
        self.assertEqual(self.t.forwardBarrier, 0)
        self.assertEqual(self.t.reverseBarrier, 0)

    def test_dE_zero(self):
        self.assertEqual(self.t.dE, 0)

    def test_KRA_zero(self):
        self.assertEqual(self.t.KRA, 0)

    def test_images_empty(self):
        self.assertEqual(self.t.images, [])

    def test_labels_empty(self):
        self.assertEqual(self.t.canLabel, "")
        self.assertEqual(self.t.nonCanLabel, "")

    def test_redecorated_zero(self):
        self.assertEqual(self.t.redecorated, 0)


# ---------------------------------------------------------------------------
# Transition.calcRate
# ---------------------------------------------------------------------------

class TestCalcRate(unittest.TestCase):

    def setUp(self):
        self.t = Transition(_make_state("A"), _make_state("B"))
        self.t.forwardBarrier = 0.5   # eV

    def test_rate_formula(self):
        """rate = prefactor * exp(-barrier / (T * kB))"""
        T = 1000.0
        prefactor = 1e13
        expected = prefactor * math.exp(-0.5 / (T * boltzmann))
        self.assertAlmostEqual(self.t.calcRate(T, prefactor), expected, places=3)

    def test_zero_barrier_gives_prefactor(self):
        self.t.forwardBarrier = 0.0
        self.assertAlmostEqual(self.t.calcRate(1000.0, 1e13), 1e13, delta=1.0)

    def test_very_high_barrier_gives_near_zero(self):
        self.t.forwardBarrier = 10.0   # 10 eV — vanishingly small rate
        rate = self.t.calcRate(300.0, 1e13)
        self.assertLess(rate, 1e-100)

    def test_rate_increases_with_temperature(self):
        r_low = self.t.calcRate(300.0)
        r_high = self.t.calcRate(1000.0)
        self.assertGreater(r_high, r_low)

    def test_rate_decreases_with_barrier(self):
        self.t.forwardBarrier = 0.1
        r_low_barrier = self.t.calcRate(1000.0)
        self.t.forwardBarrier = 1.0
        r_high_barrier = self.t.calcRate(1000.0)
        self.assertGreater(r_low_barrier, r_high_barrier)


# ---------------------------------------------------------------------------
# Transition.maxMoveAtom
# ---------------------------------------------------------------------------

class TestMaxMoveAtom(unittest.TestCase):

    def test_returns_correct_atom_index(self):
        s1 = State(2)
        s1.cellDims = np.array([100.0, 0, 0, 0, 100.0, 0, 0, 0, 100.0])
        s1.pos = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
        s1.type = np.array([1, 2], dtype=np.int32)

        s2 = s1.copy()
        # Move atom 1 by a large displacement, atom 0 stays put
        s2.pos[3] = 8.0   # atom 1 x: 5 → 8

        t = Transition(s1, s2)
        idx, typ = t.maxMoveAtom()
        self.assertEqual(idx, 1)
        self.assertEqual(typ, 2)

    def test_returns_atom_type(self):
        s1 = State(2)
        s1.cellDims = np.array([100.0, 0, 0, 0, 100.0, 0, 0, 0, 100.0])
        s1.pos = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        s1.type = np.array([3, 7], dtype=np.int32)

        s2 = s1.copy()
        s2.pos[3] = 9.0   # atom 1 moves a lot

        t = Transition(s1, s2)
        _, typ = t.maxMoveAtom()
        self.assertEqual(typ, 7)


# ---------------------------------------------------------------------------
# Transition.loadRedecoration — missing file
# ---------------------------------------------------------------------------

class TestLoadRedecoration(unittest.TestCase):

    def test_missing_file_returns_empty_dataframe(self):
        import pandas as pd
        t = Transition(_make_state("A"), _make_state("B"))
        t.redecoration = "/nonexistent/path/redecoration"
        df = t.loadRedecoration()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

class TestConnection(unittest.TestCase):

    def setUp(self):
        self.s1 = _make_state("A")
        self.s2 = _make_state("B")
        self.conn = Connection(self.s1, self.s2)

    def test_initial_state(self):
        self.assertIs(self.conn.initialState, self.s1)

    def test_final_state(self):
        self.assertIs(self.conn.finalState, self.s2)

    def test_saddle_none(self):
        self.assertIsNone(self.conn.saddleState)

    def test_len_empty(self):
        self.assertEqual(len(self.conn), 0)

    def test_len_with_transitions(self):
        t1 = Transition(self.s1, self.s2)
        t2 = Transition(self.s1, self.s2)
        self.conn.transitions.extend([t1, t2])
        self.assertEqual(len(self.conn), 2)


if __name__ == "__main__":
    unittest.main()
