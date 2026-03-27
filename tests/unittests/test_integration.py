"""
Integration tests for HopDec.

These tests exercise multiple components working together, verifying that
the State → Transition → Model pipeline behaves correctly end-to-end.

No LAMMPS is required; all States are constructed in memory.
"""

import unittest
import numpy as np

from HopDec.State import State
from HopDec.Transitions import Transition, Connection
from HopDec.Model import Model
from HopDec.Input import InputParams
from HopDec.Graphs import graphLabel, shortestPath, buildNetwork


# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------

def _params(canonical=0, max_depth=-1):
    p = InputParams()
    p.canonicalLabelling = canonical
    p.maxModelDepth = max_depth
    p.nDefectsMax = 100
    p.maxDefectAtoms = -1
    p.segmentLength = 100
    p.verbose = 0
    return p


def _state(label, pos=None):
    s = State(2)
    s.cellDims = np.array([100.0, 0, 0, 0, 100.0, 0, 0, 0, 100.0])
    s.pos = np.array(pos or [1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
    s.type = np.array([1, 2], dtype=np.int32)
    s.canLabel = label
    s.nonCanLabel = label
    s.defectPositions = np.array([])
    s.defectTypes = np.array([])
    s.defectIndices = np.array([])
    s.nDefects = 0
    s.doWork = 1
    s.time = 100
    return s


def _trans(s1, s2, label):
    t = Transition(s1, s2)
    t.canLabel = label
    t.nonCanLabel = label
    return t


def _seed_model(params, init_state):
    model = Model(params)
    model.initState = init_state
    model.stateList.append(init_state)
    model.buildModelGraph()
    return model


# ---------------------------------------------------------------------------
# 1. State copy independence
# ---------------------------------------------------------------------------

class TestStateCopyInPipeline(unittest.TestCase):
    """Copied states used in different transitions must be fully independent."""

    def test_copy_used_in_two_transitions(self):
        s1 = _state("A")
        s2 = _state("B")
        s2_copy = s2.copy()

        t1 = _trans(s1, s2, "T1")
        t2 = _trans(s1, s2_copy, "T2")

        # Mutating s2 must not affect s2_copy
        s2.pos[0] = 999.0
        self.assertNotEqual(s2_copy.pos[0], 999.0)

    def test_barriers_set_independently(self):
        s1 = _state("A")
        s2 = _state("B")
        t1 = _trans(s1, s2, "T1")
        t2 = _trans(s1, s2.copy(), "T2")
        t1.forwardBarrier = 0.3
        t2.forwardBarrier = 0.7
        self.assertAlmostEqual(t1.forwardBarrier, 0.3)
        self.assertAlmostEqual(t2.forwardBarrier, 0.7)


# ---------------------------------------------------------------------------
# 2. Linear chain: A → B → C
# ---------------------------------------------------------------------------

class TestLinearChainModel(unittest.TestCase):
    """Build a 3-state linear model and verify graph structure and depths."""

    def setUp(self):
        self.p = _params()
        self.sA = _state("A")
        self.sB = _state("B")
        self.sC = _state("C")
        self.tAB = _trans(self.sA, self.sB, "AB")
        self.tBC = _trans(self.sB, self.sC, "BC")

        self.model = _seed_model(self.p, self.sA)
        self.model.update(transitions=[self.tAB])
        self.model.update(transitions=[self.tBC])

    def test_model_has_two_transitions(self):
        self.assertEqual(len(self.model), 2)

    def test_model_has_three_states(self):
        self.assertEqual(len(self.model.stateList), 3)

    def test_depth_A_to_A(self):
        self.assertEqual(self.model.findDepth(self.sA), 0)

    def test_depth_A_to_B(self):
        self.assertEqual(self.model.findDepth(self.sB), 1)

    def test_depth_A_to_C(self):
        self.assertEqual(self.model.findDepth(self.sC), 2)

    def test_graph_nodes_correct(self):
        for label in ("A", "B", "C"):
            self.assertIn(label, self.model.graph.nodes)


# ---------------------------------------------------------------------------
# 3. Duplicate detection
# ---------------------------------------------------------------------------

class TestDuplicateDetection(unittest.TestCase):
    """Re-adding known states/transitions must not grow the model."""

    def setUp(self):
        self.p = _params(canonical=0)
        self.sA = _state("A")
        self.sB = _state("B")
        self.tAB = _trans(self.sA, self.sB, "AB")
        self.model = _seed_model(self.p, self.sA)
        self.model.update(transitions=[self.tAB])

    def test_no_duplicate_transitions(self):
        tAB2 = _trans(self.sA, self.sB, "AB")
        self.model.update(transitions=[tAB2])
        self.assertEqual(len(self.model), 1)

    def test_no_duplicate_states(self):
        sB2 = _state("B")   # same non-canonical label
        count_before = len(self.model.stateList)
        self.model.update(states=[sB2])
        self.assertEqual(len(self.model.stateList), count_before)


# ---------------------------------------------------------------------------
# 4. Canonical vs non-canonical deduplication
# ---------------------------------------------------------------------------

class TestCanonicalDeduplication(unittest.TestCase):

    def test_canonical_on_different_noncan_same_can(self):
        """
        With canonical labelling, two states that share the same canLabel
        but have different nonCanLabels should be treated as duplicates.
        """
        p = _params(canonical=1)
        sA = _state("hash1")
        sA.nonCanLabel = "nonA"
        sB = _state("hash1")   # same canLabel
        sB.nonCanLabel = "nonB"

        model = _seed_model(p, sA)
        model.stateList.append(sA)

        self.assertFalse(model.checkUniqueness(sB))

    def test_noncanonical_on_same_noncan(self):
        """
        With non-canonical labelling, two states with the same nonCanLabel
        are duplicates even if they have different canLabels.
        """
        p = _params(canonical=0)
        sA = _state("hashA")
        sA.nonCanLabel = "shared"
        sB = _state("hashB")
        sB.nonCanLabel = "shared"

        model = _seed_model(p, sA)
        model.stateList.append(sA)

        self.assertFalse(model.checkUniqueness(sB))


# ---------------------------------------------------------------------------
# 5. Branching model: A → B and A → C
# ---------------------------------------------------------------------------

class TestBranchingModel(unittest.TestCase):

    def setUp(self):
        self.p = _params()
        self.sA = _state("A")
        self.sB = _state("B")
        self.sC = _state("C")
        self.tAB = _trans(self.sA, self.sB, "AB")
        self.tAC = _trans(self.sA, self.sC, "AC")

        self.model = _seed_model(self.p, self.sA)
        self.model.update(transitions=[self.tAB, self.tAC])

    def test_two_transitions(self):
        self.assertEqual(len(self.model), 2)

    def test_B_and_C_both_depth_one(self):
        self.assertEqual(self.model.findDepth(self.sB), 1)
        self.assertEqual(self.model.findDepth(self.sC), 1)


# ---------------------------------------------------------------------------
# 6. Transition energy barrier → rate consistency
# ---------------------------------------------------------------------------

class TestBarrierRateConsistency(unittest.TestCase):

    def test_forward_reverse_KRA_relationship(self):
        """KRA = (forward + reverse) / 2 — verify we can store consistent values."""
        t = Transition(_state("A"), _state("B"))
        t.forwardBarrier = 0.8
        t.reverseBarrier = 0.4
        t.KRA = (t.forwardBarrier + t.reverseBarrier) / 2
        t.dE = t.finalState.totalEnergy or 0 - (t.initialState.totalEnergy or 0)

        self.assertAlmostEqual(t.KRA, 0.6)

    def test_rate_at_high_temperature_approaches_prefactor(self):
        """As T → ∞, exp(-Ea / kT) → 1, so rate → prefactor."""
        from HopDec.Constants import boltzmann
        t = Transition(_state("A"), _state("B"))
        t.forwardBarrier = 0.1
        # At 1e8 K, kT ≈ 8617 eV >> 0.1 eV
        rate = t.calcRate(1e8, 1e13)
        self.assertAlmostEqual(rate, 1e13, delta=1e10)


# ---------------------------------------------------------------------------
# 7. Graph connectivity (Graphs module integration)
# ---------------------------------------------------------------------------

class TestGraphsIntegration(unittest.TestCase):

    def test_path_length_matches_model_depth(self):
        """shortestPath via Graphs module should agree with Model.findDepth."""
        p = _params()
        sA = _state("A")
        sB = _state("B")
        sC = _state("C")
        t12 = _trans(sA, sB, "AB")
        t23 = _trans(sB, sC, "BC")

        model = _seed_model(p, sA)
        model.update(transitions=[t12, t23])

        graph = buildNetwork(
            [s.nonCanLabel for s in model.stateList],
            [(t.initialState.nonCanLabel, t.finalState.nonCanLabel) for t in model.transitionList],
        )
        path_len = shortestPath(graph, "A", "C")
        self.assertEqual(path_len, model.findDepth(sC))

    def test_disconnected_node_infinite_depth(self):
        """A state with no path from initState should have infinite depth."""
        p = _params()
        sA = _state("A")
        sZ = _state("Z")   # completely disconnected

        model = _seed_model(p, sA)
        model.stateList.append(sZ)
        model.buildModelGraph()

        self.assertEqual(model.findDepth(sZ), np.inf)


# ---------------------------------------------------------------------------
# 8. Connection integrates with Model
# ---------------------------------------------------------------------------

class TestConnectionWithModel(unittest.TestCase):

    def test_connection_transitions_added_to_model(self):
        p = _params()
        sA = _state("A")
        sB = _state("B")

        conn = Connection(sA, sB)
        t = _trans(sA, sB, "AB")
        conn.transitions.append(t)

        model = _seed_model(p, sA)
        model.update(connections=[conn])

        self.assertEqual(len(model), 1)
        self.assertIn(sB, model.stateList)

    def test_connection_with_multiple_transitions(self):
        p = _params()
        sA = _state("A")
        sB = _state("B")
        sC = _state("C")

        conn = Connection(sA, sC)
        conn.transitions.append(_trans(sA, sB, "AB"))
        conn.transitions.append(_trans(sB, sC, "BC"))

        model = _seed_model(p, sA)
        model.update(connections=[conn])

        self.assertEqual(len(model), 2)


if __name__ == "__main__":
    unittest.main()
