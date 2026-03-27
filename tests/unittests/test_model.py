"""
Tests for HopDec.Model — Model class unit tests.

Tests cover: init, len, checkUniqueness (canonical + non-canonical),
buildModelGraph, findDepth, and update.
"""

import unittest
import numpy as np

from HopDec.State import State
from HopDec.Transitions import Transition
from HopDec.Model import Model
from HopDec.Input import InputParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_params(canonical=0, max_depth=-1):
    p = InputParams()
    p.canonicalLabelling = canonical
    p.maxModelDepth = max_depth
    p.nDefectsMax = 100
    p.maxDefectAtoms = -1
    p.segmentLength = 100
    p.verbose = 0
    return p


def _make_state(can_label, non_can_label):
    s = State(2)
    s.cellDims = np.array([100.0, 0, 0, 0, 100.0, 0, 0, 0, 100.0])
    s.pos = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
    s.type = np.array([1, 2], dtype=np.int32)
    s.canLabel = can_label
    s.nonCanLabel = non_can_label
    s.defectPositions = np.array([])
    s.defectTypes = np.array([])
    s.defectIndices = np.array([])
    s.nDefects = 0
    s.doWork = 1
    s.time = 100
    return s


def _make_transition(s1, s2, label):
    t = Transition(s1, s2)
    t.canLabel = label
    t.nonCanLabel = label
    return t


def _bootstrap_model(params, init_state):
    """Create a Model and seed it with a single initial state."""
    model = Model(params)
    model.initState = init_state
    # Manually insert the init state so graph has at least one node
    model.stateList.append(init_state)
    model.buildModelGraph()
    return model


# ---------------------------------------------------------------------------
# Model.__init__
# ---------------------------------------------------------------------------

class TestModelInit(unittest.TestCase):

    def setUp(self):
        self.model = Model(_make_params())

    def test_stateList_empty(self):
        self.assertEqual(self.model.stateList, [])

    def test_transitionList_empty(self):
        self.assertEqual(self.model.transitionList, [])

    def test_initState_none(self):
        self.assertIsNone(self.model.initState)

    def test_graph_none(self):
        self.assertIsNone(self.model.graph)

    def test_len_zero(self):
        self.assertEqual(len(self.model), 0)


# ---------------------------------------------------------------------------
# Model.__len__
# ---------------------------------------------------------------------------

class TestModelLen(unittest.TestCase):

    def test_len_reflects_transitionList(self):
        model = Model(_make_params())
        s1 = _make_state("A", "A")
        s2 = _make_state("B", "B")
        model.transitionList.append(_make_transition(s1, s2, "T1"))
        self.assertEqual(len(model), 1)
        model.transitionList.append(_make_transition(s1, s2, "T2"))
        self.assertEqual(len(model), 2)


# ---------------------------------------------------------------------------
# Model.checkUniqueness — states
# ---------------------------------------------------------------------------

class TestCheckUniquenessStates(unittest.TestCase):

    def test_new_state_is_unique_noncanonical(self):
        model = Model(_make_params(canonical=0))
        s1 = _make_state("A", "nonA")
        self.assertTrue(model.checkUniqueness(s1))

    def test_duplicate_state_noncanonical(self):
        model = Model(_make_params(canonical=0))
        s1 = _make_state("A", "nonA")
        model.stateList.append(s1)
        s2 = _make_state("A", "nonA")  # same nonCanLabel
        self.assertFalse(model.checkUniqueness(s2))

    def test_different_noncanonical_labels_are_unique(self):
        model = Model(_make_params(canonical=0))
        s1 = _make_state("A", "nonA")
        model.stateList.append(s1)
        s2 = _make_state("A", "nonB")   # same can, different non-can
        self.assertTrue(model.checkUniqueness(s2))

    def test_new_state_is_unique_canonical(self):
        model = Model(_make_params(canonical=1))
        s1 = _make_state("hashA", "nonA")
        self.assertTrue(model.checkUniqueness(s1))

    def test_duplicate_state_canonical(self):
        model = Model(_make_params(canonical=1))
        s1 = _make_state("hashA", "nonA")
        model.stateList.append(s1)
        s2 = _make_state("hashA", "nonB")   # same canLabel but different nonCan
        self.assertFalse(model.checkUniqueness(s2))

    def test_different_canonical_labels_are_unique(self):
        model = Model(_make_params(canonical=1))
        s1 = _make_state("hashA", "nonA")
        model.stateList.append(s1)
        s2 = _make_state("hashB", "nonA")   # different canLabel
        self.assertTrue(model.checkUniqueness(s2))


# ---------------------------------------------------------------------------
# Model.checkUniqueness — transitions
# ---------------------------------------------------------------------------

class TestCheckUniquenessTransitions(unittest.TestCase):

    def test_new_transition_is_unique(self):
        model = Model(_make_params(canonical=0))
        s1, s2 = _make_state("A", "A"), _make_state("B", "B")
        t = _make_transition(s1, s2, "T1")
        self.assertTrue(model.checkUniqueness(t))

    def test_duplicate_transition_not_unique(self):
        model = Model(_make_params(canonical=0))
        s1, s2 = _make_state("A", "A"), _make_state("B", "B")
        t1 = _make_transition(s1, s2, "T1")
        model.transitionList.append(t1)
        t2 = _make_transition(s1, s2, "T1")
        self.assertFalse(model.checkUniqueness(t2))


# ---------------------------------------------------------------------------
# Model.buildModelGraph and findDepth
# ---------------------------------------------------------------------------

class TestBuildModelGraph(unittest.TestCase):

    def setUp(self):
        self.params = _make_params()
        self.s1 = _make_state("A", "A")
        self.s2 = _make_state("B", "B")
        self.s3 = _make_state("C", "C")
        self.t12 = _make_transition(self.s1, self.s2, "T12")
        self.t23 = _make_transition(self.s2, self.s3, "T23")

    def test_graph_has_correct_nodes(self):
        model = Model(self.params)
        model.stateList = [self.s1, self.s2]
        model.transitionList = [self.t12]
        model.buildModelGraph()
        self.assertIn("A", model.graph.nodes)
        self.assertIn("B", model.graph.nodes)

    def test_graph_has_correct_edges(self):
        model = Model(self.params)
        model.stateList = [self.s1, self.s2]
        model.transitionList = [self.t12]
        model.buildModelGraph()
        self.assertTrue(model.graph.has_edge("A", "B") or model.graph.has_edge("B", "A"))

    def test_isolated_node_in_graph(self):
        """A state with no transitions should still appear as a node."""
        model = Model(self.params)
        model.stateList = [self.s1, self.s3]
        model.transitionList = []
        model.buildModelGraph()
        self.assertIn("A", model.graph.nodes)
        self.assertIn("C", model.graph.nodes)


class TestFindDepth(unittest.TestCase):

    def setUp(self):
        self.params = _make_params()
        self.s1 = _make_state("A", "A")
        self.s2 = _make_state("B", "B")
        self.s3 = _make_state("C", "C")

    def _build(self, states, transitions):
        model = Model(self.params)
        model.initState = self.s1
        model.stateList = states
        model.transitionList = transitions
        model.buildModelGraph()
        return model

    def test_depth_to_self_is_zero(self):
        model = self._build([self.s1], [])
        self.assertEqual(model.findDepth(self.s1), 0)

    def test_depth_one_hop(self):
        t = _make_transition(self.s1, self.s2, "T12")
        model = self._build([self.s1, self.s2], [t])
        self.assertEqual(model.findDepth(self.s2), 1)

    def test_depth_two_hops(self):
        t12 = _make_transition(self.s1, self.s2, "T12")
        t23 = _make_transition(self.s2, self.s3, "T23")
        model = self._build([self.s1, self.s2, self.s3], [t12, t23])
        self.assertEqual(model.findDepth(self.s3), 2)

    def test_unreachable_state_gives_inf(self):
        model = self._build([self.s1, self.s3], [])   # no edge A→C
        self.assertEqual(model.findDepth(self.s3), np.inf)


# ---------------------------------------------------------------------------
# Model.update
# ---------------------------------------------------------------------------

class TestModelUpdate(unittest.TestCase):

    def setUp(self):
        self.params = _make_params()
        self.s1 = _make_state("A", "A")
        self.model = _bootstrap_model(self.params, self.s1)

    def test_update_adds_new_transition_and_state(self):
        s2 = _make_state("B", "B")
        t = _make_transition(self.s1, s2, "T12")
        result = self.model.update(transitions=[t])
        self.assertEqual(result, 1)
        self.assertEqual(len(self.model), 1)
        self.assertIn(s2, self.model.stateList)

    def test_update_duplicate_transition_not_added(self):
        s2 = _make_state("B", "B")
        t = _make_transition(self.s1, s2, "T12")
        self.model.update(transitions=[t])
        # Adding same transition again
        t2 = _make_transition(self.s1, s2, "T12")
        result = self.model.update(transitions=[t2])
        self.assertEqual(len(self.model), 1)   # still only 1

    def test_update_with_none_cleaned(self):
        """None entries in transitions list must be silently filtered."""
        result = self.model.update(transitions=[None, None])
        self.assertEqual(len(self.model), 0)

    def test_update_adds_state_directly(self):
        """update(states=[...]) with the init state already present should not double-add it."""
        count_before = len(self.model.stateList)
        self.model.update(states=[self.s1])
        # s1 is already in stateList, so nothing changes
        self.assertEqual(len(self.model.stateList), count_before)


if __name__ == "__main__":
    unittest.main()
