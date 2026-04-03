"""
Tests for HopDec.MD.main — unit tests using mocked LAMMPS.

LammpsInterface and Minimize are patched so that no real LAMMPS
installation is required. Tests cover: flag logic, state copying,
time tracking, parameter fallbacks, and loop termination.
"""

import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from HopDec.State import State
from HopDec.Input import InputParams
import HopDec.MD as MD


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_params(segment=100, temp=300, event_disp=0.5):
    p = InputParams()
    p.segmentLength = segment
    p.MDTemperature = temp
    p.eventDisplacement = event_disp
    p.verbose = 0
    return p


def _make_state():
    s = State(2)
    s.cellDims = np.array([10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0])
    s.pos = np.array([1.0, 1.0, 1.0, 5.0, 5.0, 5.0])
    s.type = np.array([1, 2], dtype=np.int32)
    s.canLabel = 'A'
    s.nonCanLabel = 'A'
    s.time = 0
    s.centroSyms = []
    s.defectPositions = np.array([])
    s.defectTypes = []
    s.defectIndices = []
    return s


# ---------------------------------------------------------------------------
# No-hop cases
# ---------------------------------------------------------------------------

class TestMDMainNoHop(unittest.TestCase):
    """maxMove stays below eventDisplacement — no transition detected."""

    @patch('HopDec.MD.Minimize')
    @patch('HopDec.MD.LammpsInterface')
    def test_flag_zero_when_below_threshold(self, MockLammps, MockMinimize):
        mock_lmp = MagicMock()
        MockLammps.return_value = mock_lmp
        mock_lmp.runMD.return_value = 0.1   # well below eventDisplacement=0.5

        params = _make_params()
        _, _, flag = MD.main(_make_state(), params, maxMDTime=params.segmentLength)

        self.assertEqual(flag, 0)

    @patch('HopDec.MD.Minimize')
    @patch('HopDec.MD.LammpsInterface')
    def test_minimize_not_called_when_no_hop(self, MockLammps, MockMinimize):
        mock_lmp = MagicMock()
        MockLammps.return_value = mock_lmp
        mock_lmp.runMD.return_value = 0.1

        MD.main(_make_state(), _make_params(), maxMDTime=_make_params().segmentLength)

        MockMinimize.main.assert_not_called()

    @patch('HopDec.MD.Minimize')
    @patch('HopDec.MD.LammpsInterface')
    def test_returns_lmp_instance(self, MockLammps, MockMinimize):
        mock_lmp = MagicMock()
        MockLammps.return_value = mock_lmp
        mock_lmp.runMD.return_value = 0.1

        params = _make_params()
        returned_lmp, _, _ = MD.main(_make_state(), params, maxMDTime=params.segmentLength)

        self.assertIs(returned_lmp, mock_lmp)


# ---------------------------------------------------------------------------
# Hop-detected cases
# ---------------------------------------------------------------------------

class TestMDMainHopDetected(unittest.TestCase):
    """maxMove exceeds eventDisplacement — transition is detected."""

    @patch('HopDec.MD.Minimize')
    @patch('HopDec.MD.LammpsInterface')
    def test_flag_one_when_above_threshold(self, MockLammps, MockMinimize):
        mock_lmp = MagicMock()
        MockLammps.return_value = mock_lmp
        mock_lmp.runMD.return_value = 1.0   # above eventDisplacement=0.5

        _, _, flag = MD.main(_make_state(), _make_params())

        self.assertEqual(flag, 1)

    @patch('HopDec.MD.Minimize')
    @patch('HopDec.MD.LammpsInterface')
    def test_minimize_called_once_on_hop(self, MockLammps, MockMinimize):
        mock_lmp = MagicMock()
        MockLammps.return_value = mock_lmp
        mock_lmp.runMD.return_value = 1.0

        MD.main(_make_state(), _make_params())

        MockMinimize.main.assert_called_once()

    @patch('HopDec.MD.Minimize')
    @patch('HopDec.MD.LammpsInterface')
    def test_minimize_receives_correct_lmp(self, MockLammps, MockMinimize):
        mock_lmp = MagicMock()
        MockLammps.return_value = mock_lmp
        mock_lmp.runMD.return_value = 1.0

        MD.main(_make_state(), _make_params())

        _, call_kwargs = MockMinimize.main.call_args
        self.assertIs(call_kwargs['lmp'], mock_lmp)

    @patch('HopDec.MD.Minimize')
    @patch('HopDec.MD.LammpsInterface')
    def test_loop_stops_immediately_after_hop(self, MockLammps, MockMinimize):
        """runMD should not be called a second time once flag is set."""
        mock_lmp = MagicMock()
        MockLammps.return_value = mock_lmp
        mock_lmp.runMD.return_value = 1.0

        MD.main(_make_state(), _make_params())

        mock_lmp.runMD.assert_called_once()


# ---------------------------------------------------------------------------
# State copying and time tracking
# ---------------------------------------------------------------------------

class TestMDMainStateHandling(unittest.TestCase):

    @patch('HopDec.MD.Minimize')
    @patch('HopDec.MD.LammpsInterface')
    def test_returned_state_is_not_input_state(self, MockLammps, MockMinimize):
        """MD works on a copy; the returned state must be a different object."""
        mock_lmp = MagicMock()
        MockLammps.return_value = mock_lmp
        mock_lmp.runMD.return_value = 0.1

        params = _make_params()
        state = _make_state()
        _, returned_state, _ = MD.main(state, params, maxMDTime=params.segmentLength)

        self.assertIsNot(returned_state, state)

    @patch('HopDec.MD.Minimize')
    @patch('HopDec.MD.LammpsInterface')
    def test_input_state_time_advances_by_one_segment(self, MockLammps, MockMinimize):
        mock_lmp = MagicMock()
        MockLammps.return_value = mock_lmp
        mock_lmp.runMD.return_value = 0.1

        params = _make_params(segment=100)
        state = _make_state()

        MD.main(state, params, maxMDTime=params.segmentLength)

        self.assertEqual(state.time, 100)

    @patch('HopDec.MD.Minimize')
    @patch('HopDec.MD.LammpsInterface')
    def test_input_state_time_advances_over_multiple_iterations(self, MockLammps, MockMinimize):
        mock_lmp = MagicMock()
        MockLammps.return_value = mock_lmp
        mock_lmp.runMD.return_value = 0.1

        params = _make_params(segment=100)
        state = _make_state()

        MD.main(state, params, maxMDTime=3 * params.segmentLength)

        self.assertEqual(state.time, 300)


# ---------------------------------------------------------------------------
# Parameter fallbacks and loop control
# ---------------------------------------------------------------------------

class TestMDMainParameters(unittest.TestCase):

    @patch('HopDec.MD.Minimize')
    @patch('HopDec.MD.LammpsInterface')
    def test_uses_params_segmentLength_by_default(self, MockLammps, MockMinimize):
        mock_lmp = MagicMock()
        MockLammps.return_value = mock_lmp
        mock_lmp.runMD.return_value = 0.1

        params = _make_params(segment=200)
        MD.main(_make_state(), params, maxMDTime=params.segmentLength)

        args, _ = mock_lmp.runMD.call_args
        self.assertEqual(args[1], 200)

    @patch('HopDec.MD.Minimize')
    @patch('HopDec.MD.LammpsInterface')
    def test_uses_params_temperature_by_default(self, MockLammps, MockMinimize):
        mock_lmp = MagicMock()
        MockLammps.return_value = mock_lmp
        mock_lmp.runMD.return_value = 0.1

        params = _make_params(temp=750)
        MD.main(_make_state(), params, maxMDTime=params.segmentLength)

        _, kwargs = mock_lmp.runMD.call_args
        self.assertEqual(kwargs['T'], 750)

    @patch('HopDec.MD.Minimize')
    @patch('HopDec.MD.LammpsInterface')
    def test_custom_segmentLength_passed_to_runMD(self, MockLammps, MockMinimize):
        mock_lmp = MagicMock()
        MockLammps.return_value = mock_lmp
        mock_lmp.runMD.return_value = 0.1

        params = _make_params(segment=100)
        MD.main(_make_state(), params, segmentLength=50, maxMDTime=100)

        args, _ = mock_lmp.runMD.call_args
        self.assertEqual(args[1], 50)

    @patch('HopDec.MD.Minimize')
    @patch('HopDec.MD.LammpsInterface')
    def test_provided_lmp_is_used_directly(self, MockLammps, MockMinimize):
        """When an lmp is passed in, LammpsInterface should not be instantiated."""
        existing_lmp = MagicMock()
        existing_lmp.runMD.return_value = 0.1

        params = _make_params()
        MD.main(_make_state(), params, lmp=existing_lmp, maxMDTime=params.segmentLength)

        MockLammps.assert_not_called()
        existing_lmp.runMD.assert_called_once()

    @patch('HopDec.MD.Minimize')
    @patch('HopDec.MD.LammpsInterface')
    def test_maxMDTime_stops_loop_at_correct_iteration_count(self, MockLammps, MockMinimize):
        """Loop must stop after maxMDTime / segmentLength iterations with no hop."""
        mock_lmp = MagicMock()
        MockLammps.return_value = mock_lmp
        mock_lmp.runMD.return_value = 0.1

        params = _make_params(segment=100)
        MD.main(_make_state(), params, maxMDTime=200)

        self.assertEqual(mock_lmp.runMD.call_count, 2)


if __name__ == '__main__':
    unittest.main()
