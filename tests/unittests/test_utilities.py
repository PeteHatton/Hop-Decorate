"""Tests for HopDec.Utilities."""

import io
import os
import tempfile
import unittest
from unittest.mock import patch

from HopDec.Utilities import log, writeTerminalBlock, writeLAMMPSDataFile


class TestLog(unittest.TestCase):

    def _capture_log(self, caller, message, indent=0):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            log(caller, message, indent)
        return buf.getvalue()

    def test_log_contains_message(self):
        out = self._capture_log("TestModule", "hello world")
        self.assertIn("hello world", out)

    def test_log_contains_caller(self):
        out = self._capture_log("TestModule", "msg")
        self.assertIn("TestModule", out)

    def test_log_strips_HopDec_prefix(self):
        """'HopDec.State' should appear as 'State' in the output."""
        out = self._capture_log("HopDec.State", "msg")
        self.assertIn("State", out)
        self.assertNotIn("HopDec.State", out)

    def test_log_contains_timestamp(self):
        out = self._capture_log("Caller", "msg")
        # Timestamps contain colons and slashes
        self.assertIn(":", out)

    def test_log_indent_adds_spaces(self):
        out_no_indent = self._capture_log("C", "m", indent=0)
        out_indented = self._capture_log("C", "m", indent=2)
        # The indented line should be longer
        self.assertGreater(len(out_indented), len(out_no_indent))

    def test_log_newline_at_end(self):
        out = self._capture_log("C", "m")
        self.assertTrue(out.endswith("\n"))


class TestWriteTerminalBlock(unittest.TestCase):

    def _capture(self, message):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            writeTerminalBlock(message)
        return buf.getvalue()

    def test_contains_message(self):
        out = self._capture("Test Message")
        self.assertIn("Test Message", out)

    def test_first_line_is_100_hashes(self):
        out = self._capture("x")
        first_line = out.split("\n")[0]
        self.assertEqual(first_line, "#" * 100)

    def test_last_non_empty_line_is_100_hashes(self):
        out = self._capture("x")
        lines = [l for l in out.split("\n") if l]
        self.assertEqual(lines[-1], "#" * 100)

    def test_two_full_hash_lines(self):
        """Top and bottom borders are solid hashes; the middle line mixes hashes and message."""
        out = self._capture("x")
        hash_lines = [l for l in out.split("\n") if l and all(c == "#" for c in l)]
        self.assertEqual(len(hash_lines), 2)


class TestWriteLAMMPSDataFile(unittest.TestCase):

    def _write_and_read(self, NAtoms, NSpecies, cellDims, types, positions):
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".dat", delete=False)
        f.close()
        try:
            writeLAMMPSDataFile(f.name, NAtoms, NSpecies, cellDims, types, positions)
            with open(f.name) as fh:
                return fh.read()
        finally:
            os.unlink(f.name)

    def setUp(self):
        self.NAtoms = 2
        self.NSpecies = 2
        self.cellDims = [10.0, 0, 0, 0, 8.0, 0, 0, 0, 6.0]
        self.types = [1, 2]
        self.positions = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        self.content = self._write_and_read(
            self.NAtoms, self.NSpecies, self.cellDims, self.types, self.positions
        )

    def test_atom_count_line(self):
        self.assertIn("2 atoms", self.content)

    def test_atom_types_line(self):
        self.assertIn("2 atom types", self.content)

    def test_xlo_xhi(self):
        self.assertIn("0.0 10.0 xlo xhi", self.content)

    def test_ylo_yhi(self):
        self.assertIn("0.0 8.0 ylo yhi", self.content)

    def test_zlo_zhi(self):
        self.assertIn("0.0 6.0 zlo zhi", self.content)

    def test_atoms_section_header(self):
        self.assertIn("Atoms", self.content)

    def test_atom_count_in_file(self):
        atom_lines = [
            l for l in self.content.split("\n")
            if l.strip() and l.strip()[0].isdigit() and len(l.split()) == 5
        ]
        self.assertEqual(len(atom_lines), self.NAtoms)

    def test_first_atom_position(self):
        self.assertIn("1.0 2.0 3.0", self.content)

    def test_second_atom_position(self):
        self.assertIn("4.0 5.0 6.0", self.content)

    def test_atom_types_written(self):
        # Type 1 and type 2 should both appear in atom lines
        self.assertIn(" 1 ", self.content)
        self.assertIn(" 2 ", self.content)


if __name__ == "__main__":
    unittest.main()
