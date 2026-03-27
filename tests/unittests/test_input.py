"""Tests for HopDec.Input — XML config parsing and InputParams."""

import os
import sys
import tempfile
import textwrap
import unittest

from HopDec.Input import InputParams, getParams


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_xml(content: str) -> str:
    """Write XML string to a temporary file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False)
    f.write(content)
    f.close()
    return f.name


MINIMAL_XML = textwrap.dedent("""\
    <InputParams>
        <specieNamesString>Fe,Ni</specieNamesString>
        <staticSpeciesString>Fe</staticSpeciesString>
        <activeSpeciesString>Ni</activeSpeciesString>
        <concentrationString>0.5,0.5</concentrationString>
        <runTime>1000</runTime>
        <MDTemperature>300</MDTemperature>
        <MDTimestep>0.002</MDTimestep>
    </InputParams>
""")

SINGLE_SPECIES_XML = textwrap.dedent("""\
    <InputParams>
        <specieNamesString>Fe</specieNamesString>
        <staticSpeciesString></staticSpeciesString>
        <activeSpeciesString>Fe</activeSpeciesString>
        <concentrationString>1.0</concentrationString>
    </InputParams>
""")

BAD_CONCENTRATION_XML = textwrap.dedent("""\
    <InputParams>
        <specieNamesString>Fe,Ni</specieNamesString>
        <staticSpeciesString>Fe</staticSpeciesString>
        <activeSpeciesString>Ni</activeSpeciesString>
        <concentrationString>0.3,0.3</concentrationString>
    </InputParams>
""")


class TestInputParamsDefaults(unittest.TestCase):
    """InputParams() must initialise to safe, predictable defaults."""

    def setUp(self):
        self.p = InputParams()

    def test_runTime_default(self):
        self.assertEqual(self.p.runTime, 0)

    def test_MDTimestep_default(self):
        self.assertEqual(self.p.MDTimestep, 0.0)

    def test_MDTemperature_default(self):
        self.assertEqual(self.p.MDTemperature, 0)

    def test_nDefectsMax_default(self):
        self.assertEqual(self.p.nDefectsMax, 1)

    def test_canonicalLabelling_default(self):
        self.assertEqual(self.p.canonicalLabelling, 0)

    def test_atomStyle_default(self):
        self.assertEqual(self.p.atomStyle, "atomic")

    def test_randomSeed_default(self):
        self.assertEqual(self.p.randomSeed, 1234)

    def test_NEBmaxBarrier_default(self):
        import math
        self.assertTrue(math.isinf(self.p.NEBmaxBarrier))

    def test_maxDefectAtoms_default(self):
        self.assertEqual(self.p.maxDefectAtoms, -1)


class TestGetParams(unittest.TestCase):
    """getParams() must correctly parse valid XML."""

    def setUp(self):
        self.xml_path = _write_xml(MINIMAL_XML)
        self.p = getParams(self.xml_path)

    def tearDown(self):
        os.unlink(self.xml_path)

    # Scalar values
    def test_runTime_parsed(self):
        self.assertEqual(self.p.runTime, 1000)

    def test_MDTemperature_parsed(self):
        self.assertEqual(self.p.MDTemperature, 300)

    def test_MDTimestep_parsed_as_float(self):
        self.assertIsInstance(self.p.MDTimestep, float)
        self.assertAlmostEqual(self.p.MDTimestep, 0.002, places=6)

    # Species names
    def test_specieNames_list(self):
        self.assertEqual(self.p.specieNames, ["Fe", "Ni"])

    def test_staticSpecies_list(self):
        self.assertEqual(self.p.staticSpecies, ["Fe"])

    def test_activeSpecies_list(self):
        self.assertEqual(self.p.activeSpecies, ["Ni"])

    # Numbered types (1-indexed positions in specieNames)
    def test_staticSpeciesTypes(self):
        self.assertEqual(self.p.staticSpeciesTypes, [1])   # Fe is index 0 → type 1

    def test_activeSpeciesTypes(self):
        self.assertEqual(self.p.activeSpeciesTypes, [2])   # Ni is index 1 → type 2

    # Concentration
    def test_concentration_list(self):
        self.assertAlmostEqual(self.p.concentration[0], 0.5)
        self.assertAlmostEqual(self.p.concentration[1], 0.5)

    def test_concentration_sums_to_one(self):
        self.assertAlmostEqual(sum(self.p.concentration), 1.0, places=10)


class TestGetParamsSingleSpecies(unittest.TestCase):
    """Edge case: single species, empty static string."""

    def setUp(self):
        self.xml_path = _write_xml(SINGLE_SPECIES_XML)
        self.p = getParams(self.xml_path)

    def tearDown(self):
        os.unlink(self.xml_path)

    def test_single_species_name(self):
        self.assertEqual(self.p.specieNames, ["Fe"])

    def test_empty_staticSpecies_gives_empty_types(self):
        self.assertEqual(self.p.staticSpeciesTypes, [])

    def test_concentration_single(self):
        self.assertAlmostEqual(self.p.concentration[0], 1.0)


class TestGetParamsErrors(unittest.TestCase):
    """getParams() must raise/exit on bad inputs."""

    def test_missing_file_raises_ValueError(self):
        with self.assertRaises(ValueError):
            getParams("/nonexistent/path/config.xml")

    def test_invalid_xml_raises_ValueError(self):
        path = _write_xml("<not valid xml <<")
        try:
            with self.assertRaises(ValueError):
                getParams(path)
        finally:
            os.unlink(path)

    def test_bad_concentration_calls_sys_exit(self):
        path = _write_xml(BAD_CONCENTRATION_XML)
        try:
            with self.assertRaises(SystemExit):
                getParams(path)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
