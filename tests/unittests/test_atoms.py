"""Tests for HopDec.Atoms — periodic-table data lookups."""

import unittest

import HopDec.Atoms as Atoms
from HopDec.Constants import massConversion


class TestAtomicLookups(unittest.TestCase):

    # ------------------------------------------------------------------
    # atomicNumber
    # ------------------------------------------------------------------

    def test_atomicNumber_iron(self):
        self.assertEqual(Atoms.atomicNumber("Fe"), 26)

    def test_atomicNumber_hydrogen(self):
        self.assertEqual(Atoms.atomicNumber("H"), 1)

    def test_atomicNumber_zirconium(self):
        self.assertEqual(Atoms.atomicNumber("Zr"), 40)

    def test_atomicNumber_unknown_raises(self):
        with self.assertRaises(KeyError):
            Atoms.atomicNumber("Xx")

    # ------------------------------------------------------------------
    # atomicMassAMU
    # ------------------------------------------------------------------

    def test_atomicMassAMU_hydrogen(self):
        """Hydrogen is ~1.008 AMU."""
        mass = Atoms.atomicMassAMU("H")
        self.assertAlmostEqual(mass, 1.008, places=1)

    def test_atomicMassAMU_iron(self):
        """Iron is ~55.845 AMU."""
        mass = Atoms.atomicMassAMU("Fe")
        self.assertAlmostEqual(mass, 55.845, places=1)

    def test_atomicMassAMU_positive(self):
        self.assertGreater(Atoms.atomicMassAMU("Fe"), 0)

    def test_atomicMassAMU_unknown_raises(self):
        with self.assertRaises(KeyError):
            Atoms.atomicMassAMU("Xx")

    # ------------------------------------------------------------------
    # atomicMass (MD units = AMU * massConversion)
    # ------------------------------------------------------------------

    def test_atomicMass_equals_amu_times_conversion(self):
        sym = "Fe"
        self.assertAlmostEqual(
            Atoms.atomicMass(sym),
            Atoms.atomicMassAMU(sym) * massConversion,
            places=8,
        )

    def test_atomicMass_positive(self):
        self.assertGreater(Atoms.atomicMass("Ni"), 0)

    # ------------------------------------------------------------------
    # atomName
    # ------------------------------------------------------------------

    def test_atomName_iron(self):
        self.assertEqual(Atoms.atomName("Fe"), "Iron")

    def test_atomName_hydrogen(self):
        self.assertEqual(Atoms.atomName("H"), "Hydrogen")

    def test_atomName_unknown_raises(self):
        with self.assertRaises(KeyError):
            Atoms.atomName("Xx")

    # ------------------------------------------------------------------
    # covalentRadius
    # ------------------------------------------------------------------

    def test_covalentRadius_returns_float(self):
        r = Atoms.covalentRadius("Fe")
        self.assertIsInstance(r, float)

    def test_covalentRadius_positive(self):
        self.assertGreater(Atoms.covalentRadius("Fe"), 0)

    def test_covalentRadius_unknown_raises(self):
        with self.assertRaises(KeyError):
            Atoms.covalentRadius("Xx")

    # ------------------------------------------------------------------
    # RGB
    # ------------------------------------------------------------------

    def test_RGB_returns_list_of_three(self):
        rgb = Atoms.RGB("Fe")
        self.assertEqual(len(rgb), 3)

    def test_RGB_values_in_range(self):
        rgb = Atoms.RGB("Fe")
        for component in rgb:
            self.assertGreaterEqual(component, 0.0)
            self.assertLessEqual(component, 1.0)

    def test_RGB_unknown_raises(self):
        with self.assertRaises(KeyError):
            Atoms.RGB("Xx")

    # ------------------------------------------------------------------
    # atomicSymbol (mass → symbol)
    # ------------------------------------------------------------------

    def test_atomicSymbol_iron(self):
        """Look up Iron by its exact AMU value from the data file."""
        amu = Atoms.atomicMassAMU("Fe")
        self.assertEqual(Atoms.atomicSymbol(amu), "Fe")

    def test_atomicSymbol_unknown_raises(self):
        with self.assertRaises(ValueError):
            Atoms.atomicSymbol(0.0)


if __name__ == "__main__":
    unittest.main()
