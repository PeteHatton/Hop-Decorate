"""Tests for HopDec.Constants"""

import math
import unittest

from HopDec.Constants import boltzmann, massConversion, twoPi


class TestConstants(unittest.TestCase):

    def test_boltzmann_value(self):
        """Boltzmann constant must match the accepted eV/K value."""
        self.assertAlmostEqual(boltzmann, 8.6173324e-5, places=12)

    def test_boltzmann_positive(self):
        self.assertGreater(boltzmann, 0)

    def test_massConversion_value(self):
        """Mass conversion factor (AMU → MD units) must match expected value."""
        self.assertAlmostEqual(massConversion, 104.363024, places=5)

    def test_massConversion_positive(self):
        self.assertGreater(massConversion, 0)

    def test_twoPi_value(self):
        """twoPi must equal 2 * math.pi."""
        self.assertAlmostEqual(twoPi, 2.0 * math.pi, places=12)


if __name__ == "__main__":
    unittest.main()
