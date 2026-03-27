"""
Pytest configuration for the HopDec test suite.

LAMMPS is an optional external dependency that requires a full compiled
installation.  We mock the `lammps` Python module here so that every
HopDec module can be imported in a normal CI / developer environment
without LAMMPS installed.  Tests that actually *exercise* LAMMPS
functionality should be marked @pytest.mark.lammps and skipped unless
the real module is available.
"""

import sys
from unittest.mock import MagicMock

# Only mock if the real package is not installed
try:
    import lammps  # noqa: F401
except ImportError:
    sys.modules["lammps"] = MagicMock()
