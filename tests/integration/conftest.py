"""
Pytest configuration for HopDec integration tests.

Integration tests are split into two categories:
  - I/O tests: no LAMMPS required (just file parsing)
  - LAMMPS tests: require real LAMMPS Python bindings

The `skip_no_lammps` marker causes LAMMPS-dependent tests to be skipped
automatically when the bindings are not installed.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
ZR_DIR = os.path.join(REPO_ROOT, "examples/main-functionality/Zr")
NEB_DIR = os.path.join(REPO_ROOT, "examples/interactive-mode/neb")


def _real_lammps_available():
    """Return True only when the real LAMMPS Python package is importable."""
    try:
        import lammps as _lammps_mod
        # If the module is a MagicMock (injected below or elsewhere), it's not real.
        return not isinstance(_lammps_mod, MagicMock)
    except ImportError:
        return False


# Mock lammps when it isn't installed so that HopDec modules can still be
# imported for I/O tests.  The mock is injected before any HopDec import.
try:
    import lammps  # noqa: F401
except ImportError:
    sys.modules["lammps"] = MagicMock()


LAMMPS_AVAILABLE = _real_lammps_available()

skip_no_lammps = pytest.mark.skipif(
    not LAMMPS_AVAILABLE,
    reason="LAMMPS Python bindings not installed",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def zr_dir(monkeypatch):
    """Change CWD to the Zr example directory for the duration of a test."""
    monkeypatch.chdir(ZR_DIR)
    return ZR_DIR


@pytest.fixture
def neb_dir(monkeypatch):
    """Change CWD to the NEB example directory for the duration of a test."""
    monkeypatch.chdir(NEB_DIR)
    return NEB_DIR
