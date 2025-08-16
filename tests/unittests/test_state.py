# test_atoms_in_sphere_triclinic.py
import numpy as np
import pytest

# ---- Helpers / stubs ---------------------------------------------------------

class DummyState:
    def __init__(self, pos, cellDims):
        self.pos = pos
        self.cellDims = cellDims

def frac_to_cart(frac, H):
    """
    Convert fractional coords (shape (...,3)) to Cartesian using row-major H=[a;b;c]
    where rows are lattice vectors in Cartesian coords.
    If frac is shape (3,), returns (3,). If (N,3) returns (N,3).
    """
    frac = np.asarray(frac, dtype=float)
    H = np.asarray(H, dtype=float).reshape(3, 3)
    return frac @ H  # row-vector convention

def brute_min_distance(p, c, H):
    """
    Brute-force minimum distance between point p and center c under PBC
    by checking the 27 neighbor images using a triclinic cell H (row-major).
    """
    # All integer shifts in {-1,0,1}^3
    shifts = np.array(np.meshgrid([-1,0,1], [-1,0,1], [-1,0,1], indexing="ij")).reshape(3,-1).T
    # Image vectors (cartesian)
    images = c + shifts @ H
    # Distances
    d2 = np.sum((p - images) ** 2, axis=1)
    return np.sqrt(d2.min())

# ---- Import the function under test -----------------------------------------
# Ensure your atomsInSphere implementation is importable, e.g.:
# from mymodule import atomsInSphere
# For demonstration here, we inline a placeholder import path:
from HopDec.State import atomsInSphere  # <-- replace with your actual module name

# ---- Test data: a strongly skewed triclinic cell -----------------------------
@pytest.fixture(scope="module")
def triclinic_H():
    # Deliberately skewed cell:
    # a = (10,  0, 0)
    # b = ( 3, 10, 0)   -> xy tilt
    # c = ( 2,  1, 9)   -> full 3D tilt
    H = np.array([
        [10.0, 0.0, 0.0],
        [ 3.0,10.0, 0.0],
        [ 2.0, 1.0, 9.0],
    ])
    return H

@pytest.fixture
def state_with_random_atoms(triclinic_H, rng_seed=1234):
    rng = np.random.default_rng(rng_seed)
    N = 200  # number of atoms
    # Random fractional positions in [0,1)^3
    frac = rng.random((N, 3))
    cart = frac_to_cart(frac, triclinic_H)
    # Flatten pos as expected by your State
    pos_flat = cart.reshape(-1).tolist()
    cellDims_flat = triclinic_H.reshape(-1).tolist()
    return DummyState(pos_flat, cellDims_flat), cart, triclinic_H

# ---- Tests -------------------------------------------------------------------

def test_triclinic_matches_bruteforce_inside_cell(state_with_random_atoms):
    """
    Random atoms + random center inside the primary triclinic cell.
    atomsInSphere must match a brute-force 27-image check.
    """
    state, cart, H = state_with_random_atoms
    rng = np.random.default_rng(42)

    # Random center in fractional space, then map to cart
    center_frac = rng.random(3)
    center = frac_to_cart(center_frac, H)

    # Choose a radius that is not trivial but includes some atoms
    radius = 3.5

    # Function under test
    got = atomsInSphere(state, center, radius)
    got_atom_indices = np.array(got) // 3  # convert x-index back to atom index

    # Brute-force ground truth
    truth = []
    for i, p in enumerate(cart):
        if brute_min_distance(p, center, H) <= radius + 1e-12:
            truth.append(i)
    truth = np.array(truth, dtype=int)

    assert np.array_equal(np.sort(got_atom_indices), np.sort(truth)), \
        "atomsInSphere disagrees with brute-force 27-image search for a triclinic cell."

def test_triclinic_matches_bruteforce_center_outside_cell(state_with_random_atoms):
    state, cart, H = state_with_random_atoms

    # Center far outside [0,1)^3 (fractional)
    center_frac = np.array([1.2, -0.3, 2.7])

    # --- FIX: wrap center into primary cell before converting to Cartesian ---
    center_frac_wrapped = center_frac - np.floor(center_frac)
    center = frac_to_cart(center_frac_wrapped, H)

    radius = 4.0

    got = atomsInSphere(state, center, radius)
    got_atom_indices = np.array(got) // 3

    truth = []
    for i, p in enumerate(cart):
        if brute_min_distance(p, center, H) <= radius + 1e-12:
            truth.append(i)
    truth = np.array(truth, dtype=int)

    assert np.array_equal(np.sort(got_atom_indices), np.sort(truth)), \
        "Center outside the cell should still yield correct triclinic minimum-image results."

def test_triclinic_edge_wrap_due_to_skew(triclinic_H):
    """
    Construct a deterministic edge case where the nearest image uses a skewed wrap.
    We place the center near the 'origin' in fractional space and one atom near the
    opposite corner; in a skewed cell the nearest image isn't a pure axis wrap.
    """
    H = triclinic_H

    # One atom near (0.98, 0.97, 0.96) in fractional
    atom_frac = np.array([0.98, 0.97, 0.96])
    atom_cart = frac_to_cart(atom_frac, H)

    # Center near (0.02, 0.03, 0.04) so the minimum-image displacement wraps across all axes
    center_frac = np.array([0.02, 0.03, 0.04])
    center = frac_to_cart(center_frac, H)

    # Put a couple of far-away atoms as distractors
    far_fracs = np.array([
        [0.50, 0.50, 0.50],
        [0.25, 0.75, 0.25],
        [0.10, 0.10, 0.90],
    ])
    far_carts = frac_to_cart(far_fracs, H)

    # Build state
    cart = np.vstack([atom_cart, far_carts])
    pos_flat = cart.reshape(-1).tolist()
    cellDims_flat = H.reshape(-1).tolist()
    state = DummyState(pos_flat, cellDims_flat)

    # Compute the true min distance for the first atom
    d_true = brute_min_distance(atom_cart, center, H)

    # Set radius just above the true distance to include only the first atom
    radius = d_true + 1e-6

    got = atomsInSphere(state, center, radius)
    got_atom_indices = np.array(got) // 3

    assert np.array_equal(got_atom_indices, np.array([0])), \
        "Skewed triclinic wrap failed: the near-corner atom should be included, distractors excluded."

def test_returns_flat_x_indices_convention(state_with_random_atoms):
    """
    The function should return x-component indices into the flattened pos array (i*3).
    Check a small hand-crafted system.
    """
    H = np.array([
        [8.0, 0.0, 0.0],
        [2.0, 7.0, 0.0],
        [1.0, 1.0, 6.0],
    ])
    # Two atoms: one at fractional origin, one at center
    fracs = np.array([
        [0.00, 0.00, 0.00],  # atom 0
        [0.50, 0.50, 0.50],  # atom 1
    ])
    cart = frac_to_cart(fracs, H)
    state = DummyState(cart.reshape(-1).tolist(), H.reshape(-1).tolist())

    # Center near atom 1
    center = cart[1] + np.array([0.1, -0.05, 0.02])
    radius = 0.2

    got = atomsInSphere(state, center, radius)
    # Should include only atom 1, and return index 1*3 = 3
    assert got == [3], f"Expected [3] (x-index for atom 1), got {got}"