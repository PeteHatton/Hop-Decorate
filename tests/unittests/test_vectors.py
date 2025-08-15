import unittest
import numpy as np
from HopDec.Vectors import *

def ref_connectivity(pos, cutoff, cellDims):
    """Slow, brute-force reference using the user's minimum-image logic."""
    R = np.asarray(pos, dtype=float).reshape(-1, 3)
    N = len(R)
    pairs = []
    for i in range(N - 1):
        for j in range(i + 1, N):
            if distance(R[i], R[j], cellDims) <= cutoff:
                pairs.append([i, j])
    return pairs

def as_sorted_pairs(pairs):
    """Sort for comparison deterministically."""
    return sorted([tuple(p) for p in pairs])

def hexagonal_like_cell(a=10.0, c=10.0, angle_deg=60.0):
    """
    Skew (non-orthogonal) cell:
    a along x
    b in x-y plane at 'angle_deg' from a
    c along z
    """
    angle = np.deg2rad(angle_deg)
    b = np.array([a * np.cos(angle), a * np.sin(angle), 0.0])
    return np.vstack([np.array([a, 0.0, 0.0]), b, np.array([0.0, 0.0, c])])


def frac_to_cart(frac_pts, cell):
    """Convert fractional -> Cartesian for [a; b; c] as rows. Accepts (...,3)."""
    frac = np.asarray(frac_pts, dtype=float)
    cell = np.asarray(cell, dtype=float)
    # Row-vector convention: cart = frac @ cell
    return frac @ cell

def orthorhombic_cell(Lx=10.0, Ly=10.0, Lz=10.0):
    """Return a 3x3 cell with lattice vectors as rows [a; b; c]."""
    return np.array([[Lx, 0.0, 0.0],
                    [0.0, Ly, 0.0],
                    [0.0, 0.0, Lz]])

def is_wrapped_inside(cart, cell, atol=1e-9):
    """
    Check COM is wrapped inside the unit cell: 0 <= fractional < 1 (within tol).
    Convert via fractional = cart @ cell^{-1} with row-vector convention.
    """
    cart = np.asarray(cart, dtype=float)
    cell = np.asarray(cell, dtype=float)
    frac = cart @ np.linalg.inv(cell)
    return np.all(frac >= -atol) and np.all(frac < 1.0 + atol)

class TestVectors(unittest.TestCase):
    
    ''' Vectors.distance '''
    def test_distance_without_periodic(self):
        pos1 = [1.0, 2.0, 3.0]
        pos2 = [4.0, 5.0, 6.0]
        cellDims = [10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]
        
        self.assertAlmostEqual(distance(pos1, pos2, cellDims), np.sqrt(27), places=5)
        
    def test_distance_with_periodic_1(self):
        pos1 = [1.0, 2.0, 3.0]
        pos2 = [9.0, 9.0, 9.0]
        cellDims = [10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]

        self.assertAlmostEqual(distance(pos1, pos2, cellDims), np.sqrt(29), places=5)

    def test_distance_with_periodic_2(self):
        pos1 = [1.0, 1.0, 1.0]
        pos2 = [3.0, 9.0, 9.0]
        cellDims = [10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]

        self.assertAlmostEqual(distance(pos1, pos2, cellDims), np.sqrt(12), places=5)

    def test_distance_with_periodic_edge(self):
        pos1 = [0.0, 0.0, 0.0]
        pos2 = [10.0, 10.0, 10.0]
        cellDims = [10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]

        self.assertAlmostEqual(distance(pos1, pos2, cellDims), 0, places=5)

    def test_distance_with_prism_without_periodic(self):
        pos1 = [1.0, 2.0, 3.0]
        pos2 = [4.0, 5.0, 6.0]
        cellDims = [10.0, 0, 0, -3, 10.0, 0, 0, 0, 10.0]
        
        self.assertAlmostEqual(distance(pos1, pos2, cellDims), np.sqrt(27), places=5)

    def test_distance_with_prism_with_periodic(self):
        pos1 = [20.9467, 9.51674, 11.5123]
        pos2 = [-6.62855, 14.8236, 10.5677]
        cellDims = [27.5753, 0, 0, -13.7876, 23.8809, 0, 0, 0, 22.5865]
        
        self.assertAlmostEqual(distance(pos1, pos2, cellDims), 5.39027, places=5)

    ''' Vectors.randomVector '''
    def test_randomVector(self):
        N = 3
        vec = randomVector(N, randomSeed=42)
        self.assertEqual(len(vec), N)
        mag = magnitude(vec)
        self.assertAlmostEqual(mag, 1, places=5)  # Vector should be normalized
    
    ''' Vectors.normalise '''
    def test_normalise(self):
        vect = [3, 4, 0]
        norm_vect = normalise(vect)
        self.assertAlmostEqual(magnitude(norm_vect), 1, places=5)
    
    def test_normalise_zero_vector(self):
        vect = [0, 0, 0]
        norm_vect = normalise(vect)
        self.assertEqual(vect, norm_vect)
    
    ''' Vectors.magnitude '''
    def test_magnitude(self):
        vect = [3, 4, 0]
        self.assertEqual(magnitude(vect), 5)
    
    ''' Vectors.displacement '''
    def test_displacement(self):
        v1 = [1.0, 2.0, 3.0]
        v2 = [9.0, 9.0, 9.0]
        cellDims = [10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]
        displacement_vec = displacement(v1, v2, cellDims)
        np.testing.assert_almost_equal(displacement_vec, [-2.0, -3.0, -4.0], decimal=5)
    
    def test_maxMoveAtom(self):
        class MockState:
            def __init__(self, pos, cellDims, NAtoms):
                self.pos = pos
                self.cellDims = cellDims
                self.NAtoms = NAtoms
        
        pos1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        pos2 = [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]
        cellDims = [10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]
        
        s1 = MockState(pos1, cellDims, 2)
        s2 = MockState(pos2, cellDims, 2)
        
        max_move = maxMoveAtom(s1, s2)[0]
        self.assertAlmostEqual(max_move, np.sqrt(27), places=5)
    
    def test_maxMoveAtomPos(self):
        pos1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        pos2 = [4.0, 5.0, 6.0, 1.0, 2.0, 3.0]
        cellDims = [10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]
        
        max_move = maxMoveAtomPos(pos1, pos2, cellDims)
        self.assertAlmostEqual(max_move, np.sqrt(27), places=5)
    
    def test_COM_periodic(self):
        points = np.array([1.0, 1.0, 1.0, 9.0, 9.0, 9.0])
        cellDims = [10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]
        
        com = COM(points, cellDims)
        np.testing.assert_almost_equal(com, [0.0, 0.0, 0.0], decimal=5)

    def test_COM_realistic(self):
        points = np.array([1.53e-15, 12.6525, 16.2675, 1.66e-15, 10.845, 18.075, 1.57e-15, 12.6525, 19.8825, 1.79e-15, 14.46, 18.075, 18.075, 12.6525, 16.2675, 19.8825, 10.845, 16.2675, 18.075, 10.845, 18.075, 18.075, 12.6525, 19.8825, 19.8825, 10.845, 19.8825, 19.8825, 14.46, 16.2675])
        cellDims = [21.69, 0, 0, 0, 21.69, 0, 0, 0, 21.69]
        
        com = COM(points, cellDims)
        np.testing.assert_almost_equal(com, [20.06325, 12.291  , 17.89425], decimal=5)
    
    def test_COM_realistic_triclinic(self):
        points = np.array([27.5753, 2.65343, 0.00171535, 27.5753, 2.65343, 2.8216, 22.9794, 2.65343, 0.00171535, 22.9794, 2.65343, 2.8216, 25.2773, 6.63358, 0.00171535, 29.8732, -1.32672, 0.00171535, 20.6815, -1.32672, 0.00171535, 25.2773, -1.32672, 0.00171535, 25.2773, -1.32672, 2.8216, 25.2773, 1.32672, 3.76614, 27.5753, 2.65343, -4.70724, 22.9794, 2.65343, -4.70724, 25.2773, 1.32672, -3.76271, 25.2773, -1.32672, -4.70724, 29.8732, 1.32672, -0.942821, 20.6815, 1.32672, -0.942821, 25.2773, 1.32672, -0.942821, 27.5753, 5.30686, -0.942821, 22.9794, 5.30686, -0.942821, 22.9794, -2.65343, -0.942821])
        cellDims = [27.5753, 0, 0, 13.7876, 23.8809, 0, 0, 0, 22.5865]
        
        com = COM(points, cellDims)
        np.testing.assert_almost_equal(com, [25.16243,  1.52572, 22.02149], decimal=5)

    def test_findConnectivity(self):
        pos = [1.0, 2.0, 3.0, 1.5, 2.5, 3.5, 8.0, 8.0, 8.0]
        cutoff = 1.0
        cellDims = [10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]
        
        connected_points = findConnectivity(pos, cutoff, cellDims)
        expected_connections = [[0, 1]]
        self.assertEqual(connected_points, expected_connections)
    
    def test_findConnectivity_periodic(self):
        pos = [1.0, 2.0, 3.0, 9.5, 2.0, 3.0]  # Close when considering periodic boundaries
        cutoff = 2.0
        cellDims = [10.0, 0, 0, 0, 10.0, 0, 0, 0, 10.0]
        
        connected_points = findConnectivity(pos, cutoff, cellDims)
        expected_connections = [[0, 1]]  # These points should be connected
        self.assertEqual(connected_points, expected_connections)

    def test_findConnectivity_realistic(self):
        pos = [1.54e-15, 9.0375, 16.2675, 1.78e-15, 7.23, 18.075, 1.58e-15, 9.0375, 19.8825, 1.66e-15, 10.845, 18.075, 18.075, 9.0375, 16.2675, 19.8825, 7.23, 16.2675, 18.075, 7.23, 18.075, 18.075, 9.0375, 19.8825, 19.8825, 7.23, 19.8825, 19.8825, 10.845, 16.2675, 18.075, 10.845, 18.075, 19.8825, 10.845, 19.8825, 9.0375, 7.23, 5.4225, 9.0375, 9.0375, 3.615, 9.0375, 9.0375, 7.23, 9.0375, 10.845, 5.4225, 10.845, 7.23, 3.615, 12.6525, 7.23, 5.4225, 12.6525, 9.0375, 3.615, 10.845, 7.23, 7.23, 12.6525, 9.0375, 7.23, 10.845, 10.845, 3.615, 12.6525, 10.845, 5.4225, 10.845, 10.845, 7.23]
        cutoff = 2.7
        cellDims = [21.69, 0, 0, 0, 21.69, 0, 0, 0, 21.69]

        connected_points = findConnectivity(pos, cutoff, cellDims)
        expected_connections = [[0, 1], [0, 3], [0, 5], [0, 9], [1, 2], [1, 5], [1, 8], [2, 3], [2, 8], [2, 11], 
         [3, 9], [3, 11], [4, 5], [4, 6], [4, 9], [4, 10], [5, 6], [6, 7], [6, 8], [7, 8], 
         [7, 10], [7, 11], [9, 10], [10, 11], [12, 13], [12, 14], [12, 16], [12, 19], 
         [13, 15], [13, 16], [13, 21], [14, 15], [14, 19], [14, 23], [15, 21], [15, 23], 
         [16, 17], [16, 18], [17, 18], [17, 19], [17, 20], [18, 21], [18, 22], [19, 20], 
         [20, 22], [20, 23], [21, 22], [22, 23]]  # These points should be connected
        self.assertEqual(connected_points, expected_connections)

class TestTriclinicBoxes(unittest.TestCase):
    """Additional tests focusing on triclinic (fully skewed) unit cells.
    The cell is supplied as a 3x3 row-major matrix: [ax, ay, az, bx, by, bz, cx, cy, cz].
    """
    # Set up a generic triclinic cell:
    # a = (10, 0, 0)
    # b = ( 2,10, 0)
    # c = ( 3, 4,10)
    triclinic = [10.0, 0.0, 0.0,
                  2.0,10.0, 0.0,
                  3.0, 4.0,10.0]

    def test_distance_triclinic_minimum_image(self):
        # Two points that are close across all three boundaries.
        r1 = [0.5, 0.5, 0.5]
        # Put r2 just outside the box in all three directions to force wrapping.
        r2 = [10.5, 10.5, 10.5]

        # Expected: compute minimum-image displacement using lattice vectors.
        a = np.array([10.0, 0.0, 0.0])
        b = np.array([ 2.0,10.0, 0.0])
        c = np.array([ 3.0, 4.0,10.0])
        H = np.column_stack([a,b,c])  # 3x3 cell matrix with columns as lattice vectors
        Hinv = np.linalg.inv(H)

        d_cart = np.array(r2) - np.array(r1)
        d_frac = Hinv @ d_cart                      # into fractional coords
        d_frac -= np.round(d_frac)                  # wrap to [-0.5,0.5)
        d_min = H @ d_frac                          # back to Cartesian

        expected = np.linalg.norm(d_min)
        self.assertAlmostEqual(distance(r1, r2, self.triclinic), expected, places=6)

    def test_displacement_triclinic_antisymmetry(self):
        r1 = [8.0, 1.0, 9.0]
        r2 = [2.0, 9.0, 3.0]
        d12 = displacement(r1, r2, self.triclinic)
        d21 = displacement(r2, r1, self.triclinic)
        np.testing.assert_allclose(d12, -np.array(d21), atol=1e-8)

    def test_distance_triclinic_boundary_zero(self):
        # Same physical position expressed with a cell-vector offset should have zero distance.
        r1 = [0.0, 0.0, 0.0]
        # r2 = r1 + a + b + c (i.e. outside by one of each lattice vector)
        r2 = [10.0+2.0+3.0, 0.0+10.0+4.0, 0.0+0.0+10.0]
        self.assertAlmostEqual(distance(r1, r2, self.triclinic), 0.0, places=12)

    def test_distance_triclinic_reduces_to_orthorhombic(self):
        # If off-diagonals are zero, should match the simple orthorhombic calculation.
        ortho = [10.0,0,0, 0,20.0,0, 0,0,30.0]
        r1 = [1.0, 2.0, 3.0]
        r2 = [11.0, 18.0, -24.0]  # separated by (+10,-4,-27) -> wrapped to (0, -4, 3)
        self.assertAlmostEqual(distance(r1, r2, ortho), np.linalg.norm([0.0, -4.0, 3.0]), places=12)

class TestMaxMoveAtomTriclinic(unittest.TestCase):
    @staticmethod
    def _state(positions, cellDims):
        class State:
            def __init__(self, pos, cellDims):
                self.pos = pos
                self.cellDims = cellDims
        return State(positions, cellDims)

    def setUp(self):
        # Triclinic cell: rows are lattice vectors a, b, c (row-major)
        # a = (10, 0, 0), b = (2, 10, 0), c = (3, 4, 10)
        self.H = [10.0, 2.0, 3.0,
                  0.0, 10.0, 4.0,
                  0.0, 0.0, 10.0]

    def test_maxMoveAtom_triclinic_basic(self):
        # Small displacements inside the same image
        pos1 = [[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]]
        pos2 = [[1.4, 1.7, 3.3],   # ~0.64 distance
                [4.0, 5.8, 5.0]]   # larger move
        s1 = self._state(pos1, self.H)
        s2 = self._state(pos2, self.H)

        # Compute expected per-atom PBC-aware distances using distance()
        d0 = distance(pos1[0], pos2[0], self.H)
        d1 = distance(pos1[1], pos2[1], self.H)
        exp_max = max(d0, d1)
        exp_idx = 0 if d0 >= d1 else 1

        got_d, got_i = maxMoveAtom(s1, s2)
        self.assertAlmostEqual(got_d, exp_max, places=12)
        self.assertEqual(got_i, exp_idx)

    def test_maxMoveAtom_triclinic_wrap_across_a_boundary(self):
        # Atom 0 crosses the +a boundary (x ~ 10), minimal image should be small
        pos1 = [[9.8, 0.0, 0.0],
                [2.0, 2.0, 2.0]]
        pos2 = [[0.3, 0.0, 0.0],   # wrapped Δ near +0.5 along a
                [2.0, 2.0, 3.0]]   # 1 Å along z
        s1 = self._state(pos1, self.H)
        s2 = self._state(pos2, self.H)

        d0 = distance(pos1[0], pos2[0], self.H)  # should be ~0.5
        d1 = distance(pos1[1], pos2[1], self.H)  # should be 1.0
        exp_max = max(d0, d1)
        exp_idx = 0 if d0 >= d1 else 1

        got_d, got_i = maxMoveAtom(s1, s2)
        self.assertAlmostEqual(got_d, exp_max, places=12)
        self.assertEqual(got_i, exp_idx)

    def test_maxMoveAtom_invariance_under_lattice_translation(self):
        # Adding integer combinations of *row-major* lattice vectors to pos2
        # must not change the PBC distance.
        # Rows of H are a, b, c
        Hm = np.asarray(self.H, float).reshape(3, 3)
        a = Hm[0].tolist()
        b = Hm[1].tolist()
        c = Hm[2].tolist()

        def add(v, w):
            return [v[0]+w[0], v[1]+w[1], v[2]+w[2]]

        pos1 = [[1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0]]
        pos2_base = [[1.2, 1.9, 3.1],
                    [4.0, 5.0, 6.0]]  # identical to atom 1 in pos1

        # Shift atom 1 in pos2 by a + b - c (row-major lattice vectors)
        pos2_shifted = [pos2_base[0], add(add(add(pos2_base[1], a), b), [-c[0], -c[1], -c[2]])]

        s1 = self._state(pos1, self.H)
        s2 = self._state(pos2_shifted, self.H)

        d0 = distance(pos1[0], pos2_base[0], self.H)      # small, nonzero
        d1_base = distance(pos1[1], pos2_base[1], self.H) # exactly zero
        d1_shift = distance(pos1[1], pos2_shifted[1], self.H)

        # The lattice-translated atom should remain at zero distance under PBC,
        # and the overall max should come from atom 0.
        self.assertAlmostEqual(d1_base, 0.0, places=12)
        self.assertAlmostEqual(d1_shift, 0.0, places=12)

        got_d, got_i = maxMoveAtom(s1, s2)
        self.assertAlmostEqual(got_d, d0, places=12)
        self.assertEqual(got_i, 0)

    def test_maxMoveAtom_selects_correct_atom_among_many(self):
        # Craft several atoms with increasing movement; ensure the max and index are correct.
        pos1 = [[0.1, 0.2, 0.3],
                [1.0, 2.0, 3.0],
                [2.0, 4.0, 6.0],
                [9.9, 0.0, 0.0]]  # near +a boundary
        pos2 = [[0.2, 0.2, 0.3],    # tiny move
                [1.0, 2.0, 4.0],    # 1 Å along z
                [1.8, 3.0, 5.0],    # larger move
                [0.2, 0.0, 0.0]]    # wraps across a; ~0.3 distance
        s1 = self._state(pos1, self.H)
        s2 = self._state(pos2, self.H)

        dists = [distance(p1, p2, self.H) for p1, p2 in zip(pos1, pos2)]
        exp_idx = int(np.argmax(dists))
        exp_max = dists[exp_idx]

        got_d, got_i = maxMoveAtom(s1, s2)
        self.assertAlmostEqual(got_d, exp_max, places=12)
        self.assertEqual(got_i, exp_idx)

    def test_maxMoveAtom_tie_breaks_to_first_max(self):
        # Two atoms moved by exactly the same (PBC) distance; np.argmax should pick the first.
        pos1 = [[0.0, 0.0, 0.0],
                [5.0, 5.0, 5.0]]
        pos2 = [[0.5, 0.0, 0.0],
                [5.0, 5.5, 5.0]]  # both 0.5 Å moves under this cell
        s1 = self._state(pos1, self.H)
        s2 = self._state(pos2, self.H)

        d0 = distance(pos1[0], pos2[0], self.H)
        d1 = distance(pos1[1], pos2[1], self.H)
        self.assertAlmostEqual(d0, d1, places=12)

        got_d, got_i = maxMoveAtom(s1, s2)
        self.assertAlmostEqual(got_d, d0, places=12)
        self.assertEqual(got_i, 0)  # first index wins ties


    def test_basic_average_in_orthorhombic_cell_no_wrapping_needed(self):
        cell = orthorhombic_cell(10, 10, 10)
        pts = np.array([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                        [2.5, 3.5, 4.5]])
        expected = pts.mean(axis=0)  # already well inside the box
        com = COM(pts, cell)
        assert np.allclose(com, expected, atol=1e-12)
        assert is_wrapped_inside(com, cell)


    def test_minimum_image_across_boundary_orthorhombic(self):
        cell = orthorhombic_cell(10, 10, 10)
        # Two points straddling the x-boundary. Minimum image mean should land at x ~ 0 (wrapped).
        pts = np.array([[9.5, 0.0, 0.0],
                        [0.5, 0.0, 0.0]])
        com = COM(pts, cell)
        # Expect COM at the boundary, wrapped inside the cell -> x close to 0 (or 10 wrapped to 0)
        assert np.allclose(com, np.array([0.0, 0.0, 0.0]), atol=1e-7) or \
            np.allclose(com, np.array([10.0, 0.0, 0.0]), atol=1e-7)
        assert is_wrapped_inside(com, cell)


    def test_nonorthogonal_minimum_image_along_a_direction(self):
        cell = hexagonal_like_cell(a=10.0, c=12.0, angle_deg=60.0)
        # Two fractional points near 0 and 1 along a-direction; same b,c.
        f1 = np.array([0.95, 0.25, 0.40])
        f2 = np.array([0.05, 0.25, 0.40])
        pts = frac_to_cart(np.vstack([f1, f2]), cell)
        com = COM(pts, cell)

        # Expected COM in fractional coords: mean of (0.95, 0.05) under MIC -> ~0.00 (wrapped),
        # b and c unchanged (0.25, 0.40)
        f_expected = np.array([0.0, 0.25, 0.40])
        cart_expected = frac_to_cart(f_expected, cell)

        # Because COM is wrapped into the unit cell, it should match expected in Cartesian.
        assert np.allclose(com, cart_expected, atol=1e-6)
        assert is_wrapped_inside(com, cell)


    def test_invariance_to_point_order(self):
        cell = orthorhombic_cell(10, 10, 10)
        pts = np.array([[9.8, 9.9, 0.1],
                        [0.2, 0.1, 9.9],
                        [0.0, 0.0, 0.0],
                        [9.9, 0.1, 0.1]])
        rng = np.random.default_rng(123)
        com_ref = COM(pts, cell)
        for _ in range(10):
            shuffled = pts[rng.permutation(len(pts))]
            com = COM(shuffled, cell)
            # COM should be the same regardless of which point is chosen as reference internally
            assert np.allclose(com, com_ref, atol=1e-9)

    def test_single_point_is_wrapped(self):
        cell = orthorhombic_cell(10, 10, 10)
        # Already inside; COM should be the same point (wrapped inside)
        p = np.array([[9.9999999, 0.0000001, 5.0]])
        com = COM(p, cell)
        assert np.allclose(com, p[0], atol=1e-9)
        assert is_wrapped_inside(com, cell)

    def test_empty_and_single_point(self):
        
        assert findConnectivity([], 1.0, np.eye(3)) == []
        assert findConnectivity([0, 0, 0], 1.0, np.eye(3)) == []

    def test_no_self_connections_and_zero_cutoff(self):
        
        R = np.array([[0, 0, 0], [0.5, 0, 0]])
        pairs = findConnectivity(R, 0.0, np.eye(3))  # zero cutoff: nothing unless identical
        assert pairs == []

    def test_orthorhombic_wrap_across_boundary(self):
        
        L = np.array([10.0, 10.0, 10.0])
        # Points near opposite faces along x; minimum-image separation is 0.5
        R = np.array([
            [9.75, 0.0, 0.0],
            [0.25, 0.0, 0.0],
            [5.0,  5.0, 5.0],  # far away
        ])
        # Accepts a 1x3 "lengths" input
        pairs = as_sorted_pairs(findConnectivity(R, cutoff=1.0, cellDims=L))
        assert pairs == [(0, 1)]

    def test_triclinic_simple_connectivity_along_a_vector(self):
        
        # Triclinic cell with skew (rows are a, b, c)
        H = np.array([
            [5.0, 0.0, 0.0],   # a
            [1.5, 4.5, 0.0],   # b (skewed)
            [0.7, 0.8, 6.0],   # c (skewed)
        ])
        # Put one point at origin; second at fractional s=[0.92, 0, 0]
        # Minimum-image separation should be |0.08 * a|
        s2 = np.array([0.92, 0.0, 0.0])
        r2 = H.T @ s2  # Cartesian
        R = np.array([[0.0, 0.0, 0.0], r2, [3.0, 3.0, 3.0]])

        d_expected = np.linalg.norm(H.T @ np.array([0.08, 0, 0]))
        assert 0.0 < d_expected < 5.0

        pairs = as_sorted_pairs(findConnectivity(R, cutoff=d_expected + 1e-9, cellDims=H))
        assert (0, 1) in pairs
        # Tighten cutoff just below expected -> should drop the pair
        pairs_tight = as_sorted_pairs(findConnectivity(R, cutoff=d_expected - 1e-6, cellDims=H))
        assert (0, 1) not in pairs_tight

    def test_accepts_3x3_and_flat_9_cell_formats(self):
        
        H = np.array([
            [4.0, 0.0, 0.0],
            [0.5, 3.5, 0.0],
            [0.2, 0.3, 5.0],
        ])
        R = np.array([[0, 0, 0], (H.T @ np.array([0.95, 0, 0]))])  # close via wrap

        pairs_3x3 = as_sorted_pairs(findConnectivity(R, 1.0, H))
        pairs_flat = as_sorted_pairs(findConnectivity(R, 1.0, H.reshape(-1)))
        assert pairs_3x3 == pairs_flat == [(0, 1)]

    def test_matches_bruteforce_reference_small_random_triclinic(self,seed=123):
        
        rng = np.random.default_rng(seed)
        # Random triclinic with moderate skew (well-conditioned)
        A = rng.normal(size=(3, 3))
        # Make sure it's not degenerate; scale to typical sizes
        H = np.array([
            [5.0, 0.0, 0.0],
            [0.8, 4.5, 0.0],
            [0.4, 0.6, 6.0],
        ]) + 0.15 * A

        # Random points inside the cell: r = H.T @ s, with s in [0,1)
        N = 20
        S = rng.random((N, 3))
        R = (H.T @ S.T).T

        cutoff = 2.2  # a few Angstroms
        pairs_fast = as_sorted_pairs(findConnectivity(R, cutoff, H))
        pairs_ref  = as_sorted_pairs(ref_connectivity(R, cutoff, H))
        assert pairs_fast == pairs_ref

    def test_no_duplicates_and_ordering(self):
        
        L = np.array([10.0, 10.0, 10.0])
        R = np.array([
            [0, 0, 0],
            [0.2, 0, 0],
            [0.3, 0, 0],
        ])
        pairs = findConnectivity(R, cutoff=0.25, cellDims=L)
        # Should only contain (0,1) not (1,0); and either (0,2) or not depending on cutoff
        for i, j in pairs:
            assert i < j
        # Pairs must be unique
        assert len(pairs) == len(set(map(tuple, pairs)))


if __name__ == "__main__":
    unittest.main()
