# test_graph_utils.py
import numpy as np
import networkx as nx
import pytest
from HopDec.Graphs import graphLabel, nDefectVolumes, buildNetwork, shortestPath

############################
# Tests for graphLabel
############################

def test_graphlabel_canonical_isomorph_equal():
    # Two isomorphic triangles with different node ids
    edges_a = [(0, 1), (1, 2), (2, 0)]
    edges_b = [(10, 11), (11, 12), (12, 10)]
    ha = graphLabel(edges_a, canonical=1)
    hb = graphLabel(edges_b, canonical=1)
    assert isinstance(ha, str) and isinstance(hb, str)
    assert ha == hb  # canonical labeling should ignore raw ids


def test_graphlabel_noncanonical_uses_indices_and_can_differ():
    edges = [(0, 1), (1, 2), (2, 0)]
    # Same edges, but different "indices" labels should change the hash
    h1 = graphLabel(edges, canonical=0, indices=[0, 1, 2])
    h2 = graphLabel(edges, canonical=0, indices=[2, 1, 0])
    assert isinstance(h1, str) and isinstance(h2, str)
    assert h1 != h2  # labels differ → hash should differ


def test_graphlabel_with_types_changes_hash_when_types_change():
    edges = [(0, 1), (1, 2), (2, 0)]
    # Same topology, different node types should change the hash
    h1 = graphLabel(edges, canonical=1, types=[0, 0, 1])
    h2 = graphLabel(edges, canonical=1, types=[0, 1, 1])
    assert h1 != h2


def test_graphlabel_noncanonical_without_indices_raises():
    edges = [(0, 1)]
    # Per the docstring, this should raise; the implementation currently
    # calls ValueError(...) without raising, so this test will catch that bug.
    with pytest.raises(ValueError):
        graphLabel(edges, canonical=0, indices=None)

############################
# Tests for nDefectVolumes
############################

def test_nDefectVolumes_connected_graph_returns_1():
    # A square is connected
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    assert nDefectVolumes(edges) == 1


def test_nDefectVolumes_disconnected_graph_returns_2():
    # Two disconnected edges (two components)
    edges = [(0, 1), (2, 3)]
    assert nDefectVolumes(edges) == 2


def test_nDefectVolumes_single_edge_is_connected():
    edges = [(0, 1)]
    assert nDefectVolumes(edges) == 1

############################
# Tests for buildNetwork
############################

def test_buildNetwork_adds_nodes_and_edges():
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2)]
    G = buildNetwork(nodes, edges)
    assert isinstance(G, nx.Graph)
    assert set(G.nodes()) == set(nodes)
    assert set(map(tuple, G.edges())) == {(0, 1), (1, 2)}


def test_buildNetwork_adds_implicit_nodes_from_edges():
    # NetworkX will add nodes from edges even if not listed in nodes
    nodes = [0, 1]
    edges = [(0, 2)]  # node 2 not in 'nodes'
    G = buildNetwork(nodes, edges)
    assert set(G.nodes()) == {0, 1, 2}
    assert (0, 2) in map(tuple, G.edges())

############################
# Tests for shortestPath
############################

def test_shortestPath_basic_length():
    G = nx.path_graph(5)  # nodes 0-4 in a line
    assert shortestPath(G, 0, 4) == 4
    assert shortestPath(G, 1, 3) == 2


def test_shortestPath_same_node_zero_length():
    G = nx.Graph()
    G.add_nodes_from([1, 2])
    assert shortestPath(G, 1, 1) == 0


def test_shortestPath_disconnected_returns_inf():
    G = nx.Graph()
    G.add_nodes_from([0, 1, 2])
    G.add_edge(0, 1)
    # 2 is disconnected
    assert shortestPath(G, 0, 2) == np.inf


def test_shortestPath_nonexistent_node_returns_inf():
    G = nx.path_graph(3)  # nodes 0,1,2
    assert shortestPath(G, -1, 2) == np.inf
    assert shortestPath(G, 0, 99) == np.inf