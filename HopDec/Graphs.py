import networkx as nx
import numpy as np

def graphLabel(graphEdges, canonical = 1, types = None, indices = None):

    """
    Generates a Weisfeiler-Lehman hash label for a graph defined by its edges.

    Parameters
    ----------
    graphEdges : array-like
        List or array of edge pairs defining the graph.
    canonical : int, optional
        If 1 (default), use canonical labeling. If 0, use user-defined node indices.
    types : array-like, optional
        Node type labels to use as attributes for canonical labeling.
    indices : array-like, optional
        Node identifiers to use when canonical is False.

    Returns
    -------
    str
        Weisfeiler-Lehman graph hash string uniquely identifying the graph structure.
    """

    G = nx.Graph([list(row) for row in graphEdges])

    if not canonical:
        if indices is None:
            raise ValueError('Non-canonical labeling requires indices to be set.')

        nx.set_node_attributes(G, np.array(indices), "IDs")
        L = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(G, node_attr = "IDs")

    else:
        if types is None:
            L = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(G)

        else:
            nx.set_node_attributes(G, np.array(types), "types")
            L = nx.algorithms.graph_hashing.weisfeiler_lehman_graph_hash(G, node_attr = "types")

    return L


def nDefectVolumes(graphEdges):

    """
    Determines whether the graph defined by `graphEdges` is fully connected.

    Parameters
    ----------
    graphEdges : array-like
        List of edge pairs defining the graph.

    Returns
    -------
    int
        1 if the graph is fully connected (single defect volume), 
        2 if it is disconnected (multiple volumes).
    """

    def dfs(graph, start, visited):
        visited[start] = True
        for neighbor in graph[start]:
            if not visited[neighbor]:
                dfs(graph, neighbor, visited)

    def check_reachability(graph):
        num_nodes = len(graph)
        for start_node in range(num_nodes):
            visited = [False] * num_nodes
            dfs(graph, start_node, visited)
            if not all(visited):
                return False
        return True

    num_nodes = max(max(edge) for edge in graphEdges) + 1
    graph = [[] for _ in range(num_nodes)]

    for edge in graphEdges:
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])

    if check_reachability(graph):
        return 1
    else:
        return 2
    
def buildNetwork(nodes,edges):

    """
    Constructs an undirected NetworkX graph from a list of nodes and edges.

    Parameters
    ----------
    nodes : array-like
        List of node identifiers.
    edges : array-like
        List of edge pairs connecting the nodes.

    Returns
    -------
    networkx.Graph
        A NetworkX graph object with the given nodes and edges.
    """

    G = nx.Graph()

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    return G

def shortestPath(graph, source, target):

    """
    Computes the shortest path length between two nodes in a graph.

    Parameters
    ----------
    graph : networkx.Graph
        A NetworkX graph object.
    source : int or str
        The starting node.
    target : int or str
        The destination node.

    Returns
    -------
    int or float
        Length of the shortest path (number of hops), or np.inf if no path exists.
    """
        
    if source not in graph.nodes or target not in graph.nodes: 
        return np.inf
    try:
        path_length = len(nx.shortest_path(graph, source=source, target=target)) - 1
        return path_length
    except nx.NetworkXNoPath:
        return np.inf