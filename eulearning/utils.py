import numpy as np
import networkx as nx

from sklearn.neighbors import KernelDensity

from scipy.sparse import csgraph
from scipy.linalg import eigh
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci


# Filtrations and multi-parameter persistence
## Point clouds
def codensity(X, density_to_codensity=lambda x: -x, **kwargs):
    '''
    Returns the KDE estimate built on X postcomposed with a decreasing function density_to_codensity.
    '''
    density = KernelDensity(**kwargs).fit(X)
    return density_to_codensity(density.score_samples(X))


## Graphs
def compute_vertvals_from_edgevals(n_vertices, edges, edge_vals):
    '''
    Assigns value to vertices from values on edges so that the obtained function on the graph is non-decreasing for the face order.

    Input:
        n_vertices	: int, number of vertices in the graph
        edges		: list ot tuple (u,v) representing an edge of the graph between vertices u and v.
        edge_vals	: ndarray containing the values of a function given on edges of the graph. Values should be ordered in the same order as the list edges.

    Output
        vert_vals	: ndarray containing the values of the function on vertices of the graph. Values are computed so that the overall function on the graph is non-decreasing for the face order.
    '''
    vert_vals = edge_vals.max() * np.ones(n_vertices)  # By convention, make the vertices with degree 0 appear at last
    if len(edges) == len(edge_vals):  # Rarely, the Ollivier-Ricci curvature does not assign a value to some edges
        for ie, (u, v) in enumerate(edges):
            e_val = edge_vals[ie]
            if e_val < vert_vals[u]:
                vert_vals[u] = e_val
            if e_val < vert_vals[v]:
                vert_vals[v] = e_val
        return vert_vals
    else:
        return vert_vals


### We define below several filtrations on graphs. Functions all output a ndarray of shape (N_v,) where N_v is the number of vertices in the graph.
def compute_FR_curvature(A):  # Computes Forman-Ricci curvature
    G = nx.to_networkx_graph(A)
    n_vertices = G.number_of_nodes()
    orc = FormanRicci(G)
    orc.compute_ricci_curvature()
    curv_edges = np.array(list(nx.get_edge_attributes(orc.G, "formanCurvature").values()))
    curv_edges = np.nan_to_num(curv_edges, copy=True, nan=0.0, posinf=None, neginf=None)
    if len(curv_edges) == 0:
        return np.zeros(n_vertices)
    return compute_vertvals_from_edgevals(n_vertices, list(G.edges()), curv_edges)


def compute_OR_curvature(A, alpha=0.5,
                         iterations=0):  # Computes Ollivier-Ricci curvature if iterations=0 and Ollivier-Ricci flow if iterations>0
    G = nx.to_networkx_graph(A)
    n_vertices = G.number_of_nodes()
    orc = OllivierRicci(G, alpha=alpha)
    orc.compute_ricci_curvature()
    orc.compute_ricci_flow(iterations=iterations)
    curv_edges = np.array(list(nx.get_edge_attributes(orc.G, "ricciCurvature").values()))
    if len(curv_edges) == 0:
        return np.zeros(n_vertices)
    return compute_vertvals_from_edgevals(n_vertices, list(G.edges()), curv_edges)


def compute_hks_signature(A, time):  # Computes Heat Kernel Signature on graphs
    num_vertices = A.shape[0]
    L = csgraph.laplacian(A, normed=True)
    egvals, egvectors = eigh(L)
    eigenvectors = np.zeros([num_vertices, num_vertices])
    eigenvals = np.zeros(num_vertices)
    eigenvals = np.flipud(egvals)
    eigenvectors = np.fliplr(egvectors)
    return np.square(eigenvectors).dot(np.diag(np.exp(-time * eigenvals))).sum(axis=1)


def compute_closeness_centrality(A):  # Compute closeness centrality
    G = nx.to_networkx_graph(A)
    cent = np.array(list(nx.closeness_centrality(G).values()))
    return cent


def compute_edge_betweenness(A):  # Compute edge betweenness
    G = nx.to_networkx_graph(A)
    n_vertices = G.number_of_nodes()
    edb = np.array(list(nx.edge_current_flow_betweenness_centrality(G).values()))
    return compute_vertvals_from_edgevals(n_vertices, list(G.edges()), edb)


# Vectorize simplex trees
def vectorize_st(st, filtrations=None):
    '''
    Vectorize a simplex tree from the Gudhi library.

    Input:
        st 				: Simplex tree from the Gudhi library
        filtrations		: None or List of ndarrays of size (st.num_simplices(),) or (st.num_vertices(),).
    Output:
        vec_st			: ndarray of size (st.num_simplices(), len(filtrations)+2) where for each simplex, the line is of the form [(-1)**dim(simplex), filtration value of simplex in st, filtration value of simplex as determined by ndarrays of filtrations]. If an array of filtrations is of shape (st.num_vertices(),), then the filtration value of each simplex is computed by upper star filtration. Otherwise, this array already indicates the filtration values for each simplex of st. CAUTION: vertices (or simplices) and their values must have the same order in the arrays of filtrations and in st.
    '''
    n_filts = 0 if filtrations is None else len(filtrations)
    n_splx, n_vrts = st.num_simplices(), st.num_vertices()
    vec_st = np.zeros((n_splx, n_filts + 2))
    for j, (s, t) in enumerate(st.get_filtration()):
        vec_st[j, 0] = (-1) ** (len(s) - 1)
        vec_st[j, 1] = t
        if filtrations is not None:
            for i, f in enumerate(filtrations):
                if f.shape[0] == n_splx:
                    vec_st[j, i + 2] = f[j]
                else:
                    vec_st[j, i + 2] = np.max(np.take(f, s))
    return vec_st