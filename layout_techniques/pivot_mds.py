import numpy as np
import networkx as nx


def pivot_mds(g, pivots = 5, weights = None, D = None, dim = 2):

    if not isinstance(g, nx.Graph):
        raise ValueError("Not a graph object")
    if not nx.is_connected(g):
        raise ValueError("Only connected graphs are supported.")
    if pivots is None and D is None:
        raise ValueError('Argument "pivots" is missing, with no default.')
    if pivots is not None:
        if pivots > g.number_of_nodes():
            raise ValueError('"pivots" must be less than the number of nodes in the graph.')
            
    if D is None:
        D = nx.floyd_warshall_numpy(g)

    pivs = np.random.choice(list(g.nodes()), pivots, replace=False)
    D = D[:, pivs]

    Dsq = D**2
    cmean = np.mean(Dsq, axis=0)
    rmean = np.mean(Dsq, axis=1)
    Dmat = Dsq - np.add.outer(rmean, cmean) + np.mean(Dsq)
    _, _, V = np.linalg.svd(Dmat)
    xy = np.dot(Dmat, np.transpose(V[0:dim, ]))

    return xy



