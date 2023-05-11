from layout_techniques.fa2 import forceatlas2 as fa2
import numpy as np

def run_fa2(G, pos = None, edge_weight = 0.0, max_iter = 2000, dim = 2, bh_optimize = False):

    fa2_obj = fa2.ForceAtlas2(
                            # Behavior alternatives
                            outboundAttractionDistribution = True,  # Dissuade hubs
                            linLogMode = False,  # NOT IMPLEMENTED
                            adjustSizes = False,  # Prevent overlap (NOT IMPLEMENTED)
                            edgeWeightInfluence = edge_weight,

                            # Performance
                            jitterTolerance = 1.0,  # Tolerance
                            barnesHutOptimize = bh_optimize, # IS IMPLEMENTED BUT DO NOT USE, MOST LIKELY BROKEN WITH SWITCH TO 3D DRAWING
                            barnesHutTheta = 1.2,
                            multiThreaded = False,  # NOT IMPLEMENTED

                            # Tuning
                            scalingRatio = 2.0,
                            strongGravityMode = False,
                            gravity = 1.0,

                            # Log
                            verbose = False)
    if dim == 3:
        assert (fa2_obj.barnesHutOptimize == False), "BarnesHutOptimize does not work with 3 dimensions"

    fa2_pos = fa2_obj.forceatlas2_networkx_layout(G = G, pos = pos, iterations = max_iter, dim = dim)

    # convert the given dictionary to a n by 2 matrix
    coords = np.zeros(shape=(len(fa2_pos.keys()), dim))
    for i in fa2_pos:
        for j in range(dim):
            coords[i][j] = fa2_pos[i][j]

    return coords


