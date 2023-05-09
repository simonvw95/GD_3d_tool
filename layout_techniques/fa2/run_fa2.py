from layout_techniques.fa2 import forceatlas2 as fa2


def run_fa2(G, pos = None, edge_weight = 0.0, max_iter = 2000):

    fa2_obj = fa2.ForceAtlas2(
                            # Behavior alternatives
                            outboundAttractionDistribution = True,  # Dissuade hubs
                            linLogMode = False,  # NOT IMPLEMENTED
                            adjustSizes = False,  # Prevent overlap (NOT IMPLEMENTED)
                            edgeWeightInfluence = edge_weight,

                            # Performance
                            jitterTolerance = 1.0,  # Tolerance
                            barnesHutOptimize = False, # IS IMPLEMENTED BUT DO NOT USE, MOST LIKELY BROKEN WITH SWITCH TO 3D DRAWING
                            barnesHutTheta = 1.2,
                            multiThreaded = False,  # NOT IMPLEMENTED

                            # Tuning
                            scalingRatio = 2.0,
                            strongGravityMode = False,
                            gravity = 1.0,

                            # Log
                            verbose = False)

    coords = fa2_obj.forceatlas2_networkx_layout(G = G, pos = pos, iterations = max_iter)

    # convert the given dictionary to a n by 2 matrix
    coords = np.zeros(shape=(len(pos.keys()), 3))
    for i in pos:
        coords[i][0] = pos[i][0]
        coords[i][1] = pos[i][1]
        coords[i][2] = pos[i][2]

    return coords


