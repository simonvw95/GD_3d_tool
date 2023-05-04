#!/usr/bin/env python3

import networkx as nx
import copy
import numpy as np
from layout_techniques.pivot_mds import pivot_mds
from layout_techniques.thesne import tsnet


def tsNET(graph, star = False, dimensions = 2, perplexity = 40, learning_rate = 50):

    # Global hyperparameters
    n = 5000  # Maximum #iterations before giving up
    momentum = 0.5
    tolerance = 1e-7
    window_size = 40

    # Cost function parameters
    r_eps = 0.05

    # Phase 2 cost function parameters
    lambdas_2 = [1, 1.2, 0]
    if star:
        lambdas_2[1] = 0.1

    # Phase 3 cost function parameters
    lambdas_3 = [1, 0.01, 0.6]

    # # Load the PivotMDS layout for initial placement
    Y_init = None

    if star:
        Y_init = pivot_mds(g = graph, dim = dimensions)

    # Compute the shortest-path distance matrix.
    X = nx.floyd_warshall_numpy(graph)

    # The actual optimization is done in the thesne module.
    Y = tsnet(
        X, output_dims=dimensions, random_state=1, perplexity=perplexity, n_epochs=n,
        Y=Y_init,
        initial_lr=learning_rate, final_lr=learning_rate, lr_switch=n // 2,
        initial_momentum=momentum, final_momentum=momentum, momentum_switch=n // 2,
        initial_l_kl=lambdas_2[0], final_l_kl=lambdas_3[0], l_kl_switch=n // 2,
        initial_l_c=lambdas_2[1], final_l_c=lambdas_3[1], l_c_switch=n // 2,
        initial_l_r=lambdas_2[2], final_l_r=lambdas_3[2], l_r_switch=n // 2,
        r_eps=r_eps, autostop=tolerance, window_size=window_size,
        verbose=False
    )

    Y = normalize_layout(Y)

    return Y


def normalize_layout(Y):

    Y_cpy = copy.deepcopy(Y)
    # Translate s.t. smallest values for both x and y are 0.
    for dim in range(Y.shape[1]):
        Y_cpy[:, dim] += -Y_cpy[:, dim].min()

    # Scale s.t. max(max(x, y)) = 1 (while keeping the same aspect ratio!)
    scaling = 1 / (np.absolute(Y_cpy).max())
    Y_cpy *= scaling

    return Y_cpy