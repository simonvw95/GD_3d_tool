import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import networkx as nx
import time
import constants

from sklearn import manifold
from layout_techniques.tsnet_v2 import tsNET
from layout_techniques.pivot_mds import pivot_mds
from layout_techniques.fa2 import run_fa2
from wakepy import keepawake



# projection_names = {
#                     'A': ['pivot_mds']
#                 }

projection_names = {
                    'A': ['SM', 'pivot_mds', 'FA2', 'tsNET', 'tsNETstar']
                }

# projection_names = {'A' : ['pivot_mds', 'tsNET', 'tsNETstar']}


def get_projections(graph, n_components, gtds, technique, perplexity = 15):

    layouts = {}

    if technique == 'SM':
        start = time.time()
        layouts['SM'] = manifold.smacof(dissimilarities = gtds, n_components = n_components)
        print('SM took: ' + str(time.time() - start) + ' seconds')

    if technique == 'pivot_mds':
        start = time.time()
        layouts['pivot_mds'] = pivot_mds(graph, dim = n_components, D = gtds, pivots = max(5, int(graph.number_of_nodes() / 10)))
        print('pivot_mds took: ' + str(time.time() - start) + ' seconds')

    if technique == 'FA2':
        start = time.time()
        bh_optimize = False
        if n_components == 2:
            bh_optimize = True
        layouts['FA2'] = run_fa2.run_fa2(G = graph, pos = None, edge_weight = 0.0, max_iter = 300, dim = n_components, bh_optimize = bh_optimize)
        print('FA2 took: ' + str(time.time() - start) + ' seconds')

    if technique == 'tsNET':
        start = time.time()
        layouts['tsNET'] = tsNET(graph, star = False, dimensions = n_components, gtds = gtds, n = 300, perplexity = perplexity)
        print('tsNET took: ' + str(time.time() - start) + ' seconds')

    if technique == 'tsNETstar':
        start = time.time()
        layouts['tsNETstar'] = tsNET(graph, star = True, dimensions = n_components, gtds = gtds, n = 300, perplexity = perplexity),
        print('tsNETstar took: ' + str(time.time() - start) + ' seconds')

    return layouts


def save_gtds(graph, dataset_name):

    gtds = nx.floyd_warshall_numpy(graph)
    df = pd.DataFrame(gtds)

    name = 'data/{0}/{0}-gtds.csv'.format(dataset_name)

    df.to_csv(name, index=None, sep=';')


if __name__ == '__main__':

    with keepawake(keep_screen_awake=False):
        # manual
        #all_dataset_names = ['EVA']
        # automatic every dataset
        all_dataset_names = os.listdir('data/')
        all_dataset_names.remove('3elt')
        # layout technique selection, see above
        selection = 'A'

        # get the perplexities for tsnet
        perplx = constants.perplexities

        # if we want to overwrite the exact same files, e.g. grid-src.csv already exists, we don't need to visualize it again
        overwrite = False

        for dataset_name in all_dataset_names:

            print('Working on ' + str(dataset_name))
            input_file = glob('data/{0}/*-src.csv'.format(dataset_name))[0]

            df = pd.read_csv(input_file, sep=';', header=0)
            graph = nx.from_pandas_edgelist(df, 'from', 'to', edge_attr = 'strength')
            graph = nx.convert_node_labels_to_integers(graph)

            selected_projections = projection_names[selection]

            gtds_file = 'data/{0}/{0}-gtds.csv'.format(dataset_name)
            if not os.path.isfile(gtds_file):
                print('Creating and saving the gtds')
                # save the graph theoretic shortest distances
                save_gtds(graph, dataset_name)

            gtds = pd.read_csv(gtds_file, sep=';', header=0)
            gtds = gtds.to_numpy()

            header = ['x', 'y', 'z']

            for n_components in [2, 3]:

                for proj_name in selected_projections:

                    output_file = os.path.join(constants.output_dir, '{0}-{1}-{2}d.csv'.format(dataset_name, proj_name, n_components))

                    # if it doesnt exist
                    if not os.path.isfile(output_file) or overwrite:
                        print('Doing ' + str(proj_name) + ' in ' + str(n_components) + 'd')
                        if dataset_name in perplx:
                            perp = perplx[dataset_name]
                        else:
                            perp = 15

                        projections = get_projections(graph, n_components, gtds, technique = proj_name, perplexity = perp)

                        if proj_name == 'SM':
                            p, _ = projections[proj_name]
                        else:
                            p = projections[proj_name]

                        print('dim: {0}, proj: {1}'.format(n_components, proj_name))
                        print('output_file: {0}'.format(output_file))

                        # sometimes tsnet* does weird things with the shape
                        if len(np.shape(p)) == 3:
                            p = p[0]

                        if n_components == 2:
                            # take the smallest value seen
                            smallest_val = abs(min(np.min(p[:, 0]), np.min(p[:, 1])))
                            # take the maximum of the the maximum differences (max difference of x coordinates & max difference of y coordinates)
                            scale_factor = max(np.max(p[:, 0]) - np.min(p[:, 0]), np.max(p[:, 1]) - np.min(p[:, 1]))
                        else:
                            smallest_val = abs(min(np.min(p[:, 0]), np.min(p[:, 1]), np.min(p[:, 2])))
                            scale_factor = max(np.max(p[:, 0]) - np.min(p[:, 0]), np.max(p[:, 1]) - np.min(p[:, 1]),
                                               np.max(p[:, 2]) - np.min(p[:, 2]))

                        # add the smallest value to the coordinates so it translates to coordinates >=0
                        p += smallest_val

                        # scale the layout
                        coords = p / scale_factor

                        df_layout = pd.DataFrame(coords)
                        df_layout.columns = header[:n_components]

                        df_layout.to_csv(output_file, index = None, sep = ';')
                    else:
                        print(output_file + ' already exists')
