import os
import sys
from glob import glob
import numpy as np
import pandas as pd
import networkx as nx
from sklearn import manifold
from layout_techniques.tsNET import tsNET
from layout_techniques.pivot_mds import pivot_mds

import constants


projection_names = {
                    'A': ['SM', 'pivot_mds', 'tsNET', 'tsNETstar']
                }


def get_projections(graph, n_components):

    layouts = {
                    'SM': manifold.smacof(dissimilarities = nx.floyd_warshall_numpy(graph), n_components = n_components),
                    'pivot_mds': pivot_mds(graph, dim = n_components),
                    'tsNET' : tsNET(graph, star = False, dimensions = n_components),
                    'tsNETstar' : tsNET(graph, star = True, dimensions = n_components)
    }

    return layouts


if __name__ == '__main__':

    # manual
    #all_dataset_names = ['gridaug']
    # automatic every dataset
    all_dataset_names = os.listdir('data/')

    # layout technique selection, see above
    selection = 'A'

    for dataset_name in all_dataset_names:

        input_file = glob('data/{0}/*-src.csv'.format(dataset_name))[0]

        df = pd.read_csv(input_file, sep=';', header=0)
        graph = nx.from_pandas_edgelist(df, 'from', 'to', edge_attr = 'strength')
        graph = nx.convert_node_labels_to_integers(graph)

        selected_projections = projection_names[selection]

        header = ['x', 'y', 'z']

        for n_components in [2, 3]:
            projections = get_projections(graph, n_components)

            for proj_name in selected_projections:

                if proj_name == 'SM':
                    p, _ = projections[proj_name]
                else:
                    p = projections[proj_name]

                output_file = os.path.join(constants.output_dir, '{0}-{1}-{2}d.csv'.format(dataset_name, proj_name, n_components))

                print('dim: {0}, proj: {1}'.format(n_components, proj_name))
                print('output_file: {0}'.format(output_file))

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
