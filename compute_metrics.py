import multiprocessing
import os
import sys
from glob import glob
from functools import partial

import networkx as nx
import numpy as np
import pandas as pd
import constants

from metrics_gd import *


def compute_metrics(coords, gtds):

    # normalized stress
    ns = norm_stress(coords, gtds, stress_alpha = 2)
    # crossing resolution
    cr = crossing_res(coords, gtds)
    # angular resolution
    ar = angular_resolution_metric(coords, gtds)

    # occlusions
    # node-node occlusion
    nn = node_node_occl(coords, gtds)
    # node-edge occlusion
    ne = node_edge_occl(coords, gtds)
    # edge-edge occlusion case 1: (simply crossing number)
    cr = crossings_number(coords, gtds)
    # edge-edge occlusion case 2: edges that are practically on top of each other (w.r.t. a certain margin)
    ee = edge_edge_occlusion(coords, gtds)

    # norm_stress, node_res = norm_stress_node_resolution_metric(coords, gtds, stress_alpha = 2)
    # angul_res = angular_resolution_metric(coords, gtds)
    # crossing_res, crossing_number = crossing_res_and_crossings_metric(coords, gtds)

    return ns, cr, ar, nn, ne, cr, ee


def parallel_metrics(coords, gtds, args):

    index, view = args
    ns_v, cr_v, ar_v, nn_v, ne_v, cr_v, ee_v = compute_metrics(coords, gtds)

    if index % 10 == 0:
        print(f"Calculating approximately view: {index}")

    return [index, ns_v, cr_v, ar_v, nn_v, ne_v, cr_v, ee_v]


if __name__ == '__main__':

    all_datasets = os.listdir('data/')

    #for dataset_name in ['gridaug']:
    for dataset_name in all_datasets:
        print(dataset_name)
        input_file = glob(f'data/{dataset_name}/*-src.csv')[0]

        metrics_file = os.path.join(constants.metrics_dir, F'metrics_{dataset_name}.pkl')

        df = pd.read_csv(input_file, sep = ';', header = 0)
        graph = nx.from_pandas_edgelist(df, 'from', 'to', edge_attr='strength')
        graph = nx.convert_node_labels_to_integers(graph)

        gtds_file = 'data/{0}/{0}-gtds.csv'.format(dataset_name)
        gtds = pd.read_csv(gtds_file, sep=';', header=0)
        gtds = gtds.to_numpy()

        dataset = input_file.split('/')[1]
        layouts = glob(os.path.join(constants.output_dir, '{0}*.csv'.format(dataset_name)))

        metrics_list = []
        # pool = multiprocessing.Pool(max(1, 4))

        for layout_file in layouts:
            elems = layout_file.split('-')[1:]
            n_components = int(elems[-1].replace('d.csv', ''))
            layout_name = '-'.join(elems[::-1][1:][::-1])

            print('file: {0}, dim: {1}, proj: {2}'.format(layout_file, n_components, layout_name))

            df_layout = pd.read_csv(layout_file, sep = ';', header = 0).to_numpy()

            ns, cr, ar, nn, ne, cr, ee = compute_metrics(df_layout, gtds)

            # repeat for all views of a 3D projection:
            metrics_views_list = []

            if n_components == 3:
                views_file = layout_file.replace('3d.csv', 'views.pkl')
                views = pd.read_pickle(views_file)['views'].to_numpy()

                # use multiprocessing to speed up metric calculation for the views

                # using pool.map results in each view having the same quality metric values, I don't know why
                # metrics_views_list = pool.map(partial(parallel_metrics, df_layout, gtds), zip(list(range(len(views))), views))
                metrics_views_list = [0] * len(views)

                curr_viewpoint = 0
                for i in range(len(views)):

                    if (i % 250) == 0:
                        print('Finished viewpoints ' + str(curr_viewpoint) + '-' + str(i))
                        curr_viewpoint = i

                    metrics_views_list[i] = np.array(compute_metrics(views[i], gtds))

            # normalization step: for each quality metric, the lowest seen value of all views is set to be the lowest point (0), then the highest
            # seen value of all views is set to the highest point (1)
            qms = [ns, cr, ar, nn, ne, cr, ee]
            qm_names = ['stress', 'crossing_resolution', 'angular_resolution', 'node-node_occlusion', 'node-edge_occlusion', 'crossing_number', 'edge-edge_occlusion']
            qm_idx = dict(zip(range(len(qms)), qms))

            for key, val in qm_idx.items():
                temp_array = np.append(np.array(metrics_views_list)[:, key], val)
                curr_min = np.min(temp_array)
                scale_val = np.max(temp_array - curr_min)
                metrics_views_list[:, key] = (metrics_views_list[:, key] - curr_min) / scale_val

            metrics_list.append((layout_name, n_components, ns, ar, cr, cn, np.array(metrics_views_list)))

        df_metrics = pd.DataFrame.from_records(metrics_list)
        df_metrics.columns = ['layout_technique', 'n_components'] + qm_names + ['views_metrics']
        df_metrics.to_pickle(metrics_file)

        # pool.close()
