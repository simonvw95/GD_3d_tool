import multiprocessing
import os
import sys
from glob import glob
from functools import partial

import itertools
import time
import networkx as nx
import numpy as np
import pandas as pd
import constants

from metrics_gd import *
#from metrics_gd_all import *

from wakepy import keepawake

import warnings
warnings.filterwarnings("ignore")

def compute_metrics(coords, gtds, r, width):

    # normalized stress, node resolution
    ns, nr = norm_stress_node_resolution(coords, gtds, tar_res = r * 2)

    # angular resolution
    ar = angular_resolution(coords, gtds)

    # occlusions
    # node-node occlusion
    nn = node_node_occl(coords, r)

    # node-edge occlusion
    ne = node_edge_occl(coords, gtds, r, width)

    # edge edge occlusion, crossing number, crossing resolution
    ee, cn, cr = edge_edge_occl_crossing_number_crossing_res(coords, gtds, r, width)

    return ns, cr, ar, nr, nn, ne, cn, ee


def compute_metrics_old(coords, gtds, r, width):
    # normalized stress
    ns = norm_stress(coords, gtds)

    # crossing resolution
    cr = crossing_res(coords, gtds)

    # angular resolution
    ar = angular_resolution(coords, gtds)

    # node resolution
    nr = node_resolution(coords, tar_res = r * 2)

    # occlusions
    # node-node occlusion
    nn = node_node_occl(coords, r)

    # node-edge occlusion
    ne = node_edge_occl(coords, gtds, r, width)

    # edge-edge occlusion case 1: (simply crossing number)
    cn = crossings_number(coords, gtds)

    # edge-edge occlusion case 2: edges that are practically on top of each other (w.r.t. a certain margin)
    ee = edge_edge_occl(coords, gtds, r, width)

    return ns, cr, ar, nr, nn, ne, cn, ee


def parallel_metrics(coords, gtds, r, width):

    results = {}

    # normalized stress, node resolution
    results['ns'], results['nr'] = norm_stress_node_resolution(coords, gtds, tar_res=r * 2)

    # angular resolution
    results['ar'] = angular_resolution(coords, gtds)

    # occlusions
    # node-node occlusion
    results['nn'] = node_node_occl(coords, r)

    # node-edge occlusion
    results['ne'] = node_edge_occl(coords, gtds, r, width)

    # edge edge occlusion, crossing number, and crossing resolution
    results['ee'], results['cn'], results['cr'] = edge_edge_occl_crossing_number_crossing_res(coords, gtds, r, width)

    return results


def parallel_metrics_old(coords, gtds, r, width):

    results = {}
    results['ns'] = norm_stress(coords, gtds)

    # crossing resolution
    results['cr'] = crossing_res(coords, gtds)

    # angular resolution
    results['ar'] = angular_resolution(coords, gtds)

    # node resolution
    results['nr'] = node_resolution(coords, tar_res = r * 2)

    # occlusions
    # node-node occlusion
    results['nn'] = node_node_occl(coords, r)

    # node-edge occlusion
    results['ne'] = node_edge_occl(coords, gtds, r, width)

    # edge-edge occlusion case 1: (simply crossing number)
    results['cn'] = crossings_number(coords, gtds)

    # edge-edge occlusion case 2: edges that are practically on top of each other (w.r.t. a certain margin)
    results['ee'] = edge_edge_occl(coords, gtds, r, width)

    return results


if __name__ == '__main__':
    with keepawake(keep_screen_awake=False):

        all_datasets = os.listdir('data/')
        sizes = {}
        for i in all_datasets:
            sizes[i] = os.path.getsize('data/' + i + '/' + i + '-gtds.csv')

        sorted_file_names = sorted(sizes, key = sizes.get)

        overwrite = False  # set to False if we do not want to overwrite existing metric results
        #for dataset_name in ['grafo1126']:
        for dataset_name in sorted_file_names:
            print(dataset_name)

            tot_time = 0

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

            # calculate the metrics per technique
            techniques = {}
            for layout_file in layouts:
                elems = layout_file.split('-')[1:]
                n_components = int(elems[-1].replace('d.csv', ''))
                technique = elems[-2]
                layout_name = elems[-2] + '-' + elems[-1]
                techniques[technique] = {'name' : technique, 'dim' : n_components, 'path' : layout_file}

            metrics_list = []

            # check if the metric file for this technique+dataset+#dimensions already exists
            exist_techniques = []
            if os.path.isfile(metrics_file):
                exist_df = pd.read_pickle(metrics_file)
                # when it does exist, and we're not overwriting then append the existing rows to the list
                if overwrite == False:
                    for row in range(exist_df.shape[0]):
                        curr_row = exist_df.loc[row, :].values.flatten().tolist()
                        metrics_list.append(curr_row)
                        # keep track of which technique needs to be computed
                        exist_techniques.append(curr_row[0])

            for tech in techniques:
                tech_name = techniques[tech]['name']
                layout_file = techniques[tech]['path']

                print('file: {0}, dim: {1}, technique: {2}'.format(techniques[tech]['path'].replace('-3d.csv', '-2d.csv'), 2, tech_name.replace('-3d.csv', '-2d.csv')))
                if overwrite == False:
                    if tech in exist_techniques:
                        print('This technique was already computed, using existing computations (set overwrite to True if you want to recompute)')
                        continue

                # first for 2d
                df_layout = pd.read_csv(techniques[tech]['path'].replace('-3d.csv', '-2d.csv'), sep = ';', header = 0).to_numpy()

                r = min(1 / np.sqrt(graph.number_of_nodes()), 1/150)
                width = r / 5

                ns, cr, ar, nr, nn, ne, cn, ee = compute_metrics_old(df_layout, gtds, r, width)

                print('file: {0}, dim: {1}, technique: {2}'.format(layout_file, techniques[tech]['dim'], tech_name))

                # repeat for all views of a 3D layout:
                metrics_views_list = []

                views_file = layout_file.replace('3d.csv', 'views.pkl')
                views = pd.read_pickle(views_file)['views'].to_numpy()

                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                start_parallel = time.time()
                try:
                    parallel_metrics_with_gtds = partial(parallel_metrics_old, gtds=gtds, r = r, width = width)
                    res_iter = pool.map(parallel_metrics_with_gtds, [view for view in views])
                except Exception as e:
                    pool.close()
                    pool.join()

                res_list = list(res_iter)
                pool.close()
                pool.join()

                # each value was saved in a dict, so now we extract those values
                test_parallel = [0] * len(views)
                for i in range(len(views)):
                    test_parallel[i] = [res_list[i]['ns'], res_list[i]['cr'], res_list[i]['ar'], res_list[i]['nr'], res_list[i]['nn'], res_list[i]['ne'], res_list[i]['cn'], res_list[i]['ee']]

                metrics_views_list = np.array(test_parallel)
                print('parallel processing time taken: ' + str(round(time.time() - start_parallel, 2)))
                tot_time += time.time() - start_parallel

                metrics_list.append((tech_name, 2, ns, cr, ar, nr, nn, ne, cn, ee, np.array([])))
                metrics_list.append((tech_name, 3, ns, cr, ar, nr, nn, ne, cn, ee, metrics_views_list))

                qm_names = constants.metrics
                df_metrics = pd.DataFrame.from_records(metrics_list)
                df_metrics.columns = ['layout_technique', 'n_components'] + qm_names + ['views_metrics']
                df_metrics.to_pickle(metrics_file)

            print('Total time taken for current graph: ' + str(round(tot_time, 2)))
