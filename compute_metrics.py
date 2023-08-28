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


def compute_metrics_old(coords, gtds, r, width, metrics_times):


    # normalized stress
    start = time.time()
    ns = norm_stress(coords, gtds)
    metrics_times['stress'] = time.time() - start

    # crossing resolution
    start = time.time()
    cr = crossing_res(coords, gtds)
    metrics_times['crossing_resolution'] = time.time() - start

    # angular resolution
    start = time.time()
    ar = angular_resolution(coords, gtds)
    metrics_times['angular_resolution'] = time.time() - start

    # node resolution
    start = time.time()
    #nr = node_resolution(coords, tar_res = r * 2)
    nr = node_resolution(coords)
    metrics_times['node_resolution'] = time.time() - start

    # occlusions
    # node-node occlusion
    start = time.time()
    nn = node_node_occl(coords, r)
    metrics_times['node-node_occlusion'] = time.time() - start

    # node-edge occlusion
    start = time.time()
    ne = node_edge_occl(coords, gtds, r, width)
    metrics_times['node-edge_occlusion'] = time.time() - start

    # edge-edge occlusion case 1: (simply crossing number)
    start = time.time()
    cn = crossings_number(coords, gtds)
    metrics_times['crossing_number'] = time.time() - start

    # edge-edge occlusion case 2: edges that are practically on top of each other (w.r.t. a certain margin)
    start = time.time()
    ee = edge_edge_occl(coords, gtds, r, width)
    metrics_times['edge-edge_occlusion'] = time.time() - start

    # edge length deviation
    start = time.time()
    el = edge_lengths_sd(coords, gtds)
    metrics_times['edge_length_deviation'] = time.time() - start

    return ns, cr, ar, nr, nn, ne, cn, ee, el, metrics_times


def parallel_metrics_old(coords, gtds, r, width, metrics_times):

    results = {}
    start_ns = time.time()
    results['ns'] = norm_stress(coords, gtds)
    metrics_times['stress'] = time.time() - start_ns

    # crossing resolution
    start_cr = time.time()
    results['cr'] = crossing_res(coords, gtds)
    metrics_times['crossing_resolution'] = time.time() - start_cr

    # angular resolution
    start_ar = time.time()
    results['ar'] = angular_resolution(coords, gtds)
    metrics_times['angular_resolution'] = time.time() - start_ar

    # node resolution
    start_nr = time.time()
    #results['nr'] = node_resolution_old(coords, tar_res = r * 2)
    results['nr'] = node_resolution(coords)
    metrics_times['node_resolution'] = time.time() - start_nr

    # occlusions
    # node-node occlusion
    start_nn = time.time()
    results['nn'] = node_node_occl(coords, r)
    metrics_times['node-node_occlusion'] = time.time() - start_nn

    # node-edge occlusion
    start_ne = time.time()
    results['ne'] = node_edge_occl(coords, gtds, r, width)
    metrics_times['node-edge_occlusion'] = time.time() - start_ne

    # edge-edge occlusion case 1: (simply crossing number)
    start_cn = time.time()
    results['cn'] = crossings_number(coords, gtds)
    metrics_times['crossing_number'] = time.time() - start_cn

    # edge-edge occlusion case 2: edges that are practically on top of each other (w.r.t. a certain margin)
    start_ee = time.time()
    results['ee'] = edge_edge_occl(coords, gtds, r, width)
    metrics_times['edge-edge_occlusion'] = time.time() - start_ee

    # edge length deviation
    start_el = time.time()
    results['el'] = edge_lengths_sd(coords, gtds)
    metrics_times['edge_length_deviation'] = time.time() - start_el

    results['metrics_times'] = metrics_times

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
            metrics_times = dict(zip(constants.metrics, [0] * len(constants.metrics)))

            input_file = glob(f'data/{dataset_name}/*-src.csv')[0]

            metrics_file = os.path.join(constants.metrics_dir, F'metrics_{dataset_name}.pkl')

            df = pd.read_csv(input_file, sep = ';', header = 0)
            graph = nx.from_pandas_edgelist(df, 'from', 'to', edge_attr='strength')
            graph = nx.convert_node_labels_to_integers(graph)

            gtds_file = 'data/{0}/{0}-gtds.csv'.format(dataset_name)
            gtds = pd.read_csv(gtds_file, sep=';', header=0)
            gtds = gtds.to_numpy()

            dataset = input_file.split('/')[1]
            layouts = glob(os.path.join(constants.output_dir, '{0}-*.csv'.format(dataset_name)))

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
                print('file: {0}, technique: {1}'.format(layout_file, tech_name))

                if overwrite == False:
                    if tech in exist_techniques:
                        print('This technique was already computed, using existing computations (set overwrite to True if you want to recompute)')
                        continue

                # first for 2d
                df_layout = pd.read_csv(techniques[tech]['path'].replace('-3d.csv', '-2d.csv'), sep = ';', header = 0).to_numpy()

                r = min(1 / np.sqrt(graph.number_of_nodes()), 1/150)
                width = r / 5

                ns, cr, ar, nr, nn, ne, cn, ee, el, metrics_times = compute_metrics_old(df_layout, gtds, r, width, metrics_times)



                # repeat for all views of a 3D layout:
                metrics_views_list = []

                views_file = layout_file.replace('3d.csv', 'views.pkl')
                views = pd.read_pickle(views_file)['views'].to_numpy()

                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                start_parallel = time.time()
                try:
                    parallel_metrics_with_gtds = partial(parallel_metrics_old, gtds=gtds, r = r, width = width, metrics_times = metrics_times)
                    res_iter = pool.map(parallel_metrics_with_gtds, [view for view in views])
                except Exception as e:
                    print(e)
                    pool.close()
                    pool.join()

                res_list = list(res_iter)
                pool.close()
                pool.join()

                # each value was saved in a dict, so now we extract those values
                test_parallel = [0] * len(views)
                for i in range(len(views)):
                    metrics_times = res_list[i]['metrics_times']
                    test_parallel[i] = [res_list[i]['ns'], res_list[i]['cr'], res_list[i]['ar'], res_list[i]['nr'], res_list[i]['nn'], res_list[i]['ne'], res_list[i]['cn'], res_list[i]['ee'], res_list[i]['el']]

                metrics_views_list = np.array(test_parallel)
                tot_time += time.time() - start_parallel

                metrics_list.append((tech_name, 2, ns, cr, ar, nr, nn, ne, cn, ee, el, np.array([])))
                metrics_list.append((tech_name, 3, ns, cr, ar, nr, nn, ne, cn, ee, el, metrics_views_list))

                qm_names = constants.metrics
                df_metrics = pd.DataFrame.from_records(metrics_list)
                df_metrics.columns = ['layout_technique', 'n_components'] + qm_names + ['views_metrics']
                df_metrics.to_pickle(metrics_file)

            print('Total time taken for current graph: ' + str(round(tot_time, 2)))
            for key in metrics_times:
                print('Time taken for metric ' + key + ': ' + str(round(metrics_times[key], 3)))

