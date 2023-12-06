import multiprocessing
import os
import sys
import itertools
import time
import networkx as nx
import numpy as np
import pandas as pd
import constants
import json
import warnings

from glob import glob
from functools import partial
from wakepy import keepawake

from metrics_gd import *

warnings.filterwarnings("ignore")


def compute_metrics(coords, gtds_matrix, rad, edge_width):

    # normalized stress
    ns_res = norm_stress(coords, gtds_matrix)

    # crossing resolution
    cr_res = crossing_res_dev(coords, gtds_matrix)

    # angular resolution
    ar_res = angular_resolution_dev(coords, gtds_matrix)

    # node resolution
    nr_res = node_resolution(coords)

    # occlusions
    # node-node occlusion
    nn_res = node_node_occl(coords, rad)

    # node-edge occlusion
    ne_res = node_edge_occl(coords, gtds_matrix, rad, edge_width)

    # edge-edge occlusion case 1: (simply crossing number)
    cn_res = crossings_number(coords, gtds_matrix)

    # edge-edge occlusion case 2: edges that are practically on top of each other (w.r.t. a certain margin)
    ee_res = edge_edge_occl(coords, gtds_matrix, rad, edge_width)

    # edge length deviation
    el_res = edge_lengths_sd(coords, gtds_matrix)

    return ns_res, cr_res, ar_res, nr_res, nn_res, ne_res, cn_res, ee_res, el_res


def compute_metrics_parallel(coords, gtds_matrix, rad, edge_width):

    results = {}
    results['ns'] = norm_stress(coords, gtds_matrix)

    # crossing resolution
    results['cr'] = crossing_res_dev(coords, gtds_matrix)

    # angular resolution
    results['ar'] = angular_resolution_dev(coords, gtds_matrix)

    # node resolution
    results['nr'] = node_resolution(coords)

    # occlusions
    # node-node occlusion
    results['nn'] = node_node_occl(coords, rad)

    # node-edge occlusion
    results['ne'] = node_edge_occl(coords, gtds_matrix, rad, edge_width)

    # edge-edge occlusion case 1: (simply crossing number)
    results['cn'] = crossings_number(coords, gtds_matrix)

    # edge-edge occlusion case 2: edges that are practically on top of each other (w.r.t. a certain margin)
    results['ee'] = edge_edge_occl(coords, gtds_matrix, r, edge_width)

    # edge length deviation
    results['el'] = edge_lengths_sd(coords, gtds_matrix)

    return results


if __name__ == '__main__':
    with keepawake(keep_screen_awake=False):

        all_datasets = os.listdir('data/')
        sizes = {}
        for i in all_datasets:
            sizes[i] = os.path.getsize('data/' + i + '/' + i + '-gtds.csv')

        sorted_file_names = sorted(sizes, key = sizes.get)

        overwrite = False  # set to False if we do not want to overwrite existing metric results

        for dataset_name in sorted_file_names:
            tot_time = 0

            # create the names of the edgelist file and the metric file
            input_file = glob(f'data/{dataset_name}/*-src.csv')[0]
            metrics_file = os.path.join(constants.metrics_dir, F'metrics_{dataset_name}.pkl')

            # create the graph object
            df = pd.read_csv(input_file, sep = ';', header = 0)
            graph = nx.from_pandas_edgelist(df, 'from', 'to', edge_attr='strength')
            graph = nx.convert_node_labels_to_integers(graph)
            edges = list(graph.edges())
            m = len(edges)

            # acquire the shortest path matrix
            gtds_file = 'data/{0}/{0}-gtds.csv'.format(dataset_name)
            gtds = pd.read_csv(gtds_file, sep=';', header=0)
            gtds = gtds.to_numpy()

            # create the name for the layouts
            layouts = glob(os.path.join(constants.output_dir, '{0}-*.csv'.format(dataset_name)))

            # we want to calculate the metrics per graph per technique, so put the technique names in a dictionary
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
                if overwrite is False:
                    for row in range(exist_df.shape[0]):
                        curr_row = exist_df.loc[row, :].values.flatten().tolist()
                        metrics_list.append(curr_row)
                        # keep track of which technique needs to be computed
                        exist_techniques.append(curr_row[0])

            # loop over all techniques
            for tech in techniques:
                tech_name = techniques[tech]['name']
                layout_file = techniques[tech]['path']
                print('dataset: {0}, technique: {1}'.format(dataset_name, tech_name))

                if overwrite is False:
                    if tech in exist_techniques:
                        print('This technique was already computed, using existing computations (set overwrite to True if you want to recompute)')
                        continue

                # first compute the metrics of the 2D layout
                # get the 2D layout
                df_layout = pd.read_csv(techniques[tech]['path'].replace('-3d.csv', '-2d.csv'), sep = ';', header = 0).to_numpy()

                # set the radius of any drawn node in the graph  layout and the width of an edge
                r = min(1 / np.sqrt(graph.number_of_nodes()), 1/150)
                width = r / 5

                # compute the metrics of the 2D layout
                ns, cr, ar, nr, nn, ne, cn, ee, el = compute_metrics(df_layout, gtds, r, width)

                # second compute the metrics for the viewpoints of the 3D layout
                # repeat for all views of a 3D layout
                views_file = layout_file.replace('3d.csv', 'views.pkl')
                views = pd.read_pickle(views_file)['views'].to_numpy()

                # use parallel processing
                pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                start_parallel = time.time()
                try:
                    parallel_metrics_with_gtds = partial(compute_metrics_parallel, gtds=gtds, r = r, width = width)
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
                    test_parallel[i] = [res_list[i]['ns'], res_list[i]['cr'], res_list[i]['ar'], res_list[i]['nr'], res_list[i]['nn'], res_list[i]['ne'], res_list[i]['cn'], res_list[i]['ee'], res_list[i]['el']]

                metrics_views_list = np.array(test_parallel)
                tot_time += time.time() - start_parallel

                # very specific format used for the pickled file of the metric results
                # do not tamper with this unless you are absolutely sure
                metrics_list.append((tech_name, 2, ns, cr, ar, nr, nn, ne, cn, ee, el, np.array([])))
                metrics_list.append((tech_name, 3, ns, cr, ar, nr, nn, ne, cn, ee, el, metrics_views_list))

                qm_names = constants.metrics
                df_metrics = pd.DataFrame.from_records(metrics_list)
                df_metrics.columns = ['layout_technique', 'n_components'] + qm_names + ['views_metrics']
                df_metrics.to_pickle(metrics_file)

            print('Total time taken for current graph: ' + str(round(tot_time, 2)))
