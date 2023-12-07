import os
import pandas as pd
import time
import constants
import warnings
import numpy as np

from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
consolid_metrics = os.path.join(constants.metrics_dir, 'metrics.pkl')
data_frame = pd.read_pickle(consolid_metrics)
D_P_dict = {}
datasets = set(data_frame['dataset_name'].to_list())
techniques = set(data_frame['layout_technique'].to_list())
consolid_metrics_file = os.path.join(constants.metrics_dir, 'metrics.pkl')
bounds_dict = constants.bounds_dict
mins_global = np.array(list(bounds_dict.values()))[:, 0]
maxs_global = np.array(list(bounds_dict.values()))[:, 1]

global_normalization = True

for ds in datasets:

    name = constants.metrics_projects_dir + '/' + ds.split('.')[0] + '_projcs_local.pkl'
    if global_normalization:
        name = constants.metrics_projects_dir + '/' + ds.split('.')[0] + '_projcs_global.pkl'

    if not os.path.isfile(name):
        print('Starting projections for the metrics of ' + str(ds))
        projs_list = []
        start = time.time()
        for tech in techniques:
            curr_metrics = data_frame[(data_frame['dataset_name'] == ds) & (data_frame['layout_technique'] == tech) & (data_frame['n_components'] == 3)]
            metrics_2d = curr_metrics[constants.metrics].to_numpy()[0]
            # add metrics of 2d
            stacked_met = np.vstack([curr_metrics['views_metrics'].to_numpy()[0], metrics_2d])
            # using global values seen from all viewpoints across all techniques and datasets
            if global_normalization:
                mins, maxs = mins_global, maxs_global
            # using local values only from the viewpoints from this technique and this dataset
            else:
                mins, maxs = np.min(stacked_met, axis=0), np.max(stacked_met, axis=0)
            stacked_met = (stacked_met - mins) / (maxs - mins)
            # nan values can appear if max==min
            stacked_met = np.nan_to_num(stacked_met, nan = 1)
            project = TSNE(n_components = 2, perplexity = 100).fit_transform(stacked_met)
            projs_list.append([ds, tech, project])

        print('Creating projections took: ' + str(round(time.time() - start, 2)) + 's')

        df_projs = pd.DataFrame.from_records(projs_list)
        df_projs.columns = ['dataset_name', 'layout_technique', 'projection']

        df_projs.to_pickle(name)
