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

for ds in datasets:
    if not os.path.isfile(constants.metrics_projects_dir + '/' + ds.split('.')[0] + '_projcs.pkl'):
        print('Starting projections for the metrics of ' + str(ds))
        projs_list = []
        start = time.time()
        for tech in techniques:
            curr_metrics = data_frame[(data_frame['dataset_name'] == ds) & (data_frame['layout_technique'] == tech) & (data_frame['n_components'] == 3)]
            metrics_2d = curr_metrics[constants.metrics].to_numpy()[0]
            # add metrics of 2d
            project = TSNE(n_components = 2, perplexity = 100).fit_transform(np.vstack([curr_metrics['views_metrics'].to_numpy()[0], metrics_2d]))
            projs_list.append([ds, tech, project])

        print('Creating projections took: ' + str(round(time.time() - start, 2)) + 's')

        df_projs = pd.DataFrame.from_records(projs_list)
        df_projs.columns = ['dataset_name', 'layout_technique', 'projection']
        df_projs.to_pickle(constants.metrics_projects_dir + '/' + ds.split('.')[0] + '_projcs.pkl')
