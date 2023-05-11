import os
import pandas as pd
from glob import glob

import constants

consolid_metrics_file = os.path.join(constants.metrics_dir, 'metrics.pkl')
metrics_files = glob(os.path.join(constants.metrics_dir, 'metrics_*.pkl'))

dfs = []
datasets = []
columns = []

for m in metrics_files:
    df = pd.read_pickle(m)
    dfs.append(df)
    datasets += [os.path.basename(m).split('_')[1].replace('.csv', '') for i in range(len(df))]
    columns = df.columns

df_metrics = pd.concat(dfs)
df_metrics.columns = columns
df_metrics['dataset_name'] = datasets

qm_names = ['stress', 'crossing_resolution', 'angular_resolution', 'node-node_occlusion', 'node-edge_occlusion', 'crossing_number', 'edge-edge_occlusion']

df_metrics = df_metrics.loc[:,['dataset_name', 'layout_technique', 'n_components'] + qm_names + ['views_metrics']]

df_metrics.to_pickle(consolid_metrics_file)
