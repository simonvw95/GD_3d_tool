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
df_metrics = df_metrics.loc[:,['dataset_name', 'layout_technique', 'n_components', 'normalized_stress', 'angular_resolution', 'crossing_resolution', 'crossing_number', 'views_metrics']]

df_metrics.to_pickle(consolid_metrics_file)
