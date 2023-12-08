import pandas as pd
import numpy as np
import os

import constants

# first turn the json files of the metrics to pickled files
for file in os.listdir(constants.metrics_dir):
    # only read json files
    if '.json' in file:
        temp = pd.read_json(constants.metrics_dir + '/' + file, orient = 'split', precise_float = True)
        # turn python lists to numpy arrays as these are used in the original code
        temp['views_metrics'] = temp['views_metrics'].apply(lambda x: np.array(x))
        temp.to_pickle(constants.metrics_dir + '/' + file[:-5] + '.pkl')

# second turn the json files of the projections to pickled files
for file in os.listdir(constants.metrics_projects_dir):
    if '.json' in file:
        temp = pd.read_json(constants.metrics_projects_dir + '/' + file, orient = 'split', precise_float = True)
        temp['projection'] = temp['projection'].apply(lambda x: np.array(x))
        temp.to_pickle(constants.metrics_projects_dir + '/' + file[:-5] + '.pkl')

