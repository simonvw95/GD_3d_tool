import os

import numpy as np
import pandas as pd

metrics = ['stress', 'crossing_resolution', 'angular_resolution', 'node_resolution', 'node-node_occlusion', 'node-edge_occlusion', 'crossing_number', 'edge-edge_occlusion', 'edge_length_deviation']

perplexities = {'3elt' : 120, 'bcsstk09' : 80, 'block_2000' : 120, 'cage8' : 80, 'CA-GrQc' : 120, 'can_96' : 20, 'dwt_72' : 15,
                'dwt_419' : 40, 'dwt_1005' : 40, 'EVA' : 600, 'grid17' : 40, 'jazz' : 120, 'lesmis' : 50, 'mesh3e1' : 40,
                'netscience' : 80, 'price_1000' : 80, 'rajat11' : 120, 'sierpinski3d' : 120, 'us_powergrid' : 160, 'visbrazil' : 120
                }
samples = 1000
hover_to_view = True #Toggle for switching views by either hovering over bars, or clicking on bars
scale_to_signal_range = False #Toglle for zooming in histograms
show_user_picked_viewpoints = False #Only works if evaluationdata is passed to the tool, (see evaluation_analysis.py)

metrics_dir = 'metrics'
output_dir = 'layouts'
analysis_dir = 'analysis'
metrics_projects_dir = 'metrics_projections'


user_mode = 'free' #options: ['free', 'eval_full', 'eval_half', 'image', 'evalimage']
#ordinal_datasets = ['Wine', 'Concrete', 'Software',]
ordinal_datasets = ['Grid']
#categorical_datasets = ['AirQuality', 'Reuters', 'WisconsinBreastCancer']
categorical_datasets = []

#evaluation_set = [('WisconsinBreastCancer', 'PCA'), ('Wine', 'TSNE'), ('Wine', 'PCA'), ('Concrete', 'TSNE'), ('Reuters', 'AE'), ('Reuters', 'TSNE'), ('Software', 'TSNE')]
#evaluation_set = [('Wine', 'AE'), ('Wine', 'MDS'), ('Wine', 'PCA'), ('Wine', 'UMAP')]
#required_view_count = 3
#output_file = 'evaluationdata/evaluationdata.pkl'

debug_mode = False


def get_consolid_metrics() -> pd.DataFrame:

    consolid_metrics = os.path.join(metrics_dir, 'metrics.pkl')
    df_consolid = pd.read_pickle(consolid_metrics)

    return df_consolid


def get_views_metrics(dataset, layout) -> np.ndarray:

    df_consolid = get_consolid_metrics()
    df = df_consolid.loc[(df_consolid['layout_name'] == layout)
                         & (df_consolid['dataset_name'] == f'{dataset}.pkl')
                         & (df_consolid['n_components'] == 3)]
    views_metrics = np.concatenate(df['views_metrics'].to_numpy())[:, 1:]

    return views_metrics
    #return the metric values of each view
