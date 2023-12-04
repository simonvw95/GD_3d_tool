import constants
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# get all the metric results
consolid_metrics = os.path.join(constants.metrics_dir, 'metrics.pkl')
data_frame = pd.read_pickle(consolid_metrics)
# get the names of the graphs, the names of the techniques and the names of the metrics
datasets = set(data_frame['dataset_name'].to_list())
techniques = ['SM', 'FA2', 'pivot_mds', 'tsNET', 'tsNETstar']
techniques_mapping = dict(zip(techniques, ['$SM$', '$FA2$', '$PivotMDS$', '$tsNET$', '$tsNET^{*}$']))
metrics = constants.metrics
metrics_col_idx = dict(zip(constants.metrics, range(len(constants.metrics))))
metrics_mapping = dict(zip(constants.metrics, ['ST', 'CR', 'AR', 'NR', 'NN', 'NE', 'CN', 'EE', 'EL']))

# first dataset, aggregated over metric.: {'metric1' : {'technique1' : [values], 'technique2' : [values]}, 'metric2' : ..}
metric_ds = {}
for m in metrics:
    col_idx = metrics_col_idx[m]
    res_dict = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
    for tech in techniques:
        for i in range(len(data_frame)):
            if (data_frame.iloc[i]['n_components'] == 3) and (data_frame.iloc[i]['layout_technique'] == tech):
                res_dict[tech].append(sum(data_frame.iloc[i]['views_metrics'][:, col_idx] > data_frame.iloc[i][m]) / constants.samples * 100)
    metric_ds[m] = res_dict

# 2nd dataset, aggregated over metric {'metric1' : [values], 'metric2' : ..}
metric_ds_agg = {}
for m in metrics:
    col_idx = metrics_col_idx[m]
    res = []
    for tech in techniques:
        for i in range(len(data_frame)):
            if (data_frame.iloc[i]['n_components'] == 3) and (data_frame.iloc[i]['layout_technique'] == tech):
                met_values = np.append(data_frame.iloc[i]['views_metrics'][:, col_idx], data_frame.iloc[i][m])
                res.append(met_values)
    metric_ds_agg[m] = np.array(res).flatten()


# 3rd dataset, aggregated over metric and subtracting the view values from the 2d value, metric distributions scaled to 0 and 1 {'metric1' : [values], 'metric2' : ..}
metric_ds_agg_rel_norm = {}
for m in metrics:
    col_idx = metrics_col_idx[m]
    res = []
    for graph in list(set(data_frame.dataset_name)):
        all_2d_vals = []
        all_viewpoint_vals = []
        all_vals = []
        for tech in techniques:
            for i in range(len(data_frame)):
                if (data_frame.iloc[i]['n_components'] == 3) and (data_frame.iloc[i]['layout_technique'] == tech) and (data_frame.iloc[i]['dataset_name'] == graph):
                    all_viewpoint_vals.append(data_frame.iloc[i]['views_metrics'][:, col_idx])
                    all_2d_vals.append(data_frame.iloc[i][m])

        # normalize for each graph in order to compare across graphs
        all_vals = np.append(np.array(all_viewpoint_vals).flatten(), np.array(all_2d_vals).flatten())
        min_local = np.min(all_vals)
        max_local = np.max(all_vals)
        scal_val_local = max_local - min_local
        all_viewpoint_vals = (np.array(all_viewpoint_vals).T - min_local) / scal_val_local
        all_2d_vals = (np.array(all_2d_vals).flatten() - min_local) / scal_val_local

        res.append(all_viewpoint_vals - all_2d_vals)

    metric_ds_agg_rel_norm[m] = np.array(res).flatten()


#################################################################################################################################
# boxplot for each metric side by side, all datapoints,
results_list = []
metric_list = []
for m in metrics:
    results_list += list(metric_ds_agg[m])
    metric_list += [m] * len(metric_ds_agg[m])

end_data = pd.DataFrame({'Metric values' : results_list, 'Metric' : metric_list})

plt.figure(figsize=(1000 / 96, 1000 / 96), dpi=96)
plt.xticks(rotation = 45, ha = "right")
ax = plt.gca()
#ax.set_title('Distributions of all datapoints', fontsize = 30)
g = sns.boxplot(x="Metric", y="Metric values", data=end_data)
sns.despine(offset=10, trim=True)
ax.set_xticklabels(list(metrics_mapping.values()))
ax.set_xticklabels(ax.get_xticklabels(), fontsize = 22)
#plt.xlabel('Metric', fontsize =26)
plt.ylabel('Metric value', fontsize = 26)
plt.yticks(fontsize = 22)
plt.tight_layout()
plt.savefig('evaluations/all_metrics_boxplot.png', bbox_inches = 'tight')
plt.clf()
plt.close('all')

#################################################################################################################################
# layout technique on x axis, one plot for each metric, checking the % of viewpoints that are better than 2d
for m in metrics:
    layout_technique_list = []
    results_list = []
    for tech in techniques:
        layout_technique_list += [techniques_mapping[tech]] * len(metric_ds['stress'][tech])
        results_list += metric_ds[m][tech]

    end_data = pd.DataFrame({'Layout technique': layout_technique_list, '% of viewpoints better than 2D' : results_list})

    plt.figure(figsize=(1000 / 96, 1000 / 96), dpi=96)
    # plt.xticks(rotation=45, ha="right")
    ax = plt.gca()
    # ax.set_title('Distributions of better viewpoints for ' + str(metrics_mapping[m]), fontsize = 30)
    sns.stripplot(data=end_data, x='Layout technique', y='% of viewpoints better than 2D', size=10, ax=ax, jitter=0.15, color = 'royalblue')
    plt.ylabel('% of viewpoints better than 2D ' + str(metrics_mapping[m]), fontsize = 26)
    #plt.xlabel('Layout Technique', fontsize=26)
    plt.xticks(fontsize = 22)
    plt.yticks(fontsize = 22)
    plt.savefig('evaluations/indiv_metric_' + str(metrics_mapping[m]) + '.png', bbox_inches = 'tight')
    plt.clf()
    plt.close('all')


#################################################################################################################################
# histogram for each metric, all datapoints
for m in metrics:

    plt.figure(figsize=(1000 / 96, 1000 / 96), dpi=96)
    plt.xticks(rotation=45, ha="right")
    ax = plt.gca()
    ax.set_title('All datapoints of ' + str(m))
    plt.hist(x = metric_ds_agg[m], bins = 100)
    plt.xlabel('Metric value')
    plt.ylabel('Frequency')
    plt.savefig('evaluations/indiv_metric_histogram_' + str(m) + '.png')
    plt.clf()
    plt.close('all')

#################################################################################################################################
# histogram for each metric, all normalized datapoints relative to their 2d Counterpart
for m in metrics:

    plt.figure(figsize=(1000 / 96, 1000 / 96), dpi=96)
    plt.xticks(rotation=45, ha="right")
    ax = plt.gca()
    ax.set_title('All datapoints of ' + str(m))
    plt.hist(x = metric_ds_agg_rel_norm[m], bins = 100)
    plt.xlabel('Metric value')
    plt.ylabel('Frequency')
    plt.savefig('evaluations/indiv_metric_rel_histogram_' + str(m) + '.png')
    plt.clf()
    plt.close('all')


#################################################################################################################################
# all datapoints relative to the 2d layout normalized w.r.t. graphs, boxplot

results_list = []
for m in metrics:
    temp = metric_ds_agg_rel_norm[m]
    results_list.append(temp)

results_list = np.array(results_list).T
end_data = pd.DataFrame(results_list, columns = metrics)

# splitting it up in 2 plots, adjust first and second line and title of the saved file
# fig, axes = plt.subplots(1, 9, figsize = (1000 / 80, 1000 / 40), dpi = 96)
fig, axes = plt.subplots(1, 9, figsize = (1000, 1000), dpi = 96)
for i, col in enumerate(metrics[:9]):
    ax = sns.boxplot(y=end_data[col], ax=axes.flatten()[i])
    curr_min = end_data[col].min()
    # ax.set_ylim(curr_min - (curr_min * 0.1), abs(curr_min) + (curr_min * 0.1))
    ax.set_ylim(-0.9, 0.9)
    ax.set_xlabel(metrics_mapping[col])
    ax.yaxis.label.set_visible(False)
    ax.xaxis.label.set_visible(False)
    ax.axhline(0, c = 'r')
    ax.set_xticklabels([plt.Text(0, 0, metrics_mapping[col])], fontsize = 26)
    ax.tick_params(labelsize = 22)

plt.tight_layout(pad = 2)
fig.subplots_adjust(top=0.95, bottom=0.1, left=0.13)
#plt.suptitle('Distributions of normalized differences', y = 0.995, fontsize = 32)
fig.supylabel('Normalized difference', fontsize = 26)
#fig.supxlabel('Metric', fontsize = 26)
plt.savefig('evaluations/all_values_agg_rel_boxplot_norm_all.png', bbox_inches = 'tight')
plt.clf()
plt.close('all')




######################################################################################################################
######################################################################################################################
######################################################################################################################
# projection plots
import os
import pandas as pd
import time
import constants
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib as mpl
import pickle
import pyqtgraph as pg
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

averages = {}
norm_metrics = {}
min_average = 1
max_average = 0
# first normalize the data
for ds in datasets:
    norm_metrics[ds] = {}
    averages[ds] = {}
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
        norm_metrics[ds][tech] = stacked_met
        means = np.mean(stacked_met, axis = 1)
        averages[ds][tech] = means
        if np.min(means) < min_average:
            min_average = np.min(means)
        if np.max(means) > max_average:
            max_average = np.max(means)

# then we want to find the minimum and maximum averages
scal_val_average = max_average - min_average
cmap = clrs.LinearSegmentedColormap.from_list("", [(0, "darkred"), (0.2, "red"), (0.5, "yellow"), (0.75, "green"), (1, "lightblue")])

# turn off global normalization for averages
# creating a scatter plot for each dataset + technique combination
global_normalization = False
# for ds in datasets:
for ds in ['visbrazil.pkl', 'us_powergrid.pkl', 'stufe.pkl', 'mesh3em5.pkl', 'grafo10236.pkl', 'grafo10232.pkl', 'grafo10229.pkl', 'grafo7223.pkl', 'grafo3890.pkl', 'CA-GrQc.pkl', 'jazz.pkl', 'grid17.pkl', 'grid.pkl']:
    curr_ds = ds[:-4]
    curr_data = pickle.load(open('metrics_projections/' + curr_ds + '_projcs_global.pkl', 'rb'))
    print('Starting projections for the metrics of ' + str(ds))
    for tech in techniques:
        # get the projection data
        curr_data_tech = curr_data[curr_data['layout_technique'] == tech]['projection'].to_numpy()[0]

        means = averages[ds][tech]
        if global_normalization:
            means = (means - min_average) / scal_val_average
        else:
            means = (means - np.min(means)) / (np.max(means) - np.min(means))

        color_range = np.array(range(0, 256, 1)) / 256

        temp_colors = [0] * len(means)
        for i in range(len(temp_colors)):
            # find out which idx to assign the current viewpoint to
            idx = np.argmin(abs(color_range - means[i]))
            temp_colors[i] = pg.mkBrush(cmap(idx, bytes = True)).color().name()

        plt.figure(figsize=(1000 / 96, 1000 / 96), dpi = 96)
        # sc = plt.scatter(x = curr_data_tech[:, 0], y = curr_data_tech[:, 1], c = means, cmap = cmap, vmin = 0, vmax = 1)
        sc = plt.scatter(x=curr_data_tech[:, 0], y=curr_data_tech[:, 1], c=temp_colors, vmin=0, vmax=1)
        plt.axis('off')
        plt.tight_layout()
        name = curr_ds + '_' + tech
        plt.savefig('evaluations/projections/' + name + '.pdf')
        plt.clf()
        plt.close('all')


###################################################################################################################################################
# turn off global normalization for averages
# creating a scatter plot for each dataset + metric combination
global_normalization = False
metrics = ['ST', 'CR', 'AR', 'NR', 'NN', 'NE', 'CN', 'EE', 'EL']
cmap = clrs.LinearSegmentedColormap.from_list("", [(0, "darkred"), (0.2, "red"), (0.5, "yellow"), (0.75, "green"), (1, "lightblue")])
bounds_dict = constants.bounds_dict
mins_global = np.array(list(bounds_dict.values()))[:, 0]
maxs_global = np.array(list(bounds_dict.values()))[:, 1]
# rajat tsNET
for ds in ['dwt_72.pkl', 'dwt_1005.pkl', 'grafo5197.pkl', 'grid17.pkl', 'mesh3e1.pkl', 'grafo10231.pkl']:
# for ds in ['rajat11.pkl']:
    curr_ds = ds[:-4]
    curr_data = pickle.load(open('metrics_projections/' + curr_ds + '_projcs_global.pkl', 'rb'))
    print('Starting projections for the metrics of ' + str(ds))
    for tech in ['SM']:
    # for tech in ['tsNET']:
        # getting the metrics in the righ format, normalized
        curr_data_tech = curr_data[curr_data['layout_technique'] == tech]['projection'].to_numpy()[0]

        curr_metrics = data_frame[(data_frame['dataset_name'] == ds) & (data_frame['layout_technique'] == tech) & (
                data_frame['n_components'] == 3)]
        metrics_2d = curr_metrics[constants.metrics].to_numpy()[0]
        # add metrics of 2d
        stacked_met = np.vstack([curr_metrics['views_metrics'].to_numpy()[0], metrics_2d])
        # normalize according to global values
        mins, maxs = mins_global, maxs_global
        stacked_met = (stacked_met - mins) / (maxs - mins)

        #####
        # all metrics combined

        means = np.mean(stacked_met, axis=1)
        means = (means - np.min(means)) / (np.max(means) - np.min(means))

        color_range = np.array(range(0, 256, 1)) / 256
        temp_colors = [0] * len(means)
        for i in range(len(temp_colors)):
            # find out which idx to assign the current viewpoint to
            idx = np.argmin(abs(color_range - means[i]))
            temp_colors[i] = pg.mkBrush(cmap(idx, bytes = True)).color().name()

        plt.figure(figsize=(1000 / 96, 1000 / 96), dpi = 96)
        # sc = plt.scatter(x = curr_data_tech[:, 0], y = curr_data_tech[:, 1], c = means, cmap = cmap, vmin = 0, vmax = 1)
        sc = plt.scatter(x=curr_data_tech[:, 0], y=curr_data_tech[:, 1], c=temp_colors, vmin=0, vmax=1)
        plt.axis('off')
        plt.tight_layout()
        name = curr_ds + '_' + tech + '_ALL.pdf'
        plt.savefig('evaluations/projections/metric_colored/' + name)
        plt.clf()
        plt.close('all')

        # stress and crossing number
        means_st_cn = np.mean(np.vstack((stacked_met[:, 0], stacked_met[:, 6])), axis=0)

        color_range = np.array(range(0, 256, 1)) / 256
        temp_colors = [0] * len(means_st_cn)
        for i in range(len(temp_colors)):
            # find out which idx to assign the current viewpoint to
            idx = np.argmin(abs(color_range - means_st_cn[i]))
            temp_colors[i] = pg.mkBrush(cmap(idx, bytes = True)).color().name()

        plt.figure(figsize=(1000 / 96, 1000 / 96), dpi = 96)
        # sc = plt.scatter(x = curr_data_tech[:, 0], y = curr_data_tech[:, 1], c = means, cmap = cmap, vmin = 0, vmax = 1)
        sc = plt.scatter(x=curr_data_tech[:, 0], y=curr_data_tech[:, 1], c=temp_colors, vmin=0, vmax=1)
        plt.axis('off')
        plt.tight_layout()
        name = curr_ds + '_' + tech + '_STCN.pdf'
        plt.savefig('evaluations/projections/metric_colored/' + name)
        plt.clf()
        plt.close('all')


        # loop over the metrics
        for m in range(len(metrics)):
        # get the projection data
            single_curr_metric = stacked_met[:, m]
            color_range = np.array(range(0, 256, 1)) / 256

            temp_colors = [0] * len(single_curr_metric)
            for i in range(len(temp_colors)):
                # find out which idx to assign the current viewpoint to
                idx = np.argmin(abs(color_range - single_curr_metric[i]))
                temp_colors[i] = pg.mkBrush(cmap(idx, bytes = True)).color().name()

            plt.figure(figsize=(1000 / 96, 1000 / 96), dpi = 96)
            # sc = plt.scatter(x = curr_data_tech[:, 0], y = curr_data_tech[:, 1], c = means, cmap = cmap, vmin = 0, vmax = 1)
            sc = plt.scatter(x=curr_data_tech[:, 0], y=curr_data_tech[:, 1], c=temp_colors, vmin=0, vmax=1)
            plt.axis('off')
            plt.tight_layout()
            name = curr_ds + '_' + tech + '_' + metrics[m]
            plt.savefig('evaluations/projections/metric_colored/' + name + '.pdf')
            plt.clf()
            plt.close('all')

###################################################################################################################################################
# turn off global normalization for averages
# creating a scatter plot for each dataset + metric combination, EXCLUDING CR and/or AR
global_normalization = False
metrics = ['ST', 'AR', 'NR', 'NN', 'NE', 'CN', 'EE', 'EL']
cmap = clrs.LinearSegmentedColormap.from_list("", [(0, "darkred"), (0.2, "red"), (0.5, "yellow"), (0.75, "green"), (1, "lightblue")])
bounds_dict = constants.bounds_dict
mins_global = np.array(list(bounds_dict.values()))[:, 0]
maxs_global = np.array(list(bounds_dict.values()))[:, 1]
# rajat tsNET
# for ds in ['dwt_72.pkl', 'dwt_1005.pkl', 'grafo5197.pkl', 'grid17.pkl', 'mesh3e1.pkl', 'grafo10231.pkl']:
for ds in ['rajat11.pkl']:
    curr_ds = ds[:-4]
    curr_data = pickle.load(open('metrics_projections_cr/' + curr_ds + '_projcs_global.pkl', 'rb'))
    print('Starting projections for the metrics of ' + str(ds))
    # for tech in ['SM']:
    for tech in ['tsNET']:
        # getting the metrics in the righ format, normalized
        curr_data_tech = curr_data[curr_data['layout_technique'] == tech]['projection'].to_numpy()[0]

        curr_metrics = data_frame[(data_frame['dataset_name'] == ds) & (data_frame['layout_technique'] == tech) & (
                data_frame['n_components'] == 3)]
        metrics_2d = curr_metrics[constants.metrics].to_numpy()[0]
        # add metrics of 2d
        stacked_met = np.vstack([curr_metrics['views_metrics'].to_numpy()[0], metrics_2d])
        # normalize according to global values
        mins, maxs = mins_global, maxs_global
        stacked_met = (stacked_met - mins) / (maxs - mins)
        stacked_met = np.delete(stacked_met, 1, axis = 1)

        #####
        # all metrics combined

        means = np.mean(stacked_met, axis=1)
        means = (means - np.min(means)) / (np.max(means) - np.min(means))

        color_range = np.array(range(0, 256, 1)) / 256
        temp_colors = [0] * len(means)
        for i in range(len(temp_colors)):
            # find out which idx to assign the current viewpoint to
            idx = np.argmin(abs(color_range - means[i]))
            temp_colors[i] = pg.mkBrush(cmap(idx, bytes = True)).color().name()

        plt.figure(figsize=(1000 / 96, 1000 / 96), dpi = 96)
        # sc = plt.scatter(x = curr_data_tech[:, 0], y = curr_data_tech[:, 1], c = means, cmap = cmap, vmin = 0, vmax = 1)
        sc = plt.scatter(x=curr_data_tech[:, 0], y=curr_data_tech[:, 1], c=temp_colors, vmin=0, vmax=1)
        plt.axis('off')
        plt.tight_layout()
        name = curr_ds + '_' + tech + '_ALL.pdf'
        plt.savefig('evaluations/projections/metric_colored/CR_removed/' + name)
        plt.clf()
        plt.close('all')

        # stress and crossing number
        means_st_cn = np.mean(np.vstack((stacked_met[:, 0], stacked_met[:, 6])), axis=0)

        color_range = np.array(range(0, 256, 1)) / 256
        temp_colors = [0] * len(means_st_cn)
        for i in range(len(temp_colors)):
            # find out which idx to assign the current viewpoint to
            idx = np.argmin(abs(color_range - means_st_cn[i]))
            temp_colors[i] = pg.mkBrush(cmap(idx, bytes = True)).color().name()

        plt.figure(figsize=(1000 / 96, 1000 / 96), dpi = 96)
        # sc = plt.scatter(x = curr_data_tech[:, 0], y = curr_data_tech[:, 1], c = means, cmap = cmap, vmin = 0, vmax = 1)
        sc = plt.scatter(x=curr_data_tech[:, 0], y=curr_data_tech[:, 1], c=temp_colors, vmin=0, vmax=1)
        plt.axis('off')
        plt.tight_layout()
        name = curr_ds + '_' + tech + '_STCN.pdf'
        plt.savefig('evaluations/projections/metric_colored/CR_removed/' + name)
        plt.clf()
        plt.close('all')


        # loop over the metrics
        for m in range(len(metrics)):
        # get the projection data
            single_curr_metric = stacked_met[:, m]
            color_range = np.array(range(0, 256, 1)) / 256

            temp_colors = [0] * len(single_curr_metric)
            for i in range(len(temp_colors)):
                # find out which idx to assign the current viewpoint to
                idx = np.argmin(abs(color_range - single_curr_metric[i]))
                temp_colors[i] = pg.mkBrush(cmap(idx, bytes = True)).color().name()

            plt.figure(figsize=(1000 / 96, 1000 / 96), dpi = 96)
            # sc = plt.scatter(x = curr_data_tech[:, 0], y = curr_data_tech[:, 1], c = means, cmap = cmap, vmin = 0, vmax = 1)
            sc = plt.scatter(x=curr_data_tech[:, 0], y=curr_data_tech[:, 1], c=temp_colors, vmin=0, vmax=1)
            plt.axis('off')
            plt.tight_layout()
            name = curr_ds + '_' + tech + '_' + metrics[m]
            plt.savefig('evaluations/projections/metric_colored/CR_removed/' + name + '.pdf')
            plt.clf()
            plt.close('all')




# the color bar
fig = plt.figure(figsize=(1000 / 96, 1000 / 96), dpi = 96)
ax = fig.add_axes([0.05, 0.80, 0.9, 0.1])
cb = mpl.colorbar.ColorbarBase(ax, orientation='horizontal',
                               cmap=cmap,
                               norm=mpl.colors.Normalize(0, 1),  # vmax and vmin
                               extend='both',
                               label='Color map of mean of metrics',
                               ticks=[0, .25, .5, .75, 1])

plt.savefig('evaluations/projections/just_colorbar.png', bbox_inches='tight')
plt.clf()
plt.close('all')


######################################################################
# graph layouts from viewpoints

import os
import pandas as pd
import time
import constants
import warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import matplotlib as mpl
import pickle
import networkx as nx
from sklearn.manifold import TSNE

warnings.filterwarnings("ignore")
consolid_metrics = os.path.join(constants.metrics_dir, 'metrics.pkl')
data_frame = pd.read_pickle(consolid_metrics)
D_P_dict = {}
datasets = set(data_frame['dataset_name'].to_list())
techniques = set(data_frame['layout_technique'].to_list())

# for ds in datasets:
for ds in ['visbrazil.pkl', 'us_powergrid.pkl', 'stufe.pkl', 'mesh3em5.pkl', 'grafo10236.pkl', 'grafo10232.pkl', 'grafo10229.pkl', 'grafo7223.pkl', 'grafo3890.pkl', 'CA-GrQc.pkl', 'jazz.pkl', 'grid17.pkl', 'grid.pkl']:
    curr_ds = ds[:-4]


    for tech in techniques:
        # first the 2d layout
        input_file = 'data/' + curr_ds + '/' + curr_ds + '-src.csv'
        df = pd.read_csv(input_file, sep=';', header=0)
        graph = nx.from_pandas_edgelist(df, 'from', 'to', edge_attr='strength')
        graph = nx.convert_node_labels_to_integers(graph)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        edges = list(graph.edges())
        nodes = list(graph.nodes())

        # get the layout data
        layout_file_2d = 'layouts/' + curr_ds + '-' + tech + '-2d.csv'
        df_2d = pd.read_csv(layout_file_2d, sep=';').to_numpy()

        pos_dict = {}
        for i in range(len(df_2d)):
            pos_dict[nodes[i]] = (df_2d[i][0], df_2d[i][1])

        nx.draw(graph, pos = pos_dict, node_size = 10, node_color = 'gray')
        plt.axis('off')
        plt.tight_layout()
        name = curr_ds + '_' + tech
        plt.savefig('evaluations/projections/' + name + '_2d.pdf')
        plt.clf()
        plt.close('all')

        # excluding the 2D layout
        worst_idc = np.argmin(averages[ds][tech][0:1000])
        best_idc = np.argmax(averages[ds][tech][0:1000])
        viewpoints = pd.read_pickle('layouts/' + curr_ds + '-' + tech + '-views.pkl')['views'].to_numpy()

        worst_viewpoint = viewpoints[worst_idc]
        pos_dict = {}
        for i in range(len(worst_viewpoint)):
            pos_dict[nodes[i]] = (worst_viewpoint[i][0], worst_viewpoint[i][1])

        nx.draw(graph, pos = pos_dict, node_size = 10, node_color = 'gray')
        plt.axis('off')
        plt.tight_layout()
        name = curr_ds + '_' + tech
        plt.savefig('evaluations/projections/' + name + '_worst.pdf')
        plt.clf()
        plt.close('all')

        best_viewpoint = viewpoints[best_idc]
        pos_dict = {}
        for i in range(len(best_viewpoint)):
            pos_dict[nodes[i]] = (best_viewpoint[i][0], best_viewpoint[i][1])

        nx.draw(graph, pos = pos_dict, node_size = 10, node_color = 'gray')
        plt.axis('off')
        plt.tight_layout()
        name = curr_ds + '_' + tech
        plt.savefig('evaluations/projections/' + name + '_best.pdf')
        plt.clf()
        plt.close('all')


