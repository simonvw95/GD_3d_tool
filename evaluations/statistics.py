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
metrics = constants.metrics
metrics_col_idx = dict(zip(constants.metrics, range(len(constants.metrics))))

techniques = ['SM', 'FA2', 'pivot_mds', 'tsNET', 'tsNETstar']
metrics_map = dict(zip(metrics, ['ST', 'CR', 'AR', 'NR', 'NN', 'NE', 'CN', 'EE', 'EL']))
# third dataset, aggregated over metric.: {'metric1' : {'technique1' : [values], 'technique2' : [values]}, 'metric2' : ..}

# creating a text file for the stats of the quality metrics
metric_ds = {}
lines = []
for m in metrics:
    col_idx = metrics_col_idx[m]
    res_dict = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
    mins = {}
    maxs = {}
    means = {}
    stds = {}
    all = {}
    for tech in techniques:
        for i in range(len(data_frame)):
            if (data_frame.iloc[i]['n_components'] == 3) and (data_frame.iloc[i]['layout_technique'] == tech):
                curr_array = np.hstack((data_frame.iloc[i]['views_metrics'][:, col_idx], np.array(data_frame.iloc[i][m])))
                res_dict[tech].append(curr_array)

        mins[tech] = np.min(np.array(res_dict[tech]).flatten())
        maxs[tech] = np.max(np.array(res_dict[tech]).flatten())
        means[tech] = np.mean(np.array(res_dict[tech]).flatten())
        stds[tech] = np.std(np.array(res_dict[tech]).flatten())

    metric_ds[m] = res_dict
    all['min'] = np.min(np.array(list(res_dict.values())).flatten())
    all['mean'] = np.mean(np.array(list(res_dict.values())).flatten())
    all['max'] = np.max(np.array(list(res_dict.values())).flatten())
    all['std'] = np.std(np.array(list(res_dict.values())).flatten())

    curr_metric = metrics_map[m]
    first_line = r' %multirow{4}{*}{%textbf{'+curr_metric+'}} & $min$ & '+qr(mins['SM'])+' & '+qr(mins['FA2'])+' & '+qr(mins['pivot_mds'])+' & '+qr(mins['tsNET'])+' & '+qr(mins['tsNETstar'])+' & '+qr(all['min'])+'%%'
    second_line = r' %cline{2-8}'
    third_line = r' & $mean$ & '+qr(means['SM'])+' & '+qr(means['FA2'])+' & '+qr(means['pivot_mds'])+' & '+qr(means['tsNET'])+' & '+qr(means['tsNETstar'])+' & '+qr(all['mean'])+'%%'
    fourth_line = r' %cline{2-8}'
    fifth_line = r' & $max$ & '+qr(maxs['SM'])+' & '+qr(maxs['FA2'])+' & '+qr(maxs['pivot_mds'])+' & '+qr(maxs['tsNET'])+' & '+qr(maxs['tsNETstar'])+' & '+qr(all['max'])+'%%'
    sixth_line = r' %cline{2-8}'
    seventh_line = r' & $sd$ & '+qr(stds['SM'])+' & '+qr(stds['FA2'])+' & '+qr(stds['pivot_mds'])+' & '+qr(stds['tsNET'])+' & '+qr(stds['tsNETstar'])+' & '+qr(all['std'])+'%%'
    eigth_line = r' %hline'
    lines += [first_line, second_line, third_line, fourth_line, fifth_line, sixth_line, seventh_line, eigth_line]



# creating a text file for the stats of the quality metrics that have better scores than 2d
# metric_ds = {}
# lines = []
# for m in metrics:
#     col_idx = metrics_col_idx[m]
#     res_dict = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
#     mins = {}
#     maxs = {}
#     means = {}
#     stds = {}
#     all = {}
#     flattened_all = np.empty((0))
#     for tech in techniques:
#         for i in range(len(data_frame)):
#             if (data_frame.iloc[i]['n_components'] == 3) and (data_frame.iloc[i]['layout_technique'] == tech):
#                 mask = data_frame.iloc[i]['views_metrics'][:, col_idx] >= np.array(data_frame.iloc[i][m])
#                 curr_array = data_frame.iloc[i]['views_metrics'][mask, col_idx]
#                 res_dict[tech].append(curr_array)
#
#         flattened = np.hstack(np.array(res_dict[tech]))
#         mins[tech] = np.min(flattened)
#         maxs[tech] = np.max(flattened)
#         means[tech] = np.mean(flattened)
#         stds[tech] = np.std(flattened)
#         flattened_all = np.concatenate((flattened_all, flattened))
#
#     metric_ds[m] = res_dict
#     all['min'] = np.min(flattened_all)
#     all['mean'] = np.mean(flattened_all)
#     all['max'] = np.max(flattened_all)
#     all['std'] = np.std(flattened_all)
#
#     curr_metric = metrics_map[m]
#     first_line = r' %multirow{4}{*}{%textbf{' + curr_metric + '}} & $min$ & ' + qr(mins['SM']) + ' & ' + qr(
#         mins['FA2']) + ' & ' + qr(mins['pivot_mds']) + ' & ' + qr(mins['tsNET']) + ' & ' + qr(
#         mins['tsNETstar']) + ' & ' + qr(all['min']) + '%%'
#     second_line = r' %cline{2-8}'
#     third_line = r' & $mean$ & '+qr(means['SM'])+' & '+qr(means['FA2'])+' & '+qr(means['pivot_mds'])+' & '+qr(means['tsNET'])+' & '+qr(means['tsNETstar'])+' & '+qr(all['mean'])+'%%'
#     fourth_line = r' %cline{2-8}'
#     fifth_line = r' & $max$ & '+qr(maxs['SM'])+' & '+qr(maxs['FA2'])+' & '+qr(maxs['pivot_mds'])+' & '+qr(maxs['tsNET'])+' & '+qr(maxs['tsNETstar'])+' & '+qr(all['max'])+'%%'
#     sixth_line = r' %cline{2-8}'
#     seventh_line = r' & $sd$ & '+qr(stds['SM'])+' & '+qr(stds['FA2'])+' & '+qr(stds['pivot_mds'])+' & '+qr(stds['tsNET'])+' & '+qr(stds['tsNETstar'])+' & '+qr(all['std'])+'%%'
#     eigth_line = r' %hline'
#     lines += [first_line, second_line, third_line, fourth_line, fifth_line, sixth_line, seventh_line, eigth_line]


# creating a text file for the percentages of the viewpoints that are better than 2D
metric_ds = {}
metric_ds_all_min = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
metric_ds_all_median = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
metric_ds_all_max = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
metric_ds_all_std = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
metric_ds_st_cn_min = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
metric_ds_st_cn_median = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
metric_ds_st_cn_max = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
metric_ds_st_cn_std = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
lines = []
for m in metrics:
    col_idx = metrics_col_idx[m]
    res_dict = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
    mins = {}
    maxs = {}
    medians = {}
    stds = {}
    all = {}
    flattened_all = np.empty((0))
    for tech in techniques:
        for i in range(len(data_frame)):
            if (data_frame.iloc[i]['n_components'] == 3) and (data_frame.iloc[i]['layout_technique'] == tech):
                mask = data_frame.iloc[i]['views_metrics'][:, col_idx] >= np.array(data_frame.iloc[i][m])
                curr_array = np.mean(mask) * 100
                res_dict[tech].append(curr_array)

        flattened = np.hstack(np.array(res_dict[tech]))
        mins[tech] = np.min(flattened)
        maxs[tech] = np.max(flattened)
        medians[tech] = np.median(flattened)
        stds[tech] = np.std(flattened)
        flattened_all = np.concatenate((flattened_all, flattened))

    metric_ds[m] = res_dict
    all['min'] = np.min(flattened_all)
    all['median'] = np.median(flattened_all)
    all['max'] = np.max(flattened_all)
    all['std'] = np.mean(flattened_all)

    for tech in techniques:

        metric_ds_all_min[tech].append(np.min(mins[tech]))
        metric_ds_all_median[tech].append(np.mean(medians[tech]))
        metric_ds_all_max[tech].append(np.max(maxs[tech]))
        metric_ds_all_std[tech].append(np.mean(stds[tech]))

        if m == 'stress' or m == 'crossing_number':
            metric_ds_st_cn_min[tech].append(np.min(mins[tech]))
            metric_ds_st_cn_median[tech].append(np.mean(medians[tech]))
            metric_ds_st_cn_max[tech].append(np.max(maxs[tech]))
            metric_ds_st_cn_std[tech].append(np.mean(stds[tech]))

    curr_metric = metrics_map[m]
    # first_line = r' %multirow{4}{*}{%textbf{' + curr_metric + '}} & $min$ & ' + qr(mins['SM'], sci_not = False) + ' & ' + qr(
    #     mins['FA2'], sci_not = False) + ' & ' + qr(mins['pivot_mds'], sci_not = False) + ' & ' + qr(mins['tsNET'], sci_not = False) + ' & ' + qr(
    #     mins['tsNETstar'], sci_not = False) + ' & ' + qr(all['min'], sci_not = False) + '%%'
    # second_line = r' %cline{2-8}'
    third_line = r' %multirow{2}{*}{%textbf{' + curr_metric + '}} & $median$ & '+qr(medians['SM'], sci_not = False)+' & '+qr(medians['FA2'], sci_not = False)+' & '+qr(medians['pivot_mds'], sci_not = False)+' & '+qr(medians['tsNET'], sci_not = False)+' & '+qr(medians['tsNETstar'], sci_not = False)+' & '+qr(all['median'], sci_not = False)+'%%'
    fourth_line = r' %cline{2-8}'
    fifth_line = r' & $max$ & '+qr(maxs['SM'], sci_not = False)+' & '+qr(maxs['FA2'], sci_not = False)+' & '+qr(maxs['pivot_mds'], sci_not = False)+' & '+qr(maxs['tsNET'], sci_not = False)+' & '+qr(maxs['tsNETstar'], sci_not = False)+' & '+qr(all['max'], sci_not = False)+'%%'
    # sixth_line = r' %cline{2-8}'
    # seventh_line = r' & $sd$ & '+qr(stds['SM'], sci_not = False)+' & '+qr(stds['FA2'], sci_not = False)+' & '+qr(stds['pivot_mds'], sci_not = False)+' & '+qr(stds['tsNET'], sci_not = False)+' & '+qr(stds['tsNETstar'], sci_not = False)+' & '+qr(all['std'], sci_not = False)+'%%'
    eigth_line = r' %hline'
    # lines += [first_line, second_line, third_line, fourth_line, fifth_line, sixth_line, seventh_line, eigth_line]
    lines += [third_line, fourth_line, fifth_line, eigth_line]

metric_agg_min_st_cn = np.min(np.array(list(metric_ds_st_cn_min.values())).flatten())
metric_agg_median_st_cn = np.mean(np.array(list(metric_ds_st_cn_median.values())).flatten())
metric_agg_max_st_cn = np.max(np.array(list(metric_ds_st_cn_max.values())).flatten())
metric_agg_std_st_cn = np.mean(np.array(list(metric_ds_st_cn_std.values())).flatten())

# first_line = r' %multirow{4}{*}{%textbf{ST+CN}} & $min$ & ' + qr(np.min(metric_ds_st_cn_min['SM']), sci_not = False) + ' & ' + qr(
#     np.min(metric_ds_st_cn_min['FA2']), sci_not = False) + ' & ' + qr(np.min(metric_ds_st_cn_min['pivot_mds']), sci_not = False) + ' & ' + qr(np.min(metric_ds_st_cn_min['tsNET']), sci_not = False) + ' & ' + qr(
#     np.min(metric_ds_st_cn_min['tsNETstar']), sci_not = False) + ' & ' + qr(metric_agg_min_st_cn, sci_not = False) + '%%'
#
# second_line = r' %cline{2-8}'
third_line = r' %multirow{2}{*}{%textbf{ST+CN}} & $median$ & ' + qr(np.mean(metric_ds_st_cn_median['SM']), sci_not = False) + ' & ' + qr(np.mean(metric_ds_st_cn_median['FA2']), sci_not = False) + ' & ' + qr(
    np.mean(metric_ds_st_cn_median['pivot_mds']), sci_not = False) + ' & ' + qr(np.mean(metric_ds_st_cn_median['tsNET']), sci_not = False) + ' & ' + qr(np.mean(metric_ds_st_cn_median['tsNETstar']), sci_not = False) + ' & ' + qr(metric_agg_median_st_cn, sci_not = False) + '%%'

fourth_line = r' %cline{2-8}'
fifth_line = r' & $max$ & ' + qr(np.max(metric_ds_st_cn_max['SM']), sci_not = False) + ' & ' + qr(np.max(metric_ds_st_cn_max['FA2']), sci_not = False) + ' & ' + qr(
    np.max(metric_ds_st_cn_max['pivot_mds']), sci_not = False) + ' & ' + qr(np.max(metric_ds_st_cn_max['tsNET']), sci_not = False) + ' & ' + qr(np.max(metric_ds_st_cn_max['tsNETstar']), sci_not = False) + ' & ' + qr(metric_agg_max_st_cn, sci_not = False) + '%%'

# sixth_line = r' %cline{2-8}'
# seventh_line = r' & $sd$ & ' + qr(np.mean(metric_ds_st_cn_std['SM']), sci_not = False) + ' & ' + qr(np.mean(metric_ds_st_cn_std['FA2']), sci_not = False) + ' & ' + qr(
#     np.mean(metric_ds_st_cn_std['pivot_mds']), sci_not = False) + ' & ' + qr(np.mean(metric_ds_st_cn_std['tsNET']), sci_not = False) + ' & ' + qr(np.mean(metric_ds_st_cn_std['tsNETstar']), sci_not = False) + ' & ' + qr(metric_agg_std_st_cn, sci_not = False) + '%%'
eigth_line = r' %hline'
# lines += [first_line, second_line, third_line, fourth_line, fifth_line, sixth_line, seventh_line, eigth_line]
lines += [third_line, fourth_line, fifth_line, eigth_line]

metric_agg_min = np.min(np.array(list(metric_ds_all_min.values())).flatten())
metric_agg_median = np.mean(np.array(list(metric_ds_all_median.values())).flatten())
metric_agg_max = np.max(np.array(list(metric_ds_all_max.values())).flatten())
metric_agg_std = np.mean(np.array(list(metric_ds_all_std.values())).flatten())

# first_line = r' %multirow{4}{*}{%textbf{ALL}} & $min$ & ' + qr(np.min(metric_ds_all_min['SM']), sci_not = False) + ' & ' + qr(
#     np.min(metric_ds_all_min['FA2']), sci_not = False) + ' & ' + qr(np.min(metric_ds_all_min['pivot_mds']), sci_not = False) + ' & ' + qr(np.min(metric_ds_all_min['tsNET']), sci_not = False) + ' & ' + qr(
#     np.min(metric_ds_all_min['tsNETstar']), sci_not = False) + ' & ' + qr(metric_agg_min, sci_not = False) + '%%'

# second_line = r' %cline{2-8}'
third_line = r' %multirow{2}{*}{%textbf{ALL}} & $median$ & ' + qr(np.mean(metric_ds_all_median['SM']), sci_not = False) + ' & ' + qr(np.mean(metric_ds_all_median['FA2']), sci_not = False) + ' & ' + qr(
    np.mean(metric_ds_all_median['pivot_mds']), sci_not = False) + ' & ' + qr(np.mean(metric_ds_all_median['tsNET']), sci_not = False) + ' & ' + qr(np.mean(metric_ds_all_median['tsNETstar']), sci_not = False) + ' & ' + qr(metric_agg_median, sci_not = False) + '%%'

fourth_line = r' %cline{2-8}'
fifth_line = r' & $max$ & ' + qr(np.max(metric_ds_all_max['SM']), sci_not = False) + ' & ' + qr(np.max(metric_ds_all_max['FA2']), sci_not = False) + ' & ' + qr(
    np.max(metric_ds_all_max['pivot_mds']), sci_not = False) + ' & ' + qr(np.max(metric_ds_all_max['tsNET']), sci_not = False) + ' & ' + qr(np.max(metric_ds_all_max['tsNETstar']), sci_not = False) + ' & ' + qr(metric_agg_max, sci_not = False) + '%%'

# sixth_line = r' %cline{2-8}'
# seventh_line = r' & $sd$ & ' + qr(np.mean(metric_ds_all_std['SM']), sci_not = False) + ' & ' + qr(np.mean(metric_ds_all_std['FA2']), sci_not = False) + ' & ' + qr(
#     np.mean(metric_ds_all_std['pivot_mds']), sci_not = False) + ' & ' + qr(np.mean(metric_ds_all_std['tsNET']), sci_not = False) + ' & ' + qr(np.mean(metric_ds_all_std['tsNETstar']), sci_not = False) + ' & ' + qr(metric_agg_std, sci_not = False) + '%%'
# eigth_line = r' %hline'

# lines += [first_line, second_line, third_line, fourth_line, fifth_line, sixth_line, seventh_line, eigth_line]
lines += [third_line, fourth_line, fifth_line, eigth_line]





# creating a text file for the stats of the quality metrics relative to their 2D drawing (including %), normalizing based on GRAPHS
metric_ds = dict(zip(list(metrics), (dict(zip(list(techniques), ([] for _ in range(len(techniques))))) for _ in range(len(metrics)))))
metric_ds_all_min = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
metric_ds_all_median = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
metric_ds_all_max = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
metric_ds_all_std = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
metric_ds_st_cn_min = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
metric_ds_st_cn_median = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
metric_ds_st_cn_max = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
metric_ds_st_cn_std = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
lines = []
for m in metrics:
    col_idx = metrics_col_idx[m]
    mins = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
    maxs = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
    medians = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
    stds = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
    # mins_perc = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
    # maxs_perc = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
    # means_perc = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
    # stds_perc = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
    all = {}
    # all_perc = {}
    for graph in list(set(data_frame.dataset_name)):
        graph_dict = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
        # graph_dict_perc = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
        twoD_dict = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
        for tech in techniques:
            for i in range(len(data_frame)):
                if (data_frame.iloc[i]['n_components'] == 3) and (data_frame.iloc[i]['layout_technique'] == tech) and (data_frame.iloc[i]['dataset_name'] == graph):
                    curr_array = data_frame.iloc[i]['views_metrics'][:, col_idx]
                    graph_dict[tech].append(curr_array)
                    twoD_dict[tech].append(np.array(data_frame.iloc[i][m]))

        # normalize for each graph in order to compare across graphs
        all_viewpoint_values = np.array(list(graph_dict.values())).flatten()
        all_2d_values = np.array(list(twoD_dict.values())).flatten()
        min_local = np.min([np.min(all_viewpoint_values), np.min(all_2d_values)])
        max_local = np.max([np.max(all_viewpoint_values), np.max(all_2d_values)])
        scal_val_local = max_local - min_local

        for tech in techniques:
            graph_dict[tech] = (graph_dict[tech] - min_local) / scal_val_local
            twoD_scaled = ((twoD_dict[tech] - min_local) / scal_val_local)
            graph_dict[tech] = graph_dict[tech] - twoD_scaled

            mins[tech].append(np.min(graph_dict[tech]))
            maxs[tech].append(np.max(graph_dict[tech]))
            medians[tech].append(np.median(graph_dict[tech]))
            stds[tech].append(np.std(graph_dict[tech]))

            # if twoD_scaled == 0:
            #     twoD_scaled = 0.00001
            # graph_dict_perc[tech] = (graph_dict[tech] / twoD_scaled) * 100
            # mins_perc[tech].append(np.min(graph_dict_perc[tech]))
            # maxs_perc[tech].append(np.max(graph_dict_perc[tech]))
            # means_perc[tech].append(np.mean(graph_dict_perc[tech]))
            # stds_perc[tech].append(np.std(graph_dict_perc[tech]))

            metric_ds[m][tech] = medians[tech]

    mins_global = {}
    maxs_global = {}
    medians_global = {}
    stds_global = {}

    # mins_global_perc = {}
    # maxs_global_perc = {}
    # means_global_perc = {}
    # stds_global_perc = {}
    for tech in techniques:
        mins_global[tech] = np.min(mins[tech])
        maxs_global[tech] = np.max(maxs[tech])
        medians_global[tech] = np.mean(medians[tech])
        stds_global[tech] = np.std(stds[tech])

        metric_ds_all_min[tech].append(mins_global[tech])
        metric_ds_all_median[tech].append(medians_global[tech])
        metric_ds_all_max[tech].append(maxs_global[tech])
        metric_ds_all_std[tech].append(stds_global[tech])
        if m == 'stress' or m == 'crossing_number':
            metric_ds_st_cn_min[tech].append(mins_global[tech])
            metric_ds_st_cn_median[tech].append(medians_global[tech])
            metric_ds_st_cn_max[tech].append(maxs_global[tech])
            metric_ds_st_cn_std[tech].append(stds_global[tech])

        # mins_global_perc[tech] = np.min(mins_perc[tech])
        # maxs_global_perc[tech] = np.max(maxs_perc[tech])
        # means_global_perc[tech] = np.mean(means_perc[tech])
        # stds_global_perc[tech] = np.std(stds_perc[tech])

    all['min'] = np.min(np.array(list(mins_global.values())).flatten())
    all['median'] = np.mean(np.array(list(medians_global.values())).flatten())
    all['max'] = np.max(np.array(list(maxs_global.values())).flatten())
    all['std'] = np.mean(np.array(list(stds_global.values())).flatten())

    # all_perc['min'] = np.min(np.array(list(mins_global_perc.values())).flatten())
    # all_perc['mean'] = np.mean(np.array(list(means_global_perc.values())).flatten())
    # all_perc['max'] = np.max(np.array(list(maxs_global_perc.values())).flatten())
    # all_perc['std'] = np.mean(np.array(list(stds_global_perc.values())).flatten())

    curr_metric = metrics_map[m]
    # first_line = r' %multirow{4}{*}{%textbf{' + curr_metric + '}} & $min$ & ' + qr(mins_global['SM']) + '|' +qr(mins_global_perc['SM'], sci_not = True)+' & ' + qr(
    #     mins_global['FA2']) + '|' +qr(mins_global_perc['FA2'], sci_not = True)+' & ' + qr(mins_global['pivot_mds']) + '|' +qr(mins_global_perc['pivot_mds'], sci_not = True)+' & ' + qr(mins_global['tsNET']) + '|' +qr(mins_global_perc['tsNET'], sci_not = True)+' & ' + qr(
    #     mins_global['tsNETstar']) + '|' +qr(mins_global_perc['tsNETstar'], sci_not = True)+' & ' + qr(all['min']) + '|' +qr(all_perc['min'], sci_not = True)+'%%'
    #
    # second_line = r' %cline{2-8}'
    # third_line = r' & $mean$ & '+qr(means_global['SM']) + '|' +qr(means_global_perc['SM'], sci_not = True)+' & '+qr(means_global['FA2']) + '|' +qr(means_global_perc['FA2'], sci_not = True)+' & '+qr(means_global['pivot_mds']) + '|' +qr(means_global_perc['pivot_mds'], sci_not = True)+' & '+qr(means_global['tsNET']) + '|' +qr(means_global_perc['tsNET'], sci_not = True)+' & '+qr(means_global['tsNETstar'])+ '|' +qr(means_global_perc['tsNETstar'], sci_not = True)+' & '+qr(all['mean']) + '|' +qr(all_perc['mean'], sci_not = True)+'%%'
    #
    # fourth_line = r' %cline{2-8}'
    # fifth_line = r' & $max$ & '+qr(maxs_global['SM'])+ '|' +qr(maxs_global_perc['SM'], sci_not = True)+' & '+qr(maxs_global['FA2'])+ '|' +qr(maxs_global_perc['FA2'], sci_not = True)+' & '+qr(maxs_global['pivot_mds'])+ '|' +qr(maxs_global_perc['pivot_mds'], sci_not = True)+' & '+qr(maxs_global['tsNET'])+ '|' +qr(maxs_global_perc['tsNET'], sci_not = True)+' & '+qr(maxs_global['tsNETstar'])+ '|' +qr(maxs_global_perc['tsNETstar'], sci_not = True)+' & '+qr(all['max'])+ '|' +qr(all_perc['max'], sci_not = True)+'%%'
    #
    # sixth_line = r' %cline{2-8}'
    # seventh_line = r' & $sd$ & '+qr(stds_global['SM'])+ '|' +qr(stds_global_perc['SM'], sci_not = True)+' & '+qr(stds_global['FA2'])+ '|' +qr(stds_global_perc['FA2'], sci_not = True)+' & '+qr(stds_global['pivot_mds'])+ '|' +qr(stds_global_perc['pivot_mds'], sci_not = True)+' & '+qr(stds_global['tsNET'])+ '|' +qr(stds_global_perc['tsNET'], sci_not = True)+' & '+qr(stds_global['tsNETstar'])+ '|' +qr(stds_global_perc['tsNETstar'], sci_not = True)+' & '+qr(all['std'])+ '|' +qr(all_perc['std'], sci_not = True)+'%%'
    # eigth_line = r' %hline'
    # first_line = r' %multirow{4}{*}{%textbf{' + curr_metric + '}} & $min$ & ' + qr(mins_global['SM'])+' & ' + qr(
    #     mins_global['FA2'])+' & ' + qr(mins_global['pivot_mds'])+' & ' + qr(mins_global['tsNET'])+' & ' + qr(
    #     mins_global['tsNETstar'])+' & ' + qr(all['min'])+'%%'
    #
    # second_line = r' %cline{2-8}'
    third_line = r' %multirow{2}{*}{%textbf{' + curr_metric + '}} & $median$ & '+qr(medians_global['SM'])+' & '+qr(medians_global['FA2'])+' & '+qr(medians_global['pivot_mds'])+' & '+qr(medians_global['tsNET'])+' & '+qr(medians_global['tsNETstar'])+' & '+qr(all['median'])+'%%'

    fourth_line = r' %cline{2-8}'
    fifth_line = r' & $max$ & '+qr(maxs_global['SM'])+' & '+qr(maxs_global['FA2'])+' & '+qr(maxs_global['pivot_mds'])+' & '+qr(maxs_global['tsNET'])+' & '+qr(maxs_global['tsNETstar'])+' & '+qr(all['max'])+'%%'

    # sixth_line = r' %cline{2-8}'
    # seventh_line = r' & $sd$ & '+qr(stds_global['SM'])+' & '+qr(stds_global['FA2'])+' & '+qr(stds_global['pivot_mds'])+' & '+qr(stds_global['tsNET'])+' & '+qr(stds_global['tsNETstar'])+' & '+qr(all['std'])+'%%'
    eigth_line = r' %hline'
    # lines += [first_line, second_line, third_line, fourth_line, fifth_line, sixth_line, seventh_line, eigth_line]
    lines += [third_line, fourth_line, fifth_line, eigth_line]


metric_agg_min_st_cn = np.min(np.array(list(metric_ds_st_cn_min.values())).flatten())
metric_agg_median_st_cn = np.mean(np.array(list(metric_ds_st_cn_median.values())).flatten())
metric_agg_max_st_cn = np.max(np.array(list(metric_ds_st_cn_max.values())).flatten())
metric_agg_std_st_cn = np.mean(np.array(list(metric_ds_st_cn_std.values())).flatten())

# first_line = r' %multirow{4}{*}{%textbf{ST+CN}} & $min$ & ' + qr(np.min(metric_ds_st_cn_min['SM'])) + ' & ' + qr(
#     np.min(metric_ds_st_cn_min['FA2'])) + ' & ' + qr(np.min(metric_ds_st_cn_min['pivot_mds'])) + ' & ' + qr(np.min(metric_ds_st_cn_min['tsNET'])) + ' & ' + qr(
#     np.min(metric_ds_st_cn_min['tsNETstar'])) + ' & ' + qr(metric_agg_min_st_cn) + '%%'
#
# second_line = r' %cline{2-8}'
third_line = r' %multirow{2}{*}{%textbf{ST+CN}} & $median$ & ' + qr(np.mean(metric_ds_st_cn_median['SM'])) + ' & ' + qr(np.mean(metric_ds_st_cn_median['FA2'])) + ' & ' + qr(
    np.mean(metric_ds_st_cn_median['pivot_mds'])) + ' & ' + qr(np.mean(metric_ds_st_cn_median['tsNET'])) + ' & ' + qr(np.mean(metric_ds_st_cn_median['tsNETstar'])) + ' & ' + qr(metric_agg_median_st_cn) + '%%'

fourth_line = r' %cline{2-8}'
fifth_line = r' & $max$ & ' + qr(np.max(metric_ds_st_cn_max['SM'])) + ' & ' + qr(np.max(metric_ds_st_cn_max['FA2'])) + ' & ' + qr(
    np.max(metric_ds_st_cn_max['pivot_mds'])) + ' & ' + qr(np.max(metric_ds_st_cn_max['tsNET'])) + ' & ' + qr(np.max(metric_ds_st_cn_max['tsNETstar'])) + ' & ' + qr(metric_agg_max_st_cn) + '%%'

# sixth_line = r' %cline{2-8}'
# seventh_line = r' & $sd$ & ' + qr(np.mean(metric_ds_st_cn_std['SM'])) + ' & ' + qr(np.mean(metric_ds_st_cn_std['FA2'])) + ' & ' + qr(
#     np.mean(metric_ds_st_cn_std['pivot_mds'])) + ' & ' + qr(np.mean(metric_ds_st_cn_std['tsNET'])) + ' & ' + qr(np.mean(metric_ds_st_cn_std['tsNETstar'])) + ' & ' + qr(metric_agg_std_st_cn) + '%%'
eigth_line = r' %hline'
# lines += [first_line, second_line, third_line, fourth_line, fifth_line, sixth_line, seventh_line, eigth_line]
lines += [third_line, fourth_line, fifth_line, eigth_line]


metric_agg_min = np.min(np.array(list(metric_ds_all_min.values())).flatten())
metric_agg_median = np.mean(np.array(list(metric_ds_all_median.values())).flatten())
metric_agg_max = np.max(np.array(list(metric_ds_all_max.values())).flatten())
metric_agg_std = np.mean(np.array(list(metric_ds_all_std.values())).flatten())

# first_line = r' %multirow{4}{*}{%textbf{ALL}} & $min$ & ' + qr(np.min(metric_ds_all_min['SM'])) + ' & ' + qr(
#     np.min(metric_ds_all_min['FA2'])) + ' & ' + qr(np.min(metric_ds_all_min['pivot_mds'])) + ' & ' + qr(np.min(metric_ds_all_min['tsNET'])) + ' & ' + qr(
#     np.min(metric_ds_all_min['tsNETstar'])) + ' & ' + qr(metric_agg_min) + '%%'
#
# second_line = r' %cline{2-8}'
third_line = r' %multirow{2}{*}{%textbf{ALL}} & $median$ & ' + qr(np.mean(metric_ds_all_median['SM'])) + ' & ' + qr(np.mean(metric_ds_all_median['FA2'])) + ' & ' + qr(
    np.mean(metric_ds_all_median['pivot_mds'])) + ' & ' + qr(np.mean(metric_ds_all_median['tsNET'])) + ' & ' + qr(np.mean(metric_ds_all_median['tsNETstar'])) + ' & ' + qr(metric_agg_median) + '%%'

fourth_line = r' %cline{2-8}'
fifth_line = r' & $max$ & ' + qr(np.max(metric_ds_all_max['SM'])) + ' & ' + qr(np.max(metric_ds_all_max['FA2'])) + ' & ' + qr(
    np.max(metric_ds_all_max['pivot_mds'])) + ' & ' + qr(np.max(metric_ds_all_max['tsNET'])) + ' & ' + qr(np.max(metric_ds_all_max['tsNETstar'])) + ' & ' + qr(metric_agg_max) + '%%'

# sixth_line = r' %cline{2-8}'
# seventh_line = r' & $sd$ & ' + qr(np.mean(metric_ds_all_std['SM'])) + ' & ' + qr(np.mean(metric_ds_all_std['FA2'])) + ' & ' + qr(
#     np.mean(metric_ds_all_std['pivot_mds'])) + ' & ' + qr(np.mean(metric_ds_all_std['tsNET'])) + ' & ' + qr(np.mean(metric_ds_all_std['tsNETstar'])) + ' & ' + qr(metric_agg_std) + '%%'
eigth_line = r' %hline'
# lines += [first_line, second_line, third_line, fourth_line, fifth_line, sixth_line, seventh_line, eigth_line]
lines += [third_line, fourth_line, fifth_line, eigth_line]



with open('test1.txt', 'a') as f:
    f.write('\n'.join(lines))


def qr(numb, sci_not = False):

    if sci_not:
        return '{:.2e}'.format(round(numb, 2))
    return '{:.3f}'.format(round(numb, 3))



import scipy

##############################
# first we check the statistical significances for the percentages of viewpoints that are better
# assume all of them are non normally distributed
scipy.stats.normaltest(metric_ds['stress']['SM'])

# stress and tsnet
# friedmanchisquare test to see if samples differ from each other, will not tell us if one is different from others
scipy.stats.friedmanchisquare(metric_ds['stress']['tsNET'], metric_ds['stress']['SM'], metric_ds['stress']['FA2'], metric_ds['stress']['pivot_mds'], metric_ds['stress']['tsNETstar'])
# cr and fa2
scipy.stats.friedmanchisquare(metric_ds['stress']['FA2'], metric_ds['stress']['SM'], metric_ds['stress']['tsNET'], metric_ds['stress']['pivot_mds'], metric_ds['stress']['tsNETstar'])

# signed rank wilcoxon for pairs of samples
# stress
print(scipy.stats.wilcoxon(metric_ds['stress']['tsNET'], metric_ds['stress']['SM'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['stress']['tsNET'], metric_ds['stress']['FA2'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['stress']['tsNET'], metric_ds['stress']['pivot_mds'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['stress']['tsNET'], metric_ds['stress']['tsNETstar'], alternative = 'greater'))

# cr
print(scipy.stats.wilcoxon(metric_ds['crossing_resolution']['FA2'], metric_ds['crossing_resolution']['SM'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['crossing_resolution']['FA2'], metric_ds['crossing_resolution']['pivot_mds'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['crossing_resolution']['FA2'], metric_ds['crossing_resolution']['tsNET'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['crossing_resolution']['FA2'], metric_ds['crossing_resolution']['tsNETstar'], alternative = 'greater'))

# ar
print(scipy.stats.wilcoxon(metric_ds['angular_resolution']['SM'], metric_ds['angular_resolution']['FA2'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['angular_resolution']['SM'], metric_ds['angular_resolution']['pivot_mds'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['angular_resolution']['SM'], metric_ds['angular_resolution']['tsNET'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['angular_resolution']['SM'], metric_ds['angular_resolution']['tsNETstar'], alternative = 'greater'))

# nr
print(scipy.stats.wilcoxon(metric_ds['node_resolution']['FA2'], metric_ds['node_resolution']['SM'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['node_resolution']['FA2'], metric_ds['node_resolution']['pivot_mds'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['node_resolution']['FA2'], metric_ds['node_resolution']['tsNET'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['node_resolution']['FA2'], metric_ds['node_resolution']['tsNETstar'], alternative = 'greater'))

# nn
print(scipy.stats.wilcoxon(metric_ds['node-node_occlusion']['FA2'], metric_ds['node-node_occlusion']['SM'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['node-node_occlusion']['FA2'], metric_ds['node-node_occlusion']['pivot_mds'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['node-node_occlusion']['FA2'], metric_ds['node-node_occlusion']['tsNET'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['node-node_occlusion']['FA2'], metric_ds['node-node_occlusion']['tsNETstar'], alternative = 'greater'))

# ne
print(scipy.stats.wilcoxon(metric_ds['node-edge_occlusion']['FA2'], metric_ds['node-edge_occlusion']['SM'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['node-edge_occlusion']['FA2'], metric_ds['node-edge_occlusion']['pivot_mds'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['node-edge_occlusion']['FA2'], metric_ds['node-edge_occlusion']['tsNET'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['node-edge_occlusion']['FA2'], metric_ds['node-edge_occlusion']['tsNETstar'], alternative = 'greater'))

# cn
print(scipy.stats.wilcoxon(metric_ds['crossing_number']['pivot_mds'], metric_ds['crossing_number']['SM'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['crossing_number']['pivot_mds'], metric_ds['crossing_number']['FA2'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['crossing_number']['pivot_mds'], metric_ds['crossing_number']['tsNET'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['crossing_number']['pivot_mds'], metric_ds['crossing_number']['tsNETstar'], alternative = 'greater'))

# ee
print(scipy.stats.wilcoxon(metric_ds['edge-edge_occlusion']['FA2'], metric_ds['edge-edge_occlusion']['SM'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['edge-edge_occlusion']['FA2'], metric_ds['edge-edge_occlusion']['pivot_mds'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['edge-edge_occlusion']['FA2'], metric_ds['edge-edge_occlusion']['tsNET'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['edge-edge_occlusion']['FA2'], metric_ds['edge-edge_occlusion']['tsNETstar'], alternative = 'greater'))

# el
print(scipy.stats.wilcoxon(metric_ds['edge_length_deviation']['tsNET'], metric_ds['edge_length_deviation']['SM'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['edge_length_deviation']['tsNET'], metric_ds['edge_length_deviation']['FA2'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['edge_length_deviation']['tsNET'], metric_ds['edge_length_deviation']['pivot_mds'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['edge_length_deviation']['tsNET'], metric_ds['edge_length_deviation']['tsNETstar'], alternative = 'greater'))

# st + cn
print(scipy.stats.wilcoxon(metric_ds['stress']['pivot_mds'] + metric_ds['crossing_number']['pivot_mds'], metric_ds['stress']['SM'] + metric_ds['crossing_number']['SM'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['stress']['pivot_mds'] + metric_ds['crossing_number']['pivot_mds'], metric_ds['stress']['FA2'] + metric_ds['crossing_number']['FA2'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['stress']['pivot_mds'] + metric_ds['crossing_number']['pivot_mds'], metric_ds['stress']['tsNET'] + metric_ds['crossing_number']['tsNET'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['stress']['pivot_mds'] + metric_ds['crossing_number']['pivot_mds'], metric_ds['stress']['tsNETstar'] + metric_ds['crossing_number']['tsNETstar'], alternative = 'greater'))

# all
best = []
first = []
second = []
third = []
fourth = []
for m in metrics:
    best += metric_ds[m]['FA2']
    first += metric_ds[m]['SM']
    second += metric_ds[m]['pivot_mds']
    third += metric_ds[m]['tsNET']
    fourth += metric_ds[m]['tsNETstar']

print(scipy.stats.wilcoxon(best, first, alternative = 'greater'))
print(scipy.stats.wilcoxon(best, second, alternative = 'greater'))
print(scipy.stats.wilcoxon(best, third, alternative = 'greater'))
print(scipy.stats.wilcoxon(best, fourth, alternative = 'greater'))




##############################
# check the statistical significances for the relative differences of normalized quality metric scores
import scipy
# signed rank wilcoxon for pairs of samples
# stress
print(scipy.stats.wilcoxon(metric_ds['stress']['tsNET'], metric_ds['stress']['SM'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['stress']['tsNET'], metric_ds['stress']['FA2'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['stress']['tsNET'], metric_ds['stress']['pivot_mds'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['stress']['tsNET'], metric_ds['stress']['tsNETstar'], alternative = 'greater'))

# cr
print(scipy.stats.wilcoxon(metric_ds['crossing_resolution']['FA2'], metric_ds['crossing_resolution']['SM'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['crossing_resolution']['FA2'], metric_ds['crossing_resolution']['pivot_mds'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['crossing_resolution']['FA2'], metric_ds['crossing_resolution']['tsNET'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['crossing_resolution']['FA2'], metric_ds['crossing_resolution']['tsNETstar'], alternative = 'greater'))

# ar
print(scipy.stats.wilcoxon(metric_ds['angular_resolution']['FA2'], metric_ds['angular_resolution']['SM'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['angular_resolution']['FA2'], metric_ds['angular_resolution']['pivot_mds'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['angular_resolution']['FA2'], metric_ds['angular_resolution']['tsNET'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['angular_resolution']['FA2'], metric_ds['angular_resolution']['tsNETstar'], alternative = 'greater'))

# nr
print(scipy.stats.wilcoxon(metric_ds['node_resolution']['FA2'], metric_ds['node_resolution']['SM'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['node_resolution']['FA2'], metric_ds['node_resolution']['pivot_mds'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['node_resolution']['FA2'], metric_ds['node_resolution']['tsNET'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['node_resolution']['FA2'], metric_ds['node_resolution']['tsNETstar'], alternative = 'greater'))

# nn
print(scipy.stats.wilcoxon(metric_ds['node-node_occlusion']['pivot_mds'], metric_ds['node-node_occlusion']['SM'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['node-node_occlusion']['pivot_mds'], metric_ds['node-node_occlusion']['FA2'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['node-node_occlusion']['pivot_mds'], metric_ds['node-node_occlusion']['tsNET'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['node-node_occlusion']['pivot_mds'], metric_ds['node-node_occlusion']['tsNETstar'], alternative = 'greater'))

# ne
print(scipy.stats.wilcoxon(metric_ds['node-edge_occlusion']['pivot_mds'], metric_ds['node-edge_occlusion']['SM'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['node-edge_occlusion']['pivot_mds'], metric_ds['node-edge_occlusion']['FA2'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['node-edge_occlusion']['pivot_mds'], metric_ds['node-edge_occlusion']['tsNET'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['node-edge_occlusion']['pivot_mds'], metric_ds['node-edge_occlusion']['tsNETstar'], alternative = 'greater'))

# cn
print(scipy.stats.wilcoxon(metric_ds['crossing_number']['FA2'], metric_ds['crossing_number']['SM'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['crossing_number']['FA2'], metric_ds['crossing_number']['pivot_mds'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['crossing_number']['FA2'], metric_ds['crossing_number']['tsNET'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['crossing_number']['FA2'], metric_ds['crossing_number']['tsNETstar'], alternative = 'greater'))

# ee
print(scipy.stats.wilcoxon(metric_ds['edge-edge_occlusion']['pivot_mds'], metric_ds['edge-edge_occlusion']['SM'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['edge-edge_occlusion']['pivot_mds'], metric_ds['edge-edge_occlusion']['FA2'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['edge-edge_occlusion']['pivot_mds'], metric_ds['edge-edge_occlusion']['tsNET'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['edge-edge_occlusion']['pivot_mds'], metric_ds['edge-edge_occlusion']['tsNETstar'], alternative = 'greater'))

# el
print(scipy.stats.wilcoxon(metric_ds['edge_length_deviation']['tsNET'], metric_ds['edge_length_deviation']['SM'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['edge_length_deviation']['tsNET'], metric_ds['edge_length_deviation']['FA2'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['edge_length_deviation']['tsNET'], metric_ds['edge_length_deviation']['pivot_mds'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['edge_length_deviation']['tsNET'], metric_ds['edge_length_deviation']['tsNETstar'], alternative = 'greater'))

# st + cn
print(scipy.stats.wilcoxon(metric_ds['stress']['FA2'] + metric_ds['crossing_number']['FA2'], metric_ds['stress']['SM'] + metric_ds['crossing_number']['SM'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['stress']['FA2'] + metric_ds['crossing_number']['FA2'], metric_ds['stress']['pivot_mds'] + metric_ds['crossing_number']['pivot_mds'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['stress']['FA2'] + metric_ds['crossing_number']['FA2'], metric_ds['stress']['tsNET'] + metric_ds['crossing_number']['tsNET'], alternative = 'greater'))
print(scipy.stats.wilcoxon(metric_ds['stress']['FA2'] + metric_ds['crossing_number']['FA2'], metric_ds['stress']['tsNETstar'] + metric_ds['crossing_number']['tsNETstar'], alternative = 'greater'))

# all
best = []
first = []
second = []
third = []
fourth = []
for m in metrics:
    best += metric_ds[m]['FA2']
    first += metric_ds[m]['SM']
    second += metric_ds[m]['pivot_mds']
    third += metric_ds[m]['tsNET']
    fourth += metric_ds[m]['tsNETstar']

print(scipy.stats.wilcoxon(best, first, alternative = 'greater'))
print(scipy.stats.wilcoxon(best, second, alternative = 'greater'))
print(scipy.stats.wilcoxon(best, third, alternative = 'greater'))
print(scipy.stats.wilcoxon(best, fourth, alternative = 'greater'))


