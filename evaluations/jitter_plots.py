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
techniques = set(data_frame['layout_technique'].to_list())
metrics = constants.metrics
metrics_col_idx = dict(zip(constants.metrics, range(len(constants.metrics))))


# third dataset, aggregated over metric.: {'metric1' : {'technique1' : [values], 'technique2' : [values]}, 'metric2' : ..}
metric_ds = {}
for m in metrics:
    col_idx = metrics_col_idx[m]
    res_dict = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
    for tech in techniques:
        for i in range(len(data_frame)):
            if (data_frame.iloc[i]['n_components'] == 3) and (data_frame.iloc[i]['layout_technique'] == tech):
                res_dict[tech].append(sum(data_frame.iloc[i]['views_metrics'][:, col_idx] > data_frame.iloc[i][m]) / constants.samples * 100)
    metric_ds[m] = res_dict

# fourth dataset, aggregated over metric where absolute value improvements are shown, then mean taken: {'metric1' : {'technique1' : [values], 'technique2' : [values]}, 'metric2' : ..}
metric_ds_perc = {}
for m in metrics:
    col_idx = metrics_col_idx[m]
    res_dict_perc = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
    for tech in techniques:
        for i in range(len(data_frame)):
            if (data_frame.iloc[i]['n_components'] == 3) and (data_frame.iloc[i]['layout_technique'] == tech):
                if sum(data_frame.iloc[i]['views_metrics'][:, col_idx] > data_frame.iloc[i][m]) > 0 and data_frame.iloc[i][m] >= 0:
                    idcs = data_frame.iloc[i]['views_metrics'][:, col_idx] > data_frame.iloc[i][m]
                    perc_impr_avg = np.mean(data_frame.iloc[i]['views_metrics'][:, col_idx][idcs] - data_frame.iloc[i][m])
                    res_dict_perc[tech].append(perc_impr_avg)
                else:
                    res_dict_perc[tech].append(0)
    metric_ds_perc[m] = res_dict_perc


# fifth dataset, aggregated over metric where max absolute value improvements are shown: {'metric1' : {'technique1' : [values], 'technique2' : [values]}, 'metric2' : ..}
metric_ds_max = {}
for m in metrics:
    col_idx = metrics_col_idx[m]
    res_dict_max = dict(zip(list(techniques), ([] for _ in range(len(techniques)))))
    for tech in techniques:
        for i in range(len(data_frame)):
            if (data_frame.iloc[i]['n_components'] == 3) and (data_frame.iloc[i]['layout_technique'] == tech):
                if sum(data_frame.iloc[i]['views_metrics'][:, col_idx] > data_frame.iloc[i][m]) > 0 and data_frame.iloc[i][m] >= 0:
                    idcs = data_frame.iloc[i]['views_metrics'][:, col_idx] > data_frame.iloc[i][m]
                    perc_impr_max = np.max(data_frame.iloc[i]['views_metrics'][:, col_idx][idcs] - data_frame.iloc[i][m])
                    res_dict_max[tech].append(perc_impr_max)
                else:
                    res_dict_max[tech].append(0)
    metric_ds_max[m] = res_dict_max


# sixth dataset, aggregated over metric where absolute value improvements are shown: {'metric1' : [values], 'metric2' : ..}
metric_ds_max_agg = {}
for m in metrics:
    col_idx = metrics_col_idx[m]
    res_max = []
    for tech in techniques:
        for i in range(len(data_frame)):
            if (data_frame.iloc[i]['n_components'] == 3) and (data_frame.iloc[i]['layout_technique'] == tech):
                if sum(data_frame.iloc[i]['views_metrics'][:, col_idx] > data_frame.iloc[i][m]) > 0 and data_frame.iloc[i][m] >= 0:
                    idcs = data_frame.iloc[i]['views_metrics'][:, col_idx] > data_frame.iloc[i][m]
                    abs_impr_max = np.max(data_frame.iloc[i]['views_metrics'][:, col_idx][idcs] - data_frame.iloc[i][m])
                    res_max.append(abs_impr_max)
                else:
                    res_max.append(0)
    metric_ds_max_agg[m] = res_max


# seventh dataset, aggregated over metric {'metric1' : [values], 'metric2' : ..}
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


# eighth dataset, aggregated over metric with distributions of metrics scaled to 0 and 1 {'metric1' : [values], 'metric2' : ..}
metric_ds_agg_norm = {}
for m in metrics:
    col_idx = metrics_col_idx[m]
    res = []
    for tech in techniques:
        for i in range(len(data_frame)):
            if (data_frame.iloc[i]['n_components'] == 3) and (data_frame.iloc[i]['layout_technique'] == tech):
                met_values = np.append(data_frame.iloc[i]['views_metrics'][:, col_idx], data_frame.iloc[i][m])
                bounds = np.min(met_values), np.max(met_values)
                scale_val = bounds[1] - bounds[0]
                met_values = np.append(((data_frame.iloc[i]['views_metrics'][:, col_idx] - bounds[0]) / scale_val), ((data_frame.iloc[i][m] - bounds[0]) / scale_val))
                res.append(met_values)
    metric_ds_agg_norm[m] = np.array(res).flatten()


# ninth dataset, aggregated over metric and subtracting the view values from the 2d value {'metric1' : [values], 'metric2' : ..}
metric_ds_agg_rel = {}
for m in metrics:
    col_idx = metrics_col_idx[m]
    res = []
    for tech in techniques:
        for i in range(len(data_frame)):
            if (data_frame.iloc[i]['n_components'] == 3) and (data_frame.iloc[i]['layout_technique'] == tech):
                met_values = data_frame.iloc[i]['views_metrics'][:, col_idx] - data_frame.iloc[i][m]
                res.append(met_values)
    metric_ds_agg_rel[m] = np.array(res).flatten()


# tenth dataset, aggregated over metric and subtracting the view values from the 2d value, metric distributions scaled to 0 and 1 {'metric1' : [values], 'metric2' : ..}
metric_ds_agg_rel_norm = {}
for m in metrics:
    col_idx = metrics_col_idx[m]
    res = []
    for tech in techniques:
        for i in range(len(data_frame)):
            if (data_frame.iloc[i]['n_components'] == 3) and (data_frame.iloc[i]['layout_technique'] == tech):
                met_values = np.append(data_frame.iloc[i]['views_metrics'][:, col_idx], data_frame.iloc[i][m])
                bounds = np.min(met_values), np.max(met_values)
                scale_val = bounds[1] - bounds[0]
                met_values = ((data_frame.iloc[i]['views_metrics'][:, col_idx] - bounds[0]) / scale_val) - ((data_frame.iloc[i][m] - bounds[0]) / scale_val)
                res.append(met_values)
    metric_ds_agg_rel_norm[m] = np.array(res).flatten()






#################################################################################################################################
# layout technique on x axis, one plot for each metric
for m in metrics:
    layout_technique_list = []
    results_list = []
    for tech in techniques:
        layout_technique_list += [tech] * len(metric_ds['stress'][tech])
        results_list += metric_ds[m][tech]

    end_data = pd.DataFrame({'Layout technique': layout_technique_list, '% of viewpoints better than 2d' : results_list})

    plt.figure(figsize=(1000 / 96, 1000 / 96), dpi=96)
    plt.xticks(rotation=45, ha="right")
    ax = plt.gca()
    ax.set_title('Layout technique comparison for ' + str(m))
    sns.stripplot(data=end_data, x='Layout technique', y='% of viewpoints better than 2d', size=5, ax=ax, jitter=0.10)
    plt.savefig('evaluations/indiv_metric_' + str(m) + '.png')
    plt.clf()
    plt.close('all')


#################################################################################################################################
# layout technique on x axis, one plot for each metric, abs improvements mean
for m in metrics:
    layout_technique_list = []
    results_list = []
    for tech in techniques:
        layout_technique_list += [tech] * len(metric_ds_perc['stress'][tech])
        results_list += metric_ds_perc[m][tech]

    end_data = pd.DataFrame({'Layout technique': layout_technique_list, 'mean absolute improvements of better 3d views' : results_list})

    plt.figure(figsize=(1000 / 96, 1000 / 96), dpi=96)
    plt.xticks(rotation=45, ha="right")
    ax = plt.gca()
    ax.set_title('Layout technique comparison for ' + str(m))
    sns.stripplot(data=end_data, x='Layout technique', y='mean absolute improvements of better 3d views', size=5, ax=ax, jitter=0.10)
    plt.savefig('evaluations/improv_' + str(m) + '.png')
    plt.clf()
    plt.close('all')


#################################################################################################################################
# layout technique on x axis, one plot for each metric, abs improvements max
for m in metrics:
    layout_technique_list = []
    results_list = []
    for tech in techniques:
        layout_technique_list += [tech] * len(metric_ds_max['stress'][tech])
        results_list += metric_ds_max[m][tech]

    end_data = pd.DataFrame({'Layout technique': layout_technique_list, 'max absolute improvements of better 3d views' : results_list})

    plt.figure(figsize=(1000 / 96, 1000 / 96), dpi=96)
    plt.xticks(rotation=45, ha="right")
    ax = plt.gca()
    ax.set_title('Layout technique comparison for ' + str(m))
    sns.stripplot(data=end_data, x='Layout technique', y='max absolute improvements of better 3d views', size=5, ax=ax, jitter=0.10)
    plt.savefig('evaluations/max_improv_' + str(m) + '.png')
    plt.clf()
    plt.close('all')


#################################################################################################################################
# layout technique on x axis, one plot for all metrics, each metric differently colored
layout_technique_list = []
results_list = []
categories = []
for tech in techniques:
    for m in metrics:
        layout_technique_list += [tech] * len(metric_ds[m][tech])
        results_list += metric_ds[m][tech]
        categories += [m] * len(metric_ds[m][tech])

end_data = pd.DataFrame({'Layout technique': layout_technique_list, '% of viewpoints better than 2d' : results_list, 'Metric' : categories})

plt.figure(figsize=(1000 / 96, 1000 / 96), dpi=96)
plt.xticks(rotation=45, ha="right")
ax = plt.gca()
ax.set_title('Layout technique comparison')
sns.stripplot(data=end_data, x='Layout technique', y='% of viewpoints better than 2d', size=5, ax=ax, jitter=0.10, hue='Metric',
                      palette=['red', 'lavender', 'green', 'gray', 'midnightblue', 'black', 'blue', 'yellow'])
plt.savefig('evaluations/all_metrics_layout_techn_jitter.png')
plt.clf()
plt.close('all')


#################################################################################################################################
# layout technique on x axis, one plot for all metrics
results_list = []
categories = []
for m in metrics:
    results_list += metric_ds_max_agg[m]
    categories += [m] * len(metric_ds_max_agg[m])

end_data = pd.DataFrame({'max absolute improvements of better 3d views' : results_list, 'Metric' : categories})

plt.figure(figsize=(1000 / 96, 1000 / 96), dpi=96)
plt.xticks(rotation=45, ha="right")
ax = plt.gca()
ax.set_title('Metric comparison')
sns.stripplot(data=end_data, x='Metric', y='max absolute improvements of better 3d views', size=5, ax=ax, jitter=0.10)
plt.savefig('evaluations/all_metrics_agg_jitter.png')
plt.clf()
plt.close('all')


#################################################################################################################################
# layout technique on x axis, one plot for all metrics, all datapoints
results_list = np.array([])
categories = []
for m in metric_ds_agg:
    results_list = np.append(results_list, metric_ds_agg[m])
    categories += [m] * len(metric_ds_agg[m])

end_data = pd.DataFrame({'All metric values of all graphs, techniques, and viewpoints' : results_list, 'Metric' : categories})

plt.figure(figsize=(1000 / 96, 1000 / 96), dpi=96)
plt.xticks(rotation=45, ha="right")
ax = plt.gca()
ax.set_title('Metric comparison')
sns.stripplot(data=end_data, x='Metric', y='All metric values of all graphs, techniques, and viewpoints', size=5, ax=ax, jitter=0.10)
plt.savefig('evaluations/all_values_agg_jitter.png')
plt.clf()
plt.close('all')


#################################################################################################################################
# layout technique on x axis, one plot for all metrics, all datapoints scaled between 0 and 1
results_list = np.array([])
categories = []
for m in metric_ds_agg_norm:
    results_list = np.append(results_list, metric_ds_agg_norm[m])
    categories += [m] * len(metric_ds_agg_norm[m])

end_data = pd.DataFrame({'All normalized metric values of all graphs, techniques, and viewpoints' : results_list, 'Metric' : categories})

plt.figure(figsize=(1000 / 96, 1000 / 96), dpi=96)
plt.xticks(rotation=45, ha="right")
ax = plt.gca()
ax.set_title('Metric comparison')
sns.stripplot(data=end_data, x='Metric', y='All normalized metric values of all graphs, techniques, and viewpoints', size=5, ax=ax, jitter=0.10)
plt.savefig('evaluations/all_values_agg_norm_jitter.png')
plt.clf()
plt.close('all')


#################################################################################################################################
# layout technique on x axis, one plot for all metrics, all datapoints relative to the 2d layout
results_list = np.array([])
categories = []
for m in metric_ds_agg_rel:
    results_list = np.append(results_list, metric_ds_agg_rel[m])
    categories += [m] * len(metric_ds_agg_rel[m])

end_data = pd.DataFrame({'Difference between 2d layout and viewpoint values, for all graphs, techniques, and viewpoints' : results_list, 'Metric' : categories})

plt.figure(figsize=(1000 / 96, 1000 / 96), dpi=96)
plt.xticks(rotation=45, ha="right")
ax = plt.gca()
ax.set_title('Metric comparison')
sns.stripplot(data=end_data, x='Metric', y='Difference between 2d layout and viewpoint values, for all graphs, techniques, and viewpoints', size=5, ax=ax, jitter=0.10)
plt.savefig('evaluations/all_values_agg_rel_jitter.png')
plt.clf()
plt.close('all')


#################################################################################################################################
# layout technique on x axis, one plot for all metrics, all datapoints relative to the 2d layout, all datapoints between 0 and 1
results_list = np.array([])
categories = []
for m in metric_ds_agg_rel_norm:
    results_list = np.append(results_list, metric_ds_agg_rel_norm[m])
    categories += [m] * len(metric_ds_agg_rel_norm[m])

end_data = pd.DataFrame({'Difference between 2d layout and normalized viewpoint values, for all graphs, techniques, and viewpoints' : results_list, 'Metric' : categories})

plt.figure(figsize=(1000 / 96, 1000 / 96), dpi=96)
plt.xticks(rotation=45, ha="right")
ax = plt.gca()
ax.set_title('Metric comparison')
sns.stripplot(data=end_data, x='Metric', y='Difference between 2d layout and normalized viewpoint values, for all graphs, techniques, and viewpoints', size=5, ax=ax, jitter=0.10)
plt.savefig('evaluations/all_values_agg_rel_norm_jitter.png')
plt.clf()
plt.close('all')



#################################################################################################################################
# barchart with metric on x axis, each technique side by side with different colors,
layout_technique_list = []
results_list = []
categories = []
for tech in techniques:
    for m in metrics:
        layout_technique_list += [tech]
        results_list += [np.mean(metric_ds[m][tech])]
        categories += [m]

end_data = pd.DataFrame({'Layout technique': layout_technique_list, '% of viewpoints better than 2d' : results_list, 'Metric' : categories})

plt.figure(figsize=(1000 / 96, 1000 / 96), dpi=96)
plt.xticks(rotation = 45, ha = "right")
ax = plt.gca()
ax.set_title('Metric comparison')
g = sns.catplot(
    data=end_data, kind="bar",
    x="Metric", y="% of viewpoints better than 2d", hue="Layout technique",
    errorbar="sd", palette=['red', 'blue', 'green', 'black', 'grey'], alpha=.6, height=6
)
g.despine(left=True)
g.legend.set_title("")
g.set_xticklabels(rotation=30)
g.set_axis_labels(y_var = 'Average % of viewpoints better than 2d')
plt.tight_layout()
plt.savefig('evaluations/all_metrics_layout_techn_barchart.png')
plt.clf()
plt.close('all')


#################################################################################################################################
# boxplot with metric on x axis, each technique side by side with different colors
layout_technique_list = []
results_list = []
categories = []
for tech in techniques:
    for m in metrics:
        layout_technique_list += [tech] * len(metric_ds[m][tech])
        results_list += metric_ds[m][tech]
        categories += [m] * len(metric_ds[m][tech])

end_data = pd.DataFrame({'Layout technique': layout_technique_list, '% of viewpoints better than 2d' : results_list, 'Metric' : categories})

plt.figure(figsize=(1000 / 96, 1000 / 96), dpi=96)
plt.xticks(rotation = 45, ha = "right")
ax = plt.gca()
ax.set_title('Metric comparison')
g = sns.boxplot(x="Metric", y="% of viewpoints better than 2d",
            hue="Layout technique", palette=['red', 'blue', 'green', 'gray', 'black'],
            data=end_data)
sns.despine(offset=10, trim=True)
ax.set_xticklabels(metrics)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.tight_layout()
plt.savefig('evaluations/all_metrics_layout_techn_boxplot.png')
plt.clf()
plt.close('all')


#################################################################################################################################
# histogram for each metric, aggregated over each technique
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
# boxplot for each metric
results_list = []
metric_list = []
for m in metrics:
    results_list += list(metric_ds_agg[m])
    metric_list += [m] * len(metric_ds_agg[m])

end_data = pd.DataFrame({'Metric values' : results_list, 'Metric' : metric_list})

plt.figure(figsize=(1000 / 96, 1000 / 96), dpi=96)
plt.xticks(rotation = 45, ha = "right")
ax = plt.gca()
ax.set_title('Metric comparison of all datapoints')
g = sns.boxplot(x="Metric", y="Metric values", data=end_data)
sns.despine(offset=10, trim=True)
ax.set_xticklabels(metrics)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.tight_layout()
plt.savefig('evaluations/all_metrics_boxplot.png')
plt.clf()
plt.close('all')


# same as above but now for the relative difference dataset
#################################################################################################################################
# histogram for each metric, aggregated over each technique
for m in metrics:

    plt.figure(figsize=(1000 / 96, 1000 / 96), dpi=96)
    plt.xticks(rotation=45, ha="right")
    ax = plt.gca()
    ax.set_title('All datapoints of ' + str(m))
    plt.hist(x = metric_ds_agg_rel[m], bins = 100)
    plt.xlabel('Metric value relative to their 2d values')
    plt.ylabel('Frequency')
    plt.savefig('evaluations/indiv_metric_rel_histogram_' + str(m) + '.png')
    plt.clf()
    plt.close('all')


#################################################################################################################################
# boxplot for each metric
results_list = []
metric_list = []
for m in metrics:
    results_list += list(metric_ds_agg_rel[m])
    metric_list += [m] * len(metric_ds_agg_rel[m])

end_data = pd.DataFrame({'Metric values' : results_list, 'Metric' : metric_list})

plt.figure(figsize=(1000 / 96, 1000 / 96), dpi=96)
plt.xticks(rotation = 45, ha = "right")
ax = plt.gca()
ax.set_title('Metric comparison of all datapoints relative to their 2d values')
g = sns.boxplot(x="Metric", y="Metric values", data=end_data)
sns.despine(offset=10, trim=True)
ax.set_xticklabels(metrics)
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
plt.tight_layout()
plt.savefig('evaluations/all_metrics_boxplot_rel.png')
plt.clf()
plt.close('all')



def create_jitter_plots(data_dict, quality_metric, categories, save_folder):

    end_data = pd.DataFrame({'Layouts': tech_list, str(quality_metric): full_val_list, 'Graph class': g_classes})



    if categories:
        sns.stripplot(data=end_data, x='Layouts', y=str(quality_metric), size=5, ax=ax, jitter=0.10, hue='Graph class',
                      palette=['red', 'blue'])
        # if we don't have a hue list then we don't set the hue argument
    else:
        sns.stripplot(data=end_data, x='Layouts', y=str(quality_metric), size=5, ax=ax, jitter=0.10,
                      palette=['red', 'lavender', 'green', 'gray', 'midnightblue', 'black', 'blue', 'yellow'])

    median_width = 0.45

    for tick, text in zip(ax.get_xticks(), ax.get_xticklabels()):
        curr_name = text.get_text()

        median_val = end_data[end_data['Layouts'] == curr_name][str(quality_metric)].median()

        # plot horizontal lines across the column
        ax.plot([tick - median_width / 2, tick + median_width / 2], [median_val, median_val], lw=3, color='black')

    handles, labels = ax.get_legend_handles_labels()

    if categories:
        ax.legend(handles[-2:], labels[-2:])

    if median_val < 1:
        plt.ylim(0, 0.3)

    plt.savefig(save_folder + 'jitter_plots' + str(quality_metric) + '.png')
    plt.clf()
    plt.close('all')
