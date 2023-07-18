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


# first dataset, aggregated over technique e.g.: {'technique' : {'metric1 : [values], 'metric2' : [values]}}
tech_ds = {}
for tech in techniques:
    res_dict = dict(zip(constants.metrics, ([] for _ in range(len(constants.metrics)))))
    for i in range(len(data_frame)):
        if (data_frame.iloc[i]['layout_technique'] == tech) and (data_frame.iloc[i]['n_components'] == 3):
            for m in metrics:
                col_idx = metrics_col_idx[m]
                res_dict[m].append(sum(data_frame.iloc[i]['views_metrics'][:, col_idx] > data_frame.iloc[i][m]) / constants.samples * 100)
    tech_ds[tech] = res_dict


# second dataset, aggregated over technique but with all metrics in one list.: {'technique' : {'metrics : [valuesmetrics1, valuesmetrics2,..]}}
# used in combination with labels
tech_ds_cat = dict(zip(tech_ds.keys(), (0 for _ in range(len(tech_ds.keys())))))

for tech in tech_ds:
    app_list = []
    for m in tech_ds[tech]:
        app_list += tech_ds[tech][m]
    tech_ds_cat[tech] = app_list


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
    plt.savefig('evaluations/' + str(m) + '.png')
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
    errorbar="sd", palette=['red', 'blue', 'green', 'gray', 'black'], alpha=.6, height=6
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
