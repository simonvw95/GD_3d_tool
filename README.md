# 3D VS 2D Graph Drawing: An Experimental Evaluation

# Running instructions

Python version 3.9.9 \
Install packages with: 'pip install -r requirements.txt' \
Original tool from https://github.com/WouterCastelein/Proj3D_views \
We adapted the tool to work for graph drawings and enhanced it by adding new features

### Minimal example
With a single git clone, the tool should be usable for three example graphs provided in the GitHub repository. Please run the **json_to_pickle.py** script to put the datafiles in the correct format. Then Simply run **visualize_metrics.py** to use the tool for three example graphs. To use the tool for your own graphs please follow the instructions below.

### Personal dataset/graphs

1. Add source files of your graphs to the /data/ directory by adding a directory with the graphsname and name them graph file as such "graphname-src.csv". E.g. /data/3elt/3elt-src.csv . The graph file should be a .csv containing the edgelist. For the precise format (including delimiter) view one of the example graphs. The "graphname-gtds.csv" file will be automatically generated once **runner.py** is executed. 
2. To create the 2D and 3D graph layouts: Run **runner.py**
3. To compute the 1000 views of all the 3D graph layouts: Run **compute_views.py**.
4. To compute the metric values for all 2D graph layouts and all viewpoint layouts of the 3D graph layouts: Run **compute_metrics.py**. Note, this may take some time depending on the size of the graphs and number of graphs chosen, and the number of cores on your cpu.
5. To join all of the metric values into a single pickle file: Run **consolid_metrics.py**. 
6. To compute the projection scatterplots of all the viewpoints' quality metric values: Run **projections.py**.
7. Now you should be able to run the tool by executing **visualize_metrics.py**.

If you have already gone through the above steps and wish to add a new graph to your dataset, simply repeat steps 1-7. The following three (computing-intensive) scripts **runner.py**, **compute_metrics.py** and **projections.py** will check if computations for a particular graph have already been made and therefore will not repeat the same computations unless instructed with the **overwrite** variable.

Additional quality metrics can be added to **metrics_gd.py**, do ensure that these are bound between [0, 1] with 1 being the best score. Additionally, modifications will have to be made to **compute_metrics.py** and **constants.py**.

## Demonstrations

A 3D graph drawing can be rotated by dragging the cursor in the graph widget or by dragging the cursor in the quality metric sphere widget. In the quality metric sphere widget a viewpoint is given a color depending on the value of the quality metric score of that viewpoint's layout. Here, the quality metric is stress.

https://github.com/simonvw95/GD_3d_tool/assets/55921054/beba1bc3-a9f3-4d1f-b59c-2f48e3c04675


Here, the quality metric is the crossing number.

https://github.com/simonvw95/GD_3d_tool/assets/55921054/837de0ad-ce35-40f2-81ff-393bbf6e46fb

For each quality metric we have a histogram of the distribution of that quality metric's values of all the viewpoint layouts. For the histogram these values are normalized w.r.t. all viewpoint quality metric values in order to show a more visible distribution. The histograms are linked with a line to indicate that the selected viewpoint layout has the corresponding quality metric values. By rotating the quality metric sphere or 3D graph layout the values in the highlighted portion in the histogram changes to show where the quality metrics of the current viewpoint lie. It is also possible to manually move over the histogram bars to find which viewpoint(s) correspond to which values in the quality metric's histogram.

https://github.com/simonvw95/GD_3d_tool/assets/55921054/7a894e30-419b-41b5-bf99-2c4e759378b6

For each graph and layout technique combination we acquire all metric values of all the viewpoint layouts and the metric values of the 2D graph layout. The resulting 1001x9 dataset (1000 viewpoint layouts, 1 2D graph layout, 9 quality metrics) is reduced to a 2D space using t-SNE and plotted. The colors for the points are chosen based on the linear combination of the quality metrics. The rotation of the 3D graph layout and the quality metric sphere is linked to the 2D projection scatterplot, where the current viewpoint is highlighted with a yellow cross marker.

https://github.com/simonvw95/GD_3d_tool/assets/55921054/75acfe7b-6c08-4371-9fb3-10e6757c2f9c

The tool also has two additional widgets. The settings widget: here the user can choose the graph, layout technique, and the quality metric of the quality metric sphere. Additionally, the user can use buttons to find the best solution according to a linear combination of all metrics or a linear combination of normalized metrics. Moreover, the user can also set weights to the quality metrics rather than each metric having equal weight for the linear combination.

![settings_widget](https://github.com/simonvw95/GD_3d_tool/assets/55921054/5e7ebbef-52c8-444d-9fad-e49434c4a221)

The information widget: here the user is displayed some additional information about the viewpoint layouts compared to the 2D graph layout. For each metric, the percentage of viewpoint layouts with higher quality metric values than the 2D graph layout is given, as well as the difference between the best viewpoint layout and the 2D graph layout. Their non-normalized values are also given.

![statistics_widget](https://github.com/simonvw95/GD_3d_tool/assets/55921054/d2e4602a-b150-4f64-93da-97486b35dab1)

Lastly, by putting all widgets together we have an overview of the entirety of the tool:
![tool](https://github.com/simonvw95/GD_3d_tool/assets/55921054/6edfe3db-1a76-48b3-81c4-fde19feb444d)





