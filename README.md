# 3D VS 2D Graph Drawing: An Experimental Evaluation

# Running instructions

Python version 3.9.9 \
Install packages with: 'pip install -r requirements.txt' \
Original tool from https://github.com/WouterCastelein/Proj3D_views \
We adapted the tool to work for graph drawings and enhanced it by adding new features

### Minimal example
#### Note: Due to unresolved pickling errors with git please follow steps 3-7 below.  (To be solved)
With a single git clone, the tool should be usable for three example graphs provided in the GitHub repository. Simply run **visualize_metrics.py**. To use the tool for your own graphs please follow the instructions below.

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

