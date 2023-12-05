# 3D VS 2D Graph Drawing: An Experimental Evaluation

# Running instructions

Python version 3.9.9 \
Install packages with: 'pip install -r requirements.txt' \

To create the 2D and 3D graph layouts: \
Run runner.py

Next run compute_views.py, to compute the 1000 views of all the 3D graph layouts. \
Then run compute_metrics.py to compute all metric values for the 2D graph layouts and viewpoints of the 3D graph layouts. Depending on the (number of) graphs chosen, and the number of cores on your cpu, this may take some time.

Lastly, run consolid_metrics.py, to generate a file that combines all metric values for all datasets and projections in a single file.

Now it should be possible to run the tool by executing visualize_metrics.py. 

<video src='https://github.com/simonvw95/GD_3d_tool/blob/master/videos/3dlayout_example1.mp4' width=180/></video>
