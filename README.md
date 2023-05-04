# Thesis
Master thesis on evaluating 3D projections by its 2D views instead of the entire 3D projection at once.

# Running instructions

Python version 3.9.9 \
Install packages with: 'pip install -r requirements.txt' \

To generate the 2D and 3D projections: \
Run runner.py, passing arguments for the dataset name and the selection of projection techniques.

Optionally, look at plots.py for visualizing the projections. \
Next run compute_views.py, to compute the 1000 views of the 3D projection. \
Then run compute_metrics.py to compute all metric values for the 2D projection, 3D projection and views of the 3D projection. 
Make sure to specify the right dataset names in the main function of this file. This process might take a while (hours) for larger datasets. \
Lastly, run consolid_metrics.py, to generate a file that combines all metric values for all datasets and projections in a single file.

Now it should be possible to run the tool by executing visualize_metrics.py. For configuration options look at constants.py



#Troubleshooting for projection techniques

**Cannot find/open MulticoreTSNE shared library:** \
MulticoreTSNE package might not work on 64bit, in that case \
-If you installed package before, pip uninstall MulticoreTSNE \
-Clone the source: git clone https://github.com/DmitryUlyanov/Multicore-TSNE.git \
-Edit /Multicore-TSNE/setup.py and add the following bold line to setup.py:

if 0 != execute(['cmake', 
**'-DCMAKE_GENERATOR_PLATFORM=x64',**
'-DCMAKE_BUILD_TYPE={}'.format(build_type),
'-DCMAKE_VERBOSE_MAKEFILE={}'.format(int(self.verbose)),

next install the package from the local repository (pip install .) or if you use a 
virtual environment make sure you install it there.

**Tapkee executable not found:** \
Go to: https://github.com/lisitsyn/tapkee/releases/tag/1.2
Download the tapkee executable for your operating system, and replace the file tapkee/tapkee
with the executable renamed to 'tapkee.exe'