# library imports
import os
import numpy as np
import pandas as pd
import pickle
import keyboard
import networkx as nx
import copy
import pyqtgraph as pg
import pyqtgraph.exporters
from glob import glob
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QPushButton, QSlider
from pyqtgraph import mkPen
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from functools import partial

# script imports
import utils
import constants
from visualizations.graph_3d import Graph3D
from visualizations.graph_2d import Graph2D
from visualizations.projec_2d import Scatter2D
from visualizations.quality_sphere import QualitySphere
from visualizations.parallel_bar_plot import parallelBarPlot


class Tool(pg.GraphicsWindow):
    def __init__(self, dataset_name = "3elt", default_layout_technique = "FA2", default_proj_choice = 'global norm', analysis_data = None):
        super(Tool, self).__init__()

        keyboard.on_press(self.keyboard_event)

        # data
        self.dataset_name = dataset_name
        self.default_layout_technique = default_layout_technique
        self.proj_choice = default_proj_choice
        self.view_locked = False

        # grid initialization of the window
        self.setBackground((0, 0, 0, 60))
        self.layoutgb = QtWidgets.QGridLayout()
        self.layoutgb.setHorizontalSpacing(1)
        self.layoutgb.setVerticalSpacing(1)
        self.layoutgb.setContentsMargins(1, 1, 1, 1)
        self.setLayout(self.layoutgb)

        self.layoutgb.setColumnStretch(0, 3)
        self.layoutgb.setColumnStretch(1, 10)
        self.layoutgb.setColumnStretch(2, 10)
        self.layoutgb.setColumnStretch(3, 10)
        self.layoutgb.setRowStretch(0, 10)
        self.layoutgb.setRowStretch(1, 10)
        self.sphere_widgets = []

        self.analysis_data = analysis_data

        # not used, may be used in the future for user study
        if constants.user_mode != 'free':
            self.layout_index = 0
            self.dataset_name, self.default_layout_technique = constants.evaluation_set[self.layout_index]
            self.evaluation_data = []

        # get the view points and initialize the starting metric and some variables
        self.view_points = np.load(f'spheres/sphere{constants.samples}_points.npy')

        # get the minima and maxima of all the quality metrics
        bounds_dict = constants.bounds_dict
        self.mins_global = np.array(list(bounds_dict.values()))[:, 0]
        self.maxs_global = np.array(list(bounds_dict.values()))[:, 1]
        # get the minimum and maximum of the averages of all the 9 quality metrics combined
        self.global_average_min = constants.glob_averages_min
        self.global_average_max = constants.glob_averages_max

        # create the default colors for the heatmap
        self.heatmap_colors = ["darkred", "red", "yellow", "green", "lightblue"]
        # create the default color mapping for the projection
        self.proj_cmap = LinearSegmentedColormap.from_list("", self.heatmap_colors)

        self.D_P_dict = self.available_datasets_layouts()
        self.current_metric = 'crossing_number'
        self.curr_viewpoint = None
        self.initialize_menu()
        self.graph_2d = None
        self.graph_3d = None
        self.metric_proj = None
        self.stats = None

        self.curr_proj_idx = None

        self.set_data(self.dataset_name, self.default_layout_technique)
        self.highlight()

    # get all the datasets listed in the metrics file into a dictionary
    def available_datasets_layouts(self):

        consolid_metrics = os.path.join(constants.metrics_dir, 'metrics.pkl')
        data_frame = pd.read_pickle(consolid_metrics)
        D_P_dict = {}
        datasets = set(data_frame['dataset_name'].to_list())

        for dataset in datasets:
            D_P_dict[dataset.split('.')[0]] = set(data_frame[data_frame['dataset_name'] == dataset]['layout_technique'].to_list())

        return D_P_dict

    # initialize the menu, add buttons, sliders, labels etc.
    def initialize_menu(self):

        self.menu = pg.LayoutWidget()
        # width_policy = QtWidgets.QSizePolicy()
        # width_policy.setHorizontalPolicy(QtWidgets.QSizePolicy.MinimumExpanding)

        # Set background white:
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QColor(255, 255, 255, 255))
        self.menu.setPalette(palette)
        self.menu.setAutoFillBackground(True)

        # get the graph file and create a graph object
        input_file = glob('data/{0}/*-src.csv'.format(self.dataset_name))[0]
        df = pd.read_csv(input_file, sep=';', header=0)
        graph = nx.from_pandas_edgelist(df, 'from', 'to', edge_attr='strength')
        graph = nx.convert_node_labels_to_integers(graph)
        self.G = graph

        # default user mode, can be switched when doing future user study
        if constants.user_mode == 'free':
            keys = list(self.D_P_dict.keys())

            # create a drop down list to pick datasets
            datasets = list(self.D_P_dict.keys())
            datasets.sort()
            self.dataset_picker = pg.ComboBox(items=datasets, default=self.dataset_name)
            self.dataset_picker.currentIndexChanged.connect(self.data_selected)

            # create a drop down list to pick the different layout techniques
            layouts = list(self.D_P_dict[keys[0]])
            layouts.sort()
            self.layout_technique_picker = pg.ComboBox(items=layouts, default=self.default_layout_technique)
            self.layout_technique_picker.currentIndexChanged.connect(self.data_selected)

            # create a drop down list to pick the different metrics
            self.metric_picker = pg.ComboBox(items = constants.metrics + ['Weighted sum of metrics (norm)', 'Weighted sum of metrics (raw)'], default = self.current_metric)
            self.metric_picker.currentIndexChanged.connect(self.data_selected)

            # create a drop down list to pick different versions of the quality metrics projection plot
            # self.projection_picker = pg.ComboBox(items = ['local norm', 'global norm'], default = 'global norm')
            # self.projection_picker.currentIndexChanged.connect(self.change_proj_type)

            # add a few labels indicating what each drop down list does
            self.menu.addLabel(text="Dataset:", row=len(self.menu.rows), col=0)
            self.menu.addWidget(self.dataset_picker, len(self.menu.rows), 0)
            # self.menu.addLabel(text="Projection type:", row=len(self.menu.rows), col=1)
            # self.menu.addWidget(self.projection_picker, len(self.menu.rows), 1)

            self.menu.addLabel(text="Layout technique:", row=len(self.menu.rows), col=0)
            self.menu.addWidget(self.layout_technique_picker, len(self.menu.rows), 0)
            self.menu.addLabel(text="Quality Metric:", row=len(self.menu.rows), col=0)
            self.menu.addWidget(self.metric_picker, len(self.menu.rows), 0)

            # change the text of the dataset label when drop down list for dataset is used
            self.dataset_picker.currentIndexChanged.connect(self.change_label_text)

            # more labels indicating some graph stats
            self.label_n = QtWidgets.QLabel('Number of nodes (n): ' + str(self.G.number_of_nodes()))
            self.label_m = QtWidgets.QLabel('Number of edges (m): ' + str(self.G.number_of_edges()))
            self.menu.addWidget(self.label_n, row=len(self.menu.rows), col=0)
            self.menu.addWidget(self.label_m, row=len(self.menu.rows), col=0)

            # button that freezes some widgets so that it doesn't change when hovering
            self.select_button = QPushButton('Select view')
            self.select_button.pressed.connect(self.select_view)
            self.menu.addWidget(self.select_button, len(self.menu.rows), 0)
            self.select_button.setVisible(True)

            # adds a button that finds the best solution when scores are normalized
            self.best_solution_button = QPushButton('Find best solution (norm)')
            self.best_solution_button.pressed.connect(self.get_best_solution)
            self.menu.addWidget(self.best_solution_button, len(self.menu.rows), 0)

            # adds a button that finds the best solution with raw values
            self.best_solution_button_raw = QPushButton('Find best solution')
            self.best_solution_button_raw.pressed.connect(self.get_best_solution_raw)
            self.menu.addWidget(self.best_solution_button_raw, len(self.menu.rows), 0)

            # sliders for qms
            self.n_metrics = len(constants.metrics)
            # map the name of the metrics to the integer identifiers
            self.metrics_name_dict = dict(zip(range(len(constants.metrics)), constants.metrics))

            # initialization of sliders, labels etc
            self.sliders = {}
            self.sliders_labels = {}
            self.weights = {}
            self.slider_buttons_min = {}
            self.slider_buttons_max = {}

            # create sliders for the quality metrics
            for i in range(self.n_metrics):

                # add labels to the sliders and give them an object name
                self.sliders_labels[i] = QtWidgets.QLabel(self.metrics_name_dict[i] + ' 1')
                self.sliders_labels[i].setObjectName(str(i))
                self.menu.addWidget(self.sliders_labels[i], len(self.menu.rows), 0)

                # add buttons next to the sliders that set weights to 0 or 100(max)
                self.slider_buttons_min[i] = QPushButton('Min')
                self.slider_buttons_min[i].setObjectName(str(i))
                self.slider_buttons_min[i].pressed.connect(lambda: self.update_weights_button(value = 0))

                self.slider_buttons_max[i] = QPushButton('Max')
                self.slider_buttons_max[i].setObjectName(str(i))
                self.slider_buttons_max[i].pressed.connect(lambda: self.update_weights_button(value = 100))

                self.menu.addWidget(self.slider_buttons_min[i], len(self.menu.rows), 1)
                self.menu.addWidget(self.slider_buttons_max[i], len(self.menu.rows), 1)

                # set all weights to the max 100 by default
                self.weights[i] = 100

                # create the sliders
                self.sliders[i] = QSlider(Qt.Horizontal)
                # self.sliders[i].setSizePolicy(width_policy)
                self.sliders[i].setMinimum(0)
                self.sliders[i].setMaximum(100)
                self.sliders[i].setValue(100)
                self.sliders[i].setSingleStep(5)
                self.sliders[i].setTickInterval(10)
                self.sliders[i].setTickPosition(QSlider.TicksBelow)
                self.sliders[i].valueChanged.connect(self.update_weights)
                self.sliders[i].setObjectName(str(i))
                self.menu.addWidget(self.sliders[i], len(self.menu.rows), 0)
        # deprecated, may be used in the future this entire else statement
        else:
            self.evaluation_started = False #Keep track of when the tutorial is over

            self.next_button = QPushButton('Begin survey')
            self.next_button.pressed.connect(self.next_layout)
            self.menu.addWidget(self.next_button, len(self.menu.rows), 0)

            self.select_button = QPushButton('Select view')
            self.select_button.pressed.connect(self.select_view)
            self.menu.addWidget(self.select_button, len(self.menu.rows), 0)
            self.selected_counter = self.menu.addLabel(text=f"0/{constants.required_view_count} views selected", row=len(self.menu.rows), col=0)
            self.select_button.setVisible(False)
            self.selected_counter.setVisible(False)

            self.preference_label = self.menu.addLabel(text="Preference:", row=len(self.menu.rows), col=0)
            self.prefer_3d = QPushButton('3D Preference')
            self.prefer_3d.pressed.connect(partial(self.select_preference, '3D'))
            self.menu.addWidget(self.prefer_3d, len(self.menu.rows), 0)

            self.prefer_2d = QPushButton('2D Preference')
            self.prefer_2d.pressed.connect(partial(self.select_preference, '2D'))
            self.menu.addWidget(self.prefer_2d, len(self.menu.rows), 0)

            self.preference_label.setVisible(False)
            self.prefer_3d.setVisible(False)
            self.prefer_2d.setVisible(False)

        for i in range(len(self.menu.rows) - 1):
            self.menu.layout.setRowStretch(i, 0)

        self.menu.layout.setRowStretch(len(self.menu.rows), 1)
        self.layoutgb.addWidget(self.menu, 0, 0, 2, 1)

    # updating the weights according to the values given in the sliders
    def update_weights(self, value):

        # get the name of the object
        sender = self.sender()
        name = sender.objectName()
        # adjust text of the slider and then the weights according to slider value
        self.sliders_labels[int(name)].setText(self.metrics_name_dict[int(name)] + ' ' + str(round(value / 100, 2)))
        self.weights[int(name)] = value

    # updating the weights according to a button
    def update_weights_button(self, value):

        # get the name of the object
        sender = self.sender()
        name = sender.objectName()
        # adjust text of the slider and the slider value and then the weights according to slider value
        self.sliders_labels[int(name)].setText(self.metrics_name_dict[int(name)] + ' ' + str(round(value / 100, 2)))
        self.sliders[int(name)].setValue(value)
        self.weights[int(name)] = value

    # setting the text of the textboxes according to the graph details
    def change_label_text(self):

        self.label_n.setText('Number of nodes (n): ' + str(self.G.number_of_nodes()))
        self.label_m.setText('Number of edges (m): ' + str(self.G.number_of_edges()))

    # deprecated, may be used in the future for user study
    def next_layout(self):

        if not self.evaluation_started:
            self.evaluation_started = True
            self.select_button.setVisible(True)
            self.selected_counter.setVisible(True)
            self.next_button.setText('Next layout')

            constants.user_mode = 'eval_half'
            self.sphere_widget.setVisible(False)
            self.hist.setVisible(False)
            self.sphere_widget.setVisible(False)

        if constants.user_mode != 'free':
            self.layout_index += 1
            if self.layout_index < len(constants.evaluation_set):
                if self.layout_index >= 4:
                    constants.user_mode = 'eval_full'
                config = constants.evaluation_set[self.layout_index]
                self.set_data(config[0], config[1])
                self.set_tool_lock(False)
                self.next_button.setDisabled(True)
                self.update_selected_count_text()
            else:
                with open(constants.output_file, 'wb') as file:
                    pickle.dump(self.evaluation_data, file)
                self.close()

    # function attached to button or keypress, freezes the screen so hovering will not interact
    def select_view(self):

        self.set_tool_lock(not self.view_locked)
        pass

    # deprecated, may be used in the future for user study
    def selected_view_count(self):

        count = 0
        for data in self.evaluation_data:
            if data['dataset'] == self.dataset_name and data['layout_technique'] == self.default_layout_technique:
                count += 1

        return count

    # deprecated, may be used in the future for user study
    def update_selected_count_text(self):

        self.selected_counter.setText(f"{self.selected_view_count()}/{constants.required_view_count} views selected")

    # deprecated, may be used in the future for user study
    def select_preference(self, preference):

        self.preference_label.setVisible(False)
        self.prefer_3d.setVisible(False)
        self.prefer_2d.setVisible(False)
        self.set_tool_lock(False)
        self.evaluation_data.append({
            'dataset': self.dataset_name,
            'layout_technique': self.default_layout_technique,
            'viewpoint': np.array(self.graph_3d.cameraPosition()),
            'view_quality': self.current_quality(),
            '2D_quality': self.metrics_2d,
            '3D_quality': self.metrics_3d,
            'preference': preference,
            'mode': constants.user_mode,
        })
        if self.selected_view_count() >= 3:
            self.next_button.setDisabled(False)
        self.check_select_available()
        self.update_selected_count_text()

    # freezes the screen so hovering will not interact
    def set_tool_lock(self, lock):

        self.view_locked = lock
        if self.view_locked:
            self.select_button.setText('Deselect view')
        else:
            self.select_button.setText('Select view')
        self.hist.lock = self.view_locked
        self.graph_3d.lock = self.view_locked
        self.sphere.lock = self.view_locked
        # self.preference_label.setVisible(self.view_locked)
        # self.prefer_3d.setVisible(self.view_locked)
        # self.prefer_2d.setVisible(self.view_locked)

    # function for displaying the viewpoint with the best weighted solution with normalized metric values
    def get_best_solution(self):

        curr_views_metrics = copy.deepcopy(self.views_metrics)
        for i in range(self.n_metrics):
            curr_views_metrics[:, i] *= (self.weights[i] / 100)

        best_view = np.argmax(np.sum(curr_views_metrics, axis = 1))
        view = self.view_points[best_view]

        self.move_to_viewpoint(view)

    # function for displaying the viewpoint with the best weighted solution with raw values
    def get_best_solution_raw(self):

        curr_views_metrics = copy.deepcopy(self.original_metrics_views)
        for i in range(self.n_metrics):
            curr_views_metrics[:, i] *= (self.weights[i] / 100)

        best_view = np.argmax(np.sum(curr_views_metrics, axis = 1))
        view = self.view_points[best_view]

        self.move_to_viewpoint(view)

    # initializes the 3d graph layout
    def initialize_3d_layout(self):

        # get the data
        layout_file_3d = F"{constants.output_dir}/{self.dataset_name}-{self.default_layout_technique}-3d.csv"
        input_file = glob('data/{0}/*-src.csv'.format(self.dataset_name))[0]
        data = pd.read_csv(layout_file_3d, sep=';').to_numpy()

        # get the graph object and edges
        df = pd.read_csv(input_file, sep=';', header=0)
        graph = nx.from_pandas_edgelist(df, 'from', 'to', edge_attr='strength')
        graph = nx.convert_node_labels_to_integers(graph)
        edges = list(graph.edges())

        # will overwrite the existing graph (in case a new dataset is chosen in the widget)
        self.G = graph

        # create a 3d graph drawing widget if there is none, if there is then adjust the data
        if self.graph_3d is None:
            self.graph_3d = Graph3D(data, self.cmap, edges, parent=self, title="3D Layout")
            self.graph_3d.setBackgroundColor('w')
            self.layoutgb.addWidget(self.graph_3d, 0, 1)
        else:
            self.graph_3d.set_data(data, self.cmap,  edges)

    # initializes the 2d graph layout
    def initialize_2d_layout(self):

        # deprecated may be used for userstudy
        if constants.user_mode == 'evalimage':
            return

        # get the graph data
        input_file = glob('data/{0}/*-src.csv'.format(self.dataset_name))[0]
        df = pd.read_csv(input_file, sep=';', header=0)
        graph = nx.from_pandas_edgelist(df, 'from', 'to', edge_attr='strength')
        graph = nx.convert_node_labels_to_integers(graph)
        edges = list(graph.edges())

        # get the layout data
        layout_file_2d = F"{constants.output_dir}/{self.dataset_name}-{self.default_layout_technique}-2d.csv"
        df_2d = pd.read_csv(layout_file_2d, sep=';').to_numpy()

        # create a 2d graph drawing widget if there is none, if there is then adjust the data
        if self.graph_2d is None:
            self.graph_2d = Graph2D(df_2d, self.cmap, edges, title ="2D Layout")
            self.layoutgb.addWidget(self.graph_2d, 0, 2)
            self.graph_2d.setBackground('w')
        else:
            self.graph_2d.set_data(df_2d, edges)

    # initialize the qualitymetric sphere
    def initialize_sphere(self, sphere_sum = None):

        double_bar = False
        # # add the viewpoint quality metric data to the sphere
        if sphere_sum == 'Weighted sum of metrics (raw)':
            # weight the raw metric values
            curr_views_metrics = copy.deepcopy(self.original_metrics_views)
            for i in range(self.n_metrics):
                curr_views_metrics[:, i] *= (self.weights[i] / 100)
            weight_sum = np.sum(curr_views_metrics, axis = 1)
            # scale weight sum again
            mins, maxs = np.min(weight_sum), np.max(weight_sum)
            self.sphere_data = (weight_sum - mins) / (maxs - mins)
        elif sphere_sum == 'Weighted sum of metrics (norm)':
            # weight the normalized metric values
            curr_views_metrics = copy.deepcopy(self.views_metrics)
            for i in range(self.n_metrics):
                curr_views_metrics[:, i] *= (self.weights[i] / 100)

            weight_sum = np.sum(curr_views_metrics, axis=1)
            mins, maxs = np.min(weight_sum), np.max(weight_sum)
            self.sphere_data = (weight_sum - mins) / (maxs - mins)
        else:
            double_bar = True
            self.sphere_data = np.copy(self.views_metrics[:, constants.metrics.index(self.current_metric)])
            min_color_bar = np.min(self.original_metrics_views[:, constants.metrics.index(self.current_metric)])
            max_color_bar = np.max(self.original_metrics_views[:, constants.metrics.index(self.current_metric)])

        c = self.heatmap_colors
        v = [0, 0.25, 0.5, 0.75, 1]
        self.heatmap = pg.ColorMap(v, c)

        if sphere_sum:
            title_add = sphere_sum
        else:
            title_add = self.current_metric

        self.sphere = QualitySphere(self.sphere_data, self.heatmap, parent=self, title=F"Viewpoint quality ({title_add})")
        self.sphere.setBackgroundColor('w')

        # add the sphere to a widget
        self.sphere_widget = pg.LayoutWidget()
        self.sphere_widget.addWidget(self.sphere, 0, 0)
        self.sphere_widget.layout.setContentsMargins(0, 0, 0, 0)
        self.sphere_widget.layout.setHorizontalSpacing(0)

        self.cbw = pg.GraphicsLayoutWidget()
        self.color_bar = pg.ColorBarItem(colorMap=self.heatmap, interactive=False, values=(0, 1))

        # display max, min and current metric value with a horizontal line
        self.color_bar.addLine(y=np.max(self.sphere_data) * 255, pen=mkPen(255, 255, 255, width=2))
        self.color_bar.addLine(y=np.min(self.sphere_data) * 255, pen=mkPen(255, 255, 255, width=2))
        self.color_bar_line = self.color_bar.addLine(y=255, pen=mkPen(0,0,0,255))

        # if double_bar:
        #     # cbw2 = pg.GraphicsLayoutWidget()
        #     start, stop, step = min_color_bar, max_color_bar, (max_color_bar - min_color_bar) / 4
        #     v2 = list(np.arange(start, stop + step / 2, step))
        #     heatmap2 = pg.ColorMap(v2, c)
        #
        #     color_bar2 = pg.ColorBarItem(colorMap=heatmap2, interactive=False, values=(min_color_bar, max_color_bar))
        #
        #     # display max, min and current metric value with a horizontal line
        #     color_bar2.addLine(y=max_color_bar * 255, pen=mkPen(255, 255, 255, width=2))
        #     color_bar2.addLine(y=min_color_bar * 255, pen=mkPen(255, 255, 255, width=2))
        #     color_bar_line2 = color_bar2.addLine(y=255, pen=mkPen(0, 0, 0, 255))
        #     self.cbw.addItem(color_bar2, row = 0, col = 0)
        #     # cbw.setBackground('w')
        #     # self.sphere_widget.addWidget(cbw2, 0, 2)
        #     # cbw.setSizePolicy(self.sphere.sizePolicy())

        # add the color bar to the side
        self.cbw.addItem(self.color_bar, row = 0, col = 1)
        self.cbw.setBackground('w')
        self.sphere_widget.addWidget(self.cbw, 0, 1)
        self.sphere_widget.layout.setColumnStretch(1, 1)
        self.sphere_widget.layout.setColumnStretch(0, 12)

        self.cbw.setSizePolicy(self.sphere.sizePolicy())
        self.layoutgb.addWidget(self.sphere_widget, 1, 1)

        # synchronize the sphere and 3d graph rotation with each other
        self.graph_3d.sync_camera_with(self.sphere)
        self.sphere.sync_camera_with(self.graph_3d)
        self.sphere_widgets.append(self.sphere_widget)

        # deprecrated but may be used in the future for user study
        if constants.user_mode == 'eval_half':
            for widget in self.sphere_widgets:
                widget.setVisible(False)

    # initialize the histogram, uses the parallelBarPlot class from parallel_bar_plot script
    def initialize_histogram(self):

        self.hist = parallelBarPlot(self.views_metrics, self.metrics_2d, self.metrics_3d, self.view_points, parent=self)
        self.hist.setBackground('w')

        # deprecated but may be used in the future for user study
        if constants.user_mode == 'evalimage':
            self.layoutgb.addWidget(self.hist, 0, 2, 2, 1)
        else:
            self.layoutgb.addWidget(self.hist, 1, 2)

    # initialize the projection of the metric space
    def initialize_metric_proj(self, proj_choice):

        # get the data, local normalization projection or global normalization projection
        name = F"{constants.metrics_projects_dir}/{self.dataset_name}_projcs_global.pkl"
        if proj_choice[0:5] == 'local':
            name = F"{constants.metrics_projects_dir}/{self.dataset_name}_projcs_local.pkl"
        data = pd.read_pickle(name)

        curr_row = data[data['layout_technique'] == self.default_layout_technique]
        proj_data = curr_row['projection'].to_numpy()[0]

        # get the min and max value (minus the min)
        curr_min = np.min(proj_data)
        scale_val = np.max(proj_data - curr_min)

        # extra check, if we only have values of 0 then we set the scale to 1, to avoid / 0
        if scale_val == 0.0:
            scale_val = 1

        #  normalize the data (the points on the scatterplot so that they're between 0 and 1)
        proj_data -= curr_min
        proj_data /= scale_val
        self.proj_data = proj_data

        coloring_norm = self.avg_met_vals_global
        if proj_choice[0:5] == 'local':
            coloring_norm = self.avg_met_vals_local

        self.coloring_norm_proj = coloring_norm

        if self.metric_proj is None:
            self.metric_proj = Scatter2D(proj_data, coloring_norm, self.proj_cmap, parent=self)
            self.metric_proj.setBackground('w')
            self.layoutgb.addWidget(self.metric_proj, 0, 3)
        else:
            self.metric_proj.set_data(proj_data, coloring_norm, self.proj_cmap, self.nearest_viewpoint_idx)

    def initialize_stats(self):

        self.stats = pg.LayoutWidget()

        # Set background white:
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QColor(255, 255, 255, 255))
        self.stats.setPalette(palette)
        self.stats.setAutoFillBackground(True)

        # go over all the metrics
        labels_stats = {}
        for i in list(reversed(range(len(constants.metrics)))):
            # compute how many viewpoints are better than the 2d layout, get the best viewpoint and comute how much better it is
            better_raw_viewpoints_perc = round(np.sum((self.original_metrics_views[:, i] > self.original_qms[i])) / len(self.original_metrics_views[:, i]) * 100, 3)
            best_res = np.max(self.original_metrics_views[:, i])
            best_res_how_much_better = best_res - self.original_qms[i]

            # add a label for each metric, its 2d value and how much better it is
            labels_stats[constants.metrics[i]] = self.stats.addLabel(text=constants.metrics[i], row=len(self.stats.rows), col=0)
            labels_stats[constants.metrics[i]].setFont(QtGui.QFont("Times", weight=QtGui.QFont.Bold))

            labels_stats[constants.metrics[i] + '_h'] = self.stats.addLabel(text=': 2d value ' + str(
                round(self.original_qms[i], 5)) + ' | best 3d value ' + str(round(best_res, 5)),
                                                                     row=len(self.stats.rows), col=0)
            labels_stats[constants.metrics[i] + '_j'] = self.stats.addLabel(text = str(round(better_raw_viewpoints_perc, 3)) + '% of viewpoints are better than 2d value | best viewpoint is ' + str(round(best_res_how_much_better, 3)) + ' better', row = len(self.stats.rows), col = 0)

        self.layoutgb.addWidget(self.stats, 1, 3)

    # since we have a fixed number of viewpoints, rotating the sphere might select a viewpoint that has not been computed
    # simply compute the distance to the nearest viewpoint and return that
    def current_quality(self):

        eye = self.sphere.cameraPosition()
        eye.normalize()

        # Find the viewpoint for which me have metrics. that is closest to the current viewpoint
        distances = np.sum((self.view_points - np.array(eye)) ** 2, axis=1)
        nearest = np.argmin(distances)
        self.nearest_viewpoint_idx = nearest

        # Get the metric values, and highlight the corresponding histogram bars
        nearest_values = self.views_metrics[nearest]
        return nearest_values

    # deprecated but may be used in the future for user study
    # returns the unit vector of the vector
    def unit_vector(self, vector):

        return vector / np.linalg.norm(vector)

    # deprecated but may be used in the future for user study
    def angle(self, v1, v2):

        n_v1 = self.unit_vector(np.array(v1))
        n_v2 = self.unit_vector(np.array(v2))
        dot = np.dot(n_v1, n_v2)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))

        return angle

    # deprecated but may be used in the future for user study
    def check_select_available(self):
        """ Test whether the current viewpoint is close to a previously selected viewpoint, in which case we disable the select button"""
        if self.selected_view_count() >= constants.required_view_count:
            self.select_button.setDisabled(True)
            return
        self.select_button.setDisabled(False)
        self.select_button.setText('Select view')
        for data in self.evaluation_data:
            if data['dataset'] == self.dataset_name and data['layout_technique'] == self.default_layout_technique:
                if self.angle(data['viewpoint'], self.graph_3d.cameraPosition()) < 0.4:
                    self.select_button.setDisabled(True)
                    self.select_button.setText("Find a different viewpoint")

    # highlights the values of the viewpoint metrics on the histogram
    def highlight(self):

        if not self.view_locked:
            # get the closest viewpoint
            nearest_values = self.current_quality()

            self.hist.highlight_bar_with_values(nearest_values, self.original_metrics_views[self.nearest_viewpoint_idx])
            # update the line in the sphere colorbar
            metric_score = nearest_values[constants.metrics.index(self.current_metric)]
            self.color_bar_line.setValue(255 * metric_score)

            self.metric_proj.set_data(self.proj_data, self.coloring_norm_proj, self.proj_cmap, self.nearest_viewpoint_idx)

            # deprecated but may be used in the future for user study
            if constants.user_mode != 'free':
                self.check_select_available()

            # deprecated but may be used in the future for user study
            if constants.debug_mode:
                print('here')
                eye = self.sphere.cameraPosition()
                eye.normalize()

                # Find the viewpoint for which me have metrics. that is closest to the current viewpoint
                distances = np.sum((self.view_points - np.array(eye)) ** 2, axis=1)
                nearest = np.argmin(distances)
                df = pd.read_pickle(f"layouts/{self.dataset_name}-{self.default_layout_technique}-views.pkl")
                view = df['views'][nearest]
                self.graph_2d.set_data(view)

    # move to a certain view based on the parallel bar plot
    def move_to_view(self, metric_index, bin_index, metric_value_l, metric_value_r, percentage):

        if not self.view_locked:
            a = self.views_metrics[:, metric_index]
            indices = np.argwhere(np.logical_and(a >= metric_value_l, a <= metric_value_r)).flatten()
            indices = indices[np.argsort(self.views_metrics[indices, metric_index])]
            index = indices[round((len(indices) - 1) * percentage)]
            viewpoint = np.array(self.view_points[index])
            self.move_to_viewpoint(viewpoint)

    # move to a certain viewpoint based on the sphere rotation, updates the 3d graph layout and sphere
    def move_to_viewpoint(self, viewpoint):

        self.curr_viewpoint = viewpoint
        viewpoint_spherical = utils.rectangular_to_spherical(np.array([viewpoint]))[0]
        self.sphere.setCameraPosition(azimuth=viewpoint_spherical[1], elevation=viewpoint_spherical[0],
                                      distance=self.sphere.cameraParams()['distance'])
        self.graph_3d.setCameraPosition(azimuth=viewpoint_spherical[1], elevation=viewpoint_spherical[0],
                                        distance=self.graph_3d.cameraParams()['distance'])
        self.sphere.update_views()
        self.graph_3d.update_order()

    # attached to the data selection drop down menu, selects dataset
    def data_selected(self):

        dataset_name = self.dataset_picker.value()
        layout_technique = self.layout_technique_picker.value()
        self.set_data(dataset_name, layout_technique)
        self.metric_selected(self.metric_picker.value())

    def change_proj_type(self):

        self.proj_choice = self.projection_picker.value()
        self.initialize_metric_proj(self.proj_choice)

    # attached to the metric selection drop down menu, selects the metric
    def metric_selected(self, metric):

        if self.metric_picker.value() in constants.metrics:
            self.current_metric = metric
            self.initialize_sphere()
        else:
            self.initialize_sphere(sphere_sum= metric)

        self.graph_3d.update_views()
        self.sphere.update_views()

    # sets the data based on the dataset and layout technique and metric
    def set_data(self, dataset_name, layout_technique):

        """
        Update the data of all the widgets inside the tool to a new dataset and layout technique combination
        """
        self.dataset_name = dataset_name
        self.default_layout_technique = layout_technique
        metrics_file = constants.metrics_dir + '/metrics_' + self.dataset_name + '.pkl'
        metrics_file = r""+metrics_file
        #metrics_file = os.path.join(constants.metrics_dir, F'metrics_{self.dataset_name}.pkl')
        #df = pd.read_pickle(metrics_file)

        try:
            df = pd.read_pickle(metrics_file)
        except Exception as e:
            print(e)
            print(metrics_file)

        select = df.loc[df['layout_technique'] == self.default_layout_technique]

        # deprecated but may be used in the future for user study
        self.iscategorical = self.dataset_name in constants.categorical_datasets
        if self.iscategorical:
            self.cmap = cm.get_cmap('tab10')
        else:
            self.cmap = cm.get_cmap('rainbow')

        # get the views and the metrics of the views and the 2d layout
        self.views_metrics = select.iloc[1]['views_metrics']
        self.metrics_2d = select.iloc[0][constants.metrics].to_numpy()
        self.metrics_3d = select.iloc[1][constants.metrics].to_numpy()
        self.n_metrics = np.shape(self.views_metrics)[1]

        # normalization step: for each quality metric, the lowest seen value of all views is set to be the lowest point (0), then the highest
        # seen value of all views is set to the highest point (1), local normalization
        # global normalization is when we normalize according to all seen values from all views from all datasets and techniques
        self.original_qms = copy.deepcopy(self.metrics_2d)
        self.original_metrics_views = copy.deepcopy(np.array(self.views_metrics))
        self.views_metrics_global = copy.deepcopy(np.array(self.views_metrics))
        self.metrics_2d_global = copy.deepcopy(self.metrics_2d)

        qm_idx = dict(zip(range(len(self.metrics_2d)), self.metrics_2d))
        for key, val in qm_idx.items():
            temp_array = np.append(self.views_metrics[:, key], val)
            curr_min_local = np.min(temp_array)
            scale_val_local = np.max(temp_array - curr_min_local)

            # extra check, if we only have values of 0 then we set the scale to 1, to avoid / 0
            if scale_val_local == 0.0:
                scale_val_local = 1

            curr_min_global = self.mins_global[key]
            curr_max_global = self.maxs_global[key]
            scale_val_global = curr_max_global - curr_min_global

            self.views_metrics[:, key] = (self.views_metrics[:, key] - curr_min_local) / scale_val_local
            self.views_metrics_global[:, key] = (self.views_metrics_global[:, key] - curr_min_global) / scale_val_global
            self.metrics_2d[key] = (self.metrics_2d[key] - curr_min_local) / scale_val_local
            self.metrics_2d_global[key] = (self.metrics_2d_global[key] - curr_min_global) / scale_val_global
            self.metrics_3d[key] = (self.metrics_3d[key] - curr_min_local) / scale_val_local

        # now all of our metric values are scaled locally or globally
        # we want the averages of these metrics across all viewpoints in order to color them
        self.avg_met_vals_local = np.mean(np.vstack((self.views_metrics, self.metrics_2d)), axis = 1)
        # to compare the global averages we want the distribution of the global averages to be the same, so we normalize these again
        # using the minimum and maximum average seen of all layouts (global min and max of all averages)
        self.avg_met_vals_global = np.mean(np.vstack((self.views_metrics_global, self.metrics_2d_global)), axis = 1)
        self.avg_met_vals_global = (self.avg_met_vals_global - self.global_average_min) / (self.global_average_max - self.global_average_min)

        # initialize all the layouts, plots etc.
        self.initialize_3d_layout()
        self.initialize_2d_layout()

        self.initialize_histogram()
        self.initialize_sphere()
        self.initialize_metric_proj(self.proj_choice)
        self.initialize_stats()
        self.graph_3d.update_views()
        self.highlight()

    # captures the downpress of the f button, attached to locking the tool
    def keyboard_event(self, event):

        pass
        if event.event_type == 'down':
            if event.name == 'f':
                self.set_tool_lock(True)

    def indices(self, di, pi):

        pi += 1
        if pi == 5:
            di += 1
            pi = 0

        return di, pi

    def save_image(self, dataset, layout, name):

        exporter = pg.exporters.ImageExporter(self.hist.scene())
        exporter.export(f'{constants.analysis_dir}/{dataset}-{layout}-{name}.png')
        if constants.user_mode == 'image':
            for metric in constants.metrics:
                self.metric_selected(metric)
                self.sphere.readQImage().save(f"{constants.analysis_dir}/{dataset}-{layout}-{metric}-sphere1.png")
                self.move_to_viewpoint(-np.array(self.sphere.cameraPosition()))
                self.sphere.readQImage().save(f"{constants.analysis_dir}/{dataset}-{layout}-{metric}-sphere2.png")

        # QtCore.QTimer.singleShot(100, lambda: self.sphere.readQImage().save(
        #     f"{constants.analysis_dir}/{dataset}-{layout}-sphere1.png"))
        # QtCore.QTimer.singleShot(200, lambda: self.move_to_viewpoint(-np.array(self.sphere.cameraPosition())))
        # QtCore.QTimer.singleShot(300, lambda: self.sphere.readQImage().save(
        #     f"{constants.analysis_dir}/{dataset}-{layout}-sphere2.png"))

    # deprecated but may be used in the future for user study
    def save_images(self, tuple):

        di, pi = tuple
        configs = self.available_datasets_layouts()
        dataset = list(configs.keys())[di]
        layout = list(configs[dataset])[pi]
        QtCore.QTimer.singleShot(100, lambda: self.set_data(dataset, layout))
        QtCore.QTimer.singleShot(200, lambda: self.save_image(dataset, layout, 'histograms'))
        QtCore.QTimer.singleShot(300, lambda: self.save_images(self.indices(di, pi)))

    # deprecated but may be used in the future for user study
    def get_boxplot_data(self):

        data_with_tools = self.analysis_data[(self.analysis_data['dataset'] == self.dataset_name) &
                                             (self.analysis_data['layout_technique'] == self.default_layout_technique) &
                                             (self.analysis_data['mode'] == 'eval_full')]
        data_without_tools = self.analysis_data[(self.analysis_data['dataset'] == self.dataset_name) &
                                                (self.analysis_data['layout_technique'] == self.default_layout_technique) &
                                                (self.analysis_data['mode'] == 'eval_half')]
        qualities_with_tools = np.array([l for l in data_with_tools['view_quality']])
        data_without_tools = np.array([l for l in data_without_tools['view_quality']])
        box_plot_data = []
        for quality_lists in [self.views_metrics, data_without_tools, qualities_with_tools]:
            box_plot_data.append([
                quality_lists.mean(axis=0),
                np.quantile(quality_lists, 0.25, axis=0),
                np.quantile(quality_lists, 0.75, axis=0),
                quality_lists.min(axis=0),
                quality_lists.max(axis=0)
            ])

        return np.array(box_plot_data)

    # deprecated but may be used in the future for user study
    def box_plot_images(self, index=0):

        dataset, layout = constants.evaluation_set[1:][index]
        self.set_data(dataset, layout)
        self.hist.draw_box_plots()
        QtCore.QTimer.singleShot(200, lambda: self.save_image(dataset, layout, 'boxplots2'))
        QtCore.QTimer.singleShot(300, lambda: self.box_plot_images(index + 1))

    # deprecated but may be used in the future for user study
    def get_user_selected_viewpoints(self):

        """
        Return a list of all viewpoint sets from the evaluation data.
        Order: Guided 2D preference, Guided 3D preference, Blind 2D preference, Blind 3D preference
        """
        data = self.analysis_data.where((self.analysis_data['layout_technique'] == self.default_layout_technique) &
                                        (self.analysis_data['dataset'] == self.dataset_name))
        viewpoints = []
        for mode in ['eval_full', 'eval_half']:
            for preference in ['2D', '3D']:
                viewpoints_sub = data.loc[(data['mode'] == mode) & (data['preference'] == preference)]['viewpoint'].to_numpy()
                viewpoints_sub = np.array([p for p in viewpoints_sub])
                viewpoints.append(viewpoints_sub)

        return viewpoints

    # deprecated but may be used in the future for user study
    def save_snapshot(self, dataset, layout, viewpoints, i, type, preference):

        if len(viewpoints) == i:
            return

        QtCore.QTimer.singleShot(10, lambda: self.move_to_viewpoint(viewpoints[i]))
        path = f"{constants.analysis_dir}/snapshots/{type}/{preference}/{dataset}-{layout}-{i}.png"
        utils.create_folder_for_path(path)
        QtCore.QTimer.singleShot(20, lambda: self.graph_3d.readQImage().save(path))
        QtCore.QTimer.singleShot(30, lambda: self.save_snapshot(dataset, layout, viewpoints, i + 1, type, preference))

    # deprecated but may be used in the future for user study
    def save_user_selected_view_snapshots(self, index, set = 0):

        dataset, layout = constants.evaluation_set[1:][index]
        self.set_data(dataset, layout)

        #Get the right viewpoint set, and save all snapshots
        viewpoint_sets = self.get_user_selected_viewpoints()
        type = 'guided' if set <= 1 else 'blind'
        preference = '2D_preference' if set in [0, 2] else '3D_preference'
        self.save_snapshot(dataset, layout, viewpoint_sets[set], 0, type, preference)

        #Save snapshots of the 2D layout
        path = f"{constants.analysis_dir}/snapshots/2D/{dataset}-{layout}.png"
        utils.create_folder_for_path(path)
        exporter = pg.exporters.ImageExporter(self.graph_2d.scene())
        exporter.export(path)

        #Recursive call for the next dataset
        if index == len(constants.evaluation_set[1:]) - 1:
            set = set + 1
            if set >= 4:
                return

        QtCore.QTimer.singleShot(5000, lambda: self.save_user_selected_view_snapshots(index + 1, set = set))
