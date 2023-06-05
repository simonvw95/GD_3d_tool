"""
Simple example using BarGraphItem
"""
import math
from glob import glob
import os
import numpy as np
import pandas as pd
import pickle
import keyboard
import networkx as nx
import copy

import time

import pyqtgraph as pg
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QPushButton, QSlider
from pyqtgraph import mkPen
import pyqtgraph.exporters

from matplotlib import cm

import utils
import constants
#from visualizations.scatter_3d import Scatter3D
#from visualizations.scatter_3d_gd import Scatter3D
from visualizations.scatter_3d_gd_v2 import Scatter3D
#from visualizations.graph_3d import Scatter3D
from visualizations.scatter_2d import Scatter2D
from visualizations.quality_sphere import QualitySphere
from visualizations.parallel_bar_plot import parallelBarPlot
from functools import partial


class Tool(pg.GraphicsWindow):
    def __init__(self, dataset_name="grafo1126", default_layout_technique="FA2", analysis_data=None):
        super(Tool, self).__init__()

        keyboard.on_press(self.keyboard_event)

        #Setup data
        self.dataset_name = dataset_name
        self.default_layout_technique = default_layout_technique
        self.view_locked = False

        # Grid initialization
        self.setBackground((0, 0, 0, 60))
        self.layoutgb = QtWidgets.QGridLayout()
        self.layoutgb.setHorizontalSpacing(1)
        self.layoutgb.setVerticalSpacing(1)
        self.layoutgb.setContentsMargins(1, 1, 1, 1)
        self.setLayout(self.layoutgb)

        self.layoutgb.setColumnStretch(0, 2)
        self.layoutgb.setColumnStretch(1, 10)
        self.layoutgb.setColumnStretch(2, 10)
        self.layoutgb.setRowStretch(0, 10)
        self.layoutgb.setRowStretch(1, 10)
        self.sphere_widgets = []

        self.analysis_data = analysis_data

        if constants.user_mode != 'free':
            self.projection_index = 0
            self.dataset_name, self.default_layout_technique = constants.evaluation_set[self.projection_index]
            self.evaluation_data = []
        self.view_points = np.load(f'spheres/sphere{constants.samples}_points.npy')
        self.D_P_dict = self.available_datasets_layouts()
        self.current_metric = 'crossing_number'
        self.initialize_menu()
        self.scatter_2d = None
        self.scatter_3d = None

        self.set_data(self.dataset_name, self.default_layout_technique)

        self.highlight()

    def get_labels(self):
        label_file = glob(F'data/{self.dataset_name}/*-labels.csv')
        if len(label_file) == 1:
            df_label = pd.read_csv(label_file[0], sep=';', header=0)
            #return (df_label.to_numpy().flatten() / 2).astype(int)
            return df_label.to_numpy().flatten()
        else:
            return None

    def available_datasets_layouts(self):
        consolid_metrics = os.path.join(constants.metrics_dir, 'metrics.pkl')
        data_frame = pd.read_pickle(consolid_metrics)
        D_P_dict = {}
        datasets = set(data_frame['dataset_name'].to_list())
        for dataset in datasets:
            D_P_dict[dataset.split('.')[0]] = set(data_frame[data_frame['dataset_name'] == dataset]['layout_technique'].to_list())
        return D_P_dict

    def initialize_menu(self):
        self.menu = pg.LayoutWidget()

        # Set background white:
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Window, QColor(255, 255, 255, 255))
        self.menu.setPalette(palette)
        self.menu.setAutoFillBackground(True);

        # get the graph file
        input_file = glob('data/{0}/*-src.csv'.format(self.dataset_name))[0]
        df = pd.read_csv(input_file, sep=';', header=0)
        graph = nx.from_pandas_edgelist(df, 'from', 'to', edge_attr='strength')
        graph = nx.convert_node_labels_to_integers(graph)
        self.G = graph

        if constants.user_mode == 'free':
            #Options to switch between datasets and pr_techniques
            keys = list(self.D_P_dict.keys())
            datasets = list(self.D_P_dict.keys())
            datasets.sort()
            self.dataset_picker = pg.ComboBox(items=datasets, default=self.dataset_name)
            layouts = list(self.D_P_dict[keys[0]])
            layouts.sort()
            self.layout_technique_picker = pg.ComboBox(items=layouts, default=self.default_layout_technique)
            self.dataset_picker.currentIndexChanged.connect(self.data_selected)
            self.layout_technique_picker.currentIndexChanged.connect(self.data_selected)

            self.metric_picker = pg.ComboBox(items = constants.metrics, default = self.current_metric)
            self.metric_picker.currentIndexChanged.connect(self.data_selected)

            self.menu.addLabel(text="Dataset:", row=len(self.menu.rows), col=0)
            self.menu.addWidget(self.dataset_picker, len(self.menu.rows), 0)
            self.menu.addLabel(text="Layout technique:", row=len(self.menu.rows), col=0)
            self.menu.addWidget(self.layout_technique_picker, len(self.menu.rows), 0)
            self.menu.addLabel(text="Quality Metric:", row=len(self.menu.rows), col=0)
            self.menu.addWidget(self.metric_picker, len(self.menu.rows), 0)

            self.dataset_picker.currentIndexChanged.connect(self.change_label_text)

            self.label_n = QtWidgets.QLabel('Number of nodes (n): ' + str(self.G.number_of_nodes()))
            self.label_m = QtWidgets.QLabel('Number of edges (m): ' + str(self.G.number_of_edges()))
            self.menu.addWidget(self.label_n, row=len(self.menu.rows), col=0)
            self.menu.addWidget(self.label_m, row=len(self.menu.rows), col=0)

            self.best_solution_button = QPushButton('Find best solution')
            self.best_solution_button.pressed.connect(self.get_best_solution)
            self.menu.addWidget(self.best_solution_button, len(self.menu.rows), 0)

            # sliders for qms
            self.n_metrics = len(constants.metrics)
            # map the name of the metrics to the integer identifiers
            self.metrics_name_dict = dict(zip(range(len(constants.metrics)), constants.metrics))

            self.sliders = {}
            self.sliders_labels = {}
            self.weights = {}
            # create sliders for the quality metrics
            for i in range(self.n_metrics):

                self.sliders_labels[i] = QtWidgets.QLabel(self.metrics_name_dict[i] + ' 1')
                self.sliders_labels[i].setObjectName(str(i))
                self.menu.addWidget(self.sliders_labels[i], len(self.menu.rows), 0)

                self.weights[i] = 100
                self.sliders[i] = QSlider(Qt.Horizontal)
                self.sliders[i].setMinimum(0)
                self.sliders[i].setMaximum(100)
                self.sliders[i].setValue(100)
                self.sliders[i].setSingleStep(5)
                self.sliders[i].setTickInterval(10)
                self.sliders[i].setTickPosition(QSlider.TicksBelow)
                self.sliders[i].valueChanged.connect(self.update_weights)
                self.sliders[i].setObjectName(str(i))
                self.menu.addWidget(self.sliders[i], len(self.menu.rows), 0)


        else:
            self.evaluation_started = False #Keep track of when the tutorial is over

            self.next_button = QPushButton('Begin survey')
            self.next_button.pressed.connect(self.next_projection)
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

    def update_weights(self, value):
        sender = self.sender()
        name = sender.objectName()
        self.sliders_labels[int(name)].setText(self.metrics_name_dict[int(name)] + ' ' + str(round(value / 100, 2)))
        self.weights[int(name)] = value

    def change_label_text(self):

        self.label_n.setText('Number of nodes (n): ' + str(self.G.number_of_nodes()))
        self.label_m.setText('Number of edges (m): ' + str(self.G.number_of_edges()))

    def next_projection(self):

        if not self.evaluation_started:
            self.evaluation_started = True
            self.select_button.setVisible(True)
            self.selected_counter.setVisible(True)
            self.next_button.setText('Next projection')

            constants.user_mode = 'eval_half'
            self.sphere_widget.setVisible(False)
            self.hist.setVisible(False)
            self.sphere_widget.setVisible(False)

        if constants.user_mode != 'free':
            self.projection_index += 1
            if self.projection_index < len(constants.evaluation_set):
                if self.projection_index >= 4:
                    constants.user_mode = 'eval_full'
                config = constants.evaluation_set[self.projection_index]
                self.set_data(config[0], config[1])
                self.set_tool_lock(False)
                self.next_button.setDisabled(True)
                self.update_selected_count_text()
            else:
                with open(constants.output_file, 'wb') as file:
                    pickle.dump(self.evaluation_data, file)
                self.close()

    def select_view(self):
        self.set_tool_lock(not self.view_locked)
        pass

    def selected_view_count(self):
        count = 0
        for data in self.evaluation_data:
            if data['dataset'] == self.dataset_name and data['layout_technique'] == self.default_layout_technique:
                count += 1
        return count

    def update_selected_count_text(self):
        self.selected_counter.setText(f"{self.selected_view_count()}/{constants.required_view_count} views selected")

    def select_preference(self, preference):
        self.preference_label.setVisible(False)
        self.prefer_3d.setVisible(False)
        self.prefer_2d.setVisible(False)
        self.set_tool_lock(False)
        self.evaluation_data.append({
            'dataset': self.dataset_name,
            'layout_technique': self.default_layout_technique,
            'viewpoint': np.array(self.scatter_3d.cameraPosition()),
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

    def set_tool_lock(self, lock):
        self.view_locked = lock
        if self.view_locked:
            self.select_button.setText('Deselect view')
        else:
            self.select_button.setText('Select view')
        self.hist.lock = self.view_locked
        self.scatter_3d.lock = self.view_locked
        self.sphere.lock = self.view_locked
        self.preference_label.setVisible(self.view_locked)
        self.prefer_3d.setVisible(self.view_locked)
        self.prefer_2d.setVisible(self.view_locked)

    # function for displaying the viewpoint with the best weighted solution
    def get_best_solution(self):

        curr_views_metrics = copy.deepcopy(self.views_metrics)
        for i in range(self.n_metrics):
            curr_views_metrics[:, i] *= (self.weights[i] / 100)


        best_view = np.argmax(np.sum(curr_views_metrics, axis = 1))
        view = self.view_points[best_view]

        self.move_to_viewpoint(view)


    def initialize_3d_scatterplot(self):
        proj_file_3d = F"{constants.output_dir}/{self.dataset_name}-{self.default_layout_technique}-3d.csv"

        input_file = glob('data/{0}/*-src.csv'.format(self.dataset_name))[0]
        df = pd.read_csv(input_file, sep=';', header=0)
        graph = nx.from_pandas_edgelist(df, 'from', 'to', edge_attr='strength')
        graph = nx.convert_node_labels_to_integers(graph)
        edges = list(graph.edges())

        # will overwrite the existing graph (in case a new dataset is chosen in the widget)
        self.G = graph

        data = pd.read_csv(proj_file_3d, sep=';').to_numpy()

        if self.scatter_3d is None:
            self.scatter_3d = Scatter3D(data, self.labels, self.cmap, self.iscategorical, edges, parent=self, title="3D Layout")
            self.scatter_3d.setBackgroundColor('w')
            self.layoutgb.addWidget(self.scatter_3d, 0, 1)
        else:
            self.scatter_3d.set_data(data, self.labels, self.cmap, self.iscategorical, edges)

    def initialize_2d_scatterplot(self):
        # 2D Scatter
        if constants.user_mode == 'evalimage':
            return

        input_file = glob('data/{0}/*-src.csv'.format(self.dataset_name))[0]
        df = pd.read_csv(input_file, sep=';', header=0)
        graph = nx.from_pandas_edgelist(df, 'from', 'to', edge_attr='strength')
        graph = nx.convert_node_labels_to_integers(graph)
        edges = list(graph.edges())

        proj_file_2d = F"{constants.output_dir}/{self.dataset_name}-{self.default_layout_technique}-2d.csv"
        df_2d = pd.read_csv(proj_file_2d, sep=';').to_numpy()
        if self.scatter_2d is None:
            self.scatter_2d = Scatter2D(df_2d, self.labels, self.cmap, self.iscategorical, edges, title = "2D Layout")
            self.layoutgb.addWidget(self.scatter_2d, 0, 2)
            self.scatter_2d.setBackground('w')
        else:
            self.scatter_2d.set_data(df_2d, self.labels, self.cmap, self.iscategorical, edges)

    def initialize_sphere(self):
        self.sphere_data = np.copy(self.views_metrics[:, constants.metrics.index(self.current_metric)])

        c = ["darkred", "red", "yellow", "green", "darkgreen"]
        v = [ 0, 0.2, 0.5, 0.8, 1]
        self.heatmap = pg.ColorMap(v, c)

        self.sphere = QualitySphere(self.sphere_data, self.heatmap, parent=self, title=F"Viewpoint quality ({self.current_metric})")
        self.sphere.setBackgroundColor('w')

        self.sphere_widget = pg.LayoutWidget()
        self.sphere_widget.addWidget(self.sphere, 0, 0)
        self.sphere_widget.layout.setContentsMargins(0, 0, 0, 0)
        self.sphere_widget.layout.setHorizontalSpacing(0)

        self.cbw = pg.GraphicsLayoutWidget()
        self.color_bar = pg.ColorBarItem(colorMap=self.heatmap, interactive=False, values=(0, 1))

        #Display max, min and current metric value with a horizontal line
        self.color_bar.addLine(y=np.max(self.sphere_data) * 255, pen=mkPen(255, 255, 255, width=2))
        self.color_bar.addLine(y=np.min(self.sphere_data) * 255, pen=mkPen(255, 255, 255, width=2))
        self.color_bar_line = self.color_bar.addLine(y=255, pen=mkPen(0,0,0,255))

        self.cbw.addItem(self.color_bar)
        self.cbw.setBackground('w')
        self.sphere_widget.addWidget(self.cbw, 0, 1)
        self.sphere_widget.layout.setColumnStretch(1, 1)
        self.sphere_widget.layout.setColumnStretch(0, 12)

        self.cbw.setSizePolicy(self.sphere.sizePolicy())
        self.layoutgb.addWidget(self.sphere_widget, 1, 1)

        self.scatter_3d.sync_camera_with(self.sphere)
        self.sphere.sync_camera_with(self.scatter_3d)
        self.sphere_widgets.append(self.sphere_widget)
        if constants.user_mode == 'eval_half':
            for widget in self.sphere_widgets:
                widget.setVisible(False)

    def initialize_histogram(self):
        self.hist = parallelBarPlot(self.views_metrics, self.metrics_2d, self.metrics_3d, self.view_points, parent=self)
        self.hist.setBackground('w')

        if constants.user_mode == 'evalimage':
            self.layoutgb.addWidget(self.hist, 0, 2, 2, 1)
        else:
            self.layoutgb.addWidget(self.hist, 1, 2)

    def current_quality(self):
        eye = self.sphere.cameraPosition()
        eye.normalize()

        # Find the viewpoint for which me have metrics. that is closest to the current viewpoint
        distances = np.sum((self.view_points - np.array(eye)) ** 2, axis=1)
        nearest = np.argmin(distances)

        # Get the metric values, and highlight the corresponding histogram bars
        nearest_values = self.views_metrics[nearest]
        return nearest_values

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle(self, v1, v2):
        n_v1 = self.unit_vector(np.array(v1))
        n_v2 = self.unit_vector(np.array(v2))
        dot = np.dot(n_v1, n_v2)
        angle = np.arccos(np.clip(dot, -1.0, 1.0))
        return angle

    def check_select_available(self):
        """ Test whether the current viewpoint is close to a previously selected viewpoint, in which case we disable the select button"""
        if self.selected_view_count() >= constants.required_view_count:
            self.select_button.setDisabled(True)
            return
        self.select_button.setDisabled(False)
        self.select_button.setText('Select view')
        for data in self.evaluation_data:
            if data['dataset'] == self.dataset_name and data['layout_technique'] == self.default_layout_technique:
                if self.angle(data['viewpoint'], self.scatter_3d.cameraPosition()) < 0.4:
                    self.select_button.setDisabled(True)
                    self.select_button.setText("Find a different viewpoint")

    def highlight(self):
        if not self.view_locked:
            nearest_values = self.current_quality()
            self.hist.highlight_bar_with_values(nearest_values)
            #Update the line in the sphere colorbar
            metric_score = nearest_values[constants.metrics.index(self.current_metric)]
            self.color_bar_line.setValue(255 * metric_score)

            if constants.user_mode != 'free':
                self.check_select_available()

            if constants.debug_mode:
                eye = self.sphere.cameraPosition()
                eye.normalize()

                # Find the viewpoint for which me have metrics. that is closest to the current viewpoint
                distances = np.sum((self.view_points - np.array(eye)) ** 2, axis=1)
                nearest = np.argmin(distances)
                df = pd.read_pickle(f"layouts/{self.dataset_name}-{self.default_layout_technique}-views.pkl")
                view = df['views'][nearest]
                self.scatter_2d.set_data(view, self.labels)

    def move_to_view(self, metric_index, bin_index, metric_value_l, metric_value_r, percentage):
        if not self.view_locked:
            a = self.views_metrics[:, metric_index]
            indices = np.argwhere(np.logical_and(a >= metric_value_l, a <= metric_value_r)).flatten()
            indices = indices[np.argsort(self.views_metrics[indices, metric_index])]
            index = indices[round((len(indices) - 1) * percentage)]
            viewpoint = np.array(self.view_points[index])
            self.move_to_viewpoint(viewpoint)

    def move_to_viewpoint(self, viewpoint):
        viewpoint_spherical = utils.rectangular_to_spherical(np.array([viewpoint]))[0]
        self.sphere.setCameraPosition(azimuth=viewpoint_spherical[1], elevation=viewpoint_spherical[0],
                                      distance=self.sphere.cameraParams()['distance'])
        self.scatter_3d.setCameraPosition(azimuth=viewpoint_spherical[1], elevation=viewpoint_spherical[0],
                                          distance=self.scatter_3d.cameraParams()['distance'])
        self.sphere.update_views()
        self.scatter_3d.update_order()

    def data_selected(self):
        dataset_name = self.dataset_picker.value()
        projection_method = self.layout_technique_picker.value()
        self.set_data(dataset_name, projection_method)
        self.metric_selected(self.metric_picker.value())
        pass

    def metric_selected(self, metric):
        self.current_metric = metric
        self.initialize_sphere()
        self.scatter_3d.update_views()
        self.sphere.update_views()

    def get_bounds(self, individual_scaling=True, include_ticks=False):

        consolid_metrics = os.path.join(constants.metrics_dir, 'metrics.pkl')
        data_frame = pd.read_pickle(consolid_metrics)

        metrics_file = os.path.join(constants.metrics_dir, F'metrics_{self.dataset_name}.pkl')
        df = pd.read_pickle(metrics_file)
        select = df.loc[df['layout_technique'] == self.default_layout_technique]

        if individual_scaling:
            consolid_metrics = os.path.join(constants.metrics_dir, 'metrics.pkl')
            data_frame = pd.read_pickle(consolid_metrics)
        else:
            #scale equally all 30 layouts
            metrics_file = os.path.join(constants.metrics_dir, F'metrics_{self.dataset_name}.pkl')
            df = pd.read_pickle(metrics_file)
            data_frame = df.loc[df['layout_technique'] == self.default_layout_technique]

        views_metrics = data_frame['views_metrics'].values
        views_metrics = [l for l in views_metrics if len(l) > 0]
        views_metrics = np.array([l for l in views_metrics if len(l) > 0])
        #views_metrics = views_metrics.reshape((views_metrics.shape[0] * views_metrics.shape[1], views_metrics.shape[2]))[:, 1:]
        views_metrics = views_metrics.reshape((views_metrics.shape[0] * views_metrics.shape[1], views_metrics.shape[2]))
        if include_ticks:
            ticks = data_frame[constants.metrics].to_numpy()
            conc = np.concatenate((views_metrics, ticks))

        mins = np.min(views_metrics, axis=0)
        maxs = np.max(views_metrics, axis=0)
        return mins, maxs

    def set_data(self, dataset_name, projection_method):
        """
        Update the data of all the widgets inside the tool to a new dataset and projection technique combination
        """
        self.dataset_name = dataset_name
        self.default_layout_technique = projection_method
        metrics_file = os.path.join(constants.metrics_dir, F'metrics_{self.dataset_name}.pkl')
        df = pd.read_pickle(metrics_file)
        select = df.loc[df['layout_technique'] == self.default_layout_technique]

        self.iscategorical = self.dataset_name in constants.categorical_datasets
        if self.iscategorical:
            self.cmap = cm.get_cmap('tab10')
        else:
            self.cmap = cm.get_cmap('rainbow')

        # use commented out if I get the multiprocessing thing working
        #self.views_metrics = select.iloc[1]['views_metrics'][:, 1:]
        self.views_metrics = select.iloc[1]['views_metrics']
        self.metrics_2d = select.iloc[0][constants.metrics].to_numpy()
        self.metrics_3d = select.iloc[1][constants.metrics].to_numpy()
        self.n_metrics = np.shape(self.views_metrics)[1]

        #mins, maxs = self.get_bounds()
        # commented out because this does not work at the moment (probably to do with how Wouter computed his metrics)
        # self.views_metrics -= mins

        # commented out as my metrics are already scaled between 0 and 1
        # self.views_metrics = (self.views_metrics - mins) / (maxs - mins)
        # self.metrics_2d = (self.metrics_2d - mins) / (maxs - mins)
        # self.metrics_3d = (self.metrics_3d - mins) / (maxs - mins)


        self.labels = self.get_labels()
        self.initialize_3d_scatterplot()
        self.initialize_2d_scatterplot()

        self.initialize_histogram()
        self.initialize_sphere()
        self.scatter_3d.update_views()
        self.highlight()

    def keyboard_event(self, event):
        pass
        # if event.event_type == 'down':
        #     if event.name == '1':
        #         vp = self.analysis_data['viewpoint'][0]
        #         self.move_to_viewpoint(vp)
        #           print('w')

    def indices(self, di, pi):
        pi += 1
        if pi == 5:
            di += 1
            pi = 0
        return di, pi

    def save_image(self, dataset, projection, name):
        exporter = pg.exporters.ImageExporter(self.hist.scene())
        exporter.export(f'{constants.analysis_dir}/{dataset}-{projection}-{name}.png')
        if constants.user_mode == 'image':
            for metric in constants.metrics:
                self.metric_selected(metric)
                self.sphere.readQImage().save(f"{constants.analysis_dir}/{dataset}-{projection}-{metric}-sphere1.png")
                self.move_to_viewpoint(-np.array(self.sphere.cameraPosition()))
                self.sphere.readQImage().save(f"{constants.analysis_dir}/{dataset}-{projection}-{metric}-sphere2.png")

        # QtCore.QTimer.singleShot(100, lambda: self.sphere.readQImage().save(
        #     f"{constants.analysis_dir}/{dataset}-{projection}-sphere1.png"))
        # QtCore.QTimer.singleShot(200, lambda: self.move_to_viewpoint(-np.array(self.sphere.cameraPosition())))
        # QtCore.QTimer.singleShot(300, lambda: self.sphere.readQImage().save(
        #     f"{constants.analysis_dir}/{dataset}-{projection}-sphere2.png"))
    def save_images(self, tuple):
        di, pi = tuple
        configs = self.available_datasets_layouts()
        dataset = list(configs.keys())[di]
        projection = list(configs[dataset])[pi]
        QtCore.QTimer.singleShot(100, lambda: self.set_data(dataset, projection))
        QtCore.QTimer.singleShot(200, lambda: self.save_image(dataset, projection, 'histograms'))
        QtCore.QTimer.singleShot(300, lambda: self.save_images(self.indices(di, pi)))

    def get_boxplot_data(self):
        data_with_tools = self.analysis_data[(self.analysis_data['dataset'] == self.dataset_name) &
                                             (self.analysis_data['projection_method'] == self.default_layout_technique) &
                                             (self.analysis_data['mode'] == 'eval_full')]
        data_without_tools = self.analysis_data[(self.analysis_data['dataset'] == self.dataset_name) &
                                                (self.analysis_data['projection_method'] == self.default_layout_technique) &
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

    def box_plot_images(self, index=0):
        dataset, projection = constants.evaluation_set[1:][index]
        self.set_data(dataset, projection)
        self.hist.draw_box_plots()
        QtCore.QTimer.singleShot(200, lambda: self.save_image(dataset, projection, 'boxplots2'))
        QtCore.QTimer.singleShot(300, lambda: self.box_plot_images(index + 1))

    def get_user_selected_viewpoints(self):
        """
        Return a list of all viewpoint sets from the evaluation data.
        Order: Guided 2D preference, Guided 3D preference, Blind 2D preference, Blind 3D preference
        """
        data = self.analysis_data.where((self.analysis_data['projection_method'] == self.default_layout_technique) &
                                        (self.analysis_data['dataset'] == self.dataset_name))
        viewpoints = []
        for mode in ['eval_full', 'eval_half']:
            for preference in ['2D', '3D']:
                viewpoints_sub = data.loc[(data['mode'] == mode) & (data['preference'] == preference)]['viewpoint'].to_numpy()
                viewpoints_sub = np.array([p for p in viewpoints_sub])
                viewpoints.append(viewpoints_sub)
        return viewpoints

    def save_snapshot(self, dataset, projection, viewpoints, i, type, preference):
        if len(viewpoints) == i:
            return
        QtCore.QTimer.singleShot(10, lambda: self.move_to_viewpoint(viewpoints[i]))
        path = f"{constants.analysis_dir}/snapshots/{type}/{preference}/{dataset}-{projection}-{i}.png"
        utils.create_folder_for_path(path)
        QtCore.QTimer.singleShot(20, lambda: self.scatter_3d.readQImage().save(path))
        QtCore.QTimer.singleShot(30, lambda: self.save_snapshot(dataset, projection, viewpoints, i + 1, type, preference))

    def save_user_selected_view_snapshots(self, index, set = 0):
        dataset, projection = constants.evaluation_set[1:][index]
        self.set_data(dataset, projection)

        #Get the right viewpoint set, and save all snapshots
        viewpoint_sets = self.get_user_selected_viewpoints()
        type = 'guided' if set <= 1 else 'blind'
        preference = '2D_preference' if set in [0, 2] else '3D_preference'
        self.save_snapshot(dataset, projection, viewpoint_sets[set], 0, type, preference)

        #Save snapshots of the 2D projection
        path = f"{constants.analysis_dir}/snapshots/2D/{dataset}-{projection}.png"
        utils.create_folder_for_path(path)
        exporter = pg.exporters.ImageExporter(self.scatter_2d.scene())
        exporter.export(path)

        #Recursive call for the next dataset
        if index == len(constants.evaluation_set[1:]) - 1:
            set = set + 1
            if set >= 4:
                return
        QtCore.QTimer.singleShot(5000, lambda: self.save_user_selected_view_snapshots(index + 1, set = set))

















