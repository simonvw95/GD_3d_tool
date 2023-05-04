from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QPainter, QFont
from pyqtgraph.Qt import QtGui
from pyqtgraph.opengl import shaders
from pyqtgraph.opengl.shaders import ShaderProgram, VertexShader, FragmentShader

import constants
from visualizations.synced_camera_view_widget import SyncedCameraViewWidget
import pyqtgraph.opengl as gl
from matplotlib import cm
import numpy as np
import pyqtgraph as pg
import copy
import networkx as nx
from OpenGL.GL import *


class CustomScatterItem(gl.GLGraphItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initializeGL(self):
        super(CustomScatterItem, self).initializeGL()

        # Custom shader draws dark edges around points
        self.shader = ShaderProgram('pointSprite', [
            VertexShader("""
                                void main() {
                                    gl_PointSize = gl_Normal.x / 1.2;
                                    gl_Position = ftransform();
                                    gl_FrontColor = gl_Color; 
                                } 
                            """),
            FragmentShader("""
                            #version 120
                            uniform sampler2D texture;
                            void main ( )
                            {
                            float dist = sqrt(pow(gl_PointCoord.x - 0.5, 2) + pow(gl_PointCoord.y - 0.5, 2));
                            if (dist >= 0.30 && dist < 0.50)
                                {
                                    float diff = 0.05 * (0.2 / (0.50 - dist));
                                    gl_FragColor = texture2D(texture, gl_PointCoord) * gl_Color - vec4(diff,diff,diff,0);
                                }
                            else if (dist < 0.30)
                                gl_FragColor = texture2D(texture, gl_PointCoord) * gl_Color;
                            else
                                gl_FragColor = vec4(0,0,0,0);
                            }
                    """)
        ])


class Scatter3D(SyncedCameraViewWidget):
    def __init__(self, data, labels, cmap, iscategorical, edges, parent=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.data = data - (np.max(data) - np.min(data)) / 2  # Center the data around (0,0,0)
        self.labels = labels
        self.parent = parent
        self.cmap = cmap
        self.iscategorical = iscategorical
        self.opts['distance'] = 110
        #self.setCameraPosition(distance=1.8)

        self.color = np.empty((data.shape[0], 4))
        if labels is None:
            for i in range(data.shape[0]):
                self.color[i] = self.cmap(0)
        else:
            for i in range(data.shape[0]):
                if self.iscategorical:
                    self.color[i] = self.cmap(self.labels[i])
                else:
                    self.color[i] = self.cmap(self.labels[i] / max(self.labels))
        sorted_indices = self.sorted_indices()


        # lineplot
        self.edges = np.array(edges)
        # we are sorting the ordering of the coordinates based on distance from the eye, so we have to relabel edges as well
        new_edge_idcs = {}
        for i in range(len(sorted_indices)):
            new_edge_idcs[i] = sorted_indices[i]

        edges_copy = copy.deepcopy(self.edges)
        for i in range(len(edges_copy)):
            n1 = edges_copy[i][0]
            n2 = edges_copy[i][1]
            edges_copy[i] = [new_edge_idcs[n1], new_edge_idcs[n2]]

        testG = nx.Graph()
        testG.add_edges_from(self.edges)
        newG = nx.relabel_nodes(testG, new_edge_idcs)

        size = 1000
        size_test = np.zeros(shape = (np.shape(data)[0],))
        size_test.fill(np.float32(size))

        self.graph_item = CustomScatterItem(edges =self.edges, nodePositions = data, pxMode = True, edgeColor = pg.mkColor('red'), edgeWidth=3, nodeSize =  size_test, nodeColor = self.color, size = size_test)
        #self.graph_item.setGLOptions('translucent')
        self.graph_item.setData(size = size_test, pxMode = False)
        #self.graph_item = gl.GLGraphItem(edges = self.edges, nodePositions = data, edgeColor = pg.mkColor('blue'), edgeWidth=3, nodeSize = np.shape(data)[0] * [np.float(100)], nodeColor = self.color, pxMode = True)
        self.addItem(self.graph_item)

        self.update_order()

    def sorted_indices(self):
        """
        Get the indices of the data sorted by distance from the camera position,
        used for rendering closer points on top of further points
        """
        eye = self.cameraPosition()
        return np.argsort(-np.linalg.norm(self.data - eye, axis=1))

    def update_order(self):
        """
        When drawing in translucent mode, items need to be drawn in order from back to front, for proper clipping,
        this function recomputes the point order and changes the point color based on distance from the eye.
        """

        eye = self.cameraPosition()
        distances = (-np.linalg.norm(self.data - eye, axis=1))
        distances -= np.min(distances)
        distances /= np.max(distances)
        distances -= 1.0
        sorted_indices = np.argsort(distances)
        full = np.full((distances.shape[0], 4), np.array([0.35, 0.35, 0.35, 0]))
        color_adjustment = np.multiply(full, -distances[:, None])

        # we are sorting the ordering of the coordinates based on distance from the eye, so we have to relabel edges as well
        new_edge_idcs = {}
        for i in range(len(sorted_indices)):
            new_edge_idcs[i] = sorted_indices[i]

        edges_copy = copy.deepcopy(self.edges)
        for i in range(len(edges_copy)):
            n1 = edges_copy[i][0]
            n2 = edges_copy[i][1]
            edges_copy[i] = [new_edge_idcs[n1], new_edge_idcs[n2]]

        testG = nx.Graph()
        testG.add_edges_from(self.edges)
        newG = nx.relabel_nodes(testG, new_edge_idcs)

        self.graph_item.setData(edges=self.edges, nodePositions=self.data)#, nodeColor = self.color[sorted_indices] + color_adjustment[sorted_indices], size = 7)


    def on_view_change(self):
        super().on_view_change()
        self.update_order()
        self.parent.highlight()

    def set_data(self, data, labels, cmap, iscategorical, edges):

        self.data = data - (np.max(data) - np.min(data)) / 2  # Center the data around (0,0,0)
        self.labels = labels
        self.cmap = cmap
        self.iscategorical = iscategorical
        self.color = np.empty((self.data.shape[0], 4))
        if labels is None:
            for i in range(self.data.shape[0]):
                self.color[i] = self.cmap(0)
        else:
            for i in range(self.data.shape[0]):
                if self.iscategorical:
                    self.color[i] = self.cmap(self.labels[i])
                else:
                    m = max(self.labels)
                    self.color[i] = self.cmap(self.labels[i] / m)
        sorted_indices = self.sorted_indices()

        self.edges = np.array(edges)

        # we are sorting the ordering of the coordinates based on distance from the eye, so we have to relabel edges as well
        new_edge_idcs = {}
        for i in range(len(sorted_indices)):
            new_edge_idcs[i] = sorted_indices[i]

        edges_copy = copy.deepcopy(self.edges)
        for i in range(len(edges_copy)):
            n1 = edges_copy[i][0]
            n2 = edges_copy[i][1]
            edges_copy[i] = [new_edge_idcs[n1], new_edge_idcs[n2]]

        testG = nx.Graph()
        testG.add_edges_from(self.edges)
        newG = nx.relabel_nodes(testG, new_edge_idcs)

        self.graph_item.setData(edges = self.edges, nodePositions = self.data, nodeColor = self.color)#, nodeColor = self.color)

        self.on_view_change()
        self.update_views()

    def paintGL(self):
        super(Scatter3D, self).paintGL()
        if self.labels is not None:
            ulabels = np.unique(self.labels)
            painter = QPainter(self)
            font = QFont()
            font_size = int(self.deviceHeight() / 30)
            font.setPixelSize(font_size)
            painter.setFont(font)
            painter.setPen(pg.mkPen())
            painter.setPen(pg.mkPen('k'))
            alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
            for i, label in enumerate(ulabels):
                if self.iscategorical:
                    color = self.cmap(label, bytes=True)
                else:
                    color = self.cmap(label / max(self.labels), bytes=True)
                painter.setBrush(pg.mkBrush(color))
                painter.drawEllipse(self.deviceWidth() - 3.5 * font_size, 0.5 * font_size + i * 1.7 * font_size,
                                    0.7 * font_size, 0.7 * font_size)
                if self.iscategorical:
                    text = alphabet[i]
                else:
                    text = str(label)
                painter.drawText(self.deviceWidth() - 2.0 * font_size, 1.2 * font_size + i * 1.7 * font_size, text)


