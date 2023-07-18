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
from OpenGL.GL import *


class CustomScatterItem(gl.GLScatterPlotItem):
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

class CustomGraphItem(gl.GLGraphItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initializeGL(self):
        super(CustomGraphItem, self).initializeGL()

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


class Graph3D(SyncedCameraViewWidget):
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

        self.edges = np.array(edges)

        self.graph_item = CustomGraphItem(edges =self.edges, nodePositions = data, pxMode = True, edgeColor = pg.mkColor('black'), edgeWidth=3)
        self.addItem(self.graph_item)

        self.scatter_item = CustomScatterItem(pos=data[sorted_indices], size=15, color=self.color[sorted_indices],
                                              pxMode=True)
        self.scatter_item.setGLOptions('translucent')
        self.addItem(self.scatter_item)

        self.update_order()

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

        # temp testing
        distances *= 2

        sorted_indices = np.argsort(distances)
        full = np.full((distances.shape[0], 4), np.array([0.35, 0.35, 0.35, 0]))
        color_adjustment = np.multiply(full, -distances[:, None])

        self.graph_item.setData(edges=self.edges,
                                nodePositions=self.data)  # , nodeColor = self.color[sorted_indices] + color_adjustment[sorted_indices], size = 7)

        self.scatter_item.setData(pos=self.data[sorted_indices],
                                  color=self.color[sorted_indices] + color_adjustment[sorted_indices])



    def sorted_indices(self):
        """
        Get the indices of the data sorted by distance from the camera position,
        used for rendering closer points on top of further points
        """
        eye = self.cameraPosition()
        return np.argsort(-np.linalg.norm(self.data - eye, axis=1))

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

        self.edges = np.array(edges)
        self.graph_item.setData(edges=self.edges, nodePositions=self.data,
                                nodeColor=self.color)  # , nodeColor = self.color)

        sorted_indices = self.sorted_indices()
        self.scatter_item.setData(pos=self.data[sorted_indices], color=self.color[sorted_indices])

        self.on_view_change()
        self.update_views()

    def paintGL(self):
        super(Graph3D, self).paintGL()
        if self.labels is not None:
            ulabels = np.unique(self.labels)
            painter = QPainter(self)
            font = QFont()
            font_size = self.deviceHeight() / 30
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


