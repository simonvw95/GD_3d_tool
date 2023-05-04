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


class CustomScatterItem(gl.GLLinePlotItem):
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

        self.labels = labels
        self.parent = parent
        self.cmap = cmap
        self.iscategorical = iscategorical
        self.opts['distance'] = 110
        #self.setCameraPosition(distance=1.5)

        self.color = np.empty((len(edges), 4))
        if labels is None:
            for i in range(len(edges)):
                self.color[i] = self.cmap(0)
        else:
            for i in range(len(edges)):
                if self.iscategorical:
                    self.color[i] = self.cmap(self.labels[i])
                else:
                    self.color[i] = self.cmap(self.labels[i] / max(self.labels))

        # lineplot
        self.edges = edges

        temp_data = data - (np.max(data) - np.min(data)) / 2
        self.data = temp_data
        if edges is not None:
            all_lines = [0] * len(edges)

            for i in range(len(edges)):
                e_n = edges[i]
                n1_n = e_n[0]
                n2_n = e_n[1]
                new_line = np.array(
                    [[temp_data[n1_n][0], temp_data[n1_n][1], temp_data[n1_n][2]], [temp_data[n2_n][0], temp_data[n2_n][1], temp_data[n2_n][2]]])
                all_lines[i] = new_line

            self.lines = np.array(all_lines)

        sorted_indices_lines = self.sorted_indices()

        self.line_item = CustomScatterItem(pos = self.lines[sorted_indices_lines], mode = "lines", color = self.color[sorted_indices_lines], width = 3, antialias = True)
        #self.line_item = gl.GLLinePlotItem(pos=self.lines, mode="lines", color=(0.5, 0.0, 1.0, 1.0), width=3)
        #self.line_item = gl.GLLinePlotItem(pos=self.lines[sorted_indices_lines], mode="lines", color=(0.5, 0.0, 1.0, 1.0), width=7, antialias=True, pxMode = True)
        self.line_item.setGLOptions('translucent')

        self.addItem(self.line_item)

        self.update_order()

    def update_order(self):
        """
        When drawing in translucent mode, items need to be drawn in order from back to front, for proper clipping,
        this function recomputes the point order and changes the point color based on distance from the eye.
        """

        eye = self.cameraPosition()

        # line
        distances_lines = (-np.linalg.norm(np.sum(self.lines - eye, axis = 1), axis=1))
        distances_lines -= np.min(distances_lines)
        distances_lines /= np.max(distances_lines)
        distances_lines -= 1.0
        sorted_indices_lines = np.argsort(distances_lines)

        full = np.full((distances_lines.shape[0], 4), np.array([0.35, 0.35, 0.35, 0]))
        color_adjustment = np.multiply(full, -distances_lines[:, None])

        self.line_item.setData(pos = self.lines[sorted_indices_lines], color = self.color[sorted_indices_lines] + color_adjustment[sorted_indices_lines])
        #self.line_item.setData(pos = self.lines[sorted_indices_lines])

    def sorted_indices(self):
        """
        Get the indices of the data sorted by distance from the camera position,
        used for rendering closer points on top of further points
        """
        eye = self.cameraPosition()
        return np.argsort(-np.linalg.norm(np.sum(self.lines - eye, axis = 1), axis=1))

    def on_view_change(self):
        super().on_view_change()
        self.update_order()
        self.parent.highlight()

    def set_data(self, data, labels, cmap, iscategorical, edges):

        # lineplot
        self.edges = edges
        temp_data = data - (np.max(data) - np.min(data)) / 2  # Center the data around (0,0,0)

        if edges is not None:
            all_lines = [0] * len(edges)

            for i in range(len(edges)):
                e_n = edges[i]
                n1_n = e_n[0]
                n2_n = e_n[1]
                new_line = np.array(
                    [[temp_data[n1_n][0], temp_data[n1_n][1], temp_data[n1_n][2]],
                     [temp_data[n2_n][0], temp_data[n2_n][1], temp_data[n2_n][2]]])
                all_lines[i] = new_line

            self.lines = np.array(all_lines)

        sorted_indices_lines = self.sorted_indices()
        self.line_item.setData(pos = self.lines[sorted_indices_lines])
        #self.line_item.setData(pos=self.lines)

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


