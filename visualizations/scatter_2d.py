from PyQt5.QtCore import QRect
from PyQt5.QtGui import QPainter, QFont, QColor
from PyQt5.QtWidgets import QLabel
from matplotlib import cm
import pyqtgraph as pg
import numpy as np
import pyqtgraph.opengl as gl
from pyqtgraph import LabelItem

import constants


class Scatter2D(pg.PlotWidget):
    def __init__(self, data, labels, cmap, iscategorical, edges, title="2D Projection", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hideAxis('left')
        self.hideAxis('bottom')
        self.setAspectLocked()
        self.setMouseEnabled(x=False, y=False)
        padding = 0.3
        self.cmap = cmap
        self.iscategorical = iscategorical

        #self.scatter_item = pg.ScatterPlotItem(size=5, pen=pg.mkPen(0,0,0,50), hoverable=True)

        # if labels is None:
        #     colors = pg.mkBrush(self.cmap(0, bytes=True))
        # else:
        #     if self.iscategorical:
        #         colors = [pg.mkBrush(self.cmap(labels[i], bytes=True)) for i in range(data.shape[0])]
        #     else:
        #         colors = [pg.mkBrush(self.cmap(labels[i] / max(labels), bytes=True)) for i in range(data.shape[0])]

        #self.scatter_item.addPoints(pos=data, brush=colors)
        #self.addItem(self.scatter_item)

        self.getViewBox().setLimits(xMin=-0.5, xMax=1.5, yMin=np.min(data[:, 1]) - padding, yMax=np.max(data[:, 1]) + padding)
        self.title = title
        self.label = QLabel(self.title, self.viewport())

        self.edges = edges
        self.line_item = pg.GraphItem(size=5, pen=pg.mkPen(0,0,0,50), hoverable=True)
        self.line_item.setData(pos=data, adj=np.array(edges))
        self.addItem(self.line_item)

    def set_data(self, data, labels, cmap, iscategorical, edges):

        # self.cmap = cmap
        # self.iscategorical = iscategorical
        # if labels is None:
        #     colors = pg.mkBrush(self.cmap(0, bytes=True))
        # else:
        #     if self.iscategorical:
        #         colors = [pg.mkBrush(self.cmap(labels[i], bytes=True)) for i in range(data.shape[0])]
        #     else:
        #         m = max(labels)
        #         colors = [pg.mkBrush(self.cmap(labels[i] / m, bytes=True)) for i in range(data.shape[0])]

        # self.scatter_item.setData(pos=data, brush=colors, size=5, pen=pg.mkPen(0,0,0,50))

        self.setMouseEnabled(x=False, y=False)
        self.edges = edges
        self.line_item.setData(pos=data, adj=np.array(edges))

    def paintEvent(self, ev):
        super().paintEvent(ev)
        font_size = int(self.sceneRect().height() / 25)
        font = QFont()
        font.setPixelSize(font_size)
        self.label.setFont(font)
        self.label.move(10, 0)
        self.label.setText(self.title)
        self.label.adjustSize()


