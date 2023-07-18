from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel
import pyqtgraph as pg
import numpy as np


class Graph2D(pg.PlotWidget):
    def __init__(self, data, labels, cmap, iscategorical, edges, title="2D Projection", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hideAxis('left')
        self.hideAxis('bottom')
        #self.setAspectLocked()
        self.setAspectLocked(lock = True, ratio = None)
        self.setMouseEnabled(x=False, y=False)
        padding = 0.3
        self.cmap = cmap
        self.iscategorical = iscategorical

        self.colors = pg.mkBrush(self.cmap(0, bytes=True))
        #self.getViewBox().setLimits(xMin=-0.5, xMax=1.5, yMin=np.min(data[:, 1]) - padding, yMax=np.max(data[:, 1]) + padding)
        self.title = title
        self.label = QLabel(self.title, self.viewport())

        self.edges = edges

        self.line_item = pg.GraphItem(pen=pg.mkPen('black', width=3), hoverable=False, pxMode = True, size = 8, brush = self.colors)
        self.line_item.setData(pos=data, adj=np.array(edges))
        self.addItem(self.line_item)

    def set_data(self, data, labels, cmap, iscategorical, edges):

        self.setMouseEnabled(x=False, y=False)
        self.edges = edges

        # unq, count = np.unique(data.round(decimals = 6), axis=0, return_counts=True)
        # repeated_groups = unq[count > 1]
        # print(repeated_groups)
        #
        # for rp in repeated_groups:
        #     idx = np.argwhere(np.all(np.isclose(data, rp, atol = 0.000001), axis = 1))[0][0]
        #     data[idx, :] = data[idx, :] + 0.000001

        self.line_item.setData(pos=data, adj=np.array(self.edges))

    def paintEvent(self, ev):
        super().paintEvent(ev)
        font_size = int(self.sceneRect().height() / 25)
        font = QFont()
        font.setPixelSize(font_size)
        self.label.setFont(font)
        self.label.move(10, 0)
        self.label.setText(self.title)
        self.label.adjustSize()


