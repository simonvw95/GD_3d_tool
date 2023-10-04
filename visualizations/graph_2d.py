from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel
import pyqtgraph as pg
import numpy as np


class Graph2D(pg.PlotWidget):
    def __init__(self, data, cmap, edges, title="2D Graph Drawing", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hideAxis('left')
        self.hideAxis('bottom')
        #self.setAspectLocked()
        self.setAspectLocked(lock = True, ratio = None)
        self.setMouseEnabled(x=False, y=False)
        padding = 0.3
        self.cmap = cmap

        self.colors = pg.mkBrush(self.cmap(0, bytes=True))
        #self.getViewBox().setLimits(xMin=-0.5, xMax=1.5, yMin=np.min(data[:, 1]) - padding, yMax=np.max(data[:, 1]) + padding)
        self.title = title
        self.label = QLabel(self.title, self.viewport())

        self.edges = edges

        self.line_item = pg.GraphItem(pen=pg.mkPen('black', width=3), hoverable=False, pxMode = True, size = 8, brush = self.colors)
        self.line_item.setData(pos=data, adj=np.array(edges))
        self.addItem(self.line_item)

    def set_data(self, data, edges):

        self.setMouseEnabled(x=False, y=False)
        self.edges = edges

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


