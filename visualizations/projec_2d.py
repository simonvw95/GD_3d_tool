from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel
import pyqtgraph as pg
import numpy as np
import constants


class Scatter2D(pg.PlotWidget):
    def __init__(self, data, cmap, parent, title="2D Projection of metrics", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.idx_sel_view = None
        self.hideAxis('left')
        self.hideAxis('bottom')
        self.setAspectLocked()
        self.setMouseEnabled(x=False, y=False)
        padding = 0.3
        self.cmap = cmap
        self.parent = parent
        self.data = data
        colors = pg.mkBrush(self.cmap(0, bytes=True))
        self.scatter_item = pg.ScatterPlotItem(size=5, pen=pg.mkPen(0,0,0,50), hoverable=True, pxMode = True)
        self.symbol_list = ['o'] * constants.samples + ['crosshair']
        self.size_list = [5] * constants.samples + [8]
        self.color_list = [colors] * constants.samples + [pg.mkBrush('black')]
        self.getViewBox().setLimits(xMin=-0.5, xMax=1.5, yMin=np.min(data[:, 1]) - padding, yMax=np.max(data[:, 1]) + padding)

        self.scatter_item.addPoints(pos=data, brush=self.color_list)
        self.scatter_item.sigHovered.connect(self.onPointsHovered)
        self.scatter_item.setSymbol(self.symbol_list)
        self.scatter_item.setSize(self.size_list)

        self.addItem(self.scatter_item)
        self.title = title
        self.label = QLabel(self.title, self.viewport())

    def set_data(self, data, cmap, spec_idx = None):
        self.cmap = cmap
        colors = pg.mkBrush(self.cmap(0, bytes=True))

        if spec_idx is not None:
            self.symbol_list[spec_idx] = 'crosshair'
            self.size_list[spec_idx] = 8
            self.color_list[spec_idx] = pg.mkBrush('yellow')
        else:
            self.symbol_list = ['o'] * constants.samples + ['crosshair']
            self.size_list = [5] * constants.samples + [8]
            self.color_list = [colors] * constants.samples + [pg.mkBrush('black')]

        self.scatter_item.setData(pos=data, brush= self.color_list, pen=pg.mkPen(0, 0, 0, 50))
        self.scatter_item.setSymbol(self.symbol_list)
        self.scatter_item.setSize(self.size_list)
        self.setMouseEnabled(x=False, y=False)

    def paintEvent(self, ev):
        super().paintEvent(ev)
        font_size = self.sceneRect().height() / 25
        font = QFont()
        font.setPixelSize(font_size)
        self.label.setFont(font)
        self.label.move(10, 0)
        self.label.setText(self.title)
        self.label.adjustSize()

    def onPointsHovered(self, obj, points):

        # take the first point that appears
        if len(points) > 0:
            pt = points[0]
            mask = self.scatter_item._maskAt(pt.pos())
            idx = np.where(mask)[0][0]

            if idx < constants.samples:
                self.parent.move_to_viewpoint(self.parent.view_points[idx])
        #         self.set_data(self.data, self.cmap, spec_idx = idx)
        # else:
        #     self.set_data(self.data, self.cmap)


