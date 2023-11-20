from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QLabel
import pyqtgraph as pg
import numpy as np
import constants
import copy

class Scatter2D(pg.PlotWidget):
    def __init__(self, data, avg_metric_vals, cmap, parent, title="2D Projection of metrics", *args, **kwargs):
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
        self.avg_metric_vals = avg_metric_vals
        # assign the scaled average metric values (1001 values between 0 and 1) to a color of cmap
        color_range = np.array(range(0, 256, 1)) / 256

        temp_colors = [0] * len(avg_metric_vals)
        for i in range(len(temp_colors)):
            # find out which idx to assign the current viewpoint to
            idx = np.argmin(abs(color_range - avg_metric_vals[i]))
            temp_colors[i] = pg.mkBrush(self.cmap(idx, bytes = True))

        self.scatter_item = pg.ScatterPlotItem(size=3, pen=pg.mkPen(0,0,0,50), hoverable=True, pxMode = True, hoverSize = 8, hoverSymbol = 'crosshair', hoverBrush = 'yellow')

        self.symbol_list = ['o'] * constants.samples + ['crosshair']
        self.size_list = [3] * constants.samples + [8]
        self.color_list = temp_colors
        self.color_list[-1] = pg.mkBrush('black')

        self.getViewBox().setLimits(xMin=-0.5, xMax=1.5, yMin=np.min(data[:, 1]) - padding, yMax=np.max(data[:, 1]) + padding)

        # self.scatter_item.addPoints(pos=data, brush=self.color_list)
        self.scatter_item.sigHovered.connect(self.onPointsHovered)
        # self.scatter_item.setSymbol(self.symbol_list)
        # self.scatter_item.setSize(self.size_list)

        self.scatter_item.setData(pos = data, symbol = self.symbol_list, size = self.size_list, data = list(avg_metric_vals))

        self.addItem(self.scatter_item)
        self.title = title
        self.label = QLabel(self.title, self.viewport())

    def set_data(self, data,  avg_metric_vals, cmap, nearest_viewpoint_idx):

        # assign the scaled average metric values (1001 values between 0 and 1) to a color of cmap
        color_range = np.array(range(0, 256, 1)) / 256

        temp_colors = [0] * len(avg_metric_vals)
        for i in range(len(temp_colors)):
            # find out which idx to assign the current viewpoint to
            idx = np.argmin(abs(color_range - avg_metric_vals[i]))
            temp_colors[i] = pg.mkBrush(cmap(idx, bytes=True))

        temp_colors[-1] = 'black'
        temp_colors[nearest_viewpoint_idx] = 'yellow'
        symbol_list = ['o'] * constants.samples + ['crosshair']
        size_list = [3] * constants.samples + [8]
        symbol_list[nearest_viewpoint_idx] = 'crosshair'
        size_list[nearest_viewpoint_idx] = 8

        self.scatter_item.setData(pos=data, brush= temp_colors, pen=pg.mkPen(0, 0, 0, 50), data = avg_metric_vals)
        self.scatter_item.setSymbol(symbol_list)
        self.scatter_item.setSize(size_list)
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
                # colors = self.colors
                # colors[idx] = 'yellow'
                # symbols = ['o'] * constants.samples + ['crosshair']
                # symbols[idx] = 'crosshair'
                # sizes = [5] * constants.samples + [8]
                # sizes[idx] = 8
                #
                # self.scatter_item.setSymbol(symbols)
                # self.scatter_item.setSize(sizes)
                # self.scatter_item.setBrush(colors)


