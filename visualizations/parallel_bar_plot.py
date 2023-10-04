from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg
from PyQt5.QtCore import QLine, QLineF, QRectF, QPointF
from PyQt5.QtWidgets import QLabel
from matplotlib import cm
from pyqtgraph import mkPen, mkBrush
from PyQt5.QtGui import QColor, QFont
import numpy as np
from pyqtgraph.Qt import QtGui

import constants
import math

class InteractiveRect(QtWidgets.QGraphicsRectItem):
    def __init__(self, parent, metric_index, bin_index, point_count, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setToolTip(F'nr_points = {point_count}')

        # required in order to receive hoverEnter/Move/Leave events
        self.setAcceptHoverEvents(True)
        self.parent = parent
        self.metric_index = metric_index
        self.bin_index = bin_index
        self.point_count = point_count

    def hoverMoveEvent(self, event):
        pos = event.pos()
        rect = self.boundingRect()
        percentage = (pos.y() - rect.y()) / (rect.height())
        self.parent.connect_bars((self.metric_index, self.bin_index), min(1, percentage))
        if constants.hover_to_view:
            self.parent.on_rect_click(self, min(1, percentage))

    def mousePressEvent(self, event):
        if not constants.hover_to_view:
            pos = event.pos()
            rect = self.boundingRect()
            percentage = (pos.y() - rect.y()) / (rect.height())
            self.parent.connect_bars((self.metric_index, self.bin_index), min(1, percentage))
            self.parent.on_rect_click(self, min(1, percentage))

    def hoverLeaveEvent(self, event: 'QGraphicsSceneHoverEvent') -> None:
        self.parent.clear_polylines()

    def boundingRect(self) -> QtCore.QRectF:
        return self.rect() #Set the bounding rect to exactly the rectangle, so that hover works properly

    def highlight(self):
        color = self.brush().color()
        color.setAlpha(255)
        self.setBrush(pg.mkBrush(color))

    def un_highlight(self):
        color = self.brush().color()
        color.setAlpha(int(255 * 0.7))
        self.setBrush(pg.mkBrush(color))

class ClickableText(pg.TextItem):
    def __init__(self, parent, text="metric", *args, **kwargs):
        super(ClickableText, self).__init__(text=text, *args, **kwargs)
        self.setAcceptHoverEvents(True)
        self.parent = parent
        self.metric = text

    def hoverEnterEvent(self, event):
        c = self.color
        c.setAlpha(255)
        self.setColor(c)

    def hoverLeaveEvent(self, event):
        c = self.color
        c.setAlpha(0.7 * 255)
        self.setColor(c)

    def mousePressEvent(self, event):
        self.parent.on_metric_click(self.metric)

class boxPlot(pg.GraphicsObject):
    def __init__(self, data: np.ndarray, height, v_pos, text):
        """
            :param data: tuple containing boxplot data in order: [mean, 1th quantile, 3th quantile, min, max]
        """
        super(boxPlot, self).__init__()
        self.data = data
        self.height = height
        self.v_pos = v_pos
        self.text = text
        self.generatePicture()

    def generatePicture(self):
        self.picture = QtGui.QPicture()
        p = QtGui.QPainter(self.picture)

        # Mean line
        p.setPen(pg.mkPen('r', width=2))
        p.drawLine(QLineF(self.data[0], self.v_pos - self.height / 2, self.data[0], self.v_pos + self.height / 2))

        # quartile square
        p.setPen(pg.mkPen('b', width=2))
        p.drawRect(QRectF(self.data[1], self.v_pos - self.height / 2, self.data[2] - self.data[1], self.height))

        #The rest
        p.setPen(pg.mkPen('k', width=2))
        p.drawLine(QLineF(self.data[3], self.v_pos, self.data[1], self.v_pos))
        p.drawLine(QLineF(self.data[2], self.v_pos, self.data[4], self.v_pos))
        p.drawLine(QLineF(self.data[3], self.v_pos - self.height / 4, self.data[3], self.v_pos + self.height / 4))
        p.drawLine(QLineF(self.data[4], self.v_pos - self.height / 4, self.data[4], self.v_pos + self.height / 4))

        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QtCore.QRectF(self.picture.boundingRect())


class parallelBarPlot(pg.PlotWidget):
    h_gap = 1.4 #Distance between histogram plots in local coordinates
    nr_bins = 40
    offsetY = 0

    def __init__(self, views_metrics, metrics2d, metrics3d, view_points, parent=None, title="Viewpoint quality histogram", *args, **kwargs):
        super().__init__(*args, **kwargs)
        if constants.user_mode == 'evalimage':
            self.h_gap *= 2
            self.offsetY = 1
        self.views_metrics = views_metrics
        self.metrics2d = metrics2d
        self.metrics3d = metrics3d
        self.view_points = view_points
        self.parent = parent
        self.title = title
        self.label = QLabel(self.title, self.viewport())
        self.cmap = cm.get_cmap('tab10')
        self.max_points_in_bin = 0 #Defined later, and used to determine where to draw the polylines on hover
        self.lock = False
        if constants.user_mode == 'eval_half':
            self.setVisible(False)

        #References to the poyline objects
        self.lines = None
        self.highlight_line = None

        self.line_pen = mkPen(QColor(0, 0, 0, 30), width=1)
        self.hist_heights = []
        self.histograms = []
        self.highlighted_rects = []

        self.x_range = (0, 1)
        if constants.scale_to_signal_range:
            self.scale_to_signal_range()
        for metric_index in range(self.views_metrics.shape[1]):
            hist = np.histogram(self.views_metrics[:, metric_index], self.nr_bins, range=self.x_range)
            self.max_points_in_bin = max(self.max_points_in_bin, np.max(hist[0]))
            self.histograms.append(hist)
        self.metric_texts = [] #Store references to the text items to change the metric values
        self.initialize()
        self.draw_histograms()

    def initialize(self):
        self.hideAxis('left')
        self.hideAxis('bottom')
        self.getViewBox().setXRange(self.x_range[0], self.x_range[1])
        self.getViewBox().setYRange(-0.2, self.h_gap * len(self.histograms))
        self.getViewBox().setMouseEnabled(x=False, y=False)

        #Write metric names in the top left corner of each histogram
        if constants.user_mode != "image" and constants.user_mode != 'evalimage':
            for metric in range(len(self.histograms)):
                text = constants.metrics[metric]
                item = ClickableText(self, text=text, anchor=(0, 0.5), color=pg.mkColor(self.cmap(metric, bytes=True, alpha=0.7)))
                self.metric_texts.append(item)
                x = self.x_range[0]
                y = self.h_gap * metric + 1.1
                item.setPos(x, y)
                self.addItem(item)

    def scale_to_signal_range(self):
        # commented this out because currently testing if it's not better to do scaling immediately after computing the metrics

        # mins = np.min(self.views_metrics, axis=0).reshape(1, 4)
        # maxs = np.max(self.views_metrics, axis=0).reshape(1, 4)
        # scale = 1 / (maxs - mins)
        # self.normdata = mins, scale
        # self.views_metrics = self.views_metrics - mins # mins.reshape((4,1))
        # self.views_metrics = self.views_metrics * scale
        self.views_metrics = self.views_metrics.astype(float)
        #print()

    def draw_histograms(self):
        self.rect_reference = {}
        self.rect_edge_pen = pg.mkPen('w', width=2)
        for metric in constants.metrics:
            self.rect_reference[metric] = []

        for h_index, histogram in enumerate(self.histograms):
            widths = histogram[1][1:] - histogram[1][:-1]

            color = pg.mkColor(self.cmap(h_index, bytes=True, alpha=0.7))

            local_heights = histogram[0] / np.max(histogram[0])
            self.hist_heights.append(np.max(local_heights))
            for i in range(len(local_heights)):
                if local_heights[i] > 0:
                    epsilon = 0.035
                else:
                    epsilon = 0
                rect = InteractiveRect(self, h_index, i, histogram[0][i], histogram[1][i] , self.h_gap * h_index - 0.01 + self.offsetY, widths[i], local_heights[i] + epsilon) #(x, y, width, height)
                rect.setBrush(mkBrush(color))
                rect.setPen(self.rect_edge_pen)
                self.addItem(rect)
                self.rect_reference[constants.metrics[h_index]].append(rect)

            if constants.user_mode == 'free':
                #Add markers fot the quality of the 2D and 3D projection
                line2D = QtWidgets.QGraphicsLineItem(self.metrics2d[h_index], self.h_gap * h_index -0.02 + self.offsetY, self.metrics2d[h_index],
                                                     self.h_gap * h_index - 0.10)
                line2D.setPen(pg.mkPen(color, width=8))
                self.addItem(line2D)

                # uncomment for later use? depicts a marker for the 3d projection?
                # line3D = QtWidgets.QGraphicsLineItem(self.metrics3d[h_index], self.h_gap * h_index -0.02 + self.offsetY, self.metrics3d[h_index],
                #                                      self.h_gap * h_index - 0.2)
                # line3D.setPen(pg.mkPen(color, width=4))
                # self.addItem(line3D)

            self.draw_axis(h_index, (self.x_range[0], h_index * self.h_gap + self.offsetY), (self.x_range[1], h_index * self.h_gap + self.offsetY), 11)

    def draw_box_plots(self):

        box_plot_data = self.parent.get_boxplot_data()
        if constants.scale_to_signal_range:
            mins, scale = self.normdata
            box_plot_data = ((box_plot_data - mins) * scale).astype(float)

        for i, metric in enumerate(constants.metrics):
            bplot_overall = boxPlot(box_plot_data[0][:, i], 0.3, self.h_gap * i - 0.3 + self.offsetY, 'overall:')
            bplot_users_half = boxPlot(box_plot_data[1][:, i], 0.3, self.h_gap * i - 0.7 + self.offsetY, 'no tools:')
            bplot_users_full = boxPlot(box_plot_data[2][:, i], 0.3, self.h_gap * i - 1.1 + self.offsetY, 'with tools:')
            self.addItem(bplot_overall)
            self.addItem(bplot_users_half)
            self.addItem(bplot_users_full)

    def highlight_bar_with_values(self, metric_values, original_values):
        if constants.user_mode in ['image', 'evalimage']:
            return
        for rect in self.highlighted_rects:
            rect.un_highlight()
        for h_index, value in enumerate(metric_values):
            bin_index = np.argmax(self.histograms[h_index][1]>metric_values[h_index]) - 1
            rect = self.rect_reference[constants.metrics[h_index]][bin_index]
            rect.highlight()
            self.highlighted_rects.append(rect)
        for i, text in enumerate(self.metric_texts):

            text.setText(
                F"{constants.metrics[i]}: {'viewpoint value: ' + str(round(original_values[i], 5))}")

    def on_rect_click(self, rect, percentage):
        if self.lock or constants.user_mode == 'evalimage':
            return
        self.parent.move_to_view(rect.metric_index, rect.bin_index, rect.boundingRect().left(), rect.boundingRect().right(), percentage)

    def on_metric_click(self, metric):
        self.parent.metric_selected(metric)

    def draw_axis(self, h_index, p_bottom_left, p_bottom_right,ticks):
        """"
        Draw a custom horizontal axis between two local coordinate points, with a number of ticks
        with values calcualted from the specified min and max values.
        """
        pen = mkPen(0.3, width=1)
        h_line = QtWidgets.QGraphicsLineItem(p_bottom_left[0],p_bottom_left[1],p_bottom_right[0],p_bottom_right[1])
        h_line.setPen(pen)
        tick_size = (self.x_range[1] - self.x_range[0]) / (ticks - 1)
        for i in range(ticks):
            x = p_bottom_left[0] + i * tick_size
            y = p_bottom_left[1]
            if h_index == 0 and constants.user_mode != 'evalimage':
                text = str(round(self.x_range[0] + i * tick_size, 2))
                item = pg.TextItem(text=text, anchor=(0.5, 0), color=0.6)
                item.setPos(x, y - 0.14 - self.offsetY)
                self.addItem(item)
            vline = QtWidgets.QGraphicsLineItem(x,y,x,y - 0.03)
            vline.setPen(mkPen(0.5, width=1))
            self.addItem(vline)
        self.addItem(h_line)

    def connect_bars(self, indices, percentage):
        """"
        For each point in the histogram bin that is specified by 'indices', draws a line to the bins of other metrics
        containing measured values for that same data point
        ----------
        indices : Tuple(number, number)
            The index of the currently hovered histogram/metric and the index of the specific bin that is hovered
        """
        if self.lock:
            return
        #unwrap indices
        h_index, b_index = indices

        #Find the data points that go through the hovered bin
        bin_bounds = self.histograms[h_index][1][b_index:b_index + 2]
        i_hovered_points = np.argwhere(np.logical_and(self.views_metrics[:, h_index] >= bin_bounds[0], self.views_metrics[:, h_index] <= bin_bounds[1]))
        points = self.views_metrics[i_hovered_points, :]
        points = points[np.argsort(points[:, 0, h_index])]
        lines_x = []
        lines_y = []

        #Find the index of the point that will be drawn with the thick highlighted line
        highlighted_p_index = round(percentage * ((points.shape[0]) - 1))

        #Compute x and y coordinates for all the points of all polylines:
        highlight_line_x = []
        highlight_line_y = []
        for p_index, point in enumerate(points):
            for i in range(len(point[0]) - 1):
                a = self.metric_to_point(point[0][i], i, h_index, p_index)
                b = self.metric_to_point(point[0][i + 1], i + 1, h_index, p_index)
                if p_index == highlighted_p_index:
                    highlight_line_x.extend([a[0], b[0]])
                    highlight_line_y.extend([a[1], b[1]])
                else:
                    lines_x.extend([a[0], b[0]])
                    lines_y.extend([a[1], b[1]])

        #Remove any previously drawn lines, and draw the new lines:
        self.clear_polylines()
        self.lines = self.plot(lines_x, lines_y, connect='pairs', pen=self.line_pen)
        self.highlight_line = self.plot(highlight_line_x, highlight_line_y, connect='pairs', pen=pg.mkPen('k', width=2))


    def metric_to_point(self, metric_value, metric_index, hovered_metric_index, p_index):
        """"
        Compute the local coordinate point for a certain metric and histogram index
        Parameters
        """
        bin_step = (self.x_range[1] - self.x_range[0]) / self.nr_bins
        proportion = (metric_value - self.x_range[0]) / (self.x_range[1] - self.x_range[0])
        #If proportion is exactly 1 (highest value in histogram) we want to take 1 minus the bincount,
        # which is the index of the last bin
        bin_index = min(self.nr_bins - 1, math.floor(proportion * self.nr_bins))
        center_bin = self.x_range[0] + bin_index * bin_step + 0.5 * bin_step
        if hovered_metric_index == metric_index:
            # For the bin that we are hovering on, we want to draw lines that start in a range from the bottom to the top of the bin
            y_step = (1 / self.hist_heights[metric_index] / np.max(self.histograms[metric_index][0]))
            epsilon = y_step * p_index
            return (center_bin, metric_index * self.h_gap + epsilon)
        else:
            #All other lines go to the center of the corresponding other bins
            y_step = (1 / self.hist_heights[metric_index] / np.max(self.histograms[metric_index][0]))
            epsilon = 0 #y_step * self.histograms[metric_index][0][bin_index] / 2
            return (metric_value, metric_index * self.h_gap + epsilon)

    def clear_polylines(self):
        self.removeItem(self.lines)
        self.removeItem(self.highlight_line)

    def paintEvent(self, ev):
        super().paintEvent(ev)
        font_size = int(self.sceneRect().height() / 25)
        font = QFont()
        font.setPixelSize(font_size)
        self.label.setFont(font)
        self.label.move(10, 0)
        self.label.setText(self.title)
        self.label.adjustSize()