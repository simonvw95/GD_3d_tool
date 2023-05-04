import pyqtgraph.opengl as gl
from PyQt5 import Qt
from PyQt5.QtCore import QRectF
from PyQt5.QtGui import QColor, QPainter, QFont


class SyncedCameraViewWidget(gl.GLViewWidget):
    """
    Wrapper class too easily make multiple 3D views of which the viewport is synced
    """

    def __init__(self, parent=None, devicePixelRatio=None, rotationMethod='euler', title='3D view', *args, **kwds):
        self.linked_views: List[gl.GLViewWidget] = []
        super().__init__(parent, devicePixelRatio, rotationMethod, *args, **kwds)
        self.title = title
        self.lock = False
        self.opts['distance'] = 100
        self.opts['fov'] = 1
        self.opts['elevation'] = 0

    def wheelEvent(self, ev):
        """Update view on zoom event"""
        return

    def pan(self, dx, dy, dz, relative='global'):
        #disable panning
        return

    def mouseMoveEvent(self, ev):
        """Update view on move event"""
        if self.lock:
            return
        super().mouseMoveEvent(ev)
        self.update_views()

    def mouseReleaseEvent(self, ev):
        """Update view on move event"""
        if self.lock:
            return
        super().mouseReleaseEvent(ev)
        self.update_views()

    def update_views(self):
        """Take camera parameters and sync with all views"""
        camera_params = self.cameraParams()
        # Remove rotation, we can't update all params at once (Azimuth and Elevation)
        camera_params["rotation"] = None
        camera_params["distance"] = None
        self.on_view_change()
        for view in self.linked_views:
            view.setCameraParams(**camera_params)
            view.on_view_change()

    def on_view_change(self):
        #Overloadable function to define update behavior once the view is changed
        pass

    def sync_camera_with(self, view: gl.GLViewWidget):
        """Add view to sync camera with"""
        self.linked_views.append(view)

    def paintGL(self, *args, **kwds):
        gl.GLViewWidget.paintGL(self, *args, **kwds)
        painter = QPainter(self)
        font = QFont()
        font_size = int(self.deviceHeight() / 25)
        font.setPixelSize(font_size)
        painter.setFont(font)
        painter.drawText(10, font_size, self.title)
