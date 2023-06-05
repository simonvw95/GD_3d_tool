from OpenGL.raw.GL.VERSION.GL_1_0 import glClearColor
from PyQt5 import QtCore
from PyQt5.QtGui import QPainter
import pyqtgraph as pg
from pyqtgraph.opengl.shaders import ShaderProgram, VertexShader, FragmentShader

from visualizations.synced_camera_view_widget import SyncedCameraViewWidget
import constants
import numpy as np
import pyqtgraph.opengl as gl

class CustomMeshItem(gl.GLMeshItem):
    def __init__(self, texture_map, *args, **kwargs):
        super(CustomMeshItem, self).__init__(*args, **kwargs)

        #Initalize shader that used 1D texture to map metric data to colors, and adds simple shading
        self.custom_shader = ShaderProgram('custom_shader', [
            VertexShader("""
                varying vec3 normal;
                void main() {
                    // compute here for use in fragment shader
                    normal = normalize(gl_NormalMatrix * gl_Normal);
                    gl_FrontColor = gl_Color;
                    gl_BackColor = gl_Color;
                    gl_Position = ftransform();
                }
            """),
            FragmentShader("""
                varying vec3 normal;
                uniform float colorMap[20];
                void main() {
                    float p = dot(normal, normalize(vec3(0.01, 0.01, -0.50)));
                    p = p < 0. ? 0. : p;
                    float m_value = gl_Color.x;
                    int i = 0;
                    float thresholds[5] = float[5](0., 0.2, 0.5, 0.8, 1.0);
                    for(int j = 1;j<5;j++){
                        if (m_value > thresholds[j])
                            i++;
                    }
                    float ratio = (m_value - thresholds[i]) / (thresholds[i + 1] - thresholds[i]);
                    i = i * 4;
                    vec3 color1 = vec3(colorMap[0 + i], colorMap[1 + i], colorMap[2 + i]) * (1.0 - ratio);
                    vec3 color2 = vec3(colorMap[4 + i], colorMap[5 + i], colorMap[6 + i]) * ratio;
                    vec4 color = vec4(color1 + color2, 1.0);
                    
                    //shading
                    color.x = color.x * p;
                    color.y = color.y * p;
                    color.z = color.z * p;
                    gl_FragColor = color;
                }
            """)
        ], uniforms={'colorMap': texture_map}),

class QualitySphere(SyncedCameraViewWidget):
    def __init__(self, data, cmap, parent=None, *args, **kwargs, ):
        super().__init__(*args, **kwargs)

        self.data = data
        self.parent = parent
        self.setCameraPosition(distance=250)
        vertices = np.load(f'spheres/sphere{constants.samples}_points.npy')
        faces = np.load(f'spheres/sphere{constants.samples}_faces.npy')
        self.cmap = cmap

        #Store the metric data into the vertex colors
        vertex_colors = [(x, x, x, 1) for x in self.data]

        #Generate 1D texture to map metric data to actual colors:
        texture = self.cmap.mapToFloat([0, 0.2, 0.5, 0.8, 1.0])
        texture_1d = [color_value for color in texture for color_value in color]

        self.md = gl.MeshData(vertexes=vertices, faces=faces, vertexColors=vertex_colors)
        self.mesh_item = CustomMeshItem(texture_1d, meshdata=self.md, smooth=True, shader='custom_shader', glOptions='translucent')
        self.addItem(self.mesh_item)
        if constants.show_user_picked_viewpoints:
            data = parent.analysis_data.where((parent.analysis_data['projection_method'] == parent.default_layout_technique) &
                                              (parent.analysis_data['dataset'] == parent.dataset_name))
            viewpoints_with_tool =  data.loc[data['mode'] == 'eval_full']['viewpoint'].to_numpy()
            viewpoints_with_tool = np.array([p for p in viewpoints_with_tool])
            viewpoints_with_tool /= np.reshape(np.sqrt(np.sum(np.square(viewpoints_with_tool), axis=1)), (viewpoints_with_tool.shape[0], 1)) * 0.96
            viewpoints_without_tool = data.loc[data['mode'] == 'eval_half']['viewpoint'].to_numpy()
            viewpoints_without_tool = np.array([p for p in viewpoints_without_tool])
            viewpoints_without_tool /= np.reshape(np.sqrt(np.sum(np.square(viewpoints_without_tool), axis=1)), (viewpoints_without_tool.shape[0], 1)) * 0.96
            scatter_item = gl.GLScatterPlotItem()
            pos = np.concatenate([viewpoints_with_tool, viewpoints_without_tool])
            color = np.array([[1.0, 0, 0, 1.0] for p in viewpoints_with_tool] + [[0, 0, 1.0, 1.0] for p in viewpoints_without_tool])
            scatter_item.setData(pos=pos, color=color)
            scatter_item.setGLOptions('translucent')
            self.addItem(scatter_item)

    def paintGL(self):
        if constants.user_mode == 'eval_half':
            self.setWindowOpacity(0)
        super(QualitySphere, self).paintGL()

        #Draw crosshair
        painter = QPainter(self)
        painter.setPen(pg.mkPen('k'))
        painter.setBrush(pg.mkBrush('k'))
        painter.drawLine(self.deviceWidth() / 3 -3, self.deviceHeight() / 3, self.deviceWidth() / 3 +3, self.deviceHeight() / 3)
        painter.drawLine(self.deviceWidth() / 3 +0.5, self.deviceHeight() / 3 -3, self.deviceWidth() / 3 +0.5, self.deviceHeight() / 3 + 3)

    def save_image(self):
        QtCore.QTimer.singleShot(1000, lambda: self.readQImage().save("fileName.png"))
