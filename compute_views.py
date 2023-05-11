import os
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from scipy.spatial import ConvexHull
import trimesh

import constants
import utils

def compute_views(proj_file, viewpoints, labels=None, display=False, manual_rotate=False):
    cmap = plt.get_cmap('tab10')

    df = pd.read_csv(proj_file, sep=';', header=0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if labels is not None:
        ax.scatter(df['x'], df['y'], df['z'], s=10, c=[cmap(cl) for cl in labels])
    else:
        ax.scatter(df['x'], df['y'], df['z'], s=10)

    viewpoints_spherical = utils.rectangular_to_spherical(viewpoints)
    #ax.scatter(viewpoints[:, 0], viewpoints[:, 1], viewpoints[:, 2])

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    #Ensure equal aspect ratio:
    max = 1
    ax.set_xlim(-max, max)
    ax.set_ylim(-max, max)
    ax.set_zlim(-max, max)
    # ax.set_xticklabels([])
    # ax.set_yticklabels([])
    # ax.set_zticklabels([])

    viewsfile = '{0}/{1}.pkl'.format(constants.output_dir, os.path.basename(proj_file).replace('3d.csv', 'views'))
    # rotate the axes and update
    ax.set_box_aspect((4, 4, 4), zoom=1.2)
    fig2 = plt.figure(3)
    view_proj = fig2.add_subplot(111)
    view_proj.set_xticklabels([])
    view_proj.set_yticklabels([])
    view_proj.axis('equal')

    if manual_rotate:
        fig.show()
        elev, azim = None, None
        while True:
            if elev != ax.elev or azim != ax.azim:
                ax.view_init(0, 0)
                view_proj.clear()
                #Update the 2D view scatterplot if the view of the 3D projection changes
                view = proj_view(df, np.linalg.inv(ax.get_proj()))
                eye = ax.eye
                update_view(view_proj, view, cmap, df, eye, fig2)

            plt.pause(0.01)
    else:
        views = []
        #Loop over all viewpoints, save the resulting 2D view and optionally display the view in a plot.
        for index, view_sphere in enumerate(viewpoints_spherical):
            ax.view_init(view_sphere[0], view_sphere[1])
            view = proj_view(df, np.linalg.inv(ax.get_proj()))
            views.append(view)
            if display:
                fig.canvas.draw()
                fig.canvas.flush_events()
                # To make sure we draw the scatterplot points in the same order as the 3D projection (which matters for labeled layouts)
                # we need to sort them first by distance to the eye. Points closer to the eye get drawn on top of points further from the eye
                eye = viewpoints[index] * 100
                update_view(view_proj, view, cmap, df, eye, fig2)
                plt.pause(1)
        views_df = pd.DataFrame.from_records([(np.array(p), views[index]) for index, p in enumerate(viewpoints)])
        views_df.columns = ['viewpoints', 'views']
        views_df.to_pickle(viewsfile)

def proj_view(df, matrix):
    #Calculates the 2D screen projection of the 2D view.
    projection = np.ones((len(df.index), 4))
    projection[0:len(df.index), :3] = df.to_numpy()
    view = np.matmul(projection, matrix)[:, 0:2]

    #Scale exactly to the range (0, 1)
    view -= np.min(view, axis=0)
    view /= np.max(view)
    return view

def update_view(ax, view, cmap, df, eye, fig):
    #calculate the 2d view and display it in the plot
    ax.clear()
    sorted_indices = np.argsort(-np.linalg.norm(df.to_numpy() - eye, axis=1))
    if labels is not None:
        ax.scatter(view[sorted_indices, 0], view[sorted_indices, 1], s=10,
                          c=[cmap(cl) for cl in labels[sorted_indices]])
    else:
        ax.scatter(view[sorted_indices, 0], view[sorted_indices, 1], s=10)
    fig.canvas.draw()
    fig.canvas.flush_events()
    #fig.show()

def fibonacci_sphere(samples=1000):
    """
    Function to generate an arbitrary number of points, more or less equally distributed on the sphere surface
    """
    points = []
    phi = math.pi * (3. - math.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append([x, y, z])

    points = np.array(points)
    return points

def save_sphere_mesh(samples = 1000):
    """
    Compute N points evenly distributed over a sphere, and generate and save a sphere mesh from these points.
    """

    points = fibonacci_sphere(samples=samples)

    hull = ConvexHull(points)
    face_indices = np.array(hull.simplices)

    mesh = trimesh.Trimesh(vertices=points, faces=face_indices)
    mesh.fix_normals()
    np.save(f'spheres/sphere{samples}_points.npy', mesh.vertices)
    np.save(f'spheres/sphere{samples}_faces.npy', np.array([[c, b, a] for a, b, c in mesh.faces]))

if __name__ == '__main__':
    projections_3d = glob(os.path.join(constants.output_dir, '*3d.csv'))

    #save_sphere_mesh(samples=constants.samples)
    viewpoints = np.load(f'spheres/sphere{constants.samples}_points.npy')
    for proj_file in projections_3d:
        dataset_name = os.path.basename(proj_file).split('-')[0]
        label_file = glob('data/{0}/*-labels.csv'.format(dataset_name))

        if len(label_file) == 1:
            df_label = pd.read_csv(label_file[0], sep=';', header=0)
            labels = df_label.to_numpy()
        else:
            labels = None

        compute_views(proj_file, viewpoints, labels=labels, display=False, manual_rotate=False)
    plt.clf()
    plt.close('all')

    pass

