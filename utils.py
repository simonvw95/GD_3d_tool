import numpy as np
import os
from pathlib import Path


def rectangular_to_spherical(points):

    # convert rectangular coordinates (x,y,z) to spherical coordinates (Elevation, Azimuth, Distance)
    spherical = np.zeros_like(points)
    # elevation
    spherical[:, 0] = np.degrees(np.arctan2(points[:, 2], np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]))))
    # azimuth angle
    spherical[:, 1] = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    spherical[:, 2] = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]) + np.square(points[:, 2]))

    return spherical


def create_folder_for_path(filepath):
    direct = os.path.dirname(filepath)
    Path(direct).mkdir(parents = True, exist_ok = True)
