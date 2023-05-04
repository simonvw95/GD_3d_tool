import numpy as np
import os
from pathlib import Path

def rectangular_to_spherical(points):
    #Convert rectangular coordinates (x,y,z) to spherical coordinates (Elevation, Azimuth, Distance)
    spherical = np.zeros_like(points)
    spherical[:, 0] = np.degrees(np.arctan2(points[:, 2], np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1])))) # Elevation
    spherical[:, 1] = np.degrees(np.arctan2(points[:, 1], points[:, 0])) # Azimuth angle
    spherical[:, 2] = np.sqrt(np.square(points[:, 0]) + np.square(points[:, 1]) + np.square(points[:, 2]))
    #spherical = np.roll(spherical, 0)
    return spherical

def create_folder_for_path(filepath):
    dir = os.path.dirname(filepath)
    Path(dir).mkdir(parents=True, exist_ok=True)