import numpy as np
import os
import string
import random
import colorsys
# import pybullet as p

def ang_in_mpi_ppi(angle):
    """
    Convert the angle to the range [-pi, pi).
    Args:
        angle (float): angle in radians.
    Returns:
        float: equivalent angle in [-pi, pi).
    """

    angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return angle

def fill_template(template, replace):
    filepath = os.path.dirname(os.path.abspath(__file__))
    template = os.path.join(filepath, '..', template)
    with open(template, 'r') as file:
        fdata = file.read()
    for field in replace:
        fdata = fdata.replace(f'{field}', str(replace[field]))
    alphabet = string.ascii_lowercase + string.digits
    rname = ''.join(random.choices(alphabet, k=16))
    fname = f'{template}.{rname}'
    # print('To replace', replace, fdata)

    with open(fname, 'w') as file:
        file.write(fdata)
    return fname

BLACK = colorsys.hsv_to_rgb(0, 0.75, 0)
RED = colorsys.hsv_to_rgb(0, 0.75, 0.75)
BLUE = colorsys.hsv_to_rgb(0.55, 0.75, 0.75)
GREEN = colorsys.hsv_to_rgb(0.4, 0.75, 0.4)
YELLOW = [0.8, 0.8, 0.1]

def generate_color(exclude_colors = None):
    if exclude_colors is None:
       return colorsys.hsv_to_rgb(np.random.rand(), 0.75, 0.75)

    exclude_colors = [np.array(colorsys.rgb_to_hsv(*color)) for color in exclude_colors]
    found = False

    while not found:
        h, s = np.random.rand(2)
        v = np.random.uniform(0.5, 1.0)
        color = np.array([h, s, v])

        found = True
        for color_ in exclude_colors:
            if np.sum(np.abs(color - color_) < 0.5) == 3:
                found = False
                break

    return colorsys.hsv_to_rgb(h, s, v)
