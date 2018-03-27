import numpy as np
import cv2
from scipy import interpolate


def interp_data(array):

    """
    Performs cubic interpolation of data in 2D grids (images).

    Parameters
    ----------
    array : numpy.ndarray
        Array containing the data to be interpolated.

    Returns
    -------
    interpData : numpy.ndarray
        Array containing the interpolated data.

    """

    # Casting array as type float.
    array = array.astype(float)

    # Setting 0 values to nan.
    array[array == 0] = np.nan

    # Generating ranges of x and y values based on the shape of the input
    # array.
    x = np.arange(0, array.shape[1])
    y = np.arange(0, array.shape[0])
    # Mask invalid values
    array = np.ma.masked_invalid(array)
    # Generates meshgrid.
    xx, yy = np.meshgrid(x, y)

    # Get only the valid values
    x1 = xx[~array.mask]
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    # Performs cubic interpolation.
    interpData = interpolate.griddata((x1, y1), newarr.ravel(),
                                      (xx, yy), method='cubic')

    return interpData


def calculate_2D_velocity(velocity, coords):

    """
    Interpolates 2D sonic velocity from PiCUS (.pit) parsed data.

    Parameters
    ----------
    velocity : numpy.ndarray
        Square matrix containing velocity data from each emitting
        point (row) to each sensor point (column).
    coords : numpy.ndarray
        n_points * 2 array containing coordinates of emitters/sensors points.

    Returns
    -------
    velocityInterp : numpy.ndarray
        3D stack of images containing interpolated velocities for each
        emitter.
    velocityLines : numpy.ndarray
        3D stack of images containing velocities lines between each
        emitter and sensor pairs.

    """

    # Detecting largest dimension (x or y) to use as image size.
    xRange = np.max(coords[:, 0]) - np.min(coords[:, 0])
    yRange = np.max(coords[:, 1]) - np.min(coords[:, 1])
    maxRange = int(np.max([xRange, yRange]))

    # Allocating baseGrid, velocityLines and velocityInterp.baseGrid will
    # be used in intermediate steps of the interpolation.
    # velocityLines and velocityInterp will store output data.
    baseGrid = np.zeros([maxRange, maxRange], dtype=int)
    velocityLines = np.zeros([maxRange, maxRange, coords.shape[0]],
                             dtype=int)
    velocityInterp = np.zeros([maxRange, maxRange, coords.shape[0]],
                              dtype=float)

    # Looping over number of points in coords (emitters).
    for i in range(coords.shape[0]):
        # Generating temporary variables.
        img = velocityLayer = baseGrid.copy()
        # Looping over number of points in coords (sensors).
        for j in range(coords.shape[0]):
            # Setting emitter base coordinate (xbase, ybase) and sensor
            # coordinate (xend, yend).
            xbase = int(coords[i, 0])
            ybase = int(img.shape[0] - coords[i, 1])
            xend = int(coords[j, 0])
            yend = int(img.shape[0] - coords[j, 1])

            # Drawing lines from base to end using openCV line function.
            # current line will have value of 1 (background is 0).
            cv2.line(img, (xbase, ybase), (xend, yend), 1, 1)
            # Using known value of 1 to set all line's pixels to current
            # sonic velocity from emitter i to sensor j.
            # Accumulate lines in velocityLayer.
            velocityLayer = velocityLayer + img
            velocityLayer[velocityLayer == 1] = int(velocity[i, j])

        # Interpolate current layer.
        layerInterp = interp_data(velocityLayer)

        # Assigning current layerInterp and velocityLayer to output arrays.
        velocityInterp[:, :, i] = layerInterp
        velocityLines[:, :, i] = velocityLayer

    return velocityInterp, velocityLines
