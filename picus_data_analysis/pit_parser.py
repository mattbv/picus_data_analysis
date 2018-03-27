# -*- coding: utf-8 -*-
"""
@author: Matheus Boni Vicari
"""

import numpy as np
from scipy.spatial.distance import cdist


def extract_calculated_velocity(pit_file):

    """
    Function to extract calculated velocity data from a .pit file.

    Parameters
    ----------
    pit_file: string
        Path to .pit file to process.

    Returns
    -------
    img: array
        2D image with sound wave velocity information in meters per second.
    velocity_data: n-by-3 array
        Sound wave velocity data with configuration (x, y, velocity). n is
        the number of pixels/points with data.

    """

    # Reading .pit file into a list of strings.
    with open(pit_file, 'r') as f:
        file_str = f.readlines()

    # Obtaining initial row of velocity information. This is marked by the next
    # line after variable 'DW' on .pit file.
    data_line_id = [i for i, x in enumerate(file_str) if 'DW' in x][0] + 1

    # Obtaining list with velocity information.
    data_str = file_str[data_line_id:]

    # Spliting strings into separate substrings based on the separator '/'.
    # Also removing new line character '\n' and initial variable declaration.
    parsed_str = []
    for s in data_str:
        parsed_str.append(s.split('=')[-1].split('\n')[0].split('/'))

    # Initializing x, y and z (velocity) variables.
    x = []
    y = []
    z = []
    # Looping over parsed_str.
    for i in parsed_str:
        # Filtering out empty substrings.
        i = filter(None, i)
        # Checking if current list of string has at least 3 elements.
        if len(i) >= 3:
            # Looping over substrings with step = 3. The .pit file has velocity
            # information on the following order x, y, z.
            for j in np.arange(0, len(i), 3):
                # Appending current values to x,y,z variables.
                x.append(np.round(float(i[j])))
                y.append(np.round(float(i[j+1])))
                z.append(float(i[j+2]))

    # Stacking raw velocity data to output.
    velocity_data = np.vstack((x, y, z)).T

    # Obtaining number of rows and columns for output image.
    x = x - np.min(x)
    y = y - np.min(y)

    # Calculating number of rows and columns to generate output image.
    cols = int(np.max(x)) + 1
    rows = int(np.max(y)) + 1

    # Calculating maximum value of y. This will be later used to invert
    # y coordinates in order to make it compatile to an image coordinate
    # sytem (in which the origin is on the top-left corner).
    ymax = np.max(y)

    # Initializing output image as array of zeroes.
    img = np.zeros([rows, cols])

    # Looping over x,y,z data and assigining z values to their x and y
    # coordinates on img.
    for xi, yi, zi in zip(x, y, z):
        # Calculating current row and column values.
        row_i = int(ymax - yi)
        col_i = int(xi)
        # Assigning current velocity value to row_i, col_i on img. velocity
        # value (zi) is divided by 10000 to adjust units.
        img[row_i, col_i] = zi / 10000

    # Replacing 0s with NaNs (to output only calculated data).
    img[img == 0] = np.nan

    return img, velocity_data


def extract_sensors_coordinates(pit_file):

    """
    Function to extract PiCUS sensors coordinates from a .pit file.

    Parameters
    ----------
    pit_file: string
        Path to .pit file to process.

    Returns
    -------
    coords: n-by-2 array
        x and y coordiantes where n in the number of sensors used in
        .pit file measurement.

    """

    # Reading .pit file into a list of strings.
    with open(pit_file, 'r') as f:
        file_str = f.readlines()

    # Obtaining initial row of sensors coordinates information. This is
    # marked by the next line after keyword '[MPoints]' on .pit file.
    coords_line_id = [i for i,
                      x in enumerate(file_str) if '[MPoints]' in x][0] + 1

    # Initializing list of coordinates.
    coords = []

    # Looping over file_str from starting line of coordinates information
    # onwards.
    for i in file_str[coords_line_id:]:
        # Checking if line has valid information.
        if len(i) > 1:
            # Spliting line data based on separator '/'.
            line_data = i.split('=')[-1].split('\n')[0].split('/')
            # Appending coordiantes to list coords.
            coords.append(line_data)
        else:
            # If no valid data is found, break. This aims to detect the end
            # of coordinate information.
            break

    return np.asarray(coords).astype(float)


def extract_raw_data(filename):

    """
    Function to parse .pit file and extract min, max and mean sonic
    velocities and coordinates from measurement points.

    Parameters
    ----------
    filename : str
        Filename of .pit file to parse.

    Returns
    -------
    velocities : tuple
        Tuple of square matrices containing minimum, maximum and mean sonic
         velocity data from all origin points (rows) to all measurement
         points (columns).
    coods : numpy.ndarray
        n_points x 2 point coordinates.

    """

    # Reading .pit file.
    with open(filename, 'r') as f:
        pitData = f.read()

    # Splitting original pit file string to obtain sound waves travel time
    # (subset of data between [Diagnoses] and [Lines]).
    timeRawStr = pitData.split('[Diagnoses]')[1].split('[Lines]')[0]

    # Splitting time data into separate measurements.
    measurementsDataStr = [s.split(']')[-1] for s in timeRawStr.split('[')]
    # Removing empty/unwanted strings from measurementsDataStr.
    measurementsDataStr = [i for i in measurementsDataStr if len(i) > 2]

    # Calculating number of rows requires and allocating timeData array.
    nRows = len(measurementsDataStr)
    timeData = np.full([20, nRows, nRows], np.nan)

    # Looping over each set of point measurements.
    for pId, pointStr in enumerate(measurementsDataStr):
        # Splitting each set of measurements into different sensors
        # measurement.
        measurementLines = pointStr.split('\n')
        measurementLines = filter(None, measurementLines)
        # Looping over each sensor's data.
        for measurement in measurementLines[1:]:
            # Split data into sensor id (connectionId) and travel time
            # data (rawDataStr).
            connectionId, rawDataStr = measurement.split('=')
            # Filtering out empty entries and spliting set of measurements.
            rawDataStr = filter(None, rawDataStr.split('/'))
            # Looping over each measurement (hit).
            for hitId, hitData in enumerate(rawDataStr):
                # As zero denotes empty, only process non-zero values.
                if hitData != '0':
                    # Casting ids as integers to avoid indexing errors.
                    h = int(hitId)
                    p = int(pId)
                    c = int(connectionId) - 1
                    # Assigning hitData as float to current (h, p, c)
                    # position in timeData array.
                    timeData[h, p, c] = float(hitData)

    # Looping over each layer of hits in timeData (axis 0) and setting
    # diagonal values to 0.
    for h in xrange(timeData.shape[0]):
        timeData[h][np.diag_indices_from(timeData[h])] = 0

    # Calculating min, max and mean timeData.
    timeMean = np.nanmean(timeData, axis=0)
    timeMax = np.nanmax(timeData, axis=0)
    timeMin = np.nanmin(timeData, axis=0)
    timeMean[np.diag_indices_from(timeMean)] = 0
    timeMax[np.diag_indices_from(timeMax)] = 0
    timeMin[np.diag_indices_from(timeMin)] = 0

    # Splitting original pit file string to obtain sensors coordinates
    # (subset of data between [Diagnoses] and [Lines]).
    coordsRawStr = pitData.split('[MPoints]')[1].split('[ZMPoints]')[0]
    coordsRawStr = filter(None, coordsRawStr.split('\n'))

    # Allocating coords array with shape [n_points * 2].
    coords = np.zeros([len(coordsRawStr), 2], dtype=float)

    # Looping over each coordinate entry, splitting into x and y data and
    # assigning them to coords array.
    for cId, coordLineStr in enumerate(coordsRawStr):
        x, y = coordLineStr.split('=')[1].split('/')
        coords[cId] = [float(x), float(y)]

    # Calculating pointwise distance between all pairs of points in coords.
    # Converting distances units from cm to m.
    pwiseDist = cdist(coords, coords) / 100

    # Generating a zero value mask for matrix diagonal to avoid calculating
    # division of zero values.
    zeroMask = ~np.eye(nRows).astype(bool)

    # Calculating valocities. The division of time by (10**6) aims to
    # convert units from ns to s.
    minVelocity = np.zeros([nRows, nRows], dtype=float)
    minVelocity[zeroMask] = (pwiseDist[zeroMask] /
                             (timeMin[zeroMask] / (10**6)))
    maxVelocity = np.zeros([nRows, nRows], dtype=float)
    maxVelocity[zeroMask] = (pwiseDist[zeroMask] /
                             (timeMax[zeroMask] / (10**6)))
    meanVelocity = np.zeros([nRows, nRows], dtype=float)
    meanVelocity[zeroMask] = (pwiseDist[zeroMask] /
                              (timeMean[zeroMask] / (10**6)))

    return (minVelocity, maxVelocity, meanVelocity), coords


def image_as_points(img):

    """
    Transforms a 2D image in a 2D array with x, y pixel coordinates and pixel
    value.

    Parameters
    ----------
    img : numpy.ndarray
        2D image.

    Returns
    -------
    pointData : numpy.ndarray
        n_pixels/points * 2 array containing pixels coordinates and values.

    """

    # Get indices of non-zero pixels.
    nonZeroY, nonZeroX = np.where(img > 0)
    # Allocates output variable.
    pointData = np.zeros([nonZeroX.shape[0], 3], dtype=float)
    # Loop over each pair of non-zero pixel coordinates and assigns them
    # and their respective pixel value to pointData.
    for i, (x1, y1) in enumerate(zip(nonZeroX, nonZeroY)):
        pointData[i, :] = [x1, y1, img[y1, x1]]

    return pointData
