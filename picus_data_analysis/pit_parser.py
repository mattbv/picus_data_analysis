# -*- coding: utf-8 -*-
"""
@author: Matheus Boni Vicari
"""

import numpy as np
import matplotlib.pyplot as plt


def extract_speed(pit_file):

    """
    Function to extract speed data from a .pit file.

    Parameters
    ==========
    pit_file: string
        Path to .pit file to process.

    Returns
    =======
    img: array
        2D image with sound wave speed information in meters per second.
    speed_data: n-by-3 array
        Sound wave speed data with configuration (x, y, speed). n is
        the number of pixels/points with data.

    """

    # Reading .pit file into a list of strings.
    with open(pit_file, 'r') as f:
        file_str = f.readlines()

    # Obtaining initial row of speed information. This is marked by the next
    # line after variable 'DW' on .pit file.
    data_line_id = [i for i, x in enumerate(file_str) if 'DW' in x][0] + 1

    # Obtaining list with speed information.
    data_str = file_str[data_line_id:]

    # Spliting strings into separate substrings based on the separator '/'.
    # Also removing new line character '\n' and initial variable declaration.
    parsed_str = []
    for s in data_str:
        parsed_str.append(s.split('=')[-1].split('\n')[0].split('/'))

    # Initializing x, y and z (speed) variables.
    x = []
    y = []
    z = []
    # Looping over parsed_str.
    for i in parsed_str:
        # Filtering out empty substrings.
        i = filter(None, i)
        # Checking if current list of string has at least 3 elements.
        if len(i) >= 3:
            # Looping over substrings with step = 3. The .pit file has speed
            # information on the following order x, y, z.
            for j in np.arange(0, len(i), 3):
                # Appending current values to x,y,z variables.
                x.append(np.round(float(i[j])))
                y.append(np.round(float(i[j+1])))
                z.append(float(i[j+2]))

    # Stacking raw speed data to output.
    speed_data = np.vstack((x, y, z)).T

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
        # Assigning current speed value to row_i, col_i on img. Speed value
        # (zi) is divided by 10000 to adjust units.
        img[row_i, col_i] = zi / 10000

    # Replacing 0s with NaNs (to output only calculated data).
    img[img == 0] = np.nan

    return img, speed_data


def extract_sensors_coordinates(pit_file):

    """
    Function to extract PiCUS sensors coordinates from a .pit file.

    Parameters
    ==========
    pit_file: string
        Path to .pit file to process.

    Returns
    =======
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


def gen_plot(img, speed_data, coords):

    """
    Function to generate a plot using sound wave speed data and
    sensors coordinates. This plot will be similar to standard output
    of PiCUS 3 software.

    Parameters
    ==========
    img: array
        2D image with sound wave speed information in meters per second.
    speed_data: n-by-3 array
        Sound wave speed data with configuration (x, y, speed). n is
        the number of pixels/points with data.
    coords: n-by-2 array
        x and y coordiantes where n in the number of sensors used in
        .pit file measurement.


    """

    # Converting coordinates to an image (top-down, left-right) coordinate
    # system.
    xc = coords[:, 0] - np.min(speed_data[:, 0])
    yc = np.max(speed_data[:, 1]) - coords[:, 1]

    # Generating unique ids for each sensor.
    point_id = np.arange(1, xc.shape[0] + 1)

    # Plotting image and sensors information.
    img_plot = plt.imshow(img, cmap='jet_r')
    plt.plot(xc, yc, 'ro')
    cbar = plt.colorbar(img_plot)
    cbar.set_label('Speed (m/s)', labelpad=-40, y=1.05, rotation=0)

    # Annotating sensors points with unique ids.
    for i, txt in enumerate(point_id):
        plt.annotate(txt, (xc[i], yc[i]))


if __name__ == "__main__":

    pit_file = '../data/Lond StJ plane1.pit'
    img, speed_data = extract_speed(pit_file)
    coords = extract_sensors_coordinates(pit_file)

    gen_plot(img, speed_data, coords)
