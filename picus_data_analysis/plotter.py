import numpy as np
import matplotlib.pyplot as plt


def gen_plot(img, velocity_data, coords):

    """
    Function to generate a plot using sound wave velocity data and
    sensors coordinates. This plot will be similar to standard output
    of PiCUS 3 software.

    Parameters
    ----------
    img: array
        2D image with sound wave velocity information in meters per second.
    velocity_data: n-by-3 array
        Sound wave velocity data with configuration (x, y, velocity). n is
        the number of pixels/points with data.
    coords: n-by-2 array
        x and y coordiantes where n in the number of sensors used in
        .pit file measurement.


    """

    # Converting coordinates to an image (top-down, left-right) coordinate
    # system.
    xc = coords[:, 0] - np.min(velocity_data[:, 0])
    yc = np.max(velocity_data[:, 1]) - coords[:, 1]

    # Generating unique ids for each sensor.
    point_id = np.arange(1, xc.shape[0] + 1)

    # Generating new figure.
    plt.figure()

    # Plotting image and sensors information.
    img_plot = plt.imshow(img)
    plt.plot(xc, yc, 'ro')
    cbar = plt.colorbar(img_plot)
    cbar.set_label('velocity (m/s)', labelpad=-40, y=1.05, rotation=0)

    # Annotating sensors points with unique ids.
    for i, txt in enumerate(point_id):
        plt.annotate(txt, (xc[i], yc[i]))
