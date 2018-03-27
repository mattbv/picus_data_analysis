import numpy as np
import matplotlib.pyplot as plt
from pit_parser import (extract_calculated_velocity,
                        extract_sensors_coordinates, extract_raw_data,
                        image_as_points)
from velocity_analysis import calculate_2D_velocity
from plotter import gen_plot


def pre_calculated_velocity(pit_file):

    """
    Example showing how to use pre-calculated velocities (by PiCUS software).

    Parameters
    ----------
    pit_file : str
        Filesystem path and name of the .pit file to process.

    """

    # Extracts velocity and sensor coordinates from file.
    velocityImg, velocityData = extract_calculated_velocity(pit_file)
    coords = extract_sensors_coordinates(pit_file)

    # Making sure all velocities are positive.
    velocityImg = np.abs(velocityImg)

    # Plots velocity data and sensors.
    gen_plot(velocityImg, velocityData, coords)
    plt.title('Velocities from pre-calculated data')


def raw_data_velocity(pit_file):

    """
    Example showing how to extract raw data from a .pit file and calculate
    mean sonic velocity. This function aims to completely bypass PiCUS
    software.

    Parameters
    ----------
    pit_file : str
        Filesystem path and name of the .pit file to process.

    """

    # Extract raw velocities from .pit file. Velocities[0] = min,
    # Velocities[1] = max, Velocities[2] = mean
    velocities, coords = extract_raw_data(pit_file)

    # Interpolates mean velocity (velocities[2]). This will generate 2
    # stacks of images containing all velocity lines from each point to
    # all other points (velocityLines) and their 2D interpolations
    # (velocityInterp).
    velocityInterp, velocityLines = calculate_2D_velocity(velocities[2],
                                                          coords)

    # Calculates maximum sonic velocity from all interpolated data.
    maxVelocity = np.nanmean(velocityInterp, axis=2)

    # Obtain image data as points (x, y, pixel_value). This is just going
    # to be used as intermediate information for plotting.
    maxVelocityPoints = image_as_points(maxVelocity)

    # Plotting results
    gen_plot(maxVelocity, maxVelocityPoints, coords)
    plt.title('Velocities from raw data')


if __name__ == "__main__":

    pit_file = '../data/Lond StJ plane1.pit'
    pre_calculated_velocity(pit_file)

    raw_data_velocity(pit_file)
