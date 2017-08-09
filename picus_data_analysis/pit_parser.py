# -*- coding: utf-8 -*-
"""
@author: Matheus Boni Vicari
"""

import numpy as np


def extract_speed(pit_file):

    with open(pit_file, 'r') as f:
        file_str = f.readlines()

    data_row_id = [i for i, x in enumerate(file_str) if 'DW' in x][0] + 1

    data_str = file_str[data_row_id:]

    parsed_str = []
    for s in data_str:
        parsed_str.append(s.split('=')[-1].split('\n')[0].split('/'))

    x = []
    y = []
    z = []
    for i in parsed_str:
        i = filter(None, i)
        if len(i) >= 3:
            for j in np.arange(0, len(i), 3):
                x.append(np.round(float(i[j])))
                y.append(np.round(float(i[j+1])))
                z.append(float(i[j+2]))

    xmax = int(np.max(x))
    ymax = int(np.max(y))

    img = np.zeros([ymax, xmax])

    for xi, yi, zi in zip(x, y, z):
        img[int(ymax - yi - 1), int(xi - 1)] = zi

    img[img == 0] = np.nan

    return img


def extract_sensors_coordinates(pit_file):

    with open(pit_file, 'r') as f:
        file_str = f.readlines()

    coords_row_id = [i for i,
                     x in enumerate(file_str) if '[MPoints]' in x][0] + 1

    coords = []
    for i in file_str[coords_row_id:]:
        if len(i) > 1:
            row_data = i.split('=')[-1].split('\n')[0].split('/')
            coords.append(row_data)
        else:
            break

    return np.asarray(coords).astype(float)


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    pit_file = r'D:/Dropbox/PhD/Scripts/phd-geography-ucl/general_purpose_packages/pycus/data/Lond StJ plane1.pit'
    img = extract_speed(pit_file)

    plt.imshow(img)
