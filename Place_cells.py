#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.ndimage import gaussian_filter

def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx

filepath = 'D:\Рабочий стол\Программирование\dataset.hdf5'

with h5py.File(filepath, 'r') as datafile:
    x_coords = datafile["animalPosition/xOfFirstLed"][:]
    y_coords = datafile["animalPosition/yOfFirstLed"][:]
    coord_fs = datafile["animalPosition"].attrs["coordinatesSampleRate"]
    cluster_number = 14
    cluster_group =  datafile["electrode_7/spikes/cluster_"+str(cluster_number)]
    spike_train = cluster_group["train"][:] / datafile.attrs['samplingRate']

def get_position(x_coords, y_coords, coord_fs ):
    x_spike = []
    y_spike = []
    x_coords_indx = np.arange(x_coords.size) / coord_fs
    y_coords_indx = np.arange(y_coords.size) / coord_fs
    return x_coords_indx, y_coords_indx

def get_spikes(x_coords_indx, y_coords_indx):
    x_spike = []
    y_spike = []
    for sp in spike_train:
        x_spike.append( x_coords[find_nearest(x_coords_indx, sp)] )
        y_spike.append( y_coords[find_nearest(y_coords_indx, sp)] )

    x_spike = np.asarray(x_spike)
    y_spike = np.asarray(y_spike)

    x_spike = x_spike[x_spike >= 0]
    y_spike = y_spike[y_spike >= 0]
    return x_spike, y_spike

x_coords_indx, y_coords_indx = get_position(x_coords,y_coords, coord_fs)

x_spike, y_spike = get_spikes(x_coords_indx, y_coords_indx)

map_place_cell, bins_x, bins_y = np.histogram2d(x_spike, y_spike, bins=360, density=True)
map_place_cell = gaussian_filter(map_place_cell, sigma = 5, mode='constant')

fig, axes = plt.subplots(ncols=2)
axes[0].scatter(x_spike, y_spike, s=4)
axes[1].plot(x_spike, y_spike)
axes[1].pcolor(bins_x[1:], bins_y[1:], map_place_cell.T, cmap='rainbow')

plt.show()

datafile.close()