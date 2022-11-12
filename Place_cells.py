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
datafile = h5py.File(filepath, mode='r')

x_coords = datafile["animalPosition/xOfFirstLed"][:]
y_coords = datafile["animalPosition/yOfFirstLed"][:]
coord_fs = datafile["animalPosition"].attrs["coordinatesSampleRate"]


cluster_number = 14

cluster_group =  datafile["electrode_7/spikes/cluster_"+str(cluster_number)]


spike_train = cluster_group["train"][:] / datafile.attrs['samplingRate']

print(datafile.attrs['duration'], spike_train[-1])
print(cluster_group.attrs["type"], cluster_group.attrs["quality"])

x_spike = []
y_spike = []
x_coords_indx = np.arange(x_coords.size) / coord_fs
y_coords_indx = np.arange(y_coords.size) / coord_fs
for sp in spike_train:
    x_spike.append( x_coords[find_nearest(x_coords_indx, sp)] )
    y_spike.append( y_coords[find_nearest(y_coords_indx, sp)] )


x_spike = np.asarray(x_spike)
y_spike = np.asarray(y_spike)

x_spike = x_spike[x_spike >= 0]
y_spike = y_spike[y_spike >= 0]

map_place_cell, bins_x, bins_y = np.histogram2d(x_spike, y_spike, bins=360, range=None, density=True)
map_place_cell = gaussian_filter(map_place_cell, sigma = 5)

fig, axes = plt.subplots(ncols=2)


axes[0].scatter(x_spike, y_spike, s=4)
axes[0].set_xlim(np.min(x_spike), np.max(x_spike))
axes[0].set_ylim(np.min(y_spike), np.max(y_spike))
axes[1].pcolor(bins_x[1:], bins_y[1:], map_place_cell, cmap='rainbow', shading='auto')

plt.show()

datafile.close()