import skimage
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import h5py
import os
import cv2
from scipy.ndimage import gaussian_filter

import imageio.v2 as imageio

from math import sqrt

from skimage.exposure import exposure
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
from skimage.io import imread, imshow, imsave
from skimage import util


def read_transparent_png(filename):
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = image_4channel[:,:,3]
    rgb_channels = image_4channel[:,:,:3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:,:,np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate((alpha_factor,alpha_factor,alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def place_cells(N):
    filepath = "D:\Рабочий стол\Программирование\dataset.hdf5"
    datafile = h5py.File(filepath, mode='r')

    x_coords = datafile["animalPosition/xOfFirstLed"][:]
    y_coords = datafile["animalPosition/yOfFirstLed"][:]
    coord_fs = datafile["animalPosition"].attrs["coordinatesSampleRate"]

    cluster_number = N

    cluster_group = datafile["electrode_7/spikes/cluster_" + str(cluster_number)]

    spike_train = cluster_group["train"][:] / datafile.attrs['samplingRate']
    '''
    print(datafile.attrs['duration'], spike_train[-1])
    print(cluster_group.attrs["type"], cluster_group.attrs["quality"])
    '''
    x_spike = []
    y_spike = []
    x_coords_indx = np.arange(x_coords.size) / coord_fs
    y_coords_indx = np.arange(y_coords.size) / coord_fs
    for sp in spike_train:
        x_spike.append(x_coords[find_nearest(x_coords_indx, sp)])
        y_spike.append(y_coords[find_nearest(y_coords_indx, sp)])

    x_spike = np.asarray(x_spike)
    y_spike = np.asarray(y_spike)

    x_spike = x_spike[x_spike >= 0]
    y_spike = y_spike[y_spike >= 0]
    print(y_spike.size)
    map_place_cell, bins_x, bins_y = np.histogram2d(y_spike, x_spike, bins=400, density=True)
    map_place_cell = gaussian_filter(map_place_cell, sigma=5, mode='constant', cval=0.0)

    inverted_map_place_cell = np.max(map_place_cell) - map_place_cell

    plt.pcolor(bins_y[:-1], bins_x[:-1], map_place_cell, cmap='rainbow', shading='auto')
    #ax = plt.gca()
    #ax.axes.xaxis.set_ticks([])
    plt.axis('off')
    #plt.tight_layout()
    plt.savefig(f"{cluster_number}.png", dpi='figure',  bbox_inches='tight', pad_inches=0)
    print(max(bins_y[:-1]), max(bins_x[:-1]), map_place_cell.shape)
    #img = imread(f'{cluster_number}.png')
    #inverted_img = util.invert(img)
    #imsave(f'100{N}.png', inverted_img)
    plt.show()
    datafile.close()
    #print(map_place_cell)
    return map_place_cell, inverted_map_place_cell, bins_x, bins_y

def blobs(N):
    #prepare_image = read_transparent_png(f'{N}.png')

    prepare_image = imageio.imread(f'{N}.png')[:, :, :3]
    prepare_image = prepare_image[::-1, ::1]
    prepare_image = cv2.resize(prepare_image, (400, 400))
    image_inverted = np.max(prepare_image) - prepare_image
    print(prepare_image.shape)
    print(image_inverted.shape)


    image_gray_inverted = rgb2gray(image_inverted)
    image_gray = rgb2gray(prepare_image)
    #image = exposure.equalize(image)  # improves detection

    blobs_log_min = blob_log(image_gray_inverted, min_sigma=14.5, max_sigma=50, num_sigma=20, threshold=0.2, overlap=0.5, log_scale=False, threshold_rel=None, exclude_border=False)
    blobs_log_min[:, 2] = blobs_log_min[:, 2] * sqrt(2)

    #blobs_log_max = blob_log(image_gray, min_sigma=10, max_sigma=50, num_sigma=10, threshold=0.1, overlap=0.5, log_scale=False, threshold_rel=None, exclude_border=False)
    blobs_log_max = blob_log(image_gray, min_sigma=14.5, max_sigma=50, num_sigma=20, threshold=0.2, overlap=0.5, log_scale=False, threshold_rel=None, exclude_border=False)
    blobs_log_max[:, 2] = blobs_log_max[:, 2] * sqrt(2)

    print(blobs_log_max)
    print(blobs_log_min)
    blobs_list = [blobs_log_min, blobs_log_max]
    colors = ['yellow', 'lime']
    titles = ['blobs_log_min','blobs_log_max']

    sequence = zip(blobs_list, colors, titles)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(prepare_image)
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()

    plt.tight_layout()

    plt.savefig(f"{N}00.png", bbox_inches='tight', pad_inches = 0)
    #plt.show()

    '''
    ax[1].set_title(blobs)
    ax[1].imshow(prepare_image)
    y, x, r = blobs_log_min
    f = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)
    y, x, r = blobs_log_max
    c = plt.Circle((x, y), r, color='yellow', linewidth=2, fill=False)
    ax[1].add_patch(c)
    ax[1].add_patch(f)
    ax[1].set_axis_off()
    ax[2].imshow(prepare_image)
'''

'''
    ax = plt.gca()
    ax.set_axis_off()
    ax.imshow(image, origin='lower')
    cnt = 0
    while cnt < len(blobs_log):
        c = plt.Circle((blobs_log[cnt][1], blobs_log[cnt][0]), blobs_log[cnt][2], color='white', linewidth=2,
                       fill=False)
        ax.add_patch(c)
        cnt = cnt + 1
'''


#map_place_cell, inverted_map_place_cell, bins_x, bins_y = place_cells(1)
place_cells(1)
blobs(1)

#mpc = mpc - log
 #x, y, sigma

#plt.pcolor(bins_y[:-1], bins_x[:-1], impc, cmap='rainbow', shading='auto')
#plt.show()


