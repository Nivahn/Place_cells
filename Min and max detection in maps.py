import skimage
import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

from skimage.io import imread, imshow, imsave
from scipy.ndimage import gaussian_filter
from skimage.feature import blob_dog, blob_log, blob_doh
from scipy import stats


def select_files(sourses_path):
    SelectedFiles = []

    for path, _, files in os.walk(sourses_path):

        for file in files:
            if file.find(".hdf5") == -1:
                continue
            if file == 'ec013.459.hdf5':
                continue

            # print(file)
            sourse_hdf5 = h5py.File(path + '/' + file, "r")
            for ele_key, electrode in sourse_hdf5.items():
                try:
                    if electrode.attrs['brainZone'] != 'CA1':
                        continue
                    if 'xOfFirstLed' not in sourse_hdf5["animalPosition/"] and 'yOfFirstLed' not in sourse_hdf5[
                        "animalPosition/"]:
                        continue
                    if sourse_hdf5.attrs["behavior_test"] != "bigSquare":
                        continue
                except KeyError:
                    continue

                for cluster in electrode['spikes'].values():
                    try:
                        if cluster.attrs['type'] != 'Int' or cluster.attrs['quality'] != 'Nice':
                            continue
                    except KeyError:
                        continue

                    SelectedFiles.append(path + '/' + file)
                    break
                break
            sourse_hdf5.close()
    return SelectedFiles


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return idx


def plot_blobs(blobs_log_min, blobs_log_max, image_inverted, source_image):
    blobs_list = [blobs_log_min, blobs_log_max]
    colors = ['black', 'lime']
    titles = ['blobs_log_min', 'blobs_log_max']

    sequence = zip(blobs_list, colors, titles)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[0].imshow(image_inverted)
        ax[1].imshow(source_image)
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
            ax[idx].set_axis_off()

    plt.tight_layout()
    # plt.savefig(f"/home/sergeydubrovin/final_{N}.png", bbox_inches='tight', pad_inches = 0)
    plt.show()
    print(blobs_log_min)


def get_blobs(electrode_number, cluster_number, filepath, animal):
    datafile = h5py.File(filepath, mode='r')

    cluster_group = datafile[str(electrode_number) + "/spikes/" + str(cluster_number)]

    spike_train = cluster_group["train"][:] / datafile.attrs['samplingRate']

    x_coords = datafile["animalPosition/xOfFirstLed"][:]
    y_coords = datafile["animalPosition/yOfFirstLed"][:]
    coord_fs = datafile["animalPosition"].attrs["coordinatesSampleRate"]

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

    map_place_cell, bins_x, bins_y = np.histogram2d(y_spike, x_spike, bins=400, density=False)  # ? ? ?
    map_place_cell = gaussian_filter(map_place_cell, sigma=5, mode='constant', cval=0.0)

    inverted_map_place_cell = np.max(map_place_cell) - map_place_cell

    source_image = map_place_cell

    image_inverted = np.max(source_image) - source_image

    # blobs_log_min = blob_log(image_inverted, min_sigma= 1, max_sigma=30, num_sigma=10, threshold=0.09, overlap=0.5, log_scale=False, threshold_rel=None, exclude_border=False)
    blobs_log_min = blob_log(image_inverted, threshold=np.mean(source_image), overlap=0.5, log_scale=False,
                             threshold_rel=None, exclude_border=False)
    # blobs_log_min[:, 2] * sqrt(2)
    mean_of_map_place_cell = np.mean(inverted_map_place_cell)

    # blobs_log_max = blob_log(prepare_image, min_sigma= 1, max_sigma=30, num_sigma=10, threshold=0.09, overlap=0.5, log_scale=False, threshold_rel=None, exclude_border=False)
    blobs_log_max = blob_log(source_image, threshold=2 * np.std(source_image), overlap=0.5, log_scale=False,
                             threshold_rel=None, exclude_border=False)
    # blobs_log_max[:, 2] * sqrt(2)
    mean_of_image_inverted = np.mean(source_image)

    datafile.close()
    bin_x_size = bins_x[1] - bins_x[0]
    bin_y_size = bins_y[1] - bins_y[0]

    # plot_blobs(blobs_log_min, blobs_log_max, image_inverted, source_image)

    return blobs_log_min, blobs_log_max, mean_of_map_place_cell, mean_of_image_inverted, bin_x_size, bin_y_size


def blobs_in_cm(blobs_log_max, blobs_log_min, bin_x_size, bin_y_size):
    blobs_log_max_width_cm = blobs_log_max[:, 2] * bin_x_size
    blobs_log_min_width_cm = blobs_log_min[:, 2] * bin_x_size

    blobs_log_max_height_cm = blobs_log_max[:, 2] * bin_y_size
    blobs_log_min_height_cm = blobs_log_min[:, 2] * bin_y_size

    return blobs_log_max_width_cm, blobs_log_min_width_cm, blobs_log_max_height_cm, blobs_log_min_height_cm


def stat_for_blobs(blobs_data, cnt):
    # Дополнить
    blobs_data = np.sort(blobs_data)
    mode = stats.mode(blobs_data)
    median = np.median(blobs_data)
    mean_blobs = np.mean(blobs_data)
    max_blobs = np.max(blobs_data)
    min_blobs = np.min(blobs_data)
    std_blobs = np.std(blobs_data)
    mean_of_blobs_in_1_map = blobs_data.size / cnt

    print('mean_blobs', mean_blobs)
    print('max_blobs', max_blobs)
    print('min_blobs', min_blobs)
    print('std_blobs', std_blobs)
    print('mean_of_blobs_in_1_map', mean_of_blobs_in_1_map)
    print('mode', mode)
    print('median', median)


def main(filepaths, ):
    filepaths = select_files('/media/usb/Data/Transformed_CRCNC/hc-3')
    print(filepaths)

    global_max_lids = np.empty(0, dtype=np.float64)
    global_min_lids = np.empty(0, dtype=np.float64)

    global_max_lids_w = np.empty(0, dtype=np.float64)
    global_min_lids_w = np.empty(0, dtype=np.float64)

    global_max_lids_h = np.empty(0, dtype=np.float64)
    global_min_lids_h = np.empty(0, dtype=np.float64)

    mean_of_map_place_cell = np.empty(0, dtype=np.float64)
    mean_of_image_inverted = np.empty(0, dtype=np.float64)

    cnt = 0

    for filepath in filepaths:
        datafile = h5py.File(filepath, mode='r')
        session = (datafile.attrs['session'])
        for data in datafile:
            if 'electrode' in data:
                if 'spikes' in datafile[f'{data}/']:
                    for clusters in datafile[f'{data}/spikes/']:

                        cluster_group = datafile[str(data) + "/spikes/" + str(clusters)]

                        spike_train = cluster_group["train"][:] / datafile.attrs['samplingRate']

                        if spike_train.size / datafile.attrs["duration"] > 30:
                            blobs_log_min, blobs_log_max, mean_normal_map, mean_inverted, bin_x_size, bin_y_size = get_blobs(
                                data, clusters, filepath, session)

                            global_max_lids = np.append(global_max_lids, blobs_log_max[:, 2])
                            global_min_lids = np.append(global_min_lids, blobs_log_min[:, 2])

                            cnt += 1

                            blobs_log_max_width_cm, blobs_log_min_width_cm, blobs_log_max_height_cm, blobs_log_min_height_cm = blobs_in_cm(
                                blobs_log_max[:, 2], blobs_log_min[:, 2], bin_x_size, bin_y_size)

                            global_max_lids_w = np.append(blobs_log_max_width_cm, blobs_log_min_width_cm)
                            global_min_lids_w = np.append(blobs_log_min_width_cm, blobs_log_min_width_cm)

                            global_max_lids_h = np.append(blobs_log_max_height_cm, blobs_log_max[:, 2])
                            global_min_lids_h = np.append(blobs_log_min_height_cm, blobs_log_min[:, 2])

                            mean_of_image_inverted = np.append(mean_of_image_inverted, mean_inverted)
                            mean_of_map_place_cell = np.append(mean_of_map_place_cell, mean_normal_map)

                            print(f'{cnt}/201')

                        else:
                            continue
                    else:
                        continue
                else:
                    continue
            else:
                continue

    with h5py.File('14_big_Square_.hdf5', 'w') as data:
        data.create_dataset('max_blobs', data=global_max_lids)
        data.create_dataset('min_blobs', data=global_min_lids)
        data.create_dataset('blobs_log_max_width_cm', data=global_max_lids_w)
        data.create_dataset('blobs_log_min_width_cm', data=global_min_lids_w)
        data.create_dataset('blobs_log_max_height_cm', data=global_max_lids_h)
        data.create_dataset('blobs_log_min_height_cm', data=global_min_lids_h)
        data.close()

    stat_for_blobs(global_max_lids, cnt)
    stat_for_blobs(global_min_lids, cnt)
    stat_for_blobs(global_max_lids_w, cnt)
    stat_for_blobs(global_min_lids_w, cnt)
    stat_for_blobs(global_max_lids_h, cnt)
    stat_for_blobs(global_min_lids_h, cnt)


filepaths = select_files('/media/usb/Data/Transformed_CRCNC/hc-3')

main(filepaths)


