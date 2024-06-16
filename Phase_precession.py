# Import ----------------------------------------------------------------------
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import lib  # Assuming lib has the necessary functions for processing
import scipy.fft
import Filter_fields as ff
from scipy.ndimage import gaussian_filter1d
from scipy.signal import chirp, find_peaks, peak_widths
import Plot_phase_precession as ppp



filepath = 'C:\\Users\\Nivahn\\PycharmProjects\\spike_sorting\\.pytest_cache/dataset.hdf5'
hdffile = h5py.File(filepath, mode='r')
# Open file -------------------------------------------------------------------
print(hdffile)
# get fs, x, y ----------------------------------------------------------------
fs_coords = hdffile['animalPosition'].attrs['coordinatesSampleRate']
x = hdffile['animalPosition/xOfFirstLed'][:]
y = hdffile['animalPosition/yOfFirstLed'][:]

# prepare coordinates ---------------------------------------------------------
x, cut_idxes = lib.prepare_coordinates(x)
y, _ = lib.prepare_coordinates(y)


# Get lfp, and spike_train ----------------------------------------------------
for electrode_name, electrode_values in hdffile.items():
    if electrode_name.find('electrode') == -1:
        continue

    if electrode_values.attrs['brainZone'] != 'CA1':
        continue

    if len(electrode_values['lfp'].keys()) < 2:
        continue
    print(electrode_name, electrode_values)

    for clusters_name in electrode_values['spikes']:
        print(clusters_name)

        if hdffile[f'{electrode_name}/spikes/{clusters_name}'].attrs['quality'] == 'Bad':
            print(hdffile[f'{electrode_name}/spikes/{clusters_name}'].attrs['quality'])
            continue

        if hdffile[f'{electrode_name}/spikes/{clusters_name}'].attrs['type'] == 'Int':
            print(hdffile[f'{electrode_name}/spikes/{clusters_name}'].attrs['type'])
            continue
        # Get spike_train -----------------------------------------------------
        spike_train = hdffile[f'{electrode_name}/spikes/{clusters_name}']["train"][:] / hdffile.attrs['samplingRate']
        print(spike_train.size)
        if spike_train.size < 600:
            continue
        
        print("spike_train - proceed")
        # Get fs_signal, fs_for_map -------------------------------------------
        
        fs_signal = electrode_values['lfp'].attrs['lfpSamplingRate']
        fs_for_map = int(fs_signal // fs_coords)
        dt = 1 / fs_for_map
        ms_of_coords = 1 / fs_for_map

        # Get lfp, clear artifacts --------------------------------------------
        for ch_idx, (channel_name, channel_data) in enumerate(electrode_values['lfp'].items()):
            lfp = 0.001 * channel_data[:].astype(np.float32)
            lfp = lib.clear_articacts(lfp)
        
        # Preapre lfp to size of x coords -------------------------------------
        t = np.linspace(0, lfp.size / fs_signal, lfp.size)
        lfp_size = lfp  
        lfp_size = np.asarray(lfp_size).astype(np.float32)
        lfp_size = lfp_size[(int(lfp_size.size) - (int(lfp_size.size//fs_for_map) * int(fs_for_map))):]
        lfp_size = np.reshape(lfp_size, (int(lfp_size.size//fs_for_map), int(fs_for_map)))
        lfp_size = np.mean(lfp_size, axis=1)
        lfp_size = lfp_size[cut_idxes[0] : cut_idxes[1]]
        
        # Get theta-rhytm -----------------------------------------------------
        range_lfp = lib.butter_bandpass_filter(lfp_size, 4, 12, fs_signal, 3)
        
        # Get analytic_signal, and phases -------------------------------------
        
        analytic_signal = lib.hilbert(lfp_size)
        theta_phase = np.angle(analytic_signal)
        
        print(electrode_name)
        print(electrode_values)
        print('Spike train size:', spike_train.size, "x.size", x.size)
        
        # Get spikes in time --------------------------------------------------
        spike_rate_bins = np.arange(0, x.size * ms_of_coords, ms_of_coords)
        spike_rate, _ = np.histogram(spike_train, bins=spike_rate_bins)
        spike_rate = spike_rate / ms_of_coords
        
        #spike_rate = gaussian_filter(spike_rate, sigma=3)
        
        # Filter for speed > 5 cm/s -------------------------------------------
        
        v_abs = np.sqrt((np.square(np.diff(x) / ms_of_coords)) + (np.square(np.diff(x) / ms_of_coords)))

        norm_v = np.argwhere(v_abs > 5)
        x_coords_ = x[norm_v]
        y_coords_ = y[norm_v]
        spike_rate_ = spike_rate[norm_v]

        x_coords_ = np.asarray(x_coords_)[:, 0]
        y_coords_ = np.asarray(y_coords_)[:, 0]
        spike_rate_ = np.asarray(spike_rate_)[:, 0]
        
        # Plot spikes over x, y -----------------------------------------------
        
        ppp.plot_spikes_and_x_y(x_coords_, y_coords_, spike_rate_)
        
        # Get map of spike rate -----------------------------------------------
        
        number_of_coords, bins_x, bins_y = np.histogram2d(x_coords_, y_coords_, bins=82)
        source_image, bins_x, bins_y = np.histogram2d(x_coords_, y_coords_, bins=82, density=False, weights=spike_rate_)
        
        # Prepare map, because of np.histogram2d ------------------------------
        
        source_image = np.flip(np.rot90(source_image), 0)
        source_image = gaussian_filter(source_image, sigma=3)
        
        # Criteria of place fields 
        min_peak_rate = 1  # Hz
        min_field_area = 25  # cm²
        fields, max_rate  = ff.find_fields(source_image, 0.9, min_peak_rate, min_field_area)
        print(fields)
        if fields is not None:
            labels, counts = np.unique(fields, return_counts=True)
            print(labels, counts)
            for (lab, count) in zip(labels, counts):
                print("lab", lab)
                if lab != 0:
                    
                    print("lab != 0")
                    
                    # Get field -----------------------------------------------
                    
                    field = np.where(fields == lab, fields, 0).T
                    
                    # Bins size -----------------------------------------------
                    
                    bs_x = bins_x[1] - bins_x[0]
                    bs_y = bins_y[1] - bins_y[0]
                    
                    # Center of field -----------------------------------------
                    
                    center_of_mass = np.unravel_index(np.argmax(source_image * field), source_image.shape)
                    center_x = (bins_x[center_of_mass[0]] + bins_x[center_of_mass[0] + 1]) / 2
                    center_y = (bins_y[center_of_mass[1]] + bins_y[center_of_mass[1] + 1]) / 2
                    
                    # Field eges, diameter ------------------------------------
                    
                    place_field_indices = np.argwhere(field)
                    min_x, min_y = np.min(place_field_indices, axis=0)
                    max_x, max_y = np.max(place_field_indices, axis=0)
                    diameter_x = (bins_x[max_x] - bins_x[min_x]) * bs_x
                    diameter_y = (bins_y[max_y] - bins_y[min_y]) * bs_y
                    
                    # Distances to center of field ----------------------------
                    
                    distances = np.sqrt((x_coords_ - center_x)**2 + (y_coords_ - center_y)**2)
                    
                    #indices_within_field = np.argwhere((distances <= max(diameter_x, diameter_y) / 2))
                    
                    # Spikes in coords ----------------------------------------
                    
                    spike_x_coords = x_coords_[spike_rate_ > 0]
                    spike_y_coords = y_coords_[spike_rate_ > 0]
                    
                    # Distances to center of field of spikes ------------------
                    
                    spike_distances = np.sqrt((spike_x_coords - center_x)**2 + (spike_y_coords - center_y)**2)
                    
                    # Diameter of field * -1, that min dist closer to 0 -------
                    
                    diameter = -1 * (diameter_x + diameter_y) / 2
                    print("diameter:", diameter)
                    
                    # Neg spikes dist -----------------------------------------
                    
                    spike_dist_neg = -1 * spike_distances
                    
                    # Find peaks ----------------------------------------------
                    peaks, _ = find_peaks(spike_dist_neg, height=-10, distance=1000)
                    print(f"diameter x:{diameter_x}", f"diameter y:{diameter_y}")
                    
                    # Sections around peaks -----------------------------------
                    sections_around_peaks = ff.extract_arr_around_peaks(spike_dist_neg, peaks, diameter)
                    
                    # Отображение участков вокруг пиков (опционально)
                    '''
                    plt.plot(spike_dist_neg)
                    plt.plot(peaks, spike_dist_neg[peaks], "x")
                    plt.plot(np.zeros_like(spike_dist_neg), "--", color="gray")
                    plt.show()
                    '''
                    
                    # Нахождение пиков в сглаженных отрицательных расстояниях до центра поля
                    distances_neg = -1 * gaussian_filter1d(distances, 6)
                    peaks, _ = find_peaks(distances_neg, height=-10, distance=1000)
                    x_around_peaks = ff.extract_arr_around_peaks(distances_neg, peaks, diameter)
                    
                    # Определение индексов в поле
                    indices_in_field = []
                    for arr in x_around_peaks:
                        if len(arr) > 1:
                            for value in arr:
                                indices_in_field.extend(np.where(distances_neg == value)[0])
                    
                    # Преобразование в массив NumPy
                    indices_in_field = np.array(indices_in_field, dtype=int)
                    
                    # Координаты и спайковые показатели для индексов в поле
                    field_x_coords = x_coords_[indices_in_field].astype(int)
                    field_y_coords = y_coords_[indices_in_field].astype(int)
                    field_spikes = spike_rate_[indices_in_field].astype(int)
                    
                    # Вычисление производных
                    derivatives_of_x = ff.second_derivative(x_around_peaks, ff.calculate_average_dx(x_around_peaks))
                    
                    # Инициализация списков для значений в поле и вне поля
                    into_field = []
                    out_of_field = []
                    
                    # Итерация по производным отрезков
                    for section, derivative in zip(x_around_peaks, derivatives_of_x):
                        if len(section) > 1:
                            into_field_tmp = []  # Создаем временный список для значений в поле
                            out_of_field_tmp = []  # Создаем временный список для значений вне поля
                            for idx, value in enumerate(derivative):
                                if idx + 1 < len(section):
                                    if value >= 0:
                                        into_field_tmp.append(section[idx + 1])  # Добавление значения из section в поле
                                    else:
                                        out_of_field_tmp.append(section[idx + 1])  # Добавление значения из section вне поля
                    
                            # Добавляем временные списки в общие списки
                            if len(into_field_tmp) > 4:
                                into_field.append(into_field_tmp)
                            else:
                                into_field.append([])
                            if len(out_of_field_tmp) > 4:
                                out_of_field.append(out_of_field_tmp)
                            else:
                                out_of_field.append([])
                    
                    # Инициализация списка для индексов в поле
                    indices_into_field = []
                    indices_out_of_field = []
                    
                    # Итерация по массиву into_field
                    for arr in into_field:
                        if len(arr) > 1:
                            for value in arr:
                                #print(f"value: {value}")
                                # Поиск индексов, где distances_neg равно значению value, и добавление их в список
                                indices_into_field.extend(np.where(distances_neg == value)[0])
                    
                    # Итерация по массиву into_field
                    for arr in out_of_field:
                        if len(arr) > 1:
                            for value in arr:
                                #print(f"value: {value}")
                                # Поиск индексов, где distances_neg равно значению value, и добавление их в список
                                indices_out_of_field.extend(np.where(distances_neg == value)[0])
                                
                                
                    
                    # Преобразование списка индексов в массив NumPy
                    indices_into_field = np.array(indices_into_field)
                    
                    spike_phases = theta_phase[indices_in_field].astype(float)
                    
                    # Выбираем только положительные значения фаз
                    spike_phases_1 = np.where(spike_phases > 0)[0]
                    
                    # Выбираем соответствующие координаты для выбранных положительных значений фаз
                    field_x_coords = x_coords_[spike_phases_1].astype(float)
                    
                    # Нормализуем позиции спайков между 0 и 1
                    field_min_x = np.min(x_coords_[spike_phases_1])
                    field_max_x = np.max(x_coords_[spike_phases_1])
                    normalized_positions = (field_x_coords - field_min_x) / (field_max_x - field_min_x)
                    
                    # Выбираем только положительные значения фаз
                    spike_phases = spike_phases[spike_phases_1]
                    
                    # Считаем фазы относительно нуля
                    spike_phases_from_zero = np.angle(np.exp(1j * spike_phases))
                    

                    # Координаты и фазы спайков для индексов в поле
                    field_x_coords = x_coords_[spike_phases_1].astype(float)
                    
                    # Нормализация позиций спайков между 0 и 1
                    field_min_x = np.min(x_coords_[spike_phases_1])
                    field_max_x = np.max(x_coords_[spike_phases_1])
                    
                    # Нормализация позиций
                    normalized_positions = (field_x_coords - field_min_x) / (field_max_x - field_min_x)
                    from scipy.stats import rayleigh, circmean
                    from scipy.optimize import curve_fit

                    rayleigh_p_value = ff.rayleightest(spike_phases_from_zero)
                    if rayleigh_p_value < 0.05:
                        print(f"Significant phase modulation detected (p = {rayleigh_p_value})")
                        # 3. Построить линейно-круговую регрессию для фазовой прецессии
                        def circular_linear_regression(x, a, b):
                            return a * x + b
                        
                        initial_guess = [1, 0]
                        params, _ = curve_fit(circular_linear_regression, normalized_positions, spike_phases_from_zero, p0=initial_guess)
                        slope, intercept = params
                        print(f"Circular-linear regression: slope = {slope}, intercept = {intercept}")
                        
                        # Построить график фазовой прецессии
                        fig, ax = plt.subplots(figsize=(10, 8))
                        ax.scatter(normalized_positions, spike_phases_from_zero, s=10, c='blue', alpha=0.5)
                        ax.plot(normalized_positions, circular_linear_regression(normalized_positions, slope, intercept), color='red')
                        ax.set_title(f'Phase Precession for {clusters_name} in {electrode_name} (Wrapped)')
                        ax.set_xlabel('Normalized Position in Place Field')
                        ax.set_ylabel('Theta Phase (wrapped)')
                        plt.show()
                    else:
                        print("No significant phase modulation detected")
                    
                    
                    
                    

                    