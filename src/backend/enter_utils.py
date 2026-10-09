import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from scipy import fft
import keras
from keras.layers import Input, Dense
import tensorflow as tf
from keras import layers
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
try:
    import matlab.engine
except Exception:
    matlab = None
from scipy.signal import butter, filtfilt, argrelextrema, hilbert
from scipy.stats import kurtosis
import pandas as pd
import seaborn as sns
from keras.models import Model
import utils_explainability
import itertools


def _dataset_file_name(name):
    name = str(name)
    mapping = {
        "XJTU2-1": "XJTU_SY_2_1",
        "XJTU2-3": "XJTU_SY_2_3",
        "XJTU3-1": "XJTU_SY_3_1",
        "XJTU3-4": "XJTU_SY_3_4",
    }
    return mapping.get(name, name)


def _read_csv_rows(path, start=0, count=None, dtype=np.float32):
    with open(path, 'r') as f:
        end = None if count is None else start + int(count)
        lines = list(itertools.islice(f, int(start), end))
    if not lines:
        return np.array([])
    data = np.loadtxt(lines, delimiter=',', dtype=dtype)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def getDataset(name, samples, first_sample):
    file_name = _dataset_file_name(name)
    data_path = 'prog_analizador/data/' + file_name + '.csv'
    healthy_path = 'prog_analizador/data/healthy' + file_name + '.csv'
    data = _read_csv_rows(data_path, int(first_sample), int(samples))
    denoised_data = _read_csv_rows(healthy_path, 0, None)
    return data, denoised_data


def getDatasetNew(name, samples, first_sample, user):
    data_path = 'prog_analizador/saved_data/' + user + '/' + str(name) + '.csv'
    healthy_path = 'prog_analizador/saved_data/' + user + '/healthy' + str(name) + '.csv'
    data = _read_csv_rows(data_path, int(first_sample), int(samples))
    denoised_data = _read_csv_rows(healthy_path, 0, None)
    return data, denoised_data


def getDatasetTmp(name, samples, first_sample, user):
    data_path = 'prog_analizador/tmp/tmp' + str(name) + '.csv'
    healthy_path = 'prog_analizador/saved_data/' + user + '/healthy' + str(name) + '.csv'
    data = _read_csv_rows(data_path, int(first_sample), int(samples))
    denoised_data = _read_csv_rows(healthy_path, 0, None)
    return data, denoised_data


def getNMax(name, user):
    data = np.array(pd.read_csv('prog_analizador/saved_data/' + user + '/' +str(name), header=None, index_col=None))
    return data.shape[0]


def getNMaxTmp(name):
    data = np.array(pd.read_csv('prog_analizador/tmp/' +str(name), header=None, index_col=None))
    return data.shape[0]


def createModel(name, input_data):
    input_layer = Input(shape=(input_data.shape[0],))
    encoder = MonotonicityLayer2(units=3500)(input_layer)
    encoder = Dense(700, activation='sigmoid')(encoder)
    encoder = SmoothingLayer(200)(encoder)
    encoder = Dense(1, activation='sigmoid')(encoder)
    encoder = Model(input_layer, encoder)
    encoder.compile(optimizer='adam', loss='mse')
    tf.keras.models.save_model(encoder, str(name))
    return encoder


def build_ms2ae_table1(input_dim):
    """MS2AE Table 1: IS-3500-700-200-1-200-700-3500-IS."""
    input_layer = Input(shape=(input_dim,), name="input_signal")
    x = Dense(3500, activation='relu', name="encoder_3500")(input_layer)
    x = Dense(700, activation='relu', name="encoder_700")(x)
    x = Dense(200, activation='relu', name="encoder_200")(x)
    hi = Dense(1, activation='sigmoid', name="hi_output")(x)
    x = Dense(200, activation='relu', name="decoder_200")(hi)
    x = Dense(700, activation='relu', name="decoder_700")(x)
    x = Dense(3500, activation='relu', name="decoder_3500")(x)
    output_layer = Dense(input_dim, activation='linear', name="reconstruction")(x)

    autoencoder = Model(inputs=input_layer, outputs=output_layer, name="MS2AE_Table1")
    autoencoder.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse'
    )
    encoder = Model(inputs=input_layer, outputs=hi, name="MS2AE_Table1_Encoder")
    return autoencoder, encoder


def get_power_spectrum(data, fs, padding=0):
    xdft, freqs = _get_two_sided_amplitude_spectrum(data, fs, padding)

    xdft = xdft * xdft
    return xdft, freqs


def _get_two_sided_amplitude_spectrum(data, fs, n_padding):
    N = len(data) + n_padding
    xdft = fft.rfft(data, n=N, norm="forward")
    freqs = fft.rfftfreq(N, d=1./fs)
    return xdft, freqs


class MonotonicityLayer2(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(MonotonicityLayer2, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.mask = self.add_weight(name="adfsadfa", shape=input_shape[1:], initializer=tf.keras.initializers.Ones(), trainable=True)
        super(MonotonicityLayer2, self).build(input_shape)

    def call(self, inputs, **kwargs):
        masked_inputs = tf.multiply(inputs, self.mask)
        return masked_inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


def from_config(config):
    return MonotonicityLayer2(**config)


class SmoothingLayer(keras.layers.Layer):
    def __init__(self, window_size, **kwargs):
        super(SmoothingLayer, self).__init__(**kwargs)
        self.window_size = window_size

    def call(self, inputs):
        inputs = inputs
        smoothed_output = tf.nn.moments(inputs, axes=[1], keepdims=True)[1]
        return smoothed_output

    def get_config(self):
        config = super().get_config()
        config.update({"window_size": self.window_size})
        return config


def get_custom_objects():
    return {
        'MonotonicityLayer2': MonotonicityLayer2,
        'SmoothingLayer': SmoothingLayer,
        'from_config': from_config
    }


def load_analysis_model(model_dir, dataset_name, healthy_samples):
    """Load or train the MS2AE Table-1 encoder used to compute the HI."""
    model_dir = str(model_dir)
    windowed_autoencoder_path = os.path.join(model_dir, str(dataset_name) + '.windowed_ms2ae_autoencoder.keras')
    windowed_encoder_path = os.path.join(model_dir, str(dataset_name) + '.windowed_ms2ae_encoder.keras')
    ms2ae_encoder_path = os.path.join(model_dir, str(dataset_name) + '.ms2ae_encoder.keras')
    h5_path = os.path.join(model_dir, str(dataset_name) + '.h5')

    # Project rerun models use fixed 2048-point windows and sample-level P95
    # aggregation. Prefer reconstruction HI because the project tables and FFP
    # refresh were computed with recon_p95.
    if os.path.isfile(windowed_autoencoder_path):
        return keras.models.load_model(windowed_autoencoder_path, compile=False), 'windowed_recon_p95'

    if os.path.isfile(windowed_encoder_path):
        return keras.models.load_model(windowed_encoder_path, compile=False), 'windowed_latent_p95'

    if os.path.isfile(ms2ae_encoder_path):
        return keras.models.load_model(ms2ae_encoder_path, compile=False), 'direct_hi'

    if os.path.isfile(h5_path):
        return keras.models.load_model(h5_path, custom_objects=get_custom_objects(), compile=False), 'direct_hi'

    healthy_samples = np.asarray(healthy_samples, dtype=np.float32)
    autoencoder, encoder = build_ms2ae_table1(int(healthy_samples.shape[1]))
    autoencoder.fit(healthy_samples, healthy_samples, epochs=5, batch_size=64, verbose=0, shuffle=True)
    keras.models.save_model(encoder, ms2ae_encoder_path)
    return encoder, 'direct_hi'


def predict_hi(model, samples, mode):
    samples = np.asarray(samples, dtype=np.float32)
    if mode in ('windowed_recon_p95', 'windowed_latent_p95'):
        window_size = int(os.environ.get("MS2AE_WINDOW_SIZE", "2048"))
        usable_dim = (samples.shape[1] // window_size) * window_size
        if usable_dim <= 0:
            raise ValueError(f"MS2AE window_size={window_size} is larger than input_dim={samples.shape[1]}")
        windows_per_sample = usable_dim // window_size
        windows = samples[:, :usable_dim].reshape(samples.shape[0] * windows_per_sample, window_size)
        pred = model.predict(windows, verbose=0, batch_size=64)
        if mode == 'windowed_recon_p95':
            window_hi = np.mean((windows - np.asarray(pred)) ** 2, axis=1)
        else:
            window_hi = np.asarray(pred).reshape(-1)
        sample_hi = np.percentile(window_hi.reshape(-1, windows_per_sample), 95, axis=1)
        return np.asarray(sample_hi).reshape(-1, 1)

    pred = model.predict(samples, verbose=0)
    if mode == 'reconstruction':
        pred = np.asarray(pred)
        if pred.ndim == 3:
            pred = pred[:, :, 0]
        return np.mean((samples - pred) ** 2, axis=1).reshape(-1, 1)
    return np.asarray(pred).reshape(-1, 1)


def differenceSignals(signal1, signal2):
    signal1 = np.asarray(signal1, dtype=np.float32).flatten()
    signal2 = np.asarray(signal2, dtype=np.float32).flatten()
    if signal1.size == 0:
        return signal2
    if signal2.size == 0:
        return np.array([])

    if len(signal1) != len(signal2):
        fixed = np.zeros(len(signal2), dtype=np.float32)
        n = min(len(signal1), len(signal2))
        fixed[:n] = signal1[:n]
        if n < len(signal2):
            fixed[n:] = signal1[n - 1]
        signal1 = fixed

    try:
        _, path = fastdtw(signal1, signal2, dist=lambda x, y: abs(x - y))
        aligned = np.zeros(len(signal2), dtype=np.float32)
        counts = np.zeros(len(signal2), dtype=np.float32)
        for x, y in path:
            if 0 <= x < len(signal1) and 0 <= y < len(signal2):
                aligned[y] += signal1[x]
                counts[y] += 1
        missing = counts == 0
        counts[missing] = 1
        aligned = aligned / counts
        aligned[missing] = signal1[missing]
        return signal2 - aligned
    except Exception:
        return signal2 - signal1


def compute_pdf(time_series):
    hist, bin_edges = np.histogram(time_series, bins='auto', density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_centers, hist


def find_threshold_using_percentile(pdf_values, percentile):
    valid = np.asarray(pdf_values).flatten()
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return 1.0
    return np.percentile(valid, percentile)


def getThreshold(data):
    return find_threshold_using_percentile(data, 95)


def get_stage_thresholds(healthydata):
    """Paper Eqs. (5)-(6): thresholds from healthy HI spread."""
    healthydata = np.asarray(healthydata).flatten()
    early_threshold = find_threshold_using_percentile(healthydata, 95)
    max_healthy = float(np.max(healthydata)) if healthydata.size > 0 else early_threshold
    delta = max_healthy - early_threshold
    medium_threshold = delta * 50.0
    last_threshold = delta * 100.0
    return early_threshold, medium_threshold, last_threshold


def checkStage(HI_analyzed_samples, threshold):
    counter = 0
    faulty = False
    bk = None
    for index, elem in enumerate(np.asarray(HI_analyzed_samples).flatten()):
        if elem >= threshold:
            counter = counter + 1
            if counter >= 5:
                bk = index - 4
                faulty = True
                break
        else:
            counter = 0
    return faulty, bk


def generate_1_3_binary_tree_bands(fs):
    """Return Kurtogram bands with explicit 1/3-binary-tree level labels."""
    nyq = fs / 2.0
    level_parts = [
        (0.0, 1),
        (1.0, 2),
        (1.6, 3),
        (2.0, 4),
        (2.6, 6),
        (3.0, 8),
    ]
    bands = []
    for level, parts in level_parts:
        width = nyq / parts
        for index in range(parts):
            bands.append({
                "level": level,
                "index": index,
                "parts": parts,
                "low": index * width,
                "high": (index + 1) * width,
            })
    return bands


def computeKurtogram(data, fs, level):
    """
    1/3-binary-tree Kurtogram band selection following the paper:
    ignore levels 0, 1, and 1.6; evaluate levels 2, 2.6, and 3;
    discard the first and last band at each evaluated level.
    """
    data = np.asarray(data).flatten()
    nyq = fs / 2.0
    bands = generate_1_3_binary_tree_bands(fs)

    best_kurt = -np.inf
    best_band = (nyq / 4.0, nyq / 2.0)
    for band in bands:
        if band["level"] not in (2.0, 2.6, 3.0):
            continue
        if band["index"] == 0 or band["index"] == band["parts"] - 1:
            continue

        low, high = band["low"], band["high"]
        low_n = max(low / nyq, 0.001)
        high_n = min(high / nyq, 0.99)
        if low_n >= high_n:
            continue
        try:
            b, a = butter(4, [low_n, high_n], btype='band')
            filtered = filtfilt(b, a, data)
            env = np.abs(hilbert(filtered))
            k_val = kurtosis(env, fisher=True, nan_policy='omit')
            if np.isfinite(k_val) and k_val > best_kurt:
                best_kurt = k_val
                best_band = (low, high)
        except Exception:
            continue
    return best_band


def filteredFFT(order, fs, low_freq, high_freq, signal):
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    high = min(high, 0.99)
    low = max(low, 0.001)
    b, a = butter(order, [low, high], btype='band')
    filtered_signal = filtfilt(b, a, signal)
    envelope = np.abs(hilbert(filtered_signal))
    fft_env, freq = np.abs(get_power_spectrum(np.hanning(len(envelope))*envelope, fs, 0))
    fft_env[0:5] = 0
    return fft_env, freq


def getFilterBands(vals, fs, init_level):
    if isinstance(vals, tuple) and len(vals) == 2:
        return vals[0], vals[1]
    vals2 = [np.array(d) for d in vals]
    vals2 = np.stack(vals2, axis=1)[0]
    for elem in vals2:
        first = elem[0]
        last = elem[len(elem)-1]
        vals2[vals2 == first] = 0
        vals2[vals2 == last] = 0
    indices = np.where(vals2[init_level:] == vals2[init_level:].max())
    first_position = indices[1][0] % len(vals2[0])
    last_position = indices[1][len(indices[1])-1] % len(vals2[0])
    first_frequency = fs/2/len(vals2[0])*first_position
    last_frequency = fs/2/len(vals2[0])*(last_position+1)
    return first_frequency, last_frequency


def common_member(a, b):
    result = [i for i in a if i in b]
    return result


def determineFailure(ruta_carpeta, data, healthydata, hi_value, fs, fstart, fend, freq_interests, interval):
    bpfo_freq = [freq_interests[0], freq_interests[0]*2, freq_interests[0]*3, freq_interests[0]*4, freq_interests[0]*5, freq_interests[0]*6]
    bpfi_freq = [freq_interests[1], freq_interests[1]*2, freq_interests[1]*3, freq_interests[1]*4, freq_interests[1]*5, freq_interests[1]*6]
    bsf_freq = [freq_interests[2], freq_interests[2]*2, freq_interests[2]*3, freq_interests[2]*4, freq_interests[2]*5, freq_interests[2]*6]
    ftf_freq = [freq_interests[3], freq_interests[3]*2, freq_interests[3]*3, freq_interests[3]*4, freq_interests[3]*5, freq_interests[3]*6]
    freq_total = np.concatenate([bpfo_freq, bpfi_freq, bsf_freq, ftf_freq])

    fft_f, freqs = filteredFFT(4, fs, fstart, fend, data)
    indices = argrelextrema(fft_f, np.greater)
    amplitudes = []
    for elem in indices[0]:
        amplitudes.append(fft_f[elem])
    ind = np.argsort(amplitudes)[::-1][:10]
    top10 = indices[0][ind]
    important_f_real_peaks = []
    important_f_expected_peaks = []
    for freq in freq_total:
        for elem in top10:
            elem = int(elem*(freqs[1]-freqs[0]))
            if elem >= freq-interval and elem <= freq+interval:
                important_f_real_peaks.append(elem)
                important_f_expected_peaks.append(freq)

    fft_nf, freqs = get_power_spectrum(data, fs)
    indices = argrelextrema(fft_nf, np.greater)
    amplitudes = []
    for elem in indices[0]:
        amplitudes.append(fft_nf[elem])
    ind = np.argsort(amplitudes)[::-1][:10]
    top10 = indices[0][ind]
    important_nf_real_peaks = []
    important_nf_expected_peaks = []
    for freq in freq_total:
        for elem in top10:
            elem = int(elem*(freqs[1]-freqs[0]))
            if elem >= freq-interval and elem <= freq+interval:
                important_nf_real_peaks.append(elem)
                important_nf_expected_peaks.append(freq)

    bpfo_status = 0
    bpfi_status = 0
    bsf_status = 0
    ftf_status = 0
    if (len(important_f_expected_peaks) > 0):
        if (len(common_member(important_f_expected_peaks, bpfo_freq)) > 0):
            bpfo_status = bpfo_status+1
        if (len(common_member(important_f_expected_peaks, bpfi_freq)) > 0):
            bpfi_status = bpfi_status+1
        if (len(common_member(important_f_expected_peaks, bsf_freq)) > 0):
            bsf_status = bsf_status+1
        if (len(common_member(important_f_expected_peaks, ftf_freq)) > 0):
            ftf_status = ftf_status+1
    if (len(important_nf_expected_peaks) > 0):
        if (len(common_member(important_nf_expected_peaks, bpfo_freq)) > 0):
            bpfo_status = bpfo_status+2
        if (len(common_member(important_nf_expected_peaks, bpfi_freq)) > 0):
            bpfi_status = bpfi_status+2
        if (len(common_member(important_nf_expected_peaks, bsf_freq)) > 0):
            bsf_status = bsf_status+2
        if (len(common_member(important_nf_expected_peaks, ftf_freq)) > 0):
            ftf_status = ftf_status+2

    early_threshold, mid_threshold, last_threshold = get_stage_thresholds(healthydata)
    stop = False

    result = {
        'fault_detected': False,
        'fault_info': None,
        'fault_type': [],
        'fault_details': [],
        'analysis_result': None,
        'stage_thresholds': {
            'early_threshold': early_threshold,
            'medium_threshold': mid_threshold,
            'last_threshold': last_threshold
        }
    }

    if (hi_value < early_threshold):
        result['analysis_result'] = "No fault detected"
        stop = True
    elif (hi_value < mid_threshold):
        result['fault_info'] = "A fault has been detected in an early stage"
    elif (hi_value < last_threshold):
        result['fault_info'] = "A fault has been detected in a medium stage"
    else:
        result['fault_info'] = "A fault has been detected in a last degradation stage"

    if not stop:
        if (bpfo_status+bpfi_status+ftf_status+bsf_status == 0):
            result['analysis_result'] = "There is no failure detected, and the motor is completely healthy"
        else:
            result['fault_detected'] = True
            result['analysis_result'] = "A fault has been detected"

            if (bpfo_status > 0):
                result['fault_type'].append("Outer_race")
                if (len(common_member(important_f_expected_peaks, bpfo_freq))>0):
                    result['fault_details'].append(common_member(important_f_expected_peaks, bpfo_freq))
                    generateImg(common_member(important_f_expected_peaks, bpfo_freq), fft_f, freqs, ruta_carpeta, 1, "Outer-race")
                elif (len(common_member(important_nf_expected_peaks, bpfo_freq))>0):
                    result['fault_details'].append(common_member(important_nf_expected_peaks, bpfo_freq))
                    generateImg(common_member(important_nf_expected_peaks, bpfo_freq), fft_f, freqs, ruta_carpeta, 1, "Outer-race")
            if (bpfi_status > 0):
                result['fault_type'].append("Inner_race")
                if (len(common_member(important_f_expected_peaks, bpfi_freq))>0):
                    result['fault_details'].append(common_member(important_f_expected_peaks, bpfi_freq))
                    generateImg(common_member(important_f_expected_peaks, bpfi_freq), fft_f, freqs, ruta_carpeta, 2, "Inner-race")
                elif (len(common_member(important_nf_expected_peaks, bpfi_freq))>0):
                    result['fault_details'].append(common_member(important_nf_expected_peaks, bpfi_freq))
                    generateImg(common_member(important_nf_expected_peaks, bpfi_freq), fft_f, freqs, ruta_carpeta, 2, "Inner-race")
            if (bsf_status > 0):
                result['fault_type'].append("Bearing_Balls")
                if (len(common_member(important_f_expected_peaks, bsf_freq))>0):
                    result['fault_details'].append(common_member(important_f_expected_peaks, bsf_freq))
                    generateImg(common_member(important_f_expected_peaks, bsf_freq), fft_f, freqs, ruta_carpeta, 3, "Bearing Balls")
                elif (len(common_member(important_nf_expected_peaks, bsf_freq))>0):
                    result['fault_details'].append(common_member(important_nf_expected_peaks, bsf_freq))
                    generateImg(common_member(important_nf_expected_peaks, bsf_freq), fft_f, freqs, ruta_carpeta, 3, "Bearing Balls")
            if (ftf_status > 0):
                result['fault_type'].append("Cage")
                if (len(common_member(important_f_expected_peaks, ftf_freq))>0):
                    result['fault_details'].append(common_member(important_f_expected_peaks, ftf_freq))
                    generateImg(common_member(important_f_expected_peaks, ftf_freq), fft_f, freqs, ruta_carpeta, 4, "Cage")
                elif (len(common_member(important_nf_expected_peaks, ftf_freq))>0):
                    result['fault_details'].append(common_member(important_nf_expected_peaks, ftf_freq))
                    generateImg(common_member(important_nf_expected_peaks, ftf_freq), fft_f, freqs, ruta_carpeta, 4, "Cage")

    return result


def generateImg(members, fft_env, freq, carpeta, flag, name):
    plt.figure(figsize=(10.24, 7.68))

    for member in members:
        plt.axvline(member, color="red")

    last_member = int(members[-1])

    if (last_member*5 > 5000):
        plt.plot(freq[0:5000], fft_env[0:5000], color="blue")
    else:
        plt.plot(freq[0:last_member*5], fft_env[0:last_member*5], color="blue")

    plt.title("FFT " + name)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.savefig(os.path.join(carpeta, f'plot{flag}.png'))
    plt.close()


def matriz_full(bear3, hi_curve_ims1, ruta_carpeta, flag, sampling_frequency, shaft_frequency, BPFO, BPFI, BSF, FTF):
    num_subsamples = 16
    overlap = False
    percentage = 0.5
    res = utils_explainability.getCorrelationTime(bear3, hi_curve_ims1, num_subsamples, overlap, percentage, sampling_frequency, BPFO, BPFI, BSF, FTF, shaft_frequency)
    columns = utils_explainability.create_columnname(num_subsamples)
    index = ['HI', 'Fund. Filtered', 'BPFO Filtered', 'BPFI Filtered', 'FTF Filtered', 'BSF Filtered',
             'Fundamental', 'BPFO', 'BPFI', 'FTF', 'BSF']
    cm = pd.DataFrame(np.abs(res), columns=columns, index=index)
    plt.figure(figsize=(10.24, 7.68))
    sns.heatmap(cm, annot=False, cmap='coolwarm', vmin=0, vmax=1)
    plt.savefig(os.path.join(ruta_carpeta, f'plot{flag}.png'))
    plt.close()


def matriz_full2(bear3, hi_curve_ims1, ruta_carpeta, flag):
    num_subsamples = 16
    overlap = False
    percentage = 0.5
    res = utils_explainability.getTimeCorrelationTimeDomain(bear3, hi_curve_ims1, num_subsamples, overlap, percentage)
    columns = utils_explainability.create_columnname(num_subsamples)
    index = ['HI', 'RMS', 'Sk', 'K', 'CF', 'SF', 'IF', 'MF']
    cm = pd.DataFrame(np.abs(res), columns=columns, index=index)
    plt.figure(figsize=(10.24, 7.68))
    sns.heatmap(cm, annot=False, cmap='coolwarm', vmin=0, vmax=1)
    plt.savefig(os.path.join(ruta_carpeta, f'plot{flag}.png'))
    plt.close()


def matriz_simple(bear3, hi_curve_ims1, ruta_carpeta, flag):
    correlation_matrix = utils_explainability.getCorrelationTimeDomain(bear3, hi_curve_ims1)
    labels = ['HI', 'RMS', 'Sk', 'K', 'CF', 'SF', 'IF', 'MF']
    cm = pd.DataFrame(np.abs(correlation_matrix.values), columns=labels, index=labels)
    plt.figure(figsize=(10.24, 7.68))
    sns.heatmap(cm, annot=True, cmap='coolwarm', vmin=0, vmax=1)
    plt.savefig(os.path.join(ruta_carpeta, f'plot{flag}.png'))
    plt.close()
    first_column = correlation_matrix.iloc[:, 0]
    rounded_numbers = [round(num, 2) for num in first_column.tolist()]
    return rounded_numbers


def matriz_simple2(bear3, hi_curve_ims1, ruta_carpeta, flag, sampling_frequency, shaft_frequency, BPFO, BPFI, BSF, FTF):
    correlation_matrix = utils_explainability.getCorrelationFreqDomain(bear3, hi_curve_ims1, sampling_frequency, BPFO, BPFI, BSF, FTF, shaft_frequency)
    labels = ['HI', 'Fund. Filtered', 'BPFO Filtered', 'BPFI Filtered', 'FTF Filtered', 'BSF Filtered',
              'Fundamental', 'BPFO', 'BPFI', 'FTF', 'BSF']
    cm = pd.DataFrame(np.abs(correlation_matrix.values), columns=labels, index=labels)
    plt.figure(figsize=(10.24, 7.68))
    sns.heatmap(cm, annot=True, cmap='coolwarm', vmin=0, vmax=1)
    plt.savefig(os.path.join(ruta_carpeta, f'plot{flag}.png'))
    plt.close()
    first_column = correlation_matrix.iloc[:, 0]
    rounded_numbers = [round(num, 2) for num in first_column.tolist()]
    return rounded_numbers
