import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from LSTM.util import device


def generate_ecg_data(duration_sec, sampling_rate=1000, noise_level=0.01):
    # Generate time axis
    t = np.linspace(0, duration_sec, int(duration_sec * sampling_rate), endpoint=False)

    # Generate ECG waveform
    ecg_waveform = (
            np.sin(2 * np.pi * 0.5 * t) +  # P wave
            np.sin(2 * np.pi * 1.0 * t) +  # QRS complex
            0.6 * np.sin(2 * np.pi * 1.5 * t) +  # T wave
            0.3 * np.sin(2 * np.pi * 2.0 * t)  # U wave
    )

    # Add noise
    noise = np.random.normal(0, noise_level, ecg_waveform.shape)
    ecg_with_noise = ecg_waveform + noise

    return t, ecg_with_noise


def extract_random_segment(array, sample_length: int = 2500):
    array_length = len(array)
    if array_length <= sample_length:
        raise ValueError("Array length must be greater than 1000.")

    # Generate a random starting index within the valid range
    start_index = np.random.randint(0, array_length - sample_length)

    # Extract a segment of length 1000 starting from the random index
    random_segment = array[start_index:(start_index + sample_length)]

    return random_segment


def generate_irregular_waveform(duration_sec, sampling_rate=1000, noise_level=0.01):
    # Generate time axis
    t = np.linspace(0, duration_sec, int(duration_sec * sampling_rate), endpoint=False)

    # Generate irregular waveform
    irregular_waveform = (
        np.sin(2 * np.pi * 0.5 * t) +  # Sinusoidal component
        np.cos(2 * np.pi * 0.3 * t)    # Random noise
    )

    # # Add noise
    noise = np.random.normal(0, noise_level, irregular_waveform.shape)
    waveform_with_noise = irregular_waveform + noise

    return t, waveform_with_noise


def get_data_loader(num_samples, feature_num, seq_length: int = 6000, batch_size: int = 10) -> DataLoader[tuple[Tensor, ...]]:
    _, ecg_data = generate_ecg_data(seq_length/1000, 1000)
    _, irregular_data = generate_irregular_waveform(6, 1000)
    x_train = []
    y_train = []
    for _ in range(num_samples):
        if np.random.rand() > 0.5:
            sequence = np.column_stack([extract_random_segment(ecg_data) for _ in range(feature_num)])
            label = 0
        else:
            sequence = np.column_stack([extract_random_segment(irregular_data) for _ in range(feature_num)])
            label = 1
        x_train.append(sequence)
        y_train.append(label)

    feature, label = np.array(x_train).astype(np.float32), np.array(y_train).astype(np.int64)
    dataset = TensorDataset(torch.from_numpy(feature).to(device), torch.from_numpy(label).to(device))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return data_loader


if __name__ == '__main__':
    # Generate ECG data for 10 seconds with default sampling rate and noise level
    duration = 8  # seconds
    sample_rate = 1000
    time_point, ecg_data = generate_ecg_data(duration, sample_rate)
    _, irregular_data = generate_irregular_waveform(duration, sample_rate)
    # print(ecg_data.shape)
    # print(extract_random_segment(ecg_data))
    # Plot the ECG data
    plt.figure(figsize=(duration, 4))
    # plt.plot(extract_random_segment(ecg_data, 3000))
    plt.plot(time_point, ecg_data, label='ECG Data')
    plt.plot(time_point, irregular_data, label='Irregular Data')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Simulated ECG Signal')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()
