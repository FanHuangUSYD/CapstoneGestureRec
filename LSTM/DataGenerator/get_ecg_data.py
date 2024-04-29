import warnings
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from scipy import signal

from LSTM.util import device, plot_data_frequency


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


def generate_other_data(duration_sec, sampling_rate=1000, noise_level=0.01, mode: int = 1):
    # Generate time axis
    t = np.linspace(0, duration_sec, int(duration_sec * sampling_rate), endpoint=False)

    if mode == 1:
        other_waveform = (
            np.sin(2 * np.pi * 0.5 * t)
        )
    elif mode == 2:
        other_waveform = (
            np.sin(2 * np.pi * 0.5 * t) +  # Sinusoidal component
            np.cos(2 * np.pi * 0.3 * t)  # Random noise
        )
    elif mode == 3:
        other_waveform = (
            np.sign(np.sin(2 * np.pi * 0.5 * t))
        )
    elif mode == 4:
        other_waveform = (
            np.abs(signal.sawtooth(2 * np.pi * 0.5 * t))
        )
    else:
        raise ValueError(f"Unknown mode: {mode}, use mode = 1")

    # Add noise
    noise = np.random.normal(0, noise_level, other_waveform.shape)
    waveform_with_noise = other_waveform + noise
    return t, waveform_with_noise


def extract_random_segment(array, sample_length: int = 2500):
    array_length = len(array)
    if array_length < sample_length:
        raise ValueError(f"Array length must be greater or equal than {sample_length}.")

    # Generate a random starting index within the valid range
    start_index = np.random.randint(0, array_length - sample_length)

    # Extract a segment of length 1000 starting from the random index
    random_segment = array[start_index:(start_index + sample_length)]

    return random_segment


def get_data_loader(num_samples, feature_num, seq_length: int = 6000, batch_size: int = 10, data_class: int = 2,
                    tag: str = "TrainDataset") -> DataLoader[tuple[Tensor, ...]]:
    _, ecg_data = generate_ecg_data(seq_length / 1000, 1000)
    other_date_list = []
    for mode in range(1, data_class):
        _, other_data = generate_other_data(seq_length / 1000, 1000, mode=mode)
        other_date_list.append(other_data)
    x_train = []
    y_train = []
    for _ in range(num_samples):
        random_number = random.randint(0, data_class - 1)
        if random_number == 0:
            sequence = np.column_stack([extract_random_segment(ecg_data) for _ in range(feature_num)])
            label = 0
        else:
            sequence = np.column_stack([extract_random_segment(other_date_list[random_number - 1]) for _ in range(feature_num)])
            label = random_number
        x_train.append(sequence)
        y_train.append(label)

    feature, label = np.array(x_train).astype(np.float32), np.array(y_train).astype(np.int64)
    dataset = TensorDataset(torch.from_numpy(feature).to(device), torch.from_numpy(label).to(device))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    plot_data_frequency(y_train, ["ECG Data"] + ["Mock Data " + str(i) for i in range(1, data_class)], tag=tag)
    return data_loader


if __name__ == '__main__':
    # # Generate ECG data for 10 seconds with default sampling rate and noise level
    # duration = 8  # seconds
    # sample_rate = 1000
    # time_point, ecg_data = generate_ecg_data(duration, sample_rate)
    # other_date_list = []
    # for mode in range(1, 5):
    #     _, other_data = generate_other_data(duration, sample_rate, mode=mode)
    #     other_date_list.append(other_data)
    # # Plot the ECG data
    # plt.figure(figsize=(duration, 4))
    # # plt.plot(extract_random_segment(ecg_data, 3000))
    # plt.plot(time_point, ecg_data, label='ECG DataGenerator')
    # for index, other_data in enumerate(other_date_list):
    #     plt.plot(time_point, other_data, label=f'Mock DataGenerator {index+1}', alpha=0.7)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('Simulated ECG Signal')
    # plt.legend(loc='upper right')
    # plt.grid(True)
    # plt.show()
    get_data_loader(num_samples=6000, feature_num=2, seq_length=3000, batch_size=10,
                    data_class=5)
    get_data_loader(num_samples=1000, feature_num=2, seq_length=3000, batch_size=10,
                    data_class=5, tag="ValidationDataset")
