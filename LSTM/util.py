import os
from collections import Counter
from datetime import datetime

import torch
from matplotlib import pyplot as plt
from torch import device

ROOT = os.path.dirname(__file__)

def get_directory() -> str:
    now = datetime.now()
    month_day = now.strftime("%m-%d")

    directory_path = os.path.join(ROOT, "Output", month_day)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    return directory_path


def get_device() -> device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available! Using GPU.')
    else:
        device = torch.device('cpu')
        print('CUDA is not available. Using CPU.')
    return device


def plot_data_frequency(label_list: list[int], name_list: list[str] = None, tag: str = None) -> None:
    frequency_counter = Counter(label_list)
    sorted_items = sorted(frequency_counter.items(), key=lambda x: x[0])
    labels, counts = zip(*sorted_items)

    if name_list and max(labels) < len(name_list):
        labels = [name_list[label] for label in labels]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, counts)
    plt.xlabel(None)
    plt.ylabel('counts')
    plt.title(f'{tag if tag else "Dataset"}.png')
    plt.ylim(int(min(counts)*0.85), int(max(counts)*1.05))
    for index, count in enumerate(counts):
        plt.text(index, count + 0.1, str(count), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(os.path.join(get_directory(), f'class_frequency{"_in_" + tag if tag else ""}.png'))
    # plt.show()


device = get_device()
