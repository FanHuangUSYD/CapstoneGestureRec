import os
from datetime import datetime

import torch
from torch import device


def get_directory() -> str:
    now = datetime.now()
    month_day_hour = now.strftime("%m-%d")

    directory_path = os.path.join(os.getcwd(), month_day_hour)

    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")
    return directory_path


def get_device() -> device:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('CUDA is available! Using GPU.')
    else:
        device = torch.device('cpu')
        print('CUDA is not available. Using CPU.')
    return device


device = get_device()
