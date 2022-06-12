from __future__ import annotations

import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

# import arff file
from scipy.io import arff

# padding
import torch.nn.functional as F

import numpy as np
import tensorflow as tf
import torch


Tensor = torch.Tensor


class BaseDataset:
    """The main responsibility of the Dataset class is to load the data from disk
    and to offer a __len__ method and a __getitem__ method
    """

    def __init__(self, datapath: Path) -> None:
        self.path = datapath
        self.data = self.process_data()

    def process_data(self) -> None:
        data = arff.loadarff(self.path)
        cur_label = int(data[0][0][14])
        EEG_list = []  # Lege list voor observaties
        chunks = []  # Lege list voor chunks
        for obs in data[0]:
            if int(obs[14]) == cur_label:
                EEG_dim = [] # Lege list voor observaties binnen de forloop (behalve de labels)
                for index, i in enumerate(obs):
                    if index != 14:
                        EEG_dim.append(i)
                EEG_dim = torch.Tensor(EEG_dim)
                EEG_list.append(EEG_dim)
            else:
                chunks_label = (cur_label, torch.stack(EEG_list))
                chunks.append(chunks_label)
                cur_label = int(obs[14])
                EEG_list = []
                EEG_dim = [] # Lege list voor observaties binnen de forloop (behalve de labels)
                for index, i in enumerate(obs):
                    if index != 14:
                        EEG_dim.append(i)
                EEG_dim = torch.Tensor(EEG_dim)
                EEG_list.append(EEG_dim)
        chunks_label = (cur_label, torch.stack(EEG_list))
        chunks.append(chunks_label)
        return chunks

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple:
        return self.data[idx]


class BaseDataIterator_wind:
    """This iterator will consume all data and stop automatically.
    The dataset should have a:
        __len__ method
        __getitem__ method

    """

    def __init__(self, dataset: BaseDataset, window_size: int) -> None:
        self.dataset = dataset
        self.ws = window_size
        self.data = self.window()

    def __len__(self) -> int:
        return int(len(self.data))

    def window(self) -> None:
        data = self.dataset
        list2 = []
        ws = self.ws
        for i in range(len(self.dataset)):
            n_window = len(data[i][1])
            time = torch.arange(0, ws).reshape(1, -1)
            window = torch.arange(0, n_window).reshape(-1, 1)
            idx = time + window
            idx = idx - ws + 1
            test = data[i][1][idx]
            test2 = (data[i][0], test)
            list2.append(test2)
        return list2

    def __getitem__(self, idx: int) -> Tuple:
        return self.data[idx]


class BaseDataIterator_pad:
    """This iterator will consume all data and stop automatically.
    The dataset should have a:
        __len__ method
        __getitem__ method

    """

    def __init__(self, dataset: BaseDataset, window_size: int) -> None:
        self.dataset = dataset
        self.ws = window_size
        self.data = self.padding()

    def __len__(self) -> int:
        return len(self.data)

    def padding(self) -> None:
        data = self.dataset
        window_size = self.ws
        list_padded = []
        for i in range(len(data)):
            len_chunck = len(data[i][1])
            diff = len_chunck % window_size
            pad_value = window_size - diff
            if diff != 0:
                new_data = F.pad(
                    input=data[i][1], pad=(0, 0, 0, pad_value), mode="constant", value=0
                )
                new_data2 = (data[i][0], new_data)
                list_padded.append(new_data2)
            else:
                new_data3 = (data[i][0], data[i][1])
                list_padded.append(new_data3)
        return list_padded

    def __getitem__(self, idx: int) -> Tuple:
        return self.data[idx]


class BaseDataIterator_pad_wind:
    """This iterator will consume all data and stop automatically.
    The dataset should have a:
        __len__ method
        __getitem__ method

    """

    def __init__(self, dataset: BaseDataset, window_size: int) -> None:
        self.dataset = dataset
        self.ws = window_size
        self.data = self.padding_wind()

    def __len__(self) -> int:
        return len(self.data)

    def padding_wind(self) -> None:
        data = self.dataset
        window_size = self.ws
        list2 = []
        list_padded = []
        for i in range(24):
            len_chunck = len(data[i][1])
            diff = len_chunck % window_size
            pad_value = window_size - diff
            if diff != 0:
                new_data = F.pad(
                    input=data[i][1], pad=(0, 0, 0, pad_value), mode="constant", value=0
                )
                new_data2 = (data[i][0], new_data)
                list_padded.append(new_data2)
                n_window = len(list_padded[i][1])
                time = torch.arange(0, window_size).reshape(1, -1)
                window = torch.arange(0, n_window).reshape(-1, 1)
                idx = time + window
                idx = idx - window_size + 1
                test = list_padded[i][1][idx]
                test2 = (list_padded[i][0], test)
                list2.append(test2)
            else:
                new_data3 = (data[i][0], data[i][1])
                list_padded.append(new_data3)
                n_window = len(list_padded[i][1])
                time = torch.arange(0, window_size).reshape(1, -1)
                window = torch.arange(0, n_window).reshape(-1, 1)
                idx = time + window
                idx = idx - window_size + 1
                test = list_padded[i][1][idx]
                test2 = (list_padded[i][0], test)
                list2.append(test2)
        return list2

    def __getitem__(self, idx: int) -> Tuple:
        return self.data[idx]
