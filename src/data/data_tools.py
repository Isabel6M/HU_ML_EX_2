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
        self.window_size = window_size
        self.data = self.window()

    def __len__(self) -> int:
        return int(len(self.data))

    def window(self) -> None:
        data = self.dataset
        list_window = []
        window_size = self.window_size
        for i in range(len(self.dataset)):
                n_window = len(data[i][1]) - window_size + 1
                time = torch.arange(0, window_size).reshape(1, -1)
                window = torch.arange(0, n_window).reshape(-1, 1)
                idx = time + window
                windows = data[i][1][idx]
                window_data = (data[i][0], windows)
                list_window.append(window_data)  
        return list_window

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
        self.window_size = window_size
        self.data = self.padding()

    def __len__(self) -> int:
        return len(self.data)

    def padding(self) -> None:
        data = self.dataset
        window_size = self.window_size
        list_padding = []
        for i in range(len(data)):
            seq_len = len(data[i][1])
            diff = seq_len % window_size
            pading_num = window_size - diff
            if diff <= 0:
                old_data = (data[i][0], data[i][1])
                list_padding.append(old_data)
            else:
                new_data = F.pad(input=data[i][1], pad=(0, 0, 0, pading_num), mode="constant", value=0)
                new_data_comb = (data[i][0], new_data)
                list_padding.append(new_data_comb)
        return list_padding

    def __getitem__(self, idx: int) -> Tuple:
        return self.data[idx]


class BaseDataIterator_pad_wind:
    """This iterator will consume all data and stop automatically.
    The dataset should have a:
        __len__ method
        __getitem__ method

    """

    def __init__(self, dataset: BaseDataset, window_size: int, batchsize: int) -> None:
        self.dataset = dataset
        self.ws = window_size
        self.data = self.padding()
        self.data2 = self.window()
        self.batchsize = batchsize

    def __len__(self) -> int:
        return len(self.data2)
    
    def padding(self) -> None:
        data = self.dataset
        window_size = self.ws
        list_padding = []
        for i in range(len(data)):
            seq_len = len(data[i][1])
            diff = seq_len % window_size
            pading_num = window_size - diff
            if diff > 0:
                new_data = F.pad(input=data[i][1], pad=(0, 0, 0, pading_num), mode="constant", value=0)
                new_data_comb = (data[i][0], new_data)
                list_padding.append(new_data_comb)
            else:
                old_data = (data[i][0], data[i][1])
                list_padding.append(old_data)
        return list_padding

    def window(self) -> None:
        data = self.data
        list_window = []
        window_size = self.ws
        for i in range(len(self.dataset)):
                n_window = len(data[i][1]) - window_size + 1
                time = torch.arange(0, window_size).reshape(1, -1)
                window = torch.arange(0, n_window).reshape(-1, 1)
                idx = time + window
                windows = data[i][1][idx]
                window_data = (data[i][0], windows)
                list_window.append(window_data)
        return list_window  

    # def padding_wind(self) -> None:
    #     data = self.dataset
    #     window_size = self.ws
    #     list_window_pad = []
    #     list_padding = []
    #     list_not_modified = []
    #     total_list = []
    #     for i in range(len(data)):
    #         seq_len = len(data[i][1])
    #         diff = seq_len % window_size
    #         pad_value = window_size - diff
    #         if diff > 0:
    #             new_data = F.pad(input=data[i][1], pad=(0, 0, 0, pad_value), mode="constant", value=0)
    #             new_data2 = (data[i][0], new_data)
    #             list_padding.append(new_data2)
    #             n_window = len(list_padding[i][1]) 
    #             time = torch.arange(0, window_size).reshape(1, -1)
    #             window = torch.arange(0, n_window).reshape(-1, 1)
    #             idx = time + window
    #             windows = list_padding[i][1][idx] 
    #             window_data = (list_padding[i][0], windows)
    #             list_window_pad.append(window_data)
    #         else:
    #             n_window = len(data[i][1]) 
    #             time = torch.arange(0, window_size).reshape(1, -1)
    #             window = torch.arange(0, n_window).reshape(-1, 1)
    #             idx = time + window
    #             windows = data[i][1][idx] 
    #             window_data = (data[i][0], windows)
    #             list_not_modified.append(window_data)
    #     total = (list_window_pad,list_not_modified)
    #     total_list.append(total)
    #     return list_window_pad
                

    def __getitem__(self, idx: int) -> Tuple:
        return self.data2[idx]

    def __iter__(self) -> BaseDataIterator_pad_wind:
        self.index = 0
        self.index_list = torch.randperm(len(self.data2))
        return self

    def batchloop(self) -> Tuple[List, List]:
        X = []  # noqa N806
        Y = []  # noqa N806
        for _ in range(self.batchsize):
            x, y = self.data2[int(self.index_list[self.index])]
            X.append(x)
            Y.append(y)
            self.index += 1
        return X, Y

    def __next__(self) -> Tuple[Tensor, Tensor]:
        if self.index <= (len(self.data2) - self.batchsize):
            X, Y = self.batchloop()  # noqa N806
            return torch.tensor(X), torch.tensor(Y)
        else:
            raise StopIteration