# from __future__ import annotations

# import random
# import shutil
# from datetime import datetime
# from pathlib import Path
# from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

# import numpy as np
# import tensorflow as tf
# import torch
# from loguru import logger
# from torch.nn.utils.rnn import pad_sequence
# from tqdm import tqdm

# Tensor = torch.Tensor

# class BaseDataset:
#     """The main responsibility of the Dataset class is to load the data from disk
#     and to offer a __len__ method and a __getitem__ method
#     """

#   def __init__(self, paths: List[Path])-> None:
#     self.paths = paths
#     #random.shuffle(self.paths)
#     self.dataset = []
#     self.process_data()

# def process_data(self) -> None:
#         for file in tqdm(self.paths):
#             x = torch.tensor(file).type(torch.numeric)
#             y = torch.tensor(file).type(torch.nominal)
#             self.dataset.append((x, y))
 

#   def __iter__(self):
#     # startindex; this makes the first index 0
#     self.idx = -1
#     # we return the full object when iter() is called
#     return self

#   def __next__(self):
#     # for every iteration, __next__ is called
#     # as long as the idx is not bigger than the data
#     # we need to do -1, because we will increase idx directly after this
#     if self.idx < len(self.data) - 1:
#       self.idx += 1
#       return self.data[self.idx]
#     else:
#       raise StopIteration


# myclass = BaseIterator(n=5)
# myiter = iter(myclass) # this calles the __iter__ method and sets idx to -1

# for x in myiter: # this calls the __next__ method
#   print(x)


# class PaddedDatagenerator(BaseDataIterator):
#     # again, we inherit everything from the baseclass
#     def __init__(self, dataset: BaseDataset, batchsize: int) -> None:
#         # we initialize the super class BaseDataIterator
#         # we now have everything the BaseDataIterator can do, for free
#         super().__init__(dataset, batchsize)
    
#     def __next__(self) -> Tuple[Tensor, Tensor]:
#         if self.index <= (len(self.dataset) - self.batchsize):
#             X, Y = self.batchloop()
#             # we just want to add padding
#             X_ = pad_sequence(X, batch_first=True, padding_value=0)  # noqa N806
#             return X_, torch.tensor(Y)
#         else:
#             raise StopIteration

# class BaseDataIterator:
#     def __init__(self, datapath):
#         self.data = datapath
       
#         self.dataset = dataset
#         self.batchsize = batchsize
#     def __len__(self) -> int:
#         # the lenght is the amount of batches
#         return int(len(self.dataset) / self.batchsize)

#     def proces_data(self)
#      data = arff.loadarff(datapath)

#     def __iter__(self) -> BaseDataIterator:
#         # initialize index
#         self.index = 0
#         self.index_list = torch.randperm(len(self.dataset))
#         return self
    
#     def batchloop(self) -> Tuple[Tensor, Tensor]:
#         X = []  # noqa N806
#         Y = []  # noqa N806
#         # fill the batch
#         for _ in range(self.batchsize):
#             x, y = self.dataset[int(self.index_list[self.index])]
#             X.append(x)
#             Y.append(y)
#             self.index += 1
#         return X, Y

#     def __next__(self) -> Tuple[Tensor, Tensor]:
#         if self.index <= (len(self.dataset) - self.batchsize):
#             X, Y = self.batchloop()
#             return X, Y
#         else:
#             raise StopIteration


# class PaddedDatagenerator(BaseDataIterator):
#     # again, we inherit everything from the baseclass
#     def __init__(self, dataset: BaseDataset, batchsize: int) -> None:
#         # we initialize the super class BaseDataIterator
#         # we now have everything the BaseDataIterator can do, for free
#         super().__init__(dataset, batchsize)
    
#     def __next__(self) -> Tuple[Tensor, Tensor]:
#         if self.index <= (len(self.dataset) - self.batchsize):
#             X, Y = self.batchloop()
#             # we just want to add padding
#             X_ = pad_sequence(X, batch_first=True, padding_value=0)  # noqa N806
#             return X_, torch.tensor(Y)
#         else:
#             raise StopIteration

# class Dataloader:
#     """Point the dataloader to a directory.
#     It will load all files, taking the subfolders as classes.
#     Only files in the formatslist are kept.
#     The .datagenerator returns an Iterator of batched images and labels
#     """

#     def __init__(
#         self,
#         path: Path,
#     ) -> None:
#         """
#         Initializes the class

#         Args:
#             path (Path): location of the images
#             formats (List[str], optional): Formats to keep. Defaults to [".png", ".jpg"]
#         """

#         # get all paths
#         self.paths, self.class_names = iter_valid_paths(path, formats)
#         # make a dictionary mapping class names to an integer
#         self.class_dict: Dict[str, int] = {
#             k: v for k, v in zip(self.class_names, range(len(self.class_names)))
#         }
#         # unpack generator
#         self.valid_files = [*self.paths]
#         self.data_size = len(self.valid_files)
#         self.index_list = [*range(self.data_size)]

#         random.shuffle(self.index_list)

#         n_train = int(self.data_size * split)
#         self.train = self.index_list[:n_train]
#         self.test = self.index_list[n_train:]

#     def __len__(self) -> int:
#         return len(self.valid_files)

#     def data_generator(
#         self,
#         batch_size: int,
#         image_size: Tuple[int, int],
#         channels: int,
#         channel_first: bool,
#         mode: str,
#         shuffle: bool = True,
#         transforms: Optional[Callable] = None,
#     ) -> Iterator:
#         """
#         Builds batches of images

#         Args:
#             batch_size (int): _description_
#             image_size (Tuple[int, int]): _description_
#             channels (int): _description_
#             shuffle (bool, optional): _description_. Defaults to True.

#         Yields:
#             Iterator: _description_
#         """
#         if mode == "train":
#             data_size = len(self.train)
#             index_list = self.train
#         else:
#             data_size = len(self.test)
#             index_list = self.test

#         index = 0
#         while True:
#             # prepare empty matrices
#             X = torch.zeros(  # noqa: N806
#                 (batch_size, channels, image_size[0], image_size[1])
#             )  # noqa: N806
#             Y = torch.zeros(batch_size, dtype=torch.long)  # noqa: N806

#             for i in range(batch_size):
#                 if index >= data_size:
#                     index = 0
#                     if shuffle:
#                         random.shuffle(index_list)
#                 # get path
#                 file = self.valid_files[index_list[index]]
#                 # get image from disk
#                 if transforms is not None:
#                     img = self.load_image(file, image_size, channels)
#                     X[i] = transforms(img)
#                 else:
#                     X[i] = torch.tensor(self.load_image(file, image_size, channels))
#                 # map parent directory name to integer
#                 Y[i] = self.class_dict[file.parent.name]
#                 index += 1

#             if not channel_first:
#                 X = torch.permute(X, (0, 2, 3, 1))  # noqa N806

#             yield ((X, Y))