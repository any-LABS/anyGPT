import os
import numpy as np
from torch.utils.data import Dataset

from anyGPT.data.prepare_data import DEFAULT_DATADIR


class NextTokenDataset(Dataset):
    def __init__(self, dataset_name, type, block_size):
        if os.path.exists(dataset_name):
            self.data_file = dataset_name
        else:
            self.data_file = os.path.join(DEFAULT_DATADIR, dataset_name, f"{type}.bin")
        self.data = np.memmap(self.data_file, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, index):
        x = self.data[index : index + self.block_size].astype(np.int64)
        y = self.data[index + 1 : index + self.block_size + 1].astype(np.int64)
        return x, y
