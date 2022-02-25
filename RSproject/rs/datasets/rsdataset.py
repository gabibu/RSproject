

from typing import List
from torch.utils.data import DataLoader

class RSDataset:

    def __init__(self, fields_dimension: List[int], data_loader:DataLoader, test_loader: DataLoader):
        self.fields_dimension = fields_dimension
        self.train_data_loader = data_loader
        self.test_data_loader = test_loader
