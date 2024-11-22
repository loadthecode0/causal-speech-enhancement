from data.dataset import EARSWHAMAudioDataset # custom class earlier
from torch.utils.data import DataLoader

class EARSWHAMDataLoader:
    def __init__(self, base_dir="data/resampled/EARS-WHAM-16.0kHz", seg_length=16000, batch_size=8, num_workers=4, transform=None):
        self.base_dir = base_dir
        self.seg_length = seg_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def get_loader(self, split, shuffle=True):
        dataset = EARSWHAMAudioDataset(base_dir=self.base_dir, dataset=split, transform=self.transform, seg_length=self.seg_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=(shuffle if split == "train" else False), 
                            num_workers=self.num_workers, pin_memory=True)
        return loader

'''
# Usage
data_loader = EARSWHAMDataLoader(batch_size=8, seg_length=16000, num_workers=4)
train_loader = data_loader.get_loader("train")
valid_loader = data_loader.get_loader("valid", shuffle=False)
test_loader = data_loader.get_loader("test", shuffle=False)
'''