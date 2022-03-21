import torch.utils.data

from datasets.base_dataset import CreateDataset
from datasets.base_dataloader import BaseDataLoader
from configs.paths import dataroot

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset_mode = opt.dataset_mode
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            drop_last=True,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)


    def __iter__(self):
        for i, (data) in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data

def get_data_generator(loader):
    while True:
        for data in loader:
            yield data

def CreateDataLoader(opt, drop_last=True):
    train_dataset, test_dataset = CreateDataset(opt)

    train_dl = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            drop_last=drop_last,
            num_workers=int(opt.nThreads))

    test_dl = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=opt.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=int(opt.nThreads))

    return train_dl, test_dl
