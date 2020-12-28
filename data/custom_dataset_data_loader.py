import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'parser':
        from data.data_stage1 import ParserDataset
        dataset = ParserDataset()
    elif opt.dataset_mode == 'cloth':
        from data.data_stage2 import ClothDataset
        dataset = ClothDataset()
    elif opt.dataset_mode == 'composer':
        from data.data_stage3 import ComposerDataset
        dataset = ComposerDataset()
    elif opt.dataset_mode == 'full':
        from data.data_all_stages import FullDataset
        dataset = FullDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
