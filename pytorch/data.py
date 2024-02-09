from torch.utils.data import DataLoader

from datasets import *
from utils import *
from torchvision import transforms, datasets
import torch
import torchio.transforms
import copy
import numpy

class DataGenerator:

    def __init__(self, args):
        self.args = args

    def brats_finetune(self):
        args = self.args
        dataloader = {}
        train_list, val_list, test_list = get_brats_list(self.args.data, self.args.ratio)
        train_ds = BratsFinetune(train_list, train=True)
        val_ds = BratsFinetune(val_list, train=False)
        test_ds = BratsFinetune(test_list, train=False)

        generator = torch.Generator()
        generator.manual_seed(args.seed)
        dataloader['train'] = DataLoader(train_ds, batch_size=self.args.b,
                                         num_workers=self.args.workers,
                                         worker_init_fn=seed_worker,
                                         generator=generator,
                                         pin_memory=True,
                                         shuffle=True)
        dataloader['eval'] = DataLoader(val_ds, batch_size=self.args.b,
                                        num_workers=self.args.workers,
                                        worker_init_fn=seed_worker,
                                        generator=generator,
                                        pin_memory=True,
                                        shuffle=False)
        dataloader['test'] = DataLoader(test_ds, batch_size=1, num_workers=self.args.b,
                                        worker_init_fn=seed_worker,
                                        generator=generator,
                                        pin_memory=True,
                                        shuffle=False)
        return dataloader