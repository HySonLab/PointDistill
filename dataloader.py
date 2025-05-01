from functools import partial
import torch

from pointcept.datasets.builder import build_dataset
from pointcept.datasets.utils import collate_fn, point_collate_fn
from pointcept.engines.defaults import default_config_parser, default_setup, worker_init_fn
from pointcept.utils import comm


class PTv3_Dataloader:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # create training data loader
        self.init_fn = (
            partial(
                worker_init_fn,
                # num_workers=self.cfg.num_worker_per_gpu,
                num_workers=1,
                rank=comm.get_rank(),
                seed=self.cfg.seed,
            )
            if self.cfg.seed is not None
            else None
        )
        
    def load_training_data(self):
        self.train_data = build_dataset(self.cfg.data.train)
        
        # create training dataset
        if comm.get_world_size() > 1:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_data)
        else:
            self.train_sampler = None
        
        train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.cfg.batch_size,
            shuffle=(self.train_sampler is None),
            num_workers=self.cfg.num_worker,
            sampler=self.train_sampler,
            collate_fn=partial(point_collate_fn, mix_prob=self.cfg.mix_prob),
            pin_memory=True,
            worker_init_fn=self.init_fn,
            drop_last=True,
            persistent_workers=True,
        )
        return train_loader
    
    def load_validation_data(self):
        self.val_data = build_dataset(self.cfg.data.val)
        
        # create validation dataset
        if comm.get_world_size() > 1:
            self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_data)
        else:
            self.val_sampler = None
        
        val_loader = torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.cfg.batch_size_val,
            shuffle=False,
            num_workers=self.cfg.num_worker,
            pin_memory=True,
            sampler=self.val_sampler,
            collate_fn=collate_fn,
        )
        return val_loader

    def load_test_data(self):

        self.test_data = build_dataset(self.cfg.data.test)

        # create validation dataset
        if comm.get_world_size() > 1:
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(self.test_data)
        else:
            self.test_sampler = None

        test_loader = torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.cfg.batch_size_test_per_gpu,
            shuffle=False,
            num_workers=self.cfg.num_worker,
            pin_memory=True,
            sampler=self.test_sampler,
            collate_fn=collate_fn,
        )

        return test_loader