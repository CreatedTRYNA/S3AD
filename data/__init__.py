import glob
import importlib
import torch
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import random
from util.registry import Registry
from timm.data.distributed_sampler import RepeatAugSampler

TRANSFORMS = Registry('Transforms')
DATA = Registry('Data')

files = glob.glob('data/[!_]*.py')

for file in files:
    model_lib = importlib.import_module(file.split('.')[0].replace('/', '.'))

from data.utils import get_transforms


def get_dataset(cfg):
    train_transforms = get_transforms(cfg, train=True, cfg_transforms=cfg.data.train_transforms)
    test_transforms = get_transforms(cfg, train=False, cfg_transforms=cfg.data.test_transforms)
    target_transforms = get_transforms(cfg, train=False, cfg_transforms=cfg.data.target_transforms)
    train_set = DATA.get_module(cfg.data.type)(cfg, train=True, transform=train_transforms,
                                               target_transform=target_transforms)

    if cfg.data.type == 'PCBAD':
        test_sets = {}
        # default to all test sets if cls_names is empty
        test_names = cfg.data.cls_names if len(cfg.data.cls_names) > 0 else [f'test{i}' for i in range(6)]

        original_cls_names = cfg.data.cls_names
        for test_name in test_names:
            # Temporarily modify cfg to load only one test set
            cfg.data.cls_names = [test_name]
            test_sets[test_name] = DATA.get_module(cfg.data.type)(cfg, train=False, transform=test_transforms,
                                                                  target_transform=target_transforms)
        # Restore original cfg
        cfg.data.cls_names = original_cls_names
        return train_set, test_sets
    else:
        test_set = DATA.get_module(cfg.data.type)(cfg, train=False, transform=test_transforms,
                                                  target_transform=target_transforms)
        return train_set, test_set

# ===> 【新增函数：放在 get_loader 外面】 <===
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_loader(cfg):
    train_set, test_set_or_sets = get_dataset(cfg)

    # ===> 【新增：创建生成器】 <===
    g = torch.Generator()
    g.manual_seed(cfg.seed) # 使用全局种子

    if cfg.dist:
        if cfg.data.sampler == 'naive':
            sampler = DistributedSampler
        elif cfg.data.sampler == 'ra':
            sampler = RepeatAugSampler
        else:
            raise NotImplementedError("sampler '{}' is not implemented".format(cfg.data.sampler))
        train_sampler = sampler(train_set, shuffle=True)

    # test_sampler = sampler(test_set, shuffle=False)
    else:
        train_sampler = None
    # test_sampler = None

    # ===> 【修改：Train DataLoader】 <===
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=cfg.trainer.data.batch_size_per_gpu,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.trainer.data.num_workers_per_gpu,
        pin_memory=cfg.trainer.data.pin_memory,
        drop_last=cfg.trainer.data.drop_last,
        persistent_workers=cfg.trainer.data.persistent_workers,
        prefetch_factor=getattr(cfg.trainer.data, 'prefetch_factor', 2),
        worker_init_fn=seed_worker,  # <== 关键修改点 1
        generator=g                  # <== 关键修改点 2
    )
    if isinstance(test_set_or_sets, dict):
        test_loaders = {}
        for name, test_set in test_set_or_sets.items():
            if cfg.dist:
                sampler_ = sampler(test_set, shuffle=False)
            else:
                sampler_ = None

            test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                      batch_size=cfg.trainer.data.batch_size_per_gpu_test,
                                                      shuffle=False,
                                                      sampler=sampler_,
                                                      num_workers=cfg.trainer.data.num_workers_per_gpu,
                                                      pin_memory=cfg.trainer.data.pin_memory,
                                                      drop_last=False,
                                                      persistent_workers=cfg.trainer.data.persistent_workers,
                                                      prefetch_factor=getattr(cfg.trainer.data, 'prefetch_factor', 2),
                                                      worker_init_fn=seed_worker,  # <== 关键修改点 1
                                                      generator=g)                  # <== 关键修改点 2
            test_loaders[name] = test_loader
        return train_loader, test_loaders

    else:
        test_set = test_set_or_sets
        if cfg.dist:
            test_sampler = sampler(test_set, shuffle=False)
        else:
            test_sampler = None
        test_loader = torch.utils.data.DataLoader(dataset=test_set,
                                                  batch_size=cfg.trainer.data.batch_size_per_gpu_test,
                                                  shuffle=False,
                                                  sampler=test_sampler,
                                                  num_workers=cfg.trainer.data.num_workers_per_gpu,
                                                  pin_memory=cfg.trainer.data.pin_memory,
                                                  drop_last=False,
                                                  persistent_workers=cfg.trainer.data.persistent_workers,
                                                  prefetch_factor=getattr(cfg.trainer.data, 'prefetch_factor', 2),
                                                  worker_init_fn=seed_worker,  # <== 关键修改点 1
                                                  generator=g)                  # <== 关键修改点 2
        return train_loader, test_loader

