"""Check a real training batch without creating training logs or checkpoints.

Run from the repository root:
python tools/smoke_train.py --data-root /path/to/mvtec --device cpu
"""
import argparse
import importlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from configs import get_cfg
from data import get_loader
from loss import get_loss_terms
from model import get_model, MODEL
from trainer import TRAINER


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, help='MVTec AD root containing meta.json')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)

    for name in ('mvtec', 'visa', 'mdpp', 'realiad'):
        cfg = importlib.import_module(f'configs.benchmark.s3ad.s3ad_{name}').cfg()
        assert MODEL.get_module(cfg.model.name) is not None
        assert TRAINER.get_module(cfg.trainer.name) is not None
    cfg = get_cfg(argparse.Namespace(
        cfg_path='configs/benchmark/s3ad/s3ad_mvtec.py', mode='train',
        sleep=-1, memory=-1, dist_url='env://', logger_rank=0,
        opts=[f'data.root={args.data_root}', "data.cls_names=['bottle']",
              'trainer.data.batch_size_per_gpu=2',
              'trainer.data.batch_size_per_gpu_test=2',
              'trainer.data.num_workers_per_gpu=0',
              'trainer.data.pin_memory=False',
              'trainer.scheduler_kwargs.warmup_epochs=0']
    ))
    assert cfg.trainer.scheduler_kwargs['warmup_epochs'] == 0
    cfg.dist = False
    train_loader, test_loader = get_loader(cfg)
    batch = next(iter(train_loader))
    images = batch['img'].to(args.device)
    model = get_model(cfg.model).to(args.device).train()
    losses = get_loss_terms(cfg.loss.loss_terms, device=args.device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=cfg.optim.lr)
    parameter = next(p for p in model.net_s.parameters() if p.requires_grad)
    before = parameter.detach().clone()
    optimizer.zero_grad()
    teacher, student = model(images)
    assert [x.shape for x in teacher] == [x.shape for x in student]
    loss = losses['pixel'](teacher, student)
    assert torch.isfinite(loss), loss
    loss.backward()
    assert all(p.grad is None and not p.requires_grad for p in model.net_t.parameters())
    assert not model.net_t.training
    for part in (model.Neck, model.net_s):
        gradients = [p.grad for p in part.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
    optimizer.step()
    assert not torch.equal(before, parameter.detach()), 'Student parameter did not update'
    assert not hasattr(model, 'feats_t_cache')
    model.eval()
    with torch.no_grad():
        teacher, student = model(next(iter(test_loader))['img'].to(args.device))
        assert all(torch.isfinite(f).all() for f in teacher + student)
    print(f'PASS: all four registries, CLI overrides, real data, forward/backward, '
          f'frozen teacher, student update and eval; loss={loss.item():.6f}; device={args.device}')


if __name__ == '__main__':
    main()
