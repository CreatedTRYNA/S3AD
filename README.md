# S3AD

Official code for **S3AD: Efficient Industrial Anomaly Detection via Selection, Space, and Scale**.

## Environment

Use Linux, Python 3.10 and an NVIDIA GPU with a driver compatible with your PyTorch CUDA build.
The supplied environment snapshot pins PyTorch 2.8.0, torchvision 0.23.0 and timm 0.9.16:

```bash
python -m pip install -r requirements_my.txt
```

The snapshot also contains dependencies used by other experiments (including CUDA extensions).
It is not a minimal or universally portable lockfile; match the CUDA toolkit and driver when
building extensions. Apex is optional for the default S3AD configuration, which uses native
PyTorch distributed training. A clean installation and full paper-result reproduction still
require runtime validation on a CUDA machine.

## Data and teacher weights

Run commands from the repository root. Obtain the datasets separately. The default layout is:

```text
datasets/
  mvtec/       # MVTec AD class folders and meta.json
  visa/        # VisA images/masks matching data_json/visa/meta.json
  MPDD/
  Real-IAD/
model/pretrain/resnet34-43635321.pth
```

Copy the supplied `data_json/mvtec/meta.json` or `data_json/visa/meta.json` into the
corresponding dataset directory, and check that its relative image/mask paths match your
extracted data. Preserve the lowercase directory names `mvtec` and `visa`: the dataset loader
uses those names to select its format. MPDD and Real-IAD use their respective dataset readers.
Override the parent directory using `S3AD_DATA_ROOT`, or a single dataset using `data.root=...`.

The teacher is a frozen ImageNet-pretrained ResNet34. Download its weights if missing:

```bash
mkdir -p model/pretrain
curl -fL https://download.pytorch.org/models/resnet34-43635321.pth \
  -o model/pretrain/resnet34-43635321.pth
```

Training starts with a newly initialized student and neck; no trained S3AD checkpoint is
required. It does not train the teacher from scratch. Weights and datasets are not tracked in Git.

## Training

Single GPU (select the physical GPU through the environment):

```bash
CUDA_VISIBLE_DEVICES=0 python3 run.py -c configs/benchmark/s3ad/s3ad_visa.py -m train
```

For MVTec AD, use `configs/benchmark/s3ad/s3ad_mvtec.py`. Other configurations are
`s3ad_mdpp.py` (MPDD; historical filename) and `s3ad_realiad.py`.
All four select the `S3AD` model and their corresponding `S3AD*Trainer`.
The default MVTec/VisA recipe is 256 x 256, batch size 20, seed 42, AdamW at 0.001,
1000 epochs, and evaluation every 20 epochs (plus every epoch from `test_start_epoch`).

First run a short training check on a small class before launching the full recipe:

```bash
CUDA_VISIBLE_DEVICES=0 python3 run.py -c configs/benchmark/s3ad/s3ad_mvtec.py -m train \
  data.root=/path/to/mvtec "data.cls_names=['bottle']" \
  trainer.epoch_full=1 trainer.test_start_epoch=1 trainer.test_per_epoch=1 \
  trainer.scheduler_kwargs.warmup_epochs=0 trainer.scheduler_kwargs.decay_epochs=1 \
  trainer.data.batch_size=2 trainer.data.batch_size_per_gpu_test=2 \
  trainer.data.num_workers_per_gpu=0 fvcore_is=False
```

These overrides are a smoke check, not the paper recipe. Setting nested trainer fields changes
runtime settings; changing only a top-level field such as `epoch_full` after config construction
does not recompute derived fields.

Distributed launch (batch size 20 is global and must be divisible by the GPU count):

```bash
CUDA_VISIBLE_DEVICES=0,1 nproc_per_node=2 bash run.sh configs/benchmark/s3ad/s3ad_visa.py train
```

The launcher also accepts `nnodes`, `node_rank`, `master_addr` and `master_port` environment
variables. GPU visibility is never overwritten by the entry point. Multi-GPU execution requires
separate runtime validation.

## Download trained checkpoints

Download one checkpoint per dataset from the [GitHub Release](https://github.com/CreatedTRYNA/S3AD/releases/tag/checkpoints-img-auc-best-v1).
These checkpoints were selected by the highest category-averaged **image AUROC on the test set**
within each of the four identified training runs. Pixel-level metrics may peak at other epochs.

| Dataset | Checkpoint | Epoch | Image AUROC (%) | Pixel AUROC (%) | AUPRO (%) |
|---|---|---:|---:|---:|---:|
| MVTec AD | [IMG_AUC_Best_MVTecAD.pth](https://github.com/CreatedTRYNA/S3AD/releases/download/checkpoints-img-auc-best-v1/IMG_AUC_Best_MVTecAD.pth) | 360 | 99.056 | 97.924 | 92.816 |
| VisA | [IMG_AUC_Best_VisA.pth](https://github.com/CreatedTRYNA/S3AD/releases/download/checkpoints-img-auc-best-v1/IMG_AUC_Best_VisA.pth) | 640 | 94.897 | 98.960 | 91.950 |
| MPDD | [IMG_AUC_Best_MPDD.pth](https://github.com/CreatedTRYNA/S3AD/releases/download/checkpoints-img-auc-best-v1/IMG_AUC_Best_MPDD.pth) | 280 | 94.457 | 98.025 | 93.628 |
| Real-IAD | [IMG_AUC_Best_RealIAD.pth](https://github.com/CreatedTRYNA/S3AD/releases/download/checkpoints-img-auc-best-v1/IMG_AUC_Best_RealIAD.pth) | 1000 | 88.278 | 98.435 | 91.855 |

The table reports historical results at each selected epoch. The released Real-IAD checkpoint
has also passed a new full-test-set GPU evaluation (see below). The released MVTec AD, VisA and
MPDD checkpoints have not yet been newly evaluated on their full test sets; the recent full
checks for those datasets used epoch-1000 checkpoints instead. These files are not asserted to
be the paper's final or globally best checkpoints.

All files are model state dictionaries for the ResNet34 teacher and `[3, 9, 9, 7]` decoder,
compatible with code commit `fe26c42`. Each file is approximately 168.6 MiB.
[SHA256SUMS.txt](https://github.com/CreatedTRYNA/S3AD/releases/download/checkpoints-img-auc-best-v1/SHA256SUMS.txt)
contains checksums verified against the uploaded assets. The MPDD checkpoint is epoch **280**;
the historical log's “5600 epoch” label is a reporting error.

For example, download and verify the MVTec AD checkpoint:

```bash
mkdir -p model/pretrain
curl -fL https://github.com/CreatedTRYNA/S3AD/releases/download/checkpoints-img-auc-best-v1/IMG_AUC_Best_MVTecAD.pth \
  -o model/pretrain/IMG_AUC_Best_MVTecAD.pth
curl -fL https://github.com/CreatedTRYNA/S3AD/releases/download/checkpoints-img-auc-best-v1/SHA256SUMS.txt \
  -o model/pretrain/SHA256SUMS.txt
(cd model/pretrain && sha256sum --check --ignore-missing SHA256SUMS.txt)
```

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python3 run.py -c configs/benchmark/s3ad/s3ad_mvtec.py -m test \
  data.root=/path/to/mvtec \
  model.kwargs.checkpoint_path=model/pretrain/IMG_AUC_Best_MVTecAD.pth
```

Training saves logs and checkpoints under `runs/`. Use the same dataset and architecture
configuration for training and evaluation. Training code availability alone does not guarantee
reproduction of the reported paper metrics.

## Quick validation

A bounded check with real MVTec images (no full training run or checkpoint writes):

```bash
python3 tools/smoke_train.py --data-root /path/to/mvtec --device cpu
```

This checks all four model/trainer registrations, nested CLI overrides, data loading,
forward/backward, frozen teacher parameters, a student optimizer update and evaluation forward.
Use `--device cuda` to run the same check on an available GPU. The CPU check passed with
PyTorch 2.8.0 and timm 0.9.16. A complete new GPU training run has not been validated.

### Full GPU evaluation of epoch-1000 checkpoints

Using code commit `fe26c42`, PyTorch 2.8.0 and timm 0.9.16 on an NVIDIA GeForce RTX 4090,
all **118,930 test images across 63 categories** were evaluated. The model, dataset readers and
evaluator were taken from this repository; evaluation ran class by class to bound memory use.

| Dataset | Test images | Categories | Image AUROC (%) | Pixel AUROC (%) | AUPRO (%) |
|---|---:|---:|---:|---:|---:|
| MVTec AD | 1,725 | 15 | 98.912 | 98.025 | 93.043 |
| VisA | 2,162 | 12 | 94.638 | 98.968 | 91.302 |
| MPDD | 458 | 6 | 92.689 | 98.002 | 92.973 |
| Real-IAD | 114,585 | 30 | 88.277 | 98.435 | 91.855 |

All checkpoint loads were strict and all final metrics were finite. The largest difference
from the historical epoch-1000 averages for these three metrics was less than 0.01 percentage points.
Only the Real-IAD file in the download table is the same checkpoint as in this evaluation;
the other downloads use the earlier epochs selected for image AUROC.
