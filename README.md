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

## Evaluation

```bash
CUDA_VISIBLE_DEVICES=0 python3 run.py -c configs/benchmark/s3ad/s3ad_visa.py -m test \
  model.kwargs.checkpoint_path=/path/to/net.pth
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
PyTorch 2.8.0 and timm 0.9.16; GPU training and final anomaly metrics have not been validated.
