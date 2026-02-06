import argparse
from configs import get_cfg
from util.net import init_training
from util.util import run_pre, init_checkpoint
from trainer import get_trainer
import warnings
import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--cfg_path', default='configs/benchmark/omni/omniad_MHSA_visa.py')
    # parser.add_argument('-c', '--cfg_path', default='configs/benchmark/omni/omniad_MHSA_mvtec.py')
    # parser.add_argument('-c', '--cfg_path', default='configs/benchmark/omni/omniad_MHSA_mdpp.py')
    # parser.add_argument('-c', '--cfg_path', default='configs/benchmark/omni/omniad_MHSA_realiad.py')

    
    parser.add_argument('-m', '--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--sleep', type=int, default=-1)
    parser.add_argument('--memory', type=int, default=-1)
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--logger_rank', default=1, type=int, help='GPU id to use.')
    parser.add_argument('opts', help='path.key=value', default=None, nargs=argparse.REMAINDER, )
    cfg_terminal = parser.parse_args()
    cfg = get_cfg(cfg_terminal)
    run_pre(cfg)
    init_training(cfg)
    init_checkpoint(cfg)
    trainer = get_trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()
