import glob
import importlib
import os
from util.registry import Registry
TRAINER = Registry('Trainer')

files = glob.glob('trainer/[!_]*.py')
for file in files:
    module_path = os.path.splitext(os.path.normpath(file))[0].replace(os.sep, '.')
    importlib.import_module(module_path)


def get_trainer(cfg):
	return TRAINER.get_module(cfg.trainer.name)(cfg)
