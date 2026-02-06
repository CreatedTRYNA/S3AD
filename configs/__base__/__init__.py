import glob
import importlib
from configs.__base__.cfg_common import cfg_common
from configs.__base__.cfg_dataset_default import cfg_dataset_default
import os

files = glob.glob('configs/__base__/[!_]*.py')
for file in files:
    module_path = os.path.splitext(os.path.normpath(file))[0].replace(os.sep, '.')
    module = importlib.import_module(module_path)
    for obj_name in dir(module):
        if obj_name.startswith("cfg_model"):
            globals()[obj_name] = getattr(module, obj_name)
