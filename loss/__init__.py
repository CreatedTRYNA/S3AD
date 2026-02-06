import glob
import importlib

from util.registry import Registry
LOSS = Registry('Loss')

files = glob.glob('loss/[!_]*.py')
import os
for file in files:


    module_path = os.path.splitext(os.path.normpath(file))[0].replace(os.sep, '.')
    importlib.import_module(module_path)


def get_loss_terms(loss_terms, device='cpu'):
	terms = {}
	for t in loss_terms:
		t = {k: v for k, v in t.items()}
		t_type = t.pop('type')
		t_name = t.pop('name')
		terms[t_name] = LOSS.get_module(t_type)(**t).to(device).eval()
	return terms
