import importlib
import os
import glob

ALGO_DICT = {}


def register_algorithm(name):
    def decorator(cls):
        ALGO_DICT[name] = cls
        return cls
    return decorator


_current_dir = os.path.dirname(os.path.abspath(__file__))
_module_files = glob.glob(os.path.join(_current_dir, "*.py"))

for _f in _module_files:
    _module_name = os.path.basename(_f)[:-3]
    if _module_name.startswith("_"):
        continue
    importlib.import_module(f"kdflow.algorithms.{_module_name}")
