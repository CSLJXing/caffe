import os.path as osp
import sys

pylayer_path = osp.dirname(osp.abspath(__file__))
pycaffe_path = osp.join(pylayer_path, '..')

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

add_path(pycaffe_path)
add_path(pylayer_path)
