import importlib
import numpy as np

NUMPY = 'numpy'
TORCH = 'torch'
TENSORFLOW = 'tensorflow'

DEFAULT_BACKEND = NUMPY
BACKEND = DEFAULT_BACKEND


class NumpyParameterClass(object):
    def __init__(self, *args, **kwargs):
        self._shape = None
        pass

    def _all_equal_shape(self, *args):
        if not args:
            return None
        shapes = map(lambda a: np.shape, args)
        shape = next(shapes)
        for s in shapes:
            if s != shape:
                raise ValueError('Mismatched data shapes.')
        self._shape = shape

    @property
    def shape(self):
        return self._shape

    @property
    def dim(self):
        return len(self._shape)


def set_backend(key):
    global BACKEND
    return importlib.import_module(key)


def set_default():
    set_backend(DEFAULT_BACKEND)


class Backend(object):
    def __init__(self, key):
        self._key = key
        self._outer_backend = BACKEND
        self._module = None

    def __enter__(self):
        if self._key == BACKEND:
            self._module = importlib.import_module(BACKEND)
        else:
            self._module = set_backend(self._key)
        return self._module

    def __exit__(self, exc_type, exc_val, exc_tb):
        set_backend(self._outer_backend)
        if self._key != BACKEND:
            del self._module


numpy = Backend(NUMPY)
torch = Backend(TORCH)
tensorflow = Backend(TENSORFLOW)
