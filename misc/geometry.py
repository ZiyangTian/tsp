import collections
import functools
import typing
import numpy as np


class Coordinate3D(collections.namedtuple('Coordinate3D', ('x', 'y', 'z'))):
    def __new__(cls, *args, **kwargs):
        np_float_scalar = functools.partial(np.array, dtype=np.float)
        if len(args) == 1 and not kwargs:
            arg = args[0]
            if isinstance(arg, cls):
                return arg
            if isinstance(arg, (typing.Sized, typing.Sequence)):
                return super(Coordinate3D, cls).__new__(Coordinate3D, *map(np_float_scalar, arg))

        args = tuple(map(np_float_scalar, args))
        for k, v in kwargs.items():
            kwargs[k] = np_float_scalar(v)
        self = super(Coordinate3D, cls).__new__(Coordinate3D, *args, **kwargs)
        if not self.x.shape == self.y.shape == self.z.shape:
            raise ValueError('`x`, `y` and `z` must be of the same shape, got {}, {}, and {}.'.format(
                self.x.shape, self.y.shape, self.z.shape))
        return self

    def __add__(self, other):
        other = type(self)(other)
        return Coordinate3D(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z)

    def __neg__(self):
        return type(self)(-self.x, -self.y, -self.z)

    def __sub__(self, other):
        return self + type(self)(other).__neg__()

    def __mul__(self, other):
        return type(self)(self.x * other, self.y * other, self.z * other)

    def __truediv__(self, other):
        return self.__mul__(1 / other)

    def dot(self, other):
        other = type(self)(other)
        return self.x * other.x + self.y * other.y + self.z * other.z

    @property
    def shape(self):
        return self.x.shape

    @property
    def ndim(self):
        return self.x.ndim

    @property
    def modulus(self):
        return np.sqrt(np.sum(np.square(self.numpy), axis=-1))

    @property
    def slope_in_plane(self):
        return np.where(
            np.logical_and(np.equal(self.x, 0), np.equal(self.y, 0)),
            np.zeros(self.shape),
            np.true_divide(self.y, self.x))

    @property
    def numpy(self):
        if self.shape is ():
            return np.array([self.x, self.y, self.z])
        return np.concatenate([self.x, self.y, self.z], axis=-1)

    def should_be_scalar(self):
        return self.has_shape(())

    def has_shape(self, shape):
        if self.shape != tuple(shape):
            raise AssertionError('Data should have shape {}, got shape {} instead.'.format(shape, self.shape))
        return self


class Point3D(Coordinate3D):
    pass


class Vector3D(Coordinate3D):
    pass


class Ellipsoid(collections.namedtuple('Ellipsoid', ('centre', 'a', 'b', 'c'))):
    def __new__(cls, centre, a, b, c):
        centre = Point3D(centre)
        a = np.broadcast_to(np.array(a), centre.shape)
        b = np.broadcast_to(np.array(b), centre.shape)
        c = np.broadcast_to(np.array(c), centre.shape)
        return super(Ellipsoid, cls).__new__(cls, centre, a, b, c)

    def get_location(self, vector: Vector3D):
        return self.centre + vector * (self.a, self.b, self.c)


class Solution(object):
    def __init__(self, index, vector):
        """

        :param index: (..., num_regions)
        :param vector: (..., num_regions, len_vector)
        """
        self.index = np.array(index)
        self.vector = np.array(vector)

    @classmethod
    def from_structured_data(cls, data: Tuple[int, List[float]]):
        index, vector = zip(*data)
        return cls(index, vector)
