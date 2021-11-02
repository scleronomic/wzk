import numpy as np


__str2np = {'f128': np.float64,
            'f64': np.float64,
            'f32': np.float32,
            'f16': np.float16,
            'i64': np.int64,
            'i32': np.int32,
            'i16': np.int16,
            'i8': np.int8,
            'b': np.bool,
            'cmp': object,  # byte strings can have varying length and otherwise there is a fuck up
            'zlib': object,  # TODO does this make sense to add?
            }


def str2np(s: str, strip: bool = True):
    if strip:
        s = s.split('_')[-1]
    return __str2np[s]


def astype(a, s):
    return a.astype(str2np(s))


c2np = {bool: np.bool_,
        str: np.str_,
        int: np.integer,
        float: np.floating}
