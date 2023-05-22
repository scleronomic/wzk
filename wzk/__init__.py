import os
from . import __multiprocessing2   # must be imported before multiprocessing / numpy

try:  # must be imported before skimage / did not find out why yet
    from pyOpt.pySLSQP.pySLSQP import SLSQP as _

except ImportError:
    pass


from .mpl2.figure import new_fig  # must be imported before matplotlib

from .ltd import *
from .functions import *
from .geometry import *
from .index import *
from .math2 import *
from .np2 import *
from .object2 import *
from .printing import print2, print_dict, print_stats, print_table, check_verbosity, print_array_3d  # noqa
from .strings import *
from .time2 import tic, toc, tictoc, get_timestamp  # noqa

# modules which require additional repositories
# from .pyOpt2 import *
# from .ray2 import *
