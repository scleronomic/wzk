from .__multiprocessing2 import *  # must be imported before multiprocessing / numpy

try:  # must be imported before skimage / did not find out why yet
    from pyOpt.pySLSQP.pySLSQP import SLSQP as _

except ImportError:
    pass


from .mpl.figure import *  # must be imported before matplotlib
from .mpl import plotting

from .binary import *
from .dicts_lists_tuples import *
from .files import *
from .functions import *
from .geometry import *
from .index import *
from .image import *
from .math2 import *
from .multiprocessing2 import *
from .numpy2 import *
from .object2 import *
from .perlin import *
from .printing import *
from .strings import *
from .spatial.transform import *
from .splines import *
from .testing import *
from .time import *
from .training import *

# modules which require additional repositories
# from .pyOpt2 import *
# from .ray2 import *
