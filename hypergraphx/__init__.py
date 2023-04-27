import pkg_resources

from . import (
    core,
    communities,
    dynamics,
    generation,
    filters,
    linalg,
    measures,
    motifs,
    readwrite,
    representations,
    utils,
    viz,
)

from .core import *
from .communities import *
from .dynamics import *
from .generation import *
from .filters import *
from .linalg import *
from .measures import *
from .motifs import *
from .readwrite import *
from .representations import *
from .utils import *
from .viz import *

__version__ = pkg_resources.get_distribution("hypergraphx").version

