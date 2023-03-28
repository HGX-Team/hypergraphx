import pkg_resources

from . import (
    core,
    communities,
    dynamics,
    generation,
    filter,
    linalg,
    measures,
    motifs,
    readwrite,
    representations,
    utils,
    viz,
)

__version__ = pkg_resources.get_distribution("hypergraphx").version

