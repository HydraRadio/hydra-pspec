from importlib.metadata import PackageNotFoundError, version

try:
    from ._version import version as __version__
except ModuleNotFoundError:  # pragma: no cover
    try:
        __version__ = version("hydra_pspec")
    except PackageNotFoundError:
        # package is not installed
        __version__ = "unknown"

from . import dpss, lssa, oqe, pspec, utils
