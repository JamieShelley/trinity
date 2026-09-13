"""Legacy NSAMDR V9-V13 research and source-preparation package.

V14 is the production model/trainer. Importing this package must not install or mutate
runtime model contracts. The two historical base types remain importable so neutral
source-preparation tools and archived tests do not gain side effects merely by importing
``v9``.
"""

from .config import V9Config
from .model import FidelityResidualNetV9

__all__ = ["V9Config", "FidelityResidualNetV9"]
