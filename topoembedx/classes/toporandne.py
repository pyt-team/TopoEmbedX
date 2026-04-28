"""Deprecated compatibility layer for ComplexRandNE."""

from ._deprecation import deprecated_alias
from .complexrandne import ComplexRandNE

__all__ = ["TopoRandNE"]

TopoRandNE = deprecated_alias(
    ComplexRandNE,
    "topoembedx.classes.toporandne.TopoRandNE",
    "topoembedx.classes.complexrandne.ComplexRandNE",
)
