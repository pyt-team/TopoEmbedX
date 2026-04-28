"""Deprecated compatibility layer for ComplexRep."""

from ._deprecation import deprecated_alias
from .complexrep import ComplexRep

__all__ = ["TopoRep"]

TopoRep = deprecated_alias(
    ComplexRep,
    "topoembedx.classes.toporep.TopoRep",
    "topoembedx.classes.complexrep.ComplexRep",
)
