"""Deprecated compatibility layer for ComplexNetMF."""

from ._deprecation import deprecated_alias
from .complexnetmf import ComplexNetMF

__all__ = ["TopoNetMF"]

TopoNetMF = deprecated_alias(
    ComplexNetMF,
    "topoembedx.classes.toponetmf.TopoNetMF",
    "topoembedx.classes.complexnetmf.ComplexNetMF",
)
