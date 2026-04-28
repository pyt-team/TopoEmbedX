"""Test deprecated Topo* compatibility aliases."""

import importlib
import inspect
import sys

import pytest

import topoembedx as tex
import topoembedx.classes as tex_classes
from topoembedx.classes.complexnetmf import ComplexNetMF
from topoembedx.classes.complexrandne import ComplexRandNE
from topoembedx.classes.complexrep import ComplexRep


class TestDeprecatedTopoAliases:
    """Test deprecated Topo* aliases."""

    def test_deprecated_module_alias_warns(self):
        """Using the old module path should still work with a warning."""
        sys.modules.pop("topoembedx.classes.toponetmf", None)
        legacy_module = importlib.import_module("topoembedx.classes.toponetmf")

        with pytest.warns(DeprecationWarning, match="TopoNetMF"):
            model = legacy_module.TopoNetMF(dimensions=3)

        assert isinstance(model, ComplexNetMF)

    def test_deprecated_classes_alias_warns(self):
        """Using the old classes package alias should warn."""
        with pytest.warns(DeprecationWarning, match="TopoRandNE"):
            model = tex_classes.TopoRandNE(dimensions=3)

        assert isinstance(model, ComplexRandNE)

    def test_deprecated_top_level_alias_warns(self):
        """Using the old top-level alias should warn."""
        with pytest.warns(DeprecationWarning, match="TopoRep"):
            model = tex.TopoRep(dimensions=3, order=2)

        assert isinstance(model, ComplexRep)

    def test_deprecated_alias_preserves_signature(self):
        """Deprecated aliases should preserve constructor signatures."""
        assert inspect.signature(tex_classes.TopoNetMF) == inspect.signature(
            ComplexNetMF
        )
