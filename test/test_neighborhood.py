"""Testing the neighborhood module."""

import pytest

import topoembedx as tex


class TestNeighborhood:
    """Test the neighborhood module of TopoEmbedX."""

    def test_value_error(self):
        """Testing if right assertion is raised for incorrect type."""
        with pytest.raises(TypeError) as e:
            tex.neighborhood.neighborhood_from_complex(1)

        assert (
            str(e.value)
            == """Input Complex can only be a Simplicial, Cell or Combinatorial Complex."""
        )
