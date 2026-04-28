"""Helpers for deprecated TopoEmbedX aliases."""

import inspect
import warnings
from typing import Any


def deprecated_alias(new_class: type[Any], old_name: str, new_name: str) -> type[Any]:
    """Create a deprecated class alias for a renamed public class.

    Parameters
    ----------
    new_class : type
        The new class that the alias will point to.
    old_name : str
        The fully qualified name of the old class (e.g.,
        ``topoembedx.classes.toponetmf.TopoNetMF``).
    new_name : str
        The fully qualified name of the new class (e.g.,
        ``topoembedx.classes.complexnetmf.ComplexNetMF``).

    Returns
    -------
    type
        A new class that is a deprecated alias for the new class. When instantiated, it
        will issue a deprecation warning.
    """

    class DeprecatedAlias(new_class):
        """Deprecated alias for a renamed public class.

        Parameters
        ----------
        *args : Any
            Positional arguments to pass to the new class initializer.
        **kwargs : Any
            Keyword arguments to pass to the new class initializer.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            warnings.warn(
                f"`{old_name}` is deprecated and will be removed in a future release. "
                f"Use `{new_name}` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            super().__init__(*args, **kwargs)

    DeprecatedAlias.__name__ = old_name.rsplit(".", maxsplit=1)[-1]
    DeprecatedAlias.__qualname__ = DeprecatedAlias.__name__
    DeprecatedAlias.__module__ = old_name.rsplit(".", maxsplit=1)[0]
    DeprecatedAlias.__signature__ = inspect.signature(new_class)

    return DeprecatedAlias
