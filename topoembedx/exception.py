"""Base errors and exceptions for TopoEmbedX."""

__all__ = ["TopoEmbedXError", "TopoEmbedXException", "TopoEmbedXNotImplementedError"]


class TopoEmbedXException(Exception):
    """Base class for exceptions in TopoEmbedX."""


class TopoEmbedXError(TopoEmbedXException):
    """Exception for a serious error in TopoEmbedX."""


class TopoEmbedXNotImplementedError(TopoEmbedXError):
    """Exception for methods not implemented for an object type."""
