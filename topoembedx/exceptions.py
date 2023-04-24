# Copyright © 2022 Pyt-Team
# All rights reserved.

"""
Base classes for TopoEmbedX exceptions
"""


class TopoEmbedXException(Exception):
    """Base class for exceptions in TopoEmbedX."""


class TopoEmbedXError(TopoEmbedXException):
    """Exception for a serious error in TopoEmbedX"""


class TopoEmbedXNotImplementedError(TopoEmbedXError):
    """Exception for methods not implemented for an object type."""
