"""Custom exceptions for cocode.

This module defines a hierarchy of exceptions for the cocode MCP server.
All exceptions inherit from CocodeError to allow for unified error handling.
"""


class CocodeError(Exception):
    """Base exception for all cocode errors."""


class IndexingError(CocodeError):
    """Raised when indexing a repository fails."""


class SearchError(CocodeError):
    """Raised when a search operation fails."""


class ConfigurationError(CocodeError):
    """Raised when configuration is invalid or missing."""


class PathError(CocodeError):
    """Raised when a file path is invalid or inaccessible."""
