"""Custom exceptions for cocode."""


class CocodeError(Exception):
    """Base exception."""


class IndexingError(CocodeError):
    """Indexing error."""


class SearchError(CocodeError):
    """Search error."""


class ConfigurationError(CocodeError):
    """Configuration error."""


class PathError(CocodeError):
    """File path error."""
