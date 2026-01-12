"""Custom exceptions for cocode."""


class CocodeError(Exception):
    """Base exception for cocode errors."""
    pass


class IndexingError(CocodeError):
    """Error during indexing."""
    pass


class SearchError(CocodeError):
    """Error during search."""
    pass


class ConfigurationError(CocodeError):
    """Error in configuration."""
    pass


class PathError(CocodeError):
    """Error with file path."""
    pass
