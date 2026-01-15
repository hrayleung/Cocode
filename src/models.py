"""Shared data models for the cocode package."""

from dataclasses import dataclass


@dataclass
class SearchResult:
    """Search result with metadata."""

    filename: str
    location: str
    content: str
    score: float
