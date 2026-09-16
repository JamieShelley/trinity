"""Minimal compatibility surface for the native Raven source-data builder.

The historical V9-V13 model stack has been retired. Only the small dataset
configuration object and Raven asset-name constants remain because the current
V16 source extractor still imports them while preserving old manifest formats.
"""

from .config import V9Config

__all__ = ["V9Config"]
