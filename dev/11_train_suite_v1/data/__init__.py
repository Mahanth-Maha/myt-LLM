# data/__init__.py
"""
Data utilities for MYT-LLM training ecosystem.
"""

from .streaming_dataset import StreamingTextDataset
from .collate import collate_fn

__all__ = ['StreamingTextDataset', 'collate_fn']
