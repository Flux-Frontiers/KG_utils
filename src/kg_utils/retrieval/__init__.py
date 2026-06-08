"""Shared retrieval helpers for serializing and enriching KG hits."""

from kg_utils.retrieval.hits import attach_content_by_sqlite, hit_to_dict

__all__ = ["hit_to_dict", "attach_content_by_sqlite"]
