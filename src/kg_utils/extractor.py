"""kg_utils/types/extractor.py — Abstract base class for KG extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from kg_utils.specs import EdgeSpec, NodeSpec


class KGExtractor(ABC):
    """Abstract extraction protocol for any knowledge graph domain.

    Subclass and implement :meth:`node_kinds`, :meth:`edge_kinds`, and
    :meth:`extract`.  The concrete :class:`~kg_utils.pipeline.KGModule`
    infrastructure (:class:`~kg_utils.store.GraphStore`,
    :class:`~kg_utils.semantic.SemanticIndex`, snapshot management) is
    provided by the framework — you only implement domain-specific parsing.

    :param repo_path: Absolute path to the repository or corpus root.
    :param config: Optional domain-specific configuration dict.
    """

    def __init__(self, repo_path: Path, config: dict[str, Any] | None = None) -> None:
        self.repo_path = Path(repo_path).resolve()
        self.config = config or {}

    @abstractmethod
    def node_kinds(self) -> list[str]:
        """Return canonical node kind names emitted by this extractor.

        :return: List of node kind strings (e.g. ``['module', 'class', 'function']``).
        """

    @abstractmethod
    def edge_kinds(self) -> list[str]:
        """Return canonical edge relation types emitted by this extractor.

        :return: List of edge relation strings (e.g. ``['CONTAINS', 'CALLS']``).
        """

    @abstractmethod
    def extract(self) -> Iterator[NodeSpec | EdgeSpec]:
        """Traverse the source and yield NodeSpec / EdgeSpec objects.

        Implementations must be deterministic: the same source should produce
        the same stream on every call.

        :return: Iterator of :class:`NodeSpec` and :class:`EdgeSpec` objects.
        """

    def meaningful_node_kinds(self) -> list[str]:
        """Return node kinds included in the vector index and coverage metrics.

        Default: all of :meth:`node_kinds`.  Override to exclude structural
        stubs (e.g., unresolved import placeholders) from semantic indexing.

        :return: Subset of node_kinds() to index semantically.
        """
        return self.node_kinds()

    def coverage_metric(self, nodes: list[NodeSpec]) -> float:
        """Compute a domain coverage quality metric.

        Default: fraction of meaningful nodes with a non-empty docstring.

        :param nodes: All extracted NodeSpec objects.
        :return: Coverage score in [0.0, 1.0].
        """
        meaningful = [n for n in nodes if n.kind in self.meaningful_node_kinds()]
        if not meaningful:
            return 0.0
        covered = sum(1 for n in meaningful if n.docstring.strip())
        return covered / len(meaningful)
