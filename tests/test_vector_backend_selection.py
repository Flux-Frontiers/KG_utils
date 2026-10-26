"""What vector backend a KGModule gets, now that there is one.

0.29.0 removed LanceDB: ``LanceDBBackend``, ``resolve_backend_name()``,
``make_backend()`` and the ``lancedb`` extra. These tests pin what replaced
the selection logic, and what was kept on purpose because consumers use it:
the ``vector_backend`` constructor argument (pycode_kg, tscode_kg, swift_kg
and ftree_kg pass ``"sqlite-vec"``) and the ``vector_backend`` key in
``stats()`` (pycode_kg and connectome_kg display it).

Nothing here loads a model or opens a store.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

import kg_utils.vector_backend as vb
from kg_utils.extractor import KGExtractor
from kg_utils.pipeline import KGModule
from kg_utils.specs import EdgeSpec, NodeSpec
from kg_utils.vector_backend import SqliteVecBackend


class _Extractor(KGExtractor):
    def node_kinds(self) -> list[str]:
        return ["function"]

    def edge_kinds(self) -> list[str]:
        return []

    def meaningful_node_kinds(self) -> list[str]:
        return ["function"]

    def extract(self) -> Iterator[NodeSpec | EdgeSpec]:
        yield NodeSpec(node_id="f", kind="function", name="f", qualname="f", source_path="m.py")


class _KG(KGModule):
    _default_dir = ".vbkg"

    def make_extractor(self) -> KGExtractor:
        return _Extractor(self.repo_root)

    def kind(self) -> str:
        return "test"

    def analyze(self) -> str:
        return "# test"


@pytest.mark.parametrize("name", ["auto", "sqlite-vec"])
def test_both_spellings_mean_sqlite_vec(tmp_path: Path, name: str) -> None:
    kg = _KG(tmp_path, vector_backend=name)
    kg.build_graph(wipe=True)
    try:
        assert kg.stats()["vector_backend"] == "sqlite-vec"
    finally:
        kg.close()


def test_lancedb_is_refused_naming_the_removal(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"removed in kgmodule-utils 0\.29\.0"):
        _KG(tmp_path, vector_backend="lancedb")


def test_an_unknown_backend_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="'auto' or 'sqlite-vec'"):
        _KG(tmp_path, vector_backend="faiss")


def test_a_stray_lancedb_dir_no_longer_changes_the_backend(tmp_path: Path) -> None:
    """Before 0.29.0, "auto" switched to LanceDB when this directory held anything."""
    (tmp_path / ".vbkg" / "lancedb" / "kg_nodes.lance").mkdir(parents=True)
    kg = _KG(tmp_path)
    kg.build_graph(wipe=True)
    try:
        assert kg.stats()["vector_backend"] == "sqlite-vec"
    finally:
        kg.close()


def test_the_backend_built_is_sqlite_vec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kg = _KG(tmp_path)

    class _Dim:
        dim = 8

    monkeypatch.setattr(type(kg), "embedder", property(lambda _self: _Dim()))
    backend = kg._make_vector_backend()
    assert isinstance(backend, SqliteVecBackend)


@pytest.mark.parametrize(
    "name", ["LanceDBBackend", "resolve_backend_name", "make_backend", "_pq_subvectors"]
)
def test_removed_names_are_gone(name: str) -> None:
    assert not hasattr(vb, name)
