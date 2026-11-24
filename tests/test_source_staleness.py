"""A store records each source file's hash at build; pack() never serves a changed file.

Sweep item 97: a node's snippet is cut from the live source file by the line
numbers stored at build. When the file changed since, those lines are someone
else's text under this node's name. The store now records a sha256 per source
file, and pack() omits the snippets of any file whose bytes no longer match,
with a warning. Stores written without hashes behave as before.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest

from kg_utils.extractor import KGExtractor
from kg_utils.pipeline import KGModule, read_lines, read_source
from kg_utils.specs import EdgeSpec, NodeSpec
from kg_utils.store import GraphStore

A_SRC = "def alpha():\n    return 'alpha'\n"
B_SRC = "def beta():\n    return 'beta'\n"


class _TwoFileExtractor(KGExtractor):
    """One function node per ``*.py`` file at the repo root, on lines 1-2."""

    def node_kinds(self) -> list[str]:
        return ["function"]

    def edge_kinds(self) -> list[str]:
        return []

    def meaningful_node_kinds(self) -> list[str]:
        return ["function"]

    def extract(self) -> Iterator[NodeSpec | EdgeSpec]:
        for f in sorted(self.repo_path.glob("*.py")):
            yield NodeSpec(
                node_id=f"function:{f.name}:{f.stem}",
                kind="function",
                name=f.stem,
                qualname=f.stem,
                source_path=f.name,
                lineno=1,
                end_lineno=2,
                docstring=f"The {f.stem} function returns its own name.",
            )


class _TwoFileKG(KGModule):
    _default_dir = ".twofilekg"

    def make_extractor(self) -> KGExtractor:
        return _TwoFileExtractor(self.repo_root)

    def kind(self) -> str:
        return "code"

    def analyze(self) -> str:
        return ""


def _repo(root: Path) -> Path:
    root.mkdir()
    (root / "a.py").write_text(A_SRC, encoding="utf-8")
    (root / "b.py").write_text(B_SRC, encoding="utf-8")
    return root


def _snippets(pack) -> dict[str, str]:
    return {n["module_path"]: n.get("snippet", {}).get("text", "") for n in pack.nodes}


# ---------------------------------------------------------------------------
# Store: recording and reading hashes (no embedder needed)
# ---------------------------------------------------------------------------


def _write_nodes(store: GraphStore) -> None:
    store.write(
        [
            NodeSpec("function:a.py:a", "function", "a", "a", "a.py", 1, 2, None),
            NodeSpec("function:b.py:b", "function", "b", "b", "b.py", 1, 2, None),
            NodeSpec("concept:x", "concept", "x", "x", "", None, None, None),
        ],
        [],
        wipe=True,
    )


def test_record_sources_hashes_each_distinct_file(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    with GraphStore(tmp_path / "graph.sqlite") as store:
        _write_nodes(store)
        assert store.source_hashes() is None
        assert store.record_sources(root) == 2
        assert store.source_hashes() == {
            "a.py": hashlib.sha256(A_SRC.encode()).hexdigest(),
            "b.py": hashlib.sha256(B_SRC.encode()).hexdigest(),
        }


def test_every_write_and_clear_forgets_the_hashes(tmp_path: Path) -> None:
    root = _repo(tmp_path / "repo")
    with GraphStore(tmp_path / "graph.sqlite") as store:
        _write_nodes(store)
        store.record_sources(root)
        store.write([], [], wipe=False)
        assert store.source_hashes() is None
        store.record_sources(root)
        store.clear()
        assert store.source_hashes() is None


def test_a_store_from_before_hashes_opens_unrecorded(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "old.sqlite"
    con = sqlite3.connect(db)
    con.execute(
        "CREATE TABLE nodes (id TEXT PRIMARY KEY, kind TEXT NOT NULL, name TEXT NOT NULL,"
        " qualname TEXT, module_path TEXT, lineno INTEGER, end_lineno INTEGER,"
        " docstring TEXT, metadata TEXT)"
    )
    con.commit()
    con.close()
    with GraphStore(db) as store:
        assert store.source_hashes() is None


# ---------------------------------------------------------------------------
# read_source matches read_lines
# ---------------------------------------------------------------------------


def test_read_source_decodes_like_read_lines(tmp_path: Path) -> None:
    f = tmp_path / "mixed.txt"
    data = b"one\r\ntwo\rthree\n\xff\xfebad bytes\n"
    f.write_bytes(data)
    lines, digest = read_source(f)
    assert lines == read_lines(f)
    assert digest == hashlib.sha256(data).hexdigest()
    assert read_source(tmp_path / "absent.txt") == ([], None)


# ---------------------------------------------------------------------------
# KGModule: build records, pack and stale_sources compare
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPackAfterAnEdit:
    @pytest.fixture
    def kg(self, tmp_path: Path) -> Iterator[_TwoFileKG]:
        kg = _TwoFileKG(_repo(tmp_path / "repo"))
        kg.build(wipe=True)
        yield kg
        kg.close()

    def test_a_fresh_build_packs_every_snippet_without_warnings(self, kg: _TwoFileKG) -> None:
        pack = kg.pack("function returns its name", k=4, max_nodes=None)
        assert _snippets(pack).keys() == {"a.py", "b.py"}
        assert all(_snippets(pack).values())
        assert not any("changed since" in w for w in pack.warnings)
        assert kg.stale_sources() == {}

    def test_an_edited_file_loses_its_snippet_and_warns(self, kg: _TwoFileKG) -> None:
        (kg.repo_root / "a.py").write_text("# a new first line\n" + A_SRC, encoding="utf-8")
        pack = kg.pack("function returns its name", k=4, max_nodes=None)
        snippets = _snippets(pack)
        assert snippets["a.py"] == ""
        assert "beta" in snippets["b.py"]
        assert [w for w in pack.warnings if "changed since" in w] == [
            "`a.py` changed since the graph was built; its snippets are omitted. Rebuild the graph."
        ]
        assert kg.stale_sources() == {"a.py": "changed"}

    def test_a_deleted_file_is_reported_missing(self, kg: _TwoFileKG) -> None:
        (kg.repo_root / "b.py").unlink()
        assert kg.stale_sources() == {"b.py": "missing"}

    def test_a_rebuild_makes_the_store_current_again(self, kg: _TwoFileKG) -> None:
        (kg.repo_root / "a.py").write_text("# a new first line\n" + A_SRC, encoding="utf-8")
        kg.build(wipe=True)
        assert kg.stale_sources() == {}
        assert "alpha" in _snippets(kg.pack("function returns its name", k=4))["a.py"]

    def test_a_store_written_without_hashes_packs_as_before(self, kg: _TwoFileKG) -> None:
        kg.store.write([], [], wipe=False)
        (kg.repo_root / "a.py").write_text("# a new first line\n" + A_SRC, encoding="utf-8")
        pack = kg.pack("function returns its name", k=4, max_nodes=None)
        assert all(_snippets(pack).values())
        assert not any("changed since" in w for w in pack.warnings)
        assert kg.stale_sources() is None
