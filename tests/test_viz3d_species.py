"""
Tests for kg_utils.viz3d.species — growth habits and the species presets.

The invariants a habit must never break are the ones the data relies on: every
chunk still hangs on wood, the default habit grows exactly the tree the engine
grew before habits existed, and a species only changes *where* the crown sits
and *how* the wood reaches it.
"""

from __future__ import annotations

import numpy as np
import pytest

from kg_utils.viz3d import (
    CROWN_TOP,
    DEFAULT_HABIT,
    ENVELOPES,
    SPECIES,
    Habit,
    colonize,
    crown_sections,
    crown_spacing,
    droop_skeleton,
    envelope_width,
    grow_tree,
    pipe_radii,
    section_cluster,
    species_table,
    vary_habit,
)

HEIGHT = 20.0
BRANCH = 6.0


def _crown(habit: Habit, n_sections: int = 20, per: int = 20) -> np.ndarray:
    """A book-like crown: *n_sections* limbs of *per* chunks each, placed by *habit*."""
    tips = crown_sections(n_sections, HEIGHT, BRANCH, habit)
    return np.vstack(
        [section_cluster(per, t, np.array([0.0, 0.0, t[2]]), 1.5, habit) for t in tips]
    )


def _default_step(pts: np.ndarray, habit: Habit) -> float:
    """colonize's default internode for a crown of at most 512 points."""
    extent = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
    return habit.step * max(extent / 40.0, 0.5 * crown_spacing(pts))


class TestEnvelopes:
    @pytest.mark.parametrize("envelope", ENVELOPES)
    def test_width_is_a_positive_fraction(self, envelope: str) -> None:
        widths = [envelope_width(envelope, s) for s in np.linspace(0.0, 1.0, 21)]
        assert all(isinstance(w, float) for w in widths)
        assert all(0.0 < w <= 1.0 + 1e-9 for w in widths)

    def test_shapes_differ_where_they_should(self) -> None:
        # A cone is widest at the bottom, a vase at the top, an ellipsoid mid-crown.
        assert envelope_width("cone", 0.0) > envelope_width("cone", 1.0)
        assert envelope_width("vase", 1.0) > envelope_width("vase", 0.0)
        assert envelope_width("ellipsoid", 0.5) > envelope_width("ellipsoid", 0.0)

    def test_unknown_envelope_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="Unknown envelope"):
            envelope_width("blob", 0.5)
        with pytest.raises(ValueError, match="Unknown envelope"):
            Habit(envelope="blob")


class TestCrownSections:
    @pytest.mark.parametrize("name", sorted(SPECIES))
    def test_sections_sit_inside_the_envelope(self, name: str) -> None:
        habit = SPECIES[name]
        tips = crown_sections(25, HEIGHT, BRANCH, habit)
        assert tips.shape == (25, 3)
        z = tips[:, 2]
        assert z.min() == pytest.approx(HEIGHT * habit.clear_bole)
        assert z.max() == pytest.approx(HEIGHT * CROWN_TOP)
        s = (z - z.min()) / (z.max() - z.min())
        radial = np.hypot(tips[:, 0], tips[:, 1])
        expected = [BRANCH * habit.width * envelope_width(habit.envelope, t) for t in s]
        np.testing.assert_allclose(radial, expected, rtol=1e-9)

    def test_whorls_share_a_height(self) -> None:
        tips = crown_sections(12, HEIGHT, BRANCH, Habit(whorl=4))
        assert len(np.unique(np.round(tips[:, 2], 9))) == 3

    def test_base_offsets_the_crown(self) -> None:
        here = crown_sections(5, HEIGHT, BRANCH, SPECIES["oak"])
        there = crown_sections(5, HEIGHT, BRANCH, SPECIES["oak"], base=(10.0, -4.0))
        np.testing.assert_allclose(there - here, np.tile([10.0, -4.0, 0.0], (5, 1)))

    def test_no_sections(self) -> None:
        assert crown_sections(0, HEIGHT, BRANCH).shape == (0, 3)


class TestSectionCluster:
    def test_spread_and_lift(self) -> None:
        tip = np.array([4.0, 0.0, 10.0])
        axis = np.array([0.0, 0.0, 10.0])
        base = section_cluster(40, tip, axis, 1.0)
        wide = section_cluster(40, tip, axis, 1.0, Habit(spread=2.0))
        hung = section_cluster(40, tip, axis, 1.0, Habit(lift=-1.0))
        assert np.linalg.norm(wide - tip, axis=1).max() == pytest.approx(
            2 * np.linalg.norm(base - tip, axis=1).max()
        )
        assert hung[:, 2].mean() < base[:, 2].mean() - 0.5

    def test_empty(self) -> None:
        assert section_cluster(0, np.zeros(3), np.zeros(3), 1.0, Habit(lift=2.0)).shape == (0, 3)


class TestVaryHabit:
    def test_seeded_and_bounded(self) -> None:
        oak = SPECIES["oak"]
        a, b = vary_habit(oak, "moby-dick"), vary_habit(oak, "moby-dick")
        assert a == b
        assert vary_habit(oak, "walden") != a
        assert abs(a.width / oak.width - 1.0) <= 0.12 + 1e-9
        assert a.envelope == oak.envelope and a.influence == oak.influence

    def test_amount_zero_is_identity(self) -> None:
        assert vary_habit(SPECIES["fir"], "x", amount=0) is SPECIES["fir"]


class TestGrowth:
    def test_default_habit_grows_the_pre_habit_tree(self) -> None:
        crown = _crown(DEFAULT_HABIT)
        plain = grow_tree(crown, np.zeros(3), key="same")
        with_default = grow_tree(crown, np.zeros(3), key="same", habit=DEFAULT_HABIT)
        np.testing.assert_array_equal(plain.points, with_default.points)
        np.testing.assert_array_equal(plain.radii, with_default.radii)
        np.testing.assert_array_equal(plain.crown, crown)

    @pytest.mark.parametrize("name", sorted(SPECIES))
    def test_every_chunk_hangs_on_wood(self, name: str) -> None:
        habit = vary_habit(SPECIES[name], name)
        crown = _crown(habit)
        sk = grow_tree(crown, np.zeros(3), key=name, habit=habit)
        assert sk.crown is not None and sk.crown.shape == crown.shape
        gaps = np.linalg.norm(sk.crown[:, None, :] - sk.points[None, :, :], axis=2).min(axis=1)
        # colonize's kill radius is 2 * step; droop carries each chunk with its node.
        assert gaps.max() <= 2.0 * _default_step(crown, habit) + 1e-6

    @pytest.mark.parametrize("name", [n for n, h in SPECIES.items() if h.leader > 0])
    def test_leader_species_stand_plumb(self, name: str) -> None:
        habit = SPECIES[name]
        crown = _crown(habit)
        sk = grow_tree(crown, np.zeros(3), key=name, habit=habit)
        kids = sk.children()
        k = 0
        while len(kids.get(k, [])) == 1:
            k = kids[k][0]
        # The trunk rises plumb to its first fork...
        assert np.hypot(*sk.points[k, :2]) < 1e-6
        # ...and the leader keeps a stem on the axis up through the crown.
        on_axis = sk.points[np.hypot(sk.points[:, 0], sk.points[:, 1]) < 1e-6]
        z_lo, z_hi = crown[:, 2].min(), crown[:, 2].max()
        assert on_axis[:, 2].max() >= z_lo + habit.leader * (z_hi - z_lo) - _default_step(
            crown, habit
        )

    def test_species_silhouettes_differ(self) -> None:
        def aspect(name: str) -> float:
            sk = grow_tree(_crown(SPECIES[name]), np.zeros(3), key="same", habit=SPECIES[name])
            return float(np.hypot(sk.points[:, 0], sk.points[:, 1]).max() / sk.points[:, 2].max())

        assert aspect("oak") > 1.8 * aspect("poplar")
        assert aspect("pine") > aspect("fir")


class TestDroop:
    def _grown(self) -> tuple[np.ndarray, object]:
        habit = SPECIES["willow"]
        crown = _crown(habit)
        sk = colonize(crown, np.zeros(3), seed=1, plumb_trunk=True)
        pipe_radii(sk)
        return crown, sk

    def test_segments_keep_their_length_and_twigs_hang(self) -> None:
        crown, sk = self._grown()
        before = sk.points.copy()
        (moved,) = droop_skeleton(sk, 2.5, points=(crown,))
        p = sk.parents
        seg_before = np.linalg.norm(before[1:] - before[p[1:]], axis=1)
        seg_after = np.linalg.norm(sk.points[1:] - sk.points[p[1:]], axis=1)
        clamped = sk.points[1:, 2] <= 0.3 + 1e-9
        np.testing.assert_allclose(seg_after[~clamped], seg_before[~clamped], rtol=1e-9, atol=1e-9)
        assert sk.points[sk.tips, 2].mean() < before[sk.tips, 2].mean() - 1.0
        assert moved[:, 2].mean() < crown[:, 2].mean() - 1.0
        # The trunk is stiff: its base does not move.
        np.testing.assert_array_equal(sk.points[0], before[0])

    def test_zero_droop_changes_nothing(self) -> None:
        crown, sk = self._grown()
        before = sk.points.copy()
        (same,) = droop_skeleton(sk, 0.0, points=(crown,))
        np.testing.assert_array_equal(sk.points, before)
        np.testing.assert_array_equal(same, crown)


def test_species_table_is_plain_numbers() -> None:
    table = species_table()
    assert sorted(table) == sorted(SPECIES)
    assert len(table) == 9
    assert table["willow"]["droop"] == SPECIES["willow"].droop
    assert all(
        isinstance(v, (int, float, str, bool)) for row in table.values() for v in row.values()
    )
