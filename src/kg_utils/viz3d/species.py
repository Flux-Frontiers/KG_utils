"""
species.py — growth habits that make organic trees look like species.

Space colonization (:func:`~kg_utils.viz3d.organic.colonize`) follows its
attractors: give every tree the same crown envelope and the same growth
parameters and every tree comes out the same shape, whatever its data.  A
:class:`Habit` is what a species contributes on top of the data:

- **where the crown sits** — an envelope (Weber & Penn's crown shapes: dome,
  cone, umbrella, ...), how wide it is, how much bare trunk is under it, and how
  each section's cluster of chunks spreads and lifts (:func:`crown_sections`,
  :func:`section_cluster`);
- **how the wood grows toward it** — tropism, influence radius, internode
  step, jitter and pipe-model exponent (Runions et al. 2007), a central leader
  for excurrent species, and a post-growth gravity droop for weeping ones
  (:func:`~kg_utils.viz3d.organic.grow_tree` with ``habit=``).

The data still decides everything that carries meaning — height, how many
sections, one crown point per chunk that the wood must reach.  The habit only
decides where those points sit in space and how the wood gets to them.

:data:`SPECIES` holds nine tuned presets.  :func:`vary_habit` nudges one by a
seeded few percent per tree so a grove of one species is not a row of clones;
vary once per tree and pass the same habit to both :func:`crown_sections` and
``grow_tree``.

Everything here is Z-up and pure NumPy, like the rest of the engine.  The
Knowledge Press web forest (``web/src/game/species.ts`` in knowledge_press)
mirrors :data:`SPECIES` in its own Y-up coordinates; :func:`species_table`
gives the plain numbers to compare against.

Author: Eric G. Suchanek, PhD
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from kg_utils.viz3d.organic import PIPE_EXPONENT, leaf_facing, oriented_cluster, seed_from_key

__author__ = "Eric G. Suchanek, PhD"

#: Crown envelopes :func:`envelope_width` understands.
ENVELOPES: tuple[str, ...] = (
    "column",
    "ellipsoid",
    "ovoid",
    "cone",
    "dome",
    "vase",
    "umbrella",
    "spindle",
)

#: The crown's top as a fraction of trunk height; the highest section sits here.
CROWN_TOP: float = 0.95

#: How far a section's cluster rises per unit of ``lift`` above 1, as a
#: fraction of the cluster radius.  At ``lift = 1`` clusters sit exactly where
#: :func:`~kg_utils.viz3d.organic.oriented_cluster` puts them.
LIFT_SHIFT: float = 0.275

_GOLDEN = np.pi * (3.0 - np.sqrt(5.0))


@dataclass(frozen=True)
class Habit:
    """
    How a species lays out its crown and grows its wood.

    The defaults reproduce the engine's behaviour before habits existed: a
    tapering column of sections, growth at the crown's own scale, a trunk led
    to the nearest chunk.

    :param envelope: Crown silhouette, one of :data:`ENVELOPES`.
    :param width: Crown half-width as a multiple of the caller's branch length.
    :param clear_bole: Bare trunk below the lowest section, as a fraction of
        trunk height.
    :param leader: How far into the crown the trunk rises plumb before growth
        takes over, as a fraction of the crown's height.  ``0`` forks at the
        crown's base (oak); near ``1`` keeps one central stem (fir).
    :param whorl: Sections per level tier around the stem; ``1`` spirals them
        on the golden angle, ``> 1`` sets conifer-style whorls.
    :param spread: Section cluster radius, as a multiple of the caller's.
    :param lift: Clusters rise (``> 1``) or hang (``< 1``, even negative)
        relative to where :func:`~kg_utils.viz3d.organic.oriented_cluster`
        puts them.
    :param tropism: Upward pull added to every growth step; negative droops.
    :param influence: Attraction radius in internodes.  Small grows twiggy,
        bushy wood; large grows long straight limbs.
    :param step: Internode length as a multiple of the crown-derived default.
    :param jitter: Direction noise per step: ``0.12`` clean, ``0.3`` gnarled.
    :param pipe_exponent: Pipe-model exponent; higher tapers faster.
    :param droop: Gravity after growth: thin wood bends toward the ground.
        ``0`` is stiff; ``2.5`` hangs a willow's twigs straight down.
    :param plumb_trunk: Raise the trunk straight up to the crown's base rather
        than leaning it toward the nearest chunk.  Implied by ``leader > 0``.
    """

    envelope: str = "column"
    width: float = 1.0
    clear_bole: float = 0.30
    leader: float = 0.0
    whorl: int = 1
    spread: float = 1.0
    lift: float = 1.0
    tropism: float = 0.18
    influence: float = 12.0
    step: float = 1.0
    jitter: float = 0.12
    pipe_exponent: float = PIPE_EXPONENT
    droop: float = 0.0
    plumb_trunk: bool = False

    def __post_init__(self) -> None:
        """Reject an envelope name that :func:`envelope_width` would not know."""
        if self.envelope not in ENVELOPES:
            raise ValueError(
                f"Unknown envelope {self.envelope!r}; choose from {', '.join(ENVELOPES)}"
            )


#: The engine's behaviour before habits: what ``habit=None`` means everywhere.
DEFAULT_HABIT = Habit()


#: Nine tuned species.  The web forest mirrors these numbers.
SPECIES: dict[str, Habit] = {
    # English oak: short bole, broad low dome, long gnarled horizontal limbs.
    "oak": Habit(
        plumb_trunk=True,
        envelope="dome",
        width=1.55,
        clear_bole=0.22,
        spread=1.1,
        lift=0.6,
        tropism=0.02,
        influence=16,
        step=1.1,
        jitter=0.24,
        pipe_exponent=2.0,
        droop=0.05,
    ),
    # Horse chestnut: a full rounded ellipsoid on a medium bole.
    "chestnut": Habit(
        plumb_trunk=True,
        envelope="ellipsoid",
        width=1.15,
        clear_bole=0.28,
        leader=0.15,
        tropism=0.14,
        jitter=0.14,
        pipe_exponent=2.2,
        droop=0.08,
    ),
    # Fir: a narrow cone from near the ground on one straight leader.
    "fir": Habit(
        plumb_trunk=True,
        envelope="cone",
        width=0.95,
        clear_bole=0.10,
        leader=0.92,
        spread=0.7,
        lift=-0.3,
        tropism=0.30,
        # 7 was too short for the trunk to claim the wide bottom whorl, so a
        # branch from higher up grew back down to it (0.25.1).
        influence=12,
        step=0.8,
        jitter=0.08,
        pipe_exponent=2.6,
        droop=0.05,
    ),
    # London plane: a long clean bole under a tall egg-shaped crown.
    "plane": Habit(
        plumb_trunk=True,
        envelope="ovoid",
        width=1.05,
        clear_bole=0.38,
        leader=0.30,
        lift=1.2,
        tropism=0.24,
        influence=18,
        step=1.15,
        jitter=0.10,
        pipe_exponent=2.1,
    ),
    # Blackthorn: a low, dense, twiggy thicket-tree spreading from low down.
    # Twiggy from its short internodes, not from a short reach: at influence 6
    # and jitter 0.3 each tip saw too few chunks to steer by and random-walked
    # into helical limbs twice as long as the straight path (0.25.1).
    "blackthorn": Habit(
        plumb_trunk=True,
        envelope="vase",
        width=1.30,
        clear_bole=0.12,
        spread=1.25,
        lift=0.8,
        tropism=0.10,
        influence=10,
        step=0.7,
        jitter=0.18,
        pipe_exponent=2.5,
    ),
    # Stone pine: a tall bare stem under a flat-topped parasol.
    "pine": Habit(
        plumb_trunk=True,
        envelope="umbrella",
        width=1.45,
        clear_bole=0.50,
        leader=0.45,
        spread=1.1,
        lift=0.4,
        tropism=0.12,
        influence=14,
        step=1.1,
        jitter=0.12,
        pipe_exponent=2.2,
    ),
    # Silver birch: a slender stem, a narrow open crown, twig ends that hang.
    "birch": Habit(
        plumb_trunk=True,
        envelope="ovoid",
        width=0.75,
        clear_bole=0.25,
        leader=0.70,
        spread=0.9,
        lift=0.6,
        tropism=0.30,
        influence=9,
        step=0.85,
        jitter=0.10,
        pipe_exponent=2.7,
        droop=0.45,
    ),
    # Weeping willow: arching limbs high up, curtains of twigs to the ground.
    "willow": Habit(
        plumb_trunk=True,
        envelope="dome",
        width=1.30,
        clear_bole=0.50,
        leader=0.45,
        spread=1.2,
        lift=-0.9,
        tropism=0.20,
        influence=12,
        step=1.0,
        jitter=0.12,
        pipe_exponent=2.3,
        droop=2.5,
    ),
    # Lombardy poplar: a tall narrow spindle of steeply rising branches.
    "poplar": Habit(
        plumb_trunk=True,
        envelope="spindle",
        width=0.42,
        clear_bole=0.08,
        leader=0.85,
        spread=0.7,
        lift=1.2,
        tropism=0.70,
        influence=8,
        step=0.9,
        jitter=0.08,
        pipe_exponent=2.4,
    ),
}


def envelope_width(envelope: str, s: float) -> float:
    """
    Crown half-width at crown height *s*, as a fraction of the widest.

    :param envelope: One of :data:`ENVELOPES`.
    :param s: Height within the crown, ``0`` at its base and ``1`` at its top;
        clamped to that range.
    :return: Width fraction in ``(0, 1]``.
    :raises ValueError: For an unknown envelope.
    """
    return float(_envelope_width(envelope, min(1.0, max(0.0, float(s)))))


def _envelope_width(envelope: str, t: float) -> float:
    """:func:`envelope_width` on a clamped *t*."""
    if envelope == "column":
        return 1.0 - 0.4 * t
    if envelope == "ellipsoid":
        return np.sqrt(max(0.0, 1.0 - (2.0 * t - 1.0) ** 2)) * 0.8 + 0.2
    if envelope == "ovoid":
        half = 0.4 if t < 0.4 else 0.6
        return np.sqrt(max(0.0, 1.0 - ((t - 0.4) / half) ** 2)) * 0.8 + 0.2
    if envelope == "cone":
        return 1.0 - 0.92 * t
    if envelope == "dome":
        return np.sqrt(max(0.0, 1.0 - t * t)) * 0.8 + 0.2
    if envelope == "vase":
        return 0.35 + 0.65 * np.sin(np.pi / 2.0 * min(1.0, t * 1.25))
    if envelope == "umbrella":
        if t < 0.6:
            return 0.25 + 0.75 * np.sin(np.pi / 2.0 * (t / 0.6))
        return np.sqrt(max(0.0, 1.0 - ((t - 0.6) / 0.4) ** 2)) * 0.9 + 0.1
    if envelope == "spindle":
        return float(np.sqrt(np.sin(np.pi * (0.08 + 0.84 * t))))
    raise ValueError(f"Unknown envelope {envelope!r}; choose from {', '.join(ENVELOPES)}")


def vary_habit(habit: Habit, key: str, amount: float = 1.0) -> Habit:
    """
    Nudge a habit by a seeded few percent, so no two trees are clones.

    Crown width moves up to ±12%, the clear bole ±0.04, tropism ±0.05 and
    droop ±20%, each scaled by *amount*.  Vary once per tree and give the
    result to both :func:`crown_sections` and ``grow_tree``.

    :param habit: The species habit.
    :param key: Stable per-tree key (a book slug); the same key gives the
        same tree every time.
    :param amount: Scale on every nudge; ``0`` returns *habit* unchanged.
    :return: The varied habit.
    """
    if amount == 0:
        return habit
    rng = np.random.default_rng(seed_from_key(f"{key}:habit"))
    u = rng.uniform(-1.0, 1.0, 4) * amount
    return replace(
        habit,
        width=habit.width * (1.0 + 0.12 * u[0]),
        clear_bole=float(np.clip(habit.clear_bole + 0.04 * u[1], 0.05, 0.6)),
        tropism=habit.tropism + 0.05 * u[2],
        droop=habit.droop * (1.0 + 0.2 * u[3]),
    )


def crown_sections(
    n: int,
    trunk_height: float,
    branch_length: float,
    habit: Habit = DEFAULT_HABIT,
    *,
    base: tuple[float, float] = (0.0, 0.0),
) -> np.ndarray:
    """
    Section tips inside the habit's crown envelope, lowest first.

    Sections climb from the clear bole to :data:`CROWN_TOP` of the trunk's
    height on golden-angle azimuths (or in level whorls), each as far from the
    trunk axis as the envelope is wide at its height.

    :param n: Number of sections (limbs).
    :param trunk_height: Height of the tree, in scene units.
    :param branch_length: Crown half-width before the habit's ``width``.
    :param habit: Species habit.
    :param base: Trunk base ``(x, y)``; tips are returned in world XY.
    :return: ``(n, 3)`` tip positions, Z-up.
    """
    if n <= 0:
        return np.zeros((0, 3))
    per = max(1, int(round(habit.whorl)))
    tiers = -(-n // per)
    tips = np.empty((n, 3))
    for i in range(n):
        tier = i // per
        t = 0.5 if tiers == 1 else tier / (tiers - 1)
        z = trunk_height * (habit.clear_bole + (CROWN_TOP - habit.clear_bole) * t)
        angle = i * _GOLDEN if per == 1 else (i % per) * 2.0 * np.pi / per + tier * _GOLDEN
        r = branch_length * habit.width * envelope_width(habit.envelope, t)
        tips[i] = (base[0] + r * np.cos(angle), base[1] + r * np.sin(angle), z)
    return tips


def section_cluster(
    n: int,
    tip: np.ndarray,
    axis_point: np.ndarray,
    radius: float,
    habit: Habit = DEFAULT_HABIT,
) -> np.ndarray:
    """
    One section's chunk points: a cluster facing out from the trunk, shaped
    by the habit's ``spread`` and ``lift``.

    :param n: Number of chunks.
    :param tip: ``(3,)`` section tip.
    :param axis_point: ``(3,)`` point on the trunk axis the limb leaves from;
        the cluster faces away from it.
    :param radius: Cluster radius before the habit's ``spread``.
    :param habit: Species habit.
    :return: ``(n, 3)`` chunk positions.
    """
    tip = np.asarray(tip, dtype=float)
    r = radius * habit.spread
    pts = np.asarray(
        oriented_cluster(n, tip, leaf_facing(tip - np.asarray(axis_point, dtype=float)), r)
    ).reshape(-1, 3)
    if habit.lift != 1.0 and len(pts):
        pts = pts + np.array([0.0, 0.0, (habit.lift - 1.0) * LIFT_SHIFT * r])
    return pts


def species_table() -> dict[str, dict[str, float | int | str | bool]]:
    """
    :data:`SPECIES` as plain numbers, for comparing a mirror (the web forest)
    against this source.

    :return: ``{species: {field: value}}``.
    """
    return {name: asdict(h) for name, h in SPECIES.items()}
