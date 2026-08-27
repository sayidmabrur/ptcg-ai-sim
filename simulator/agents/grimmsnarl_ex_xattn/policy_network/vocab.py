"""Enum mirrors and tensor encodings for ``Option``/``SelectData`` fields.

The parquet dataset stores these fields as the plain ints the game engine
emits over the wire (see ``cg/api.py``'s ``to_dataclass``/``json_to_dataclass``
round trip) rather than as strings, so encoding them is a matter of picking a
padding/"unknown" sentinel per field, not building a string vocabulary.

The IntEnums below (``OptionType``, ``AreaType``, etc.) are a hand-copied
mirror of ``cg/api.py`` — trivial to keep in sync by eyeballing that file, and
this way a plain dataclass/enum reader doesn't need the rest of this module's
engine dependency. The *sizes* that can't safely be eyeballed and copied
(``CARD_ID_VOCAB_SIZE``, ``ATTACK_ID_VOCAB_SIZE``) are instead read live from
``cg.api.all_card_data()``/``all_attack()`` — the native ``cg.dll``/
``libcg.so`` simulator's own database — since those counts can grow as the
competition's card pool grows, and guessing them from replay data or
``EN_Card_Data.csv`` risks silently undersizing the embedding table the day a
new card/attack id outside the sampled range shows up.
"""

import re
import sys
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any

import torch

# ``cg`` lives at the repo root, three levels up from this file
# (archetypes/lucario/policy_network/vocab.py) — not on sys.path when this
# module is imported from within policy_network (e.g. ``python dataset.py``
# run from the repo root only adds policy_network/ itself, per Python's
# script-directory rule).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from cg.api import all_attack, all_card_data  # noqa: E402


class EnergyType(IntEnum):
    COLORLESS = 0
    GRASS = 1
    FIRE = 2
    WATER = 3
    LIGHTNING = 4
    PSYCHIC = 5
    FIGHTING = 6
    DARKNESS = 7
    METAL = 8
    DRAGON = 9
    RAINBOW = 10  # Every type.
    TEAM_ROCKET = 11  # Psychic and Darkness.


class CardType(IntEnum):
    POKEMON = 0
    ITEM = 1
    TOOL = 2  # Pokémon Tool.
    SUPPORTER = 3
    STADIUM = 4
    BASIC_ENERGY = 5
    SPECIAL_ENERGY = 6


class SpecialConditionType(IntEnum):
    POISON = 0
    BURN = 1
    SLEEP = 2
    PARALYZE = 3
    CONFUSE = 4


class AreaType(IntEnum):
    DECK = 1
    HAND = 2
    DISCARD = 3
    ACTIVE = 4
    BENCH = 5
    PRIZE = 6
    STADIUM = 7
    ENERGY = 8
    TOOL = 9
    PRE_EVOLUTION = 10
    PLAYER = 11
    LOOKING = 12


class OptionType(IntEnum):
    NUMBER = 0
    YES = 1
    NO = 2
    CARD = 3
    TOOL_CARD = 4
    ENERGY_CARD = 5
    ENERGY = 6
    PLAY = 7
    ATTACH = 8
    EVOLVE = 9
    ABILITY = 10
    DISCARD = 11
    RETREAT = 12
    ATTACK = 13
    END = 14
    SKILL = 15
    SPECIAL_CONDITION = 16


class SelectType(IntEnum):
    MAIN = 0
    CARD = 1
    ATTACHED_CARD = 2
    CARD_OR_ATTACHED_CARD = 3
    ENERGY = 4
    SKILL = 5
    ATTACK = 6
    EVOLVE = 7
    COUNT = 8
    YES_NO = 9
    SPECIAL_CONDITION = 10


class SelectContext(IntEnum):
    MAIN = 0
    SETUP_ACTIVE_POKEMON = 1
    SETUP_BENCH_POKEMON = 2
    SWITCH = 3
    TO_ACTIVE = 4
    TO_BENCH = 5
    TO_FIELD = 6
    TO_HAND = 7
    DISCARD = 8
    TO_DECK = 9
    TO_DECK_BOTTOM = 10
    TO_PRIZE = 11
    NOT_MOVE = 12
    DAMAGE_COUNTER = 13
    DAMAGE_COUNTER_ANY = 14
    DAMAGE = 15
    REMOVE_DAMAGE_COUNTER = 16
    HEAL = 17
    EVOLVES_FROM = 18
    EVOLVES_TO = 19
    DEVOLVE = 20
    ATTACH_FROM = 21
    ATTACH_TO = 22
    DETACH_FROM = 23
    LOOK = 24
    EFFECT_TARGET = 25
    DISCARD_ENERGY_CARD = 26
    DISCARD_TOOL_CARD = 27
    SWITCH_ENERGY_CARD = 28
    DISCARD_CARD_OR_ATTACHED_CARD = 29
    DISCARD_ENERGY = 30
    TO_HAND_ENERGY = 31
    TO_DECK_ENERGY = 32
    SWITCH_ENERGY = 33
    SKILL_ORDER = 34
    ATTACK = 35
    DISABLE_ATTACK = 36
    EVOLVE = 37
    DRAW_COUNT = 38
    DAMAGE_COUNTER_COUNT = 39
    REMOVE_DAMAGE_COUNTER_COUNT = 40
    IS_FIRST = 41
    MULLIGAN = 42
    ACTIVATE = 43
    FIRST_EFFECT = 44
    MORE_DEVOLVE = 45
    COIN_HEAD = 46
    AFFECT_SPECIAL_CONDITION = 47
    RECOVER_SPECIAL_CONDITION = 48
    # cg/api.py notes new members may be appended during the competition.


#: Card/attack id ranges come from the game engine itself — the authoritative
#: source, not the replay data or EN_Card_Data.csv — via
#: ``cg.api.all_card_data()`` / ``cg.api.all_attack()``, which call straight
#: into the native ``cg.dll``/``libcg.so`` simulator and return its full
#: card/attack database. Both id spaces are contiguous 1..N (verified by
#: sorting the ids returned and checking against ``range(1, max+1)``), so a
#: dense embedding table indexed by id works directly; 0 is reserved as the
#: "no card"/"no attack" sentinel for optional fields.
MAX_CARD_ID = len(all_card_data())
MAX_ATTACK_ID = len(all_attack())
CARD_ID_VOCAB_SIZE = MAX_CARD_ID + 1
ATTACK_ID_VOCAB_SIZE = MAX_ATTACK_ID + 1


class CardStage(IntEnum):
    """Pokémon evolution stage. Trainer/Energy cards (``basic``/``stage1``/
    ``stage2`` all ``False`` in ``CardData`` — confirmed against
    ``all_card_data()``: 206 of 1267 cards) get ``NOT_APPLICABLE``, same as
    the ``card_id`` 0 padding/no-card sentinel."""

    NOT_APPLICABLE = 0
    BASIC = 1
    STAGE1 = 2
    STAGE2 = 3


CARD_STAGE_VOCAB_SIZE = len(CardStage)

#: ``CardType``/``EnergyType`` are always populated in ``CardData`` (never
#: ``None`` — confirmed against all 1267 cards), but their own members start
#: at 0, so a card_id-0 "no card" sentinel would collide with a real
#: ``POKEMON``/``COLORLESS`` value if looked up unshifted. +1 here, same
#: convention as every other Optional-enum vocab size in this module.
CARD_TYPE_VOCAB_SIZE = len(CardType) + 1
CARD_ENERGY_TYPE_VOCAB_SIZE = len(EnergyType) + 1

#: ``weakness``/``resistance`` are ``Optional[EnergyType]`` (246 of 1267 cards
#: have no weakness, 1047 no resistance — every Trainer/Energy plus some
#: Pokémon), so they take the same +1 shift: 0 = "none", real types at 1..12.
CARD_WEAKNESS_VOCAB_SIZE = len(EnergyType) + 1
CARD_RESISTANCE_VOCAB_SIZE = len(EnergyType) + 1

# --------------------------------------------------------------------------
# Card skills (abilities and Trainer effects)
#
# ``CardData.skills`` was previously read by nothing in this package: a card's
# ability reached the model only as whatever its ``card_id`` embedding happened
# to memorise, and two different cards printed with the same ability shared no
# representation at all. That is 426 card-skill pairs over 1267 cards — and
# only 218 of them are Pokémon abilities. The other 208 are Item / Tool /
# Supporter / Stadium effect text, so this is also the first time the policy can
# tell "this hand card searches the deck" from "this hand card draws".
#
# Three features come out of it, cheapest first:
#
# ``skill_count``   how many skills the card has (0/1/2) — "does it have an
#                   ability at all", which nothing expressed before.
# ``ability_id``    an embedding index over distinct skill *names*, so the
#                   Festival Lead on three different Pokémon is one shared
#                   vector. For a single-skill card this is a pure function of
#                   ``card_id``, so it adds no information there — the win is
#                   the cross-card sharing.
# ``skill_flags``   a multi-hot over ``_SKILL_PATTERNS`` below, matched against
#                   the rules text. This is the part that generalises: it says
#                   what an unfamiliar card *does*, not merely which card it is.
#
# The engine does not read this text either way — checked against libcg.so,
# where each card's real behaviour is a chain of typed ``Chain::effect*`` calls
# built in ``CardImpl()`` and interpreted by ``ActivateEffect(State&, Effect
# const&, int)``, with ``Skill.name``/``Skill.text`` attached alongside purely
# as display data by ``Chain::initSkill``. Those opcodes would be the ideal
# feature and are exactly what ``ApiAllCard`` does not serialise, so the text
# is the best proxy the public API exposes.

def _normalize_rules_text(text: str) -> str:
    """Lowercase, collapse whitespace, and straighten quotes.

    The quote step is load-bearing, not cosmetic: the engine's rules text uses
    the typographic apostrophe U+2019 ("can’t attack"), so a pattern written
    with the ASCII ``'`` silently matches nothing. That cost real coverage
    before it was caught — the attack-side ``attack_lock`` flag went from 0.0%
    to 5.1% (52 attacks) once normalised, and three skill patterns that looked
    dead ("can't attack", "can't retreat", "can't use") turned out to fire on 2,
    5 and 10 cards.
    """
    return " ".join(text.lower().replace("’", "'").replace("‘", "'").split())


#: Rules-text patterns, matched case-insensitively against a skill's ``text``
#: and OR-ed across a card's skills. Deliberately regex-on-templated-prose
#: rather than a learned text encoder: the whole corpus is 426 texts over 343
#: distinct words, so the phrasing is near-templated and a pattern match is
#: both exact and free at inference (it is baked into the table below at
#: import, so nothing has to run a tokeniser during a match).
#:
#: Every pattern here was checked to actually fire against the real card pool,
#: with the hit rate in the comment; two candidates that genuinely matched
#: nothing even after quote normalisation ("does N more damage", which is
#: attack-side phrasing, and a standalone "isn't affected") were dropped rather
#: than shipped as permanently-zero inputs. 16 of the 426 texts match no
#: pattern at all; those are one-off static rules ("as long as this Pokémon is
#: in play, it is {G} and {R} type") with nothing shared to capture.
_SKILL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("draw", r"draw \d|draw a card|draw card"),            # 9.2%
    ("search_deck", r"search your deck"),                   # 15.0%
    ("shuffle", r"shuffle"),                                # 22.3%
    ("to_hand", r"into your hand|to your hand"),            # 15.7%
    ("discard", r"discard"),                                # 20.0%
    ("heal", r"heal"),                                      # 4.9%
    ("damage_counter", r"damage counter"),                  # 6.8%
    ("switch", r"switch"),                                  # 5.6%
    ("attach_energy", r"attach .{0,40}energy|energy .{0,20}attach"),  # 12.4%
    ("provides_energy", r"provides"),                       # 3.1%
    ("prevent", r"prevent|no damage|isn't affected|is not affected"),  # 6.1%
    ("coin_flip", r"flip a coin"),                          # 2.6%
    ("once_per_turn", r"once during your turn"),            # 19.0%
    ("knocked_out", r"knocked out"),                        # 9.9%
    ("evolve", r"evolve|evolution"),                        # 11.0%
    ("retreat", r"retreat"),                                # 3.3%
    ("special_condition", r"poison|burn|asleep|paralyz|confus"),  # 4.5%
    ("reveal", r"reveal"),                                  # 12.4%
    ("bench", r"bench"),                                    # 16.7%
    ("active_spot", r"active spot"),                        # 12.2%
    ("prize", r"prize"),                                    # 4.7%
    ("targets_opponent", r"opponent"),                      # 44.4%
    ("extra_hp", r"gets \+\d+ hp"),                         # 1.6%
    ("cost_hand", r"discard a card from your hand|discard \d+ cards? from your hand"),  # 0.5%
    # One combined restriction flag rather than three near-dead ones: the
    # individual phrasings fire on 2 / 5 / 10 cards, too sparse to be worth a
    # dimension each, but "this ability forbids something" is worth one.
    ("restriction", r"can't attack|can't retreat|can't use"),  # 3.8%
)

SKILL_FLAG_NAMES = tuple(name for name, _ in _SKILL_PATTERNS)
SKILL_FLAG_COUNT = len(SKILL_FLAG_NAMES)

#: 0, 1 or 2 skills — 846/416/5 cards respectively in the real pool. Sized off
#: the observed maximum plus slack so a future card with three cannot index
#: out of the embedding.
SKILL_COUNT_CAP = 4
SKILL_COUNT_VOCAB_SIZE = SKILL_COUNT_CAP + 1


def _build_skill_tables():
    """``(skill_count, ability_id, skill_flags)`` keyed by ``card_id``.

    Built in a function so the per-card loop variables stay out of the module
    namespace, matching ``_build_attack_tables``.
    """
    names: dict[str, int] = {}
    count = torch.zeros(CARD_ID_VOCAB_SIZE, dtype=torch.long)
    ability = torch.zeros(CARD_ID_VOCAB_SIZE, dtype=torch.long)
    flags = torch.zeros((CARD_ID_VOCAB_SIZE, SKILL_FLAG_COUNT), dtype=torch.long)
    compiled = [(index, re.compile(pattern)) for index, (_, pattern) in enumerate(_SKILL_PATTERNS)]

    for card in all_card_data():
        if not card.skills:
            continue
        count[card.cardId] = min(len(card.skills), SKILL_COUNT_CAP)
        # The *first* skill names the card for ``ability_id``. Five cards carry
        # two skills, and an ABILITY option identifies only the card (via
        # ``area``/``index``), never which of its skills — so there is no
        # option-side signal a second id could ever be matched against. The
        # flags below are OR-ed over both, which is the part that does not lose
        # information.
        skill_name = card.skills[0].name.strip()
        ability[card.cardId] = names.setdefault(skill_name, len(names) + 1)
        for skill in card.skills:
            text = _normalize_rules_text(skill.text)
            for index, pattern in compiled:
                if pattern.search(text):
                    flags[card.cardId, index] = 1
    return count, ability, flags, len(names) + 1


(
    _CARD_SKILL_COUNT,
    _CARD_ABILITY_ID,
    _CARD_SKILL_FLAGS,
    ABILITY_ID_VOCAB_SIZE,
) = _build_skill_tables()

#: Row 0 stays all-zero / id 0, which is already the right answer for the
#: "no card" sentinel and for the 846 cards with no skill: index 0 of the
#: ability embedding is the shared "this card has no ability" vector.
_CARD_SKILL_COUNT_NORM = _CARD_SKILL_COUNT.float() / SKILL_COUNT_CAP
_CARD_SKILL_FLAGS_FLOAT = _CARD_SKILL_FLAGS.float()

#: One shared scale for every HP-like and damage-like magnitude in the
#: codebase — board HP, a card's printed HP, and an attack's damage all
#: divide by this. Sharing it is the point, not an accident: normalized
#: attack damage and normalized defender HP then live on the *same* axis, so
#: "does this attack KO that Pokémon" is the directly comparable
#: ``damage_norm >= hp_norm`` rather than a relationship the model has to
#: reconstruct across two differently-scaled inputs. Observed maxima:
#: ``CardData.hp`` 380, ``Attack.damage`` 350 (both from the engine's own
#: database), so 400 clears both with headroom.
HP_CAP = 400.0
#: ``CardData.retreatCost`` is 0-4 across all 1267 cards; 5 leaves room for a
#: costlier printing without rescaling.
RETREAT_COST_CAP = 5.0
#: ``len(Attack.energies)`` is 0-5, and the most copies of one energy type in
#: a single cost is also 5 — so this caps both the total cost and each
#: per-type count.
ENERGY_COST_CAP = 6.0

#: Per-``card_id`` static lookup tables (``cg.api.CardData``'s ``cardType``/
#: ``energyType``/``ex``/``megaEx``/``tera``/``aceSpec``/stage flags), built
#: once from ``all_card_data()`` so a board Pokémon's ``id`` can be joined
#: against its card-type flags without shipping the whole card database
#: through the dataset. Index 0 is the ``card_id`` "no card" sentinel, so
#: every table is sized ``CARD_ID_VOCAB_SIZE`` with index 0 left at its
#: default (0/False) — already correct for the shifted ``card_type``/
#: ``energy_type`` tables too, since 0 there means "no card" not a real enum
#: member.
_CARD_STAGE = torch.zeros(CARD_ID_VOCAB_SIZE, dtype=torch.long)
_CARD_TYPE = torch.zeros(CARD_ID_VOCAB_SIZE, dtype=torch.long)
_CARD_ENERGY_TYPE = torch.zeros(CARD_ID_VOCAB_SIZE, dtype=torch.long)
#: ``torch.long`` 0/1, not ``torch.bool`` — these are model-input features
#: (unlike a padding/validity mask), so they follow the same ``int(bool)``
#: convention used for every other boolean feature in this codebase
#: (``poisoned``, ``stadium_played``, etc.) rather than the mask dtype.
_CARD_EX = torch.zeros(CARD_ID_VOCAB_SIZE, dtype=torch.long)
_CARD_MEGA_EX = torch.zeros(CARD_ID_VOCAB_SIZE, dtype=torch.long)
_CARD_TERA = torch.zeros(CARD_ID_VOCAB_SIZE, dtype=torch.long)
_CARD_ACE_SPEC = torch.zeros(CARD_ID_VOCAB_SIZE, dtype=torch.long)
#: The quantities that actually decide Pokémon TCG lines, previously absent
#: from the feature set entirely: printed HP (how much damage kills it),
#: weakness/resistance (the x2 / -30 damage modifiers), and retreat cost (what
#: escaping the Active Spot costs). Without these a policy has to memorize
#: them per card id from the replays, which generalizes to nothing.
_CARD_HP = torch.zeros(CARD_ID_VOCAB_SIZE, dtype=torch.long)
_CARD_RETREAT_COST = torch.zeros(CARD_ID_VOCAB_SIZE, dtype=torch.long)
_CARD_WEAKNESS = torch.zeros(CARD_ID_VOCAB_SIZE, dtype=torch.long)
_CARD_RESISTANCE = torch.zeros(CARD_ID_VOCAB_SIZE, dtype=torch.long)
for _card in all_card_data():
    if _card.basic:
        _stage = CardStage.BASIC
    elif _card.stage1:
        _stage = CardStage.STAGE1
    elif _card.stage2:
        _stage = CardStage.STAGE2
    else:
        _stage = CardStage.NOT_APPLICABLE
    _CARD_STAGE[_card.cardId] = _stage
    _CARD_TYPE[_card.cardId] = _card.cardType + 1
    _CARD_ENERGY_TYPE[_card.cardId] = _card.energyType + 1
    _CARD_EX[_card.cardId] = int(_card.ex)
    _CARD_MEGA_EX[_card.cardId] = int(_card.megaEx)
    _CARD_TERA[_card.cardId] = int(_card.tera)
    _CARD_ACE_SPEC[_card.cardId] = int(_card.aceSpec)
    _CARD_HP[_card.cardId] = _card.hp
    _CARD_RETREAT_COST[_card.cardId] = _card.retreatCost
    # +1 shift so 0 stays "none" (see CARD_WEAKNESS_VOCAB_SIZE).
    _CARD_WEAKNESS[_card.cardId] = 0 if _card.weakness is None else _card.weakness + 1
    _CARD_RESISTANCE[_card.cardId] = 0 if _card.resistance is None else _card.resistance + 1
del _card, _stage

#: ``stage`` is genuinely ordinal (BASIC < STAGE1 < STAGE2 — more evolved),
#: but as a raw int it's used as a categorical embedding index, which
#: doesn't guarantee the model respects that order. This is a companion
#: float in [0, 1] (``stage / (CardStage members - 1)``) carrying just the
#: "how evolved is this" magnitude explicitly, alongside (not instead of)
#: the categorical ``stage`` — ``NOT_APPLICABLE`` (0, non-Pokémon) and
#: ``BASIC`` (1) both map near the low end, which is an acceptable overlap
#: since neither is "evolved."
_CARD_STAGE_NORM = _CARD_STAGE.float() / (len(CardStage) - 1)

#: Pre-normalized companions, same pattern as ``_CARD_STAGE_NORM`` — these are
#: magnitudes, so they reach the model as floats in [0, 1] rather than as
#: embedding indexes. Index 0 (the "no card" sentinel) is 0.0 in both, which
#: is already the right answer for a card with no HP / no retreat cost.
_CARD_HP_NORM = (_CARD_HP.float() / HP_CAP).clamp(0.0, 1.0)
_CARD_RETREAT_COST_NORM = (_CARD_RETREAT_COST.float() / RETREAT_COST_CAP).clamp(0.0, 1.0)

#: Per-``attack_id`` static lookups, the attack-side counterpart to the
#: ``CardData`` tables above. ``attack_id`` previously reached the model only
#: as the scalar ``attack_id / 10.0`` — an id used as a magnitude, so attack
#: 1092 arrived as "109.2" and its 200 damage / 3-energy cost were nowhere in
#: the input at all. Row 0 is the "no attack" sentinel (all zeros), which is
#: also where ``NO_VALUE`` (-1) clamps to for non-attack options.
#: ``Attack.text`` patterns — the rider effect, which nothing read before.
#: Damage and energy cost were already here, so an attack reached the model as
#: "200 damage for 3 energy" with no way to tell a clean hitter from one that
#: discards your whole board, locks itself out next turn, or scales off coin
#: flips. Two attacks with the same damage and cost were identical inputs
#: distinguishable only by memorising ``attack_id``, which is exactly the
#: generalisation failure this table exists to fix — and attacks are the win
#: condition, so it bites hardest here.
#:
#: 1023 of 1556 attacks carry text, over 299 distinct words. Same
#: validated-hit-rate discipline as ``_SKILL_PATTERNS``: rates in the comments,
#: and ``delayed_ko`` ("will be Knocked Out", 1 attack) was dropped as too
#: sparse to earn a dimension. 21 of the 1023 match nothing.
_ATTACK_EFFECT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("damage_scaling", r"for each"),                         # 18.1%
    ("more_damage", r"more damage"),                         # 18.0%
    ("discard", r"discard"),                                 # 17.1%
    ("next_turn", r"next turn"),                             # 14.0%
    ("coin_flip", r"flip a coin|flip \d+ coins"),            # 13.7%
    ("shuffle", r"shuffle"),                                 # 9.9%
    ("special_condition", r"poison|burn|asleep|paralyz|confus"),  # 9.4%
    ("ignore_weak_res", r"weakness and resistance"),         # 8.1%
    ("search_deck", r"search your deck"),                    # 7.8%
    ("damage_counter", r"damage counter"),                   # 6.4%
    ("discard_own_energy", r"discard all energy from this|discard .{0,30}energy from this pok"),  # 6.1%
    ("bench_damage", r"damage to each|damage to \d+ of your opponent|to each of your opponent"),  # 5.7%
    # Only reachable with quote normalisation — see _normalize_rules_text.
    ("attack_lock", r"can't use|can't attack|cannot attack"),  # 5.1%
    ("self_damage", r"damage to itself"),                    # 4.8%
    ("to_hand", r"into your hand|to your hand"),             # 4.4%
    ("reveal", r"reveal"),                                   # 4.0%
    ("heal", r"heal"),                                       # 3.5%
    ("draw", r"draw"),                                       # 3.5%
    ("retreat", r"retreat"),                                 # 3.0%
    ("from_discard_pile", r"discard pile"),                  # 2.5%
    ("switch", r"switch"),                                   # 2.3%
    ("prevent", r"prevent|no damage|isn't affected"),        # 2.3%
    ("basic_pokemon", r"basic pok"),                         # 2.1%
    ("knocked_out", r"knocked out"),                         # 1.6%
    ("prize", r"prize"),                                     # 1.6%
    ("attach_energy", r"attach .{0,40}energy"),              # 1.5%
    ("move_energy", r"move .{0,30}energy"),                  # 1.0%
    ("targets_opponent", r"opponent"),                       # 43.8%
)

ATTACK_EFFECT_FLAG_NAMES = tuple(name for name, _ in _ATTACK_EFFECT_PATTERNS)
ATTACK_EFFECT_FLAG_COUNT = len(ATTACK_EFFECT_FLAG_NAMES)


def _build_attack_tables() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Built in a function rather than a module-level loop so the per-attack
    loop variables don't leak into the module namespace."""
    damage = torch.zeros(ATTACK_ID_VOCAB_SIZE, dtype=torch.long)
    energy_cost = torch.zeros(ATTACK_ID_VOCAB_SIZE, dtype=torch.long)
    energy_type_counts = torch.zeros((ATTACK_ID_VOCAB_SIZE, len(EnergyType)), dtype=torch.long)
    effect_flags = torch.zeros((ATTACK_ID_VOCAB_SIZE, ATTACK_EFFECT_FLAG_COUNT), dtype=torch.long)
    compiled = [
        (index, re.compile(pattern))
        for index, (_, pattern) in enumerate(_ATTACK_EFFECT_PATTERNS)
    ]
    for attack in all_attack():
        damage[attack.attackId] = attack.damage
        energy_cost[attack.attackId] = len(attack.energies)
        # A count per energy type, not a multi-hot: a cost of {G}{C}{C} needs
        # the 2 colorless distinguished from 1, exactly like
        # ``_pad_energy_attached_chain`` already counts rather than flags.
        for energy_type in attack.energies:
            energy_type_counts[attack.attackId, energy_type] += 1
        if attack.text and attack.text.strip():
            text = _normalize_rules_text(attack.text)
            for index, pattern in compiled:
                if pattern.search(text):
                    effect_flags[attack.attackId, index] = 1
    return damage, energy_cost, energy_type_counts, effect_flags


(
    _ATTACK_DAMAGE,
    _ATTACK_ENERGY_COST,
    _ATTACK_ENERGY_TYPE_COUNTS,
    _ATTACK_EFFECT_FLAGS,
) = _build_attack_tables()

_ATTACK_EFFECT_FLAGS_FLOAT = _ATTACK_EFFECT_FLAGS.float()

#: Damage shares ``HP_CAP`` with board/card HP on purpose — see ``HP_CAP``.
_ATTACK_DAMAGE_NORM = (_ATTACK_DAMAGE.float() / HP_CAP).clamp(0.0, 1.0)
_ATTACK_ENERGY_COST_NORM = (_ATTACK_ENERGY_COST.float() / ENERGY_COST_CAP).clamp(0.0, 1.0)
_ATTACK_ENERGY_TYPE_COUNTS_NORM = (
    _ATTACK_ENERGY_TYPE_COUNTS.float() / ENERGY_COST_CAP
).clamp(0.0, 1.0)


def attack_flags(attack_id: torch.Tensor) -> dict[str, torch.Tensor]:
    """Static ``Attack`` properties for a tensor of ``attack_id``s (any shape).

    ``attack_id`` carries ``NO_VALUE`` (-1) on every option that isn't an
    attack, which is not a legal index — clamping to row 0 resolves those to
    the all-zero "no attack" row. The clamped ids come back as
    ``attack_id_safe`` so the embedding lookup downstream doesn't have to
    re-derive that, and so the clamp is applied in exactly one place.
    """
    safe = attack_id.clamp(min=0)
    return {
        "attack_id_safe": safe,
        "attack_damage_norm": _ATTACK_DAMAGE_NORM[safe],
        "attack_energy_cost_norm": _ATTACK_ENERGY_COST_NORM[safe],
        "attack_energy_type_counts": _ATTACK_ENERGY_TYPE_COUNTS_NORM[safe],
        # What the attack *does* beyond its damage number. Row 0 is all-zero,
        # so a non-attack option (``NO_VALUE`` clamped to 0) and an attack with
        # no rider text both correctly read "no effects".
        "attack_effect_flags": _ATTACK_EFFECT_FLAGS_FLOAT[safe],
    }


#: Flag name -> its static per-card lookup table, indexed by ``card_id``.
#:
#: A registry rather than a dict literal built inside ``card_type_flags`` because the
#: literal form evaluated *every* lookup and then dropped the ones a restricted
#: ``fields`` had not asked for. Profiling a rollout put ``card_type_flags`` at 41 us a
#: call and 32 calls per decision — 1.3 ms of every decision's feature build, a quarter
#: of it spent indexing tensors nothing would read. Indexing lazily is the same
#: function, so features and checkpoints are unaffected.
#:
#: ``skill_flags`` is the one entry that is not scalar per card: it carries a trailing
#: ``SKILL_FLAG_COUNT`` dim, the same shape convention ``attack_flags``'s
#: ``attack_energy_type_counts`` already uses, and ``collate.pad_stack`` pads it
#: elementwise so a ragged card list batches without special-casing.
_CARD_FLAG_TABLES: dict[str, torch.Tensor] = {
    "stage": _CARD_STAGE,
    "stage_norm": _CARD_STAGE_NORM,
    "type": _CARD_TYPE,
    "energy_type": _CARD_ENERGY_TYPE,
    "ex": _CARD_EX,
    "mega_ex": _CARD_MEGA_EX,
    "tera": _CARD_TERA,
    "ace_spec": _CARD_ACE_SPEC,
    "hp_norm": _CARD_HP_NORM,
    "retreat_cost_norm": _CARD_RETREAT_COST_NORM,
    "weakness": _CARD_WEAKNESS,
    "resistance": _CARD_RESISTANCE,
    "skill_count": _CARD_SKILL_COUNT,
    "skill_count_norm": _CARD_SKILL_COUNT_NORM,
    "ability_id": _CARD_ABILITY_ID,
    "skill_flags": _CARD_SKILL_FLAGS_FLOAT,
}


def card_type_flags(
    card_id: torch.Tensor, fields: tuple[str, ...] | None = None
) -> dict[str, torch.Tensor]:
    """Look up static ``CardData`` type flags for a tensor of ``card_id``s
    (any shape) — ``card_id`` 0 (the "no card" sentinel) resolves to
    ``CardStage.NOT_APPLICABLE``/0 (no ``CardType``/``EnergyType``)/``False``
    for every flag.

    ``fields`` restricts which flags get looked up, for card slots where
    some are confirmed structurally dead: e.g. an attached Energy card is
    always ``BASIC_ENERGY``/``SPECIAL_ENERGY`` (never a Pokémon), so
    ``stage``/``ex``/``mega_ex``/``tera`` are ``False``/``NOT_APPLICABLE``
    for all 20 energy cards in ``all_card_data()`` — but ``ace_spec`` isn't
    (3 real ACE SPEC energy cards exist), so it stays available to request.
    """
    tables = _CARD_FLAG_TABLES
    return {flag: tables[flag][card_id] for flag in (fields or tables)}

#: +1 on every enum used as an Optional field, to reserve 0 as "field not set
#: for this option/selection" — every one of these IntEnums' own members
#: already start at 0, so 0 can't double as both a real value and "unset".
OPTION_TYPE_VOCAB_SIZE = len(OptionType)
AREA_VOCAB_SIZE = len(AreaType) + 1
SELECT_TYPE_VOCAB_SIZE = len(SelectType)
SELECT_CONTEXT_VOCAB_SIZE = len(SelectContext) + 1  # +1 slack: enum may grow

#: 0/1 real values, 2 = unknown (option field was None).
TARGETS_OPPONENT_VOCAB_SIZE = 3

#: ``specialConditionType`` is set only on ``SPECIAL_CONDITION`` options, so it
#: takes the usual +1 shift with 0 = "not a special-condition option". It was
#: pulled out of the raw obs by ``features.OPTION_FIELDS`` all along but never
#: turned into a tensor, which left every ``SPECIAL_CONDITION`` option in a
#: decision *identical* in feature space — "cure the Poison" and "cure the
#: Burn" were the same input, and ``equivalence_mask`` duly called them the
#: same play.
SPECIAL_CONDITION_VOCAB_SIZE = len(SpecialConditionType) + 1

#: Sentinel for optional plain-magnitude int fields (index/count/id fields
#: with no fixed vocab) that carry no enum of their own.
NO_VALUE = -1


def _area(value: int | None) -> int:
    return 0 if value is None else int(value)


def _card_id(value: int | None) -> int:
    return 0 if value is None else int(value)


def _magnitude(value: int | None) -> int:
    return NO_VALUE if value is None else int(value)


def _targets_opponent(value: bool | None) -> int:
    return 2 if value is None else int(value)


def _special_condition(value: int | None) -> int:
    return 0 if value is None else int(value) + 1


@dataclass
class OptionsVocab:
    """One decision's option list, as parallel ``(num_options,)`` tensors.

    Fields mirror ``cg.api.Option`` (see ``features.OPTION_FIELDS``), except
    ``playerIndex`` which ``features._remap_options`` already turns into the
    POV-relative ``targets_opponent``.
    """

    type: torch.Tensor
    number: torch.Tensor
    area: torch.Tensor
    index: torch.Tensor
    targets_opponent: torch.Tensor
    tool_index: torch.Tensor
    energy_index: torch.Tensor
    count: torch.Tensor
    in_play_area: torch.Tensor
    in_play_index: torch.Tensor
    attack_id: torch.Tensor
    card_id: torch.Tensor
    serial: torch.Tensor
    special_condition_type: torch.Tensor

    @classmethod
    def from_options(cls, options: list[dict[str, Any]]) -> "OptionsVocab":
        return cls(
            type=torch.tensor([int(o["type"]) for o in options], dtype=torch.long),
            number=torch.tensor([_magnitude(o["number"]) for o in options], dtype=torch.long),
            area=torch.tensor([_area(o["area"]) for o in options], dtype=torch.long),
            index=torch.tensor([_magnitude(o["index"]) for o in options], dtype=torch.long),
            targets_opponent=torch.tensor(
                [_targets_opponent(o["targets_opponent"]) for o in options], dtype=torch.long
            ),
            # A pointer into a Pokémon's attached tools, encoded like the other
            # index fields (``NO_VALUE`` when the option isn't a TOOL_CARD).
            tool_index=torch.tensor([_magnitude(o["toolIndex"]) for o in options], dtype=torch.long),
            energy_index=torch.tensor([_magnitude(o["energyIndex"]) for o in options], dtype=torch.long),
            count=torch.tensor([_magnitude(o["count"]) for o in options], dtype=torch.long),
            in_play_area=torch.tensor([_area(o["inPlayArea"]) for o in options], dtype=torch.long),
            in_play_index=torch.tensor([_magnitude(o["inPlayIndex"]) for o in options], dtype=torch.long),
            attack_id=torch.tensor([_magnitude(o["attackId"]) for o in options], dtype=torch.long),
            card_id=torch.tensor([_card_id(o["cardId"]) for o in options], dtype=torch.long),
            serial=torch.tensor([_magnitude(o["serial"]) for o in options], dtype=torch.long),
            # +1 shift, 0 = "not a special-condition option" — see
            # SPECIAL_CONDITION_VOCAB_SIZE for why this was worth adding.
            special_condition_type=torch.tensor(
                [_special_condition(o["specialConditionType"]) for o in options], dtype=torch.long
            ),
        )


@dataclass
class SelectionVocab:
    """One decision's selection, as scalar tensors.

    ``deck`` is encoded as its length (0 unless selecting from the deck);
    ``contextCard``/``effect`` as the referenced card's id (0 if unset) — the
    game state elsewhere already carries each card's full struct, so nothing
    beyond identity is needed to place these mid-decision.
    """

    type: torch.Tensor
    context: torch.Tensor
    min_count: torch.Tensor
    max_count: torch.Tensor
    remain_damage_counter: torch.Tensor
    remain_energy_cost: torch.Tensor
    deck_size: torch.Tensor
    context_card_id: torch.Tensor
    effect_card_id: torch.Tensor

    @classmethod
    def from_selection(cls, selection: dict[str, Any]) -> "SelectionVocab":
        deck = selection["deck"]
        context_card = selection["contextCard"]
        effect = selection["effect"]
        return cls(
            type=torch.tensor(int(selection["type"]), dtype=torch.long),
            context=torch.tensor(int(selection["context"]), dtype=torch.long),
            min_count=torch.tensor(int(selection["minCount"]), dtype=torch.long),
            max_count=torch.tensor(int(selection["maxCount"]), dtype=torch.long),
            remain_damage_counter=torch.tensor(int(selection["remainDamageCounter"]), dtype=torch.long),
            remain_energy_cost=torch.tensor(int(selection["remainEnergyCost"]), dtype=torch.long),
            deck_size=torch.tensor(0 if deck is None else len(deck), dtype=torch.long),
            context_card_id=torch.tensor(_card_id(context_card and context_card["id"]), dtype=torch.long),
            effect_card_id=torch.tensor(_card_id(effect and effect["id"]), dtype=torch.long),
        )


def pad_options(per_decision_options: list[OptionsVocab]) -> dict[str, torch.Tensor]:
    """Stack a chain of ``OptionsVocab`` (one per decision, ragged option
    counts) into ``(chain_len, max_options)`` tensors plus a validity mask.

    Padded slots get each field's own "unset" sentinel (0 for enum-ish
    fields, ``NO_VALUE`` for plain magnitudes) — the mask is what a model
    should actually gate on, since 0 doubles as a real value for some fields.
    """
    chain_len = len(per_decision_options)
    max_options = max((o.type.numel() for o in per_decision_options), default=0)

    def stack(field: str, fill: int) -> torch.Tensor:
        out = torch.full((chain_len, max_options), fill, dtype=torch.long)
        for i, options in enumerate(per_decision_options):
            values = getattr(options, field)
            out[i, : values.numel()] = values
        return out

    mask = torch.zeros((chain_len, max_options), dtype=torch.bool)
    for i, options in enumerate(per_decision_options):
        mask[i, : options.type.numel()] = True

    magnitude_fields = (
        "number", "index", "energy_index", "count",
        "in_play_index", "attack_id", "serial", "tool_index",
    )
    # ``special_condition_type`` is +1-shifted so 0 already means "unset",
    # which is exactly the right fill for a padded slot.
    zero_filled_fields = (
        "area", "targets_opponent", "card_id", "special_condition_type",
    )

    result = {"type": stack("type", 0), "options_mask": mask}
    for field in magnitude_fields:
        result[field] = stack(field, NO_VALUE)
    for field in zero_filled_fields:
        result[field] = stack(field, 0)
    return result


def stack_selections(selections: list[SelectionVocab]) -> dict[str, torch.Tensor]:
    """Stack a chain of per-decision ``SelectionVocab`` into ``(chain_len,)`` tensors.

    A decision chain is empty for the first decision of an episode (no prior
    history yet) — ``torch.stack`` rejects an empty list, so that case needs
    its own empty ``(0,)`` tensor per field rather than stacking nothing.
    """
    fields = (
        "type", "context", "min_count", "max_count", "remain_damage_counter",
        "remain_energy_cost", "deck_size", "context_card_id", "effect_card_id",
    )
    if not selections:
        return {field: torch.empty(0, dtype=torch.long) for field in fields}
    return {field: torch.stack([getattr(s, field) for s in selections]) for field in fields}
