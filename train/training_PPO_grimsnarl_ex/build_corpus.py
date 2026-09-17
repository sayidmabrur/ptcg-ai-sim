"""Assemble the BC training parquet for the Alakazam pilot from raw replay JSON.

    conda activate kaggle-pokemon
    python build_corpus.py --out alakazam_yushin.parquet

This is the Alakazam equivalent of ``../dataset/build_corpus.py``, collapsed from two
passes into one. The Grimmsnarl pipeline needed ``scan_replays.py`` first because it was
hunting one 60-card list inside ~110k mixed ladder episodes; here the corpus is already
cleaned upstream — ``trajectories_dataset/<date>_cleaned_yushin/<episode_id>_yushin_ito.json``
is every game one pilot played, and nothing else. So the only questions left are which
*seat* that pilot occupied (it varies per game) and which frames are trainable, and both
are answerable from the replay itself. There is nothing to scan for.

The output columns and their meanings are byte-for-byte the contract
``bc_train.ArchetypeDataset`` is written against, so that class is reused unmodified:
``is_target_archetype`` flags the pilot's own clean decision frames, every other row stays
*readable* but unsampled because the feature builders need them.

Invariants this writer preserves, and why they are not optional
--------------------------------------------------------------
``features.py`` builds ``decision_chain`` and ``opponent_history`` by walking *physically
backwards* from a row until ``episode_id`` changes. It never consults ``frame_index`` and
never searches. A violation therefore does not raise — it silently truncates history, and
the model trains on a shorter chain than it will be served. So:

1. every row of an episode is physically contiguous, with no foreign row inside it;
2. within an episode, rows ascend in frame order;
3. both seats of a game live in that same block, tagged by ``player_index``.

One JSON file is exactly one episode, which gives 1 and 3 for free, and frames are
emitted in replay order, which gives 2. Episodes are written in ascending ``episode_id``
order (the files are sorted by the id in their own name before anything is read, so this
costs no memory), and the invariants are asserted per episode anyway.

Row groups break only on episode boundaries, so no episode straddles one. Not required
for correctness — the loader indexes rows globally — but it is what keeps a backward scan
inside the resident row-group cache instead of evicting it every few samples.

Two filters, matching the Grimmsnarl corpus exactly so the val numbers are comparable
-------------------------------------------------------------------------------------
*Forced moves are removed, not merely unflagged.* A frame offering a single legal option
is a free correct answer; leaving those in as samples inflates ``val_exact`` without the
policy having learned anything. They are deleted rather than flagged so that a backward
scan steps over the last 30 *real decisions* rather than the last 30 frames — the spacing
the Grimmsnarl checkpoints learned on.

*Concessions and disconnects are screened out.* A seat that made fewer than
``--min-decisions`` choices, or more than ``--max-decisions``, or never reached
``--min-turn``, is not a game anyone piloted to a conclusion. The screen only narrows
``is_target_archetype``; the rows survive.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "challenger" / "policy_network"))

from features import OPTION_FIELDS, SELECTION_FIELDS  # noqa: E402

#: The episode id inside a replay's filename, so the whole corpus can be put in ascending
#: episode order before a single 5MB file is parsed. Two naming conventions are present —
#: ``87557436_yushin_ito.json`` and ``2026-07-23_87557436_yushin_ito.json`` — so this
#: matches every run of >=6 digits and the *last* one is taken. Anchoring at the start
#: instead silently returns ``2026`` for half the corpus, which collapses those files onto
#: one id and makes the ascending-order check fail on unrelated episodes.
_EPISODE_ID = re.compile(r"\d{6,}")
#: ``<YYYY-MM-DD>_cleaned_<slug>`` — the parent directory, which supplies ``day_index``.
_DAY = re.compile(r"(\d{4}-\d{2}-\d{2})")


# --------------------------------------------------------------------------- schema
# Explicit, because Arrow's inference would drop any option field that happens to be
# absent from the first row it sees, and the feature builders index these by name.


def _card(pa_: Any) -> Any:
    return pa_.struct([
        pa_.field("id", pa_.int32()),
        pa_.field("playerIndex", pa_.int8()),
        pa_.field("serial", pa_.int32()),
    ])


def make_schema() -> pa.Schema:
    card = _card(pa)
    pokemon = pa.struct([
        pa.field("id", pa.int32()), pa.field("playerIndex", pa.int8()),
        pa.field("serial", pa.int32()), pa.field("hp", pa.int16()),
        pa.field("maxHp", pa.int16()), pa.field("appearThisTurn", pa.bool_()),
        pa.field("energies", pa.list_(pa.int8())),
        pa.field("energyCards", pa.list_(card)), pa.field("tools", pa.list_(card)),
        pa.field("preEvolution", pa.list_(card)),
    ])
    player = pa.struct([
        pa.field("active", pa.list_(pokemon)), pa.field("bench", pa.list_(pokemon)),
        pa.field("benchMax", pa.int8()), pa.field("deckCount", pa.int16()),
        pa.field("discard", pa.list_(card)), pa.field("prize", pa.list_(card)),
        pa.field("handCount", pa.int16()), pa.field("hand", pa.list_(card)),
        pa.field("poisoned", pa.bool_()), pa.field("burned", pa.bool_()),
        pa.field("asleep", pa.bool_()), pa.field("paralyzed", pa.bool_()),
        pa.field("confused", pa.bool_()),
    ])
    state = pa.struct([
        pa.field("energyAttached", pa.bool_()), pa.field("firstPlayer", pa.int8()),
        pa.field("looking", pa.list_(card)), pa.field("players", pa.list_(player)),
        pa.field("result", pa.int8()), pa.field("retreated", pa.bool_()),
        pa.field("stadium", pa.list_(card)), pa.field("stadiumPlayed", pa.bool_()),
        pa.field("supporterPlayed", pa.bool_()), pa.field("turn", pa.int16()),
        pa.field("turnActionCount", pa.int16()), pa.field("yourIndex", pa.int8()),
    ])
    option = pa.struct([
        pa.field("type", pa.int8()), pa.field("number", pa.int16()),
        pa.field("area", pa.int8()), pa.field("index", pa.int16()),
        pa.field("playerIndex", pa.int8()), pa.field("toolIndex", pa.int8()),
        pa.field("energyIndex", pa.int8()), pa.field("count", pa.int8()),
        pa.field("inPlayArea", pa.int8()), pa.field("inPlayIndex", pa.int8()),
        pa.field("attackId", pa.int32()), pa.field("cardId", pa.int32()),
        pa.field("serial", pa.int32()), pa.field("specialConditionType", pa.int8()),
    ])
    selection = pa.struct([
        pa.field("type", pa.int8()), pa.field("context", pa.int16()),
        pa.field("minCount", pa.int8()), pa.field("maxCount", pa.int8()),
        pa.field("remainDamageCounter", pa.int16()),
        pa.field("remainEnergyCost", pa.int16()),
        pa.field("deck", pa.list_(card)), pa.field("contextCard", card),
        pa.field("effect", card),
    ])
    # Deliberately few columns: ``_read_row`` decodes *every* column of a row on every
    # access, so a column nothing reads is a per-sample tax paid ~300k times per epoch.
    # ``step``, ``final_result`` and ``player_name`` are dropped for that reason
    # (``final_result`` is the constant -1 in these replays anyway; the outcome lives in
    # ``seat_result``, and identity is implied by ``is_target_archetype``).
    return pa.schema([
        pa.field("episode_id", pa.int64()), pa.field("frame_index", pa.int32()),
        pa.field("state", state), pa.field("selection", selection),
        pa.field("options", pa.list_(option)),
        pa.field("target_action", pa.list_(pa.int16())),
        pa.field("reward", pa.float32()),
        pa.field("player_index", pa.int8()),
        pa.field("is_target_archetype", pa.bool_()),
        pa.field("seat_result", pa.int8()), pa.field("day_index", pa.int8()),
    ])


# --------------------------------------------------------------------- replay decoding


def normalise_option(option: dict[str, Any]) -> dict[str, Any]:
    return {field: option.get(field) for field in OPTION_FIELDS}


def is_decision(record: dict[str, Any], player_index: int) -> bool:
    observation = record.get("observation", {})
    current = observation.get("current")
    return (
        record.get("status") == "ACTIVE"
        and observation.get("select") is not None
        and current is not None
        and current.get("yourIndex") == player_index
    )


def is_valid_target(action: Any, selection: dict[str, Any], option_count: int) -> bool:
    if not isinstance(action, list):
        return False
    minimum, maximum = selection.get("minCount"), selection.get("maxCount")
    if minimum is None or maximum is None or not minimum <= len(action) <= maximum:
        return False
    if len(set(action)) != len(action):
        return False
    return all(isinstance(index, int) and 0 <= index < option_count for index in action)


def _normalise(name: str) -> str:
    """Fold the two spellings of the same pilot together.

    Replay filenames carry a slug (``raihan_ramadistra``) while ``TeamNames`` carries the
    display name (``Raihan Ramadistra``), and it is the slug that is visible when choosing
    what to pass on the command line. Folding case and underscores means either spelling
    selects the pilot, instead of the slug silently matching nothing and producing an empty
    corpus that looks like a filter working.
    """
    return name.strip().casefold().replace("_", " ")


def deck_seats(replay: dict[str, Any], card_id: int) -> list[int]:
    """Which seats submitted an opening 60-card deck containing ``card_id``.

    This is the right way to select an archetype, and it replaces matching on pilot names.
    Three things names get wrong that this cannot:

    *A pilot plays more than one deck.* ``alphastarmie`` appears on the Grimmsnarl list in
    one directory and on the Alakazam list in another. Flagging every seat that account
    occupied would train one policy on two archetypes -- the failure that measured 12.5%
    against 52.8% in this repo.

    *A filename need not carry a pilot.* 158 of these replays are named ``<id>_.json`` with
    an empty slug, so there is no name to match; the deck is still right there in the
    replay's first action.

    *An archetype is not one 60-card list.* These pilots share 17 of 19 unique cards with
    ``challenger_deck.csv`` but differ in a slot or two, so exact-list equality rejects
    genuine Grimmsnarl games. Presence of the archetype's defining card does not.

    The deck is read from the opening submission -- the 60-length action the engine takes
    before play, which ``is_decision`` deliberately excludes from the training rows.
    """
    seats = []
    for seat in range(2):
        for frame in replay["steps"][:3]:
            if seat >= len(frame):
                break
            action = frame[seat].get("action")
            if isinstance(action, list) and len(action) == 60:
                if card_id in action:
                    seats.append(seat)
                break
    return seats


def target_seats(info: dict[str, Any], agents: Iterable[str]) -> list[int]:
    """Which seats the pilots being cloned occupied — not fixed across games.

    A list rather than one seat because a corpus can legitimately clone several pilots of
    the *same* decklist: ``dataset/`` holds 1,696 games from ``raihan_ramadistra`` and 369
    from ``flg``, both verified to play the identical 60 cards, and taking only one of them
    would discard most of the corpus. When two target pilots meet each other, both seats are
    trainable and both are returned.

    Matched on ``TeamNames``, falling back to ``Agents[].Name``. An empty list drops the
    file, rather than guessing a seat and cloning whoever happened to be sitting there.
    """
    names = info.get("TeamNames") or [a.get("Name") for a in info.get("Agents", [])]
    wanted = {_normalise(a) for a in agents}
    return [i for i, name in enumerate(names or [])
            if isinstance(name, str) and _normalise(name) in wanted]


def seat_outcomes(frames: list[list[dict[str, Any]]]) -> dict[int, int]:
    """``{seat: +1 win / -1 loss / 0 draw}``.

    Taken from the terminal per-seat ``reward``, not from ``state.result``: measured on
    these replays ``result`` is -1 on the final frame of every episode, i.e. it is never
    populated, while the last frame's reward is +1/-1 for the two seats.
    """
    outcomes: dict[int, int] = {}
    for frame in reversed(frames):
        for seat, record in enumerate(frame):
            if seat in outcomes:
                continue
            reward = record.get("reward")
            if reward is not None and record.get("status") in ("DONE", "INVALID", "TIMEOUT"):
                outcomes[seat] = int(np.sign(float(reward)))
        if len(outcomes) >= len(frames[-1]):
            break
    return outcomes


def episode_rows(
    path: Path,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, int]], dict[str, Any]]:
    """Every paired decision in one replay, plus per-seat stats and the ``info`` block.

    ``info`` is returned rather than re-read because these replays are ~5MB each and a
    second ``json.loads`` to recover two agent names would roughly double the build cost.

    The replay stores an action beside the observation that *resulted* from it, so each
    active selection is paired with the next action the same seat submitted. An episode
    whose pairing breaks anywhere is rejected wholesale by the caller: a broken pairing
    means the causal chain is untrustworthy, and its other frames would still carry a
    ``decision_chain`` assembled from it.
    """
    replay = json.loads(path.read_text(encoding="utf-8"))
    frames = replay["steps"]
    episode_id = int(replay.get("info", {}).get("EpisodeId", -1))
    outcomes = seat_outcomes(frames)

    pending: dict[int, dict[str, Any] | None] = {}
    rows: list[dict[str, Any]] = []
    stats: dict[int, dict[str, int]] = {}
    rejected = 0

    for frame_index, frame in enumerate(frames):
        for seat, record in enumerate(frame):
            previous = pending.pop(seat, None)
            if previous is not None:
                action = record.get("action")
                if is_valid_target(action, previous["selection"], len(previous["options"])):
                    rows.append({**previous, "target_action": action,
                                 "reward": float(record.get("reward") or 0.0)})
                else:
                    rejected += 1

            if is_decision(record, seat):
                observation = record["observation"]
                selection = observation["select"]
                state = observation["current"]
                bucket = stats.setdefault(seat, {"n_decisions": 0, "max_turn": 0})
                bucket["n_decisions"] += 1
                bucket["max_turn"] = max(bucket["max_turn"], int(state.get("turn") or 0))
                pending[seat] = {
                    "episode_id": episode_id,
                    "frame_index": frame_index,
                    "player_index": seat,
                    "state": state,
                    "selection": {key: selection.get(key) for key in SELECTION_FIELDS},
                    "options": [normalise_option(o) for o in selection["option"]],
                }
    if rejected:
        return [], stats, replay
    for seat, bucket in stats.items():
        bucket["result"] = outcomes.get(seat, 0)
    return rows, stats, replay


# ------------------------------------------------------------------------- discovery


def replay_files(root: Path) -> list[tuple[int, int, Path]]:
    """``(episode_id, day_index, path)`` for every replay, in ascending episode order.

    Read from the *names* only. ``day_index`` numbers the dated directories oldest-first,
    which is what the recency-weighting experiments key on; the sort that actually decides
    the file layout is by ``episode_id``, because that — not the date — is what the
    contiguity invariant is expressed in.

    The same episode does appear under more than one dated directory (the daily dumps
    overlap at their boundaries), and writing it twice would put two copies of one game in
    the corpus — near-duplicate frames that a game-level split cannot separate, since the
    split partitions episode *ids*. First occurrence wins, which is the earliest day.

    Two directory layouts are accepted, because the two archetype corpora are shaped
    differently and neither is worth copying 9-22GB to normalise. Either the replays sit in
    dated subdirectories (``2026-07-23_cleaned_yushin/87557436_yushin_ito.json``), or they
    sit flat in one directory with the date in each filename
    (``2026-07-26_88277379_flg.json``). The day is read from the filename when it carries
    one and from the parent directory otherwise, so both produce the same ``day_index``.
    """
    paths = sorted(root.rglob("*.json"))
    if not paths:
        return []

    def day_of(path: Path) -> str:
        own = _DAY.search(path.name)
        if own:
            return own.group(1)
        parent = _DAY.search(path.parent.name)
        # No date anywhere means recency weighting would be silently wrong rather than
        # absent, so refuse instead of bucketing everything into one day.
        if parent is None:
            raise SystemExit(f"no YYYY-MM-DD in {path.name} or its directory")
        return parent.group(1)

    days = sorted({day_of(p) for p in paths})
    if len(days) > 128:
        raise SystemExit(f"{len(days)} days does not fit day_index's int8; widen the column")
    index_of = {day: i for i, day in enumerate(days)}

    found: dict[int, tuple[int, int, Path]] = {}
    duplicates = 0
    for path in paths:
        digits = _EPISODE_ID.findall(path.stem)
        if not digits:
            continue
        episode_id = int(digits[-1])
        if episode_id in found:
            duplicates += 1
            continue
        found[episode_id] = (episode_id, index_of[day_of(path)], path)
    if duplicates:
        print(f"[corpus] {duplicates:,} replay files skipped as duplicates of an "
              f"episode already seen on an earlier day")
    return [found[key] for key in sorted(found)]


# ---------------------------------------------------------------------------- writer


class CorpusWriter:
    """Buffers whole episodes and flushes row groups only on episode boundaries."""

    def __init__(self, path: Path, schema: pa.Schema, rows_per_group: int,
                 compression: str) -> None:
        self.schema, self.rows_per_group = schema, rows_per_group
        self._writer = pq.ParquetWriter(path, schema, compression=compression)
        self._buffer: list[dict[str, Any]] = []
        self.rows_written = 0
        self.groups_written = 0

    def add_episode(self, rows: list[dict[str, Any]]) -> None:
        self._buffer.extend(rows)
        if len(self._buffer) >= self.rows_per_group:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        table = pa.Table.from_pylist(self._buffer, schema=self.schema)
        self._writer.write_table(table, row_group_size=table.num_rows)
        self.rows_written += table.num_rows
        self.groups_written += 1
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        if self.rows_written == 0:
            self._writer.write_table(pa.Table.from_pylist([], schema=self.schema))
        self._writer.close()


def check_episode(rows: list[dict[str, Any]], previous: int | None) -> int:
    """Fail loudly rather than emit a file whose history scans are quietly truncated."""
    episode = rows[0]["episode_id"]
    if any(row["episode_id"] != episode for row in rows):
        raise SystemExit(f"episode {episode}: mixed episode_id inside one file")
    if previous is not None and episode <= previous:
        raise SystemExit(f"episode {episode} follows {previous}; not ascending")
    frames = [row["frame_index"] for row in rows]
    if any(b < a for a, b in zip(frames, frames[1:])):
        raise SystemExit(f"episode {episode}: frame_index is not ascending")
    return episode


# ------------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replays", type=Path, default=_ROOT / "trajectories_dataset",
                        help="directory of <date>_cleaned_<slug>/ replay JSON directories")
    parser.add_argument("--out", type=Path, default=_ROOT / "alakazam_yushin.parquet")
    parser.add_argument("--deck-contains", type=int, default=None, metavar="CARD_ID",
                        help="select seats whose opening 60-card deck contains this card "
                             "(647 = Marnie's Grimmsnarl ex). Preferred over --agent: it "
                             "follows the deck rather than the account, so a pilot who "
                             "plays two archetypes contributes only their games on this one")
    parser.add_argument("--agent", action="extend", nargs="+", default=None,
                        help="the pilot(s) being cloned, named as in TeamNames or as the "
                             "filename slug (case and underscores are folded). Several are "
                             "allowed when they pilot the SAME decklist -- verify that "
                             "before adding one, since cloning two different decks into one "
                             "policy teaches it the average of two strategies")
    parser.add_argument("--rows-per-group", type=int, default=2400,
                        help="target rows per row group. Smaller than the Grimmsnarl "
                             "corpus's 4800 because the loader holds several groups "
                             "resident and this box has 16GB")
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--min-options", type=int, default=2,
                        help="drop frames offering fewer legal options (forced moves)")
    parser.add_argument("--min-decisions", type=int, default=15)
    parser.add_argument("--max-decisions", type=int, default=320)
    parser.add_argument("--min-turn", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0,
                        help="stop after this many replays (for a smoke test)")
    args = parser.parse_args()

    files = replay_files(args.replays)
    if not files:
        raise SystemExit(f"no <episode_id>*.json replays under {args.replays}")
    if not args.agent and args.deck_contains is None:
        raise SystemExit("pass --deck-contains CARD_ID (preferred) or --agent NAME ...")
    if args.limit:
        files = files[: args.limit]
    print(f"[corpus] {len(files):,} replays over "
          f"{len({day for _, day, _ in files})} days -> {args.out}")

    writer = CorpusWriter(args.out, make_schema(), args.rows_per_group, args.compression)
    previous: int | None = None
    seen = samples = forced = 0
    dropped_pairing = dropped_no_seat = dropped_screen = dropped_id_mismatch = 0
    wins = losses = draws = 0
    pilots: collections.Counter = collections.Counter()
    started = time.time()

    try:
        for count, (episode_id, day_index, path) in enumerate(files, start=1):
            rows, stats, replay = episode_rows(path)
            info = replay.get("info", {})
            if not rows:
                dropped_pairing += 1
                continue
            if rows[0]["episode_id"] != episode_id:
                # The ascending order this file's layout depends on was derived from the
                # names, but the rows carry ``info.EpisodeId``. If those ever disagree the
                # ordering guarantee is void, so the episode is dropped rather than
                # written out of order.
                dropped_id_mismatch += 1
                continue
            seats = (deck_seats(replay, args.deck_contains) if args.deck_contains
                     else target_seats(info, args.agent))
            if not seats:
                dropped_no_seat += 1
                continue

            # Screened per seat, not per episode: when two target pilots meet, one of them
            # can concede a game the other played out properly, and a single verdict would
            # either discard the good seat or keep the bad one.
            clean_seats = set()
            for seat in seats:
                bucket = stats.get(seat, {"n_decisions": 0, "max_turn": 0, "result": 0})
                if (args.min_decisions <= bucket["n_decisions"] <= args.max_decisions
                        and bucket["max_turn"] >= args.min_turn):
                    clean_seats.add(seat)
                else:
                    dropped_screen += 1
                result = int(bucket.get("result", 0))
                wins += result > 0
                losses += result < 0
                draws += result == 0
            for seat in seats:
                names = info.get("TeamNames") or []
                pilots[_normalise(names[seat]) if seat < len(names) else "(unnamed)"] += 1

            kept = []
            for row in rows:
                if len(row["options"]) < args.min_options:
                    forced += 1
                    continue
                is_sample = row["player_index"] in clean_seats
                samples += is_sample
                # Each row's own seat outcome, read from that seat's record rather than
                # negated from the target's. With two target seats there is no single
                # "the target", and negating would be wrong for a draw either way.
                kept.append({**row, "is_target_archetype": is_sample,
                             "seat_result": int(stats.get(row["player_index"], {})
                                                .get("result", 0)),
                             "day_index": day_index})
            if not kept:
                continue
            previous = check_episode(kept, previous)
            writer.add_episode(kept)
            seen += 1

            if count % 250 == 0:
                elapsed = time.time() - started
                print(f"  {count:>6,}/{len(files):,} replays  "
                      f"{writer.rows_written + len(writer._buffer):>9,} rows  "
                      f"{samples:>9,} samples  "
                      f"{elapsed:.0f}s ({count / max(elapsed, 1e-9):.1f} replays/s)",
                      flush=True)
    finally:
        writer.close()

    size = args.out.stat().st_size / 1e9
    print(f"\nwrote {args.out}")
    print(f"  episodes     {seen:,} kept  "
          f"({dropped_pairing:,} dropped on a broken action pairing, "
          f"{dropped_no_seat:,} with no target seat, "
          f"{dropped_id_mismatch:,} whose id disagreed with their filename)")
    print(f"  rows         {writer.rows_written:,} readable (both seats)")
    print(f"  samples      {samples:,} flagged is_target_archetype")
    print(f"  forced       {forced:,} rows dropped (<{args.min_options} legal options)")
    print(f"  screened     {dropped_screen:,} episodes kept but unflagged "
          f"(concession/disconnect screen)")
    for name, count in pilots.most_common():
        print(f"  pilot        {name!r}: {count:,} seats")
    print(f"  pilot record {wins:,}W {losses:,}L {draws:,}D "
          f"({wins / max(wins + losses, 1):.1%} of decided games)")
    print(f"  row groups   {writer.groups_written:,} "
          f"(~{writer.rows_written // max(writer.groups_written, 1):,} rows each)")
    print(f"  size         {size:.2f} GB")
    print(f"  elapsed      {(time.time() - started) / 60:.1f} min")


if __name__ == "__main__":
    main()
