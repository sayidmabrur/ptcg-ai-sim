from __future__ import annotations

import numpy as np

from extractor import (
    CATEGORICAL_COLUMNS,
    CATEGORICAL_VOCABS,
    CONDITION_COLUMNS,
    CSV_COLUMNS,
    ENERGY_COUNT_COLUMNS,
    ENERGY_PLAYED,
    GLOBAL_COLUMNS,
    MAX_ATTACHED_ENERGY,
    MAX_BENCH,
    MATCHUP_COLUMNS,
    MAX_PRE_EVOLUTION,
    MAX_TOOLS,
    MY_ACTIVE,
    MY_BENCH,
    N_SLOTS,
    NUMERIC_COLUMNS,
    OPP_ACTIVE,
    OPP_BENCH,
    OWNERS,
    STADIUM_IN_PLAY,
    STADIUM_PLAYED,
    SUPPORTER_PLAYED,
    Extractor,
    damage_after_modifiers,
    energy_shortfall,
    Observation,
    _CAT,
    _NUM,
    _SLOT_ZONE,
    to_observation_class,
)


class BoardExtractor(Extractor):
    SPECS: dict[str, tuple[tuple[int, ...], type]] = {
        "input_ids": ((N_SLOTS,), np.int64),
        "tools": ((N_SLOTS, MAX_TOOLS), np.int64),
        "energies": ((N_SLOTS, MAX_ATTACHED_ENERGY), np.int64),
        "pre_evolution": ((N_SLOTS, MAX_PRE_EVOLUTION), np.int64),
        "energy_counts": ((N_SLOTS, len(ENERGY_COUNT_COLUMNS)), np.float32),
        "matchup": ((N_SLOTS, len(MATCHUP_COLUMNS)), np.float32),
        "categorical": ((N_SLOTS, len(CATEGORICAL_COLUMNS)), np.int64),
        "numeric": ((N_SLOTS, len(NUMERIC_COLUMNS)), np.float32),
        "card_masks": ((N_SLOTS,), np.bool_),
        "globals": ((len(GLOBAL_COLUMNS),), np.float32),
    }

    NUMERIC_SCALE = np.ones(len(NUMERIC_COLUMNS), dtype=np.float32)
    NUMERIC_SCALE[[_NUM["max_hp"], _NUM["hp"]]] = 340.0
    NUMERIC_SCALE[_NUM["retreat_cost"]] = 4.0
    GLOBAL_SCALE = np.array([60.0, 60.0, 6.0, 20.0, 8.0, 60.0, 6.0, 20.0, 8.0, 1.0, 1.0, 1.0, 30.0], dtype=np.float32)
    ENERGY_COUNT_SCALE = 4.0
    MATCHUP_SCALE = np.array([1.0, 1.0, 5.0, 5.0, 2.0, 2.0, 1.0, 1.0], dtype=np.float32)

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self.normalize = normalize

    def encode(self, obs: dict | Observation) -> dict[str, np.ndarray]:
        if isinstance(obs, dict):
            obs = to_observation_class(obs)
        state = obs.current
        if state is None or obs.select is None:
            raise ValueError("observation has no board (initial deck selection); skip it")

        me = state.yourIndex
        opp = 1 - me
        turn_player = self._turn_player(state.turn, state.firstPlayer)
        plays = self._update_plays(me, obs, turn_player)

        out = {key: np.zeros(shape, dtype) for key, (shape, dtype) in self.SPECS.items()}
        out["categorical"][:, _CAT["zone"]] = _SLOT_ZONE
        out["card_masks"][:] = True

        for owner_seat, active_slot, bench_slot in ((me, MY_ACTIVE, MY_BENCH), (opp, OPP_ACTIVE, OPP_BENCH)):
            player = state.players[owner_seat]
            owner = OWNERS.index("ME" if owner_seat == me else "OPP")
            out["categorical"][active_slot : bench_slot + MAX_BENCH, _CAT["owner"]] = owner
            target = (state.players[1 - owner_seat].active or [None])[0]
            if player.active:
                self._fill_pokemon(out, active_slot, player.active[0])
                self._fill_matchup(out, active_slot, player.active[0], target)
                conditions = (player.poisoned, player.burned, player.asleep, player.paralyzed, player.confused)
                out["numeric"][active_slot, _NUM[CONDITION_COLUMNS[0]] :] = conditions
            for i, pokemon in enumerate(player.bench[:MAX_BENCH]):
                self._fill_pokemon(out, bench_slot + i, pokemon)
                self._fill_matchup(out, bench_slot + i, pokemon, target)
            out["card_masks"][bench_slot + min(player.benchMax, MAX_BENCH) : bench_slot + MAX_BENCH] = False

        if state.stadium:
            self._fill_card(out, STADIUM_IN_PLAY, state.stadium[0].id)

        if turn_player is not None:
            owner = OWNERS.index("ME" if turn_player == me else "OPP")
            out["categorical"][SUPPORTER_PLAYED:, _CAT["owner"]] = owner
            for slot, played, card_id in (
                (SUPPORTER_PLAYED, state.supporterPlayed, plays.supporter),
                (ENERGY_PLAYED, state.energyAttached, plays.energy),
                (STADIUM_PLAYED, state.stadiumPlayed, plays.stadium),
            ):
                if played:
                    self._fill_card(out, slot, card_id)

        me_state, opp_state = state.players[me], state.players[opp]
        out["globals"][:] = (
            state.turn,
            me_state.deckCount, len(me_state.prize), me_state.handCount, me_state.benchMax,
            opp_state.deckCount, len(opp_state.prize), opp_state.handCount, opp_state.benchMax,
            turn_player == me,
            state.firstPlayer == me,
            state.retreated,
            state.turnActionCount,
        )

        if self.normalize:
            out["numeric"] /= self.NUMERIC_SCALE
            out["globals"] /= self.GLOBAL_SCALE
            out["energy_counts"] /= self.ENERGY_COUNT_SCALE
            out["matchup"] /= self.MATCHUP_SCALE
        return out

    def _fill_matchup(self, out: dict[str, np.ndarray], slot: int, pokemon, target) -> None:
        card = self._card_data.get(pokemon.id) if pokemon is not None else None
        attacks = [self._attacks[a] for a in card.attacks if a in self._attacks] if card is not None else []
        if not attacks:
            return
        attached = [int(e) for e in pokemon.energies]
        defender = self._card_data.get(target.id) if target is not None and target.hp > 0 else None
        rows = []
        for attack in attacks:
            shortfall = energy_shortfall([int(e) for e in attack.energies], attached)
            damage = damage_after_modifiers(attack.damage, card, defender) if defender is not None else 0
            ratio = min(damage / target.hp, 2.0) if defender is not None else 0.0
            rows.append((shortfall, ratio, damage > 0 and defender is not None and damage >= target.hp))
        best = max(rows, key=lambda r: (r[1], -r[0]))
        ready = [r for r in rows if r[0] == 0]
        out["matchup"][slot] = (
            1.0,
            bool(ready),
            min(r[0] for r in rows),
            best[0],
            best[1],
            max((r[1] for r in ready), default=0.0),
            any(r[2] for r in ready),
            any(r[2] for r in rows if r[0] <= 1),
        )

    def decode(self, features: dict[str, np.ndarray]) -> list[dict[str, object]]:
        numeric = features["numeric"] * (self.NUMERIC_SCALE if self.normalize else 1)
        globals_ = features["globals"] * (self.GLOBAL_SCALE if self.normalize else 1)
        matchup = features["matchup"] * (self.MATCHUP_SCALE if self.normalize else 1)
        energy_counts = features["energy_counts"] * (self.ENERGY_COUNT_SCALE if self.normalize else 1)
        global_values = {name: int(round(v)) for name, v in zip(GLOBAL_COLUMNS, globals_)}
        rows = []
        for slot in range(N_SLOTS):
            row: dict[str, object] = dict(global_values, slot=slot)
            row["card_id"] = self.vocab[features["input_ids"][slot]]
            row["available"] = int(features["card_masks"][slot])
            for name, value in zip(CATEGORICAL_COLUMNS, features["categorical"][slot]):
                row[name] = CATEGORICAL_VOCABS[name][value]
            for name, value in zip(NUMERIC_COLUMNS, numeric[slot]):
                row[name] = int(round(value))
            for i in range(MAX_TOOLS):
                row[f"tool_{i + 1}"] = self.vocab[features["tools"][slot, i]]
            for i in range(MAX_ATTACHED_ENERGY):
                row[f"att_e_{i + 1}"] = self.vocab[features["energies"][slot, i]]
            for i in range(MAX_PRE_EVOLUTION):
                row[f"pre_evo_{i + 1}"] = self.vocab[features["pre_evolution"][slot, i]]
            for name, value in zip(ENERGY_COUNT_COLUMNS, energy_counts[slot]):
                row[name] = int(round(value))
            for name, value in zip(MATCHUP_COLUMNS, matchup[slot]):
                row[name] = round(float(value), 3)
            rows.append({name: row[name] for name in CSV_COLUMNS})
        return rows
