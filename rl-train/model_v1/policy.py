from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_EXTRACTORS = Path(__file__).resolve().parent / "feature_extractor"
if str(_EXTRACTORS) not in sys.path:
    sys.path.insert(0, str(_EXTRACTORS))

from board_extractor import BoardExtractor  # noqa: E402
from card_set_extractor import MAX_HAND, MAX_LOOKING, MAX_PILE, PILES, CardSetExtractor  # noqa: E402
from extractor import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    CATEGORICAL_VOCABS,
    ENERGY_COUNT_COLUMNS,
    GLOBAL_COLUMNS,
    MATCHUP_COLUMNS,
    N_AREAS,
    N_SLOTS,
    NUMERIC_COLUMNS,
    OWNERS,
)
from log_extractor import LOG_NUMERIC_COLUMNS, MAX_LOGS, N_LOG_TYPES, LogExtractor  # noqa: E402
from option_extractor import (  # noqa: E402
    MAX_OPTIONS,
    N_OPTION_TYPES,
    N_REF_TOKENS,
    N_SELECT_CONTEXTS,
    N_SELECT_TYPES,
    N_SPECIAL_CONDITIONS,
    OPTION_NUMERIC_COLUMNS,
    SELECT_NUMERIC_COLUMNS,
    OptionExtractor,
)

EXTRACTOR_SPECS: dict[str, tuple[tuple[int, ...], type]] = {}
for _cls in (BoardExtractor, CardSetExtractor, OptionExtractor, LogExtractor):
    EXTRACTOR_SPECS.update(_cls.SPECS)

N_ACTIONS = MAX_OPTIONS + 1
DONE = MAX_OPTIONS
ENV_SPECS: dict[str, tuple[tuple[int, ...], type]] = {
    "action_mask": ((N_ACTIONS,), np.bool_),
    "picked_count": ((1,), np.float32),
    "option_picked": ((MAX_OPTIONS,), np.bool_),
}
ALL_SPECS = {**EXTRACTOR_SPECS, **ENV_SPECS}
IN_KEYS = tuple(ALL_SPECS)

(SEG_GLOBAL, SEG_BOARD, SEG_HAND, SEG_LOOKING, SEG_PILE, SEG_LOG) = range(6)
N_SEGMENTS = 6
BOARD_START = 2
REF_END = BOARD_START + N_SLOTS + MAX_HAND + MAX_LOOKING
assert N_REF_TOKENS == 1 + REF_END - BOARD_START, "pointer space must match the board / hand / looking tokens"
N_TOKENS = REF_END + len(PILES) * MAX_PILE + MAX_LOGS
SUB_INDEX_SIZE = 32


@dataclass
class PolicyConfig:
    d_model: int = 128
    n_heads: int = 4
    n_encoder_layers: int = 6
    n_decoder_layers: int = 2
    ffn_mult: int = 4
    dropout: float = 0.0


class CardEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_dim, card_features=None):
        super().__init__()
        self.embedding_dim = embed_dim
        self.learned = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.static_proj = None
        if card_features is not None:
            features = torch.as_tensor(card_features, dtype=torch.float32)
            assert features.shape[0] == vocab_size, "card_features rows must match the card vocab"
            self.register_buffer('static', features)
            self.static_proj = nn.Linear(features.shape[1], embed_dim, bias=False)

    def forward(self, ids):
        x = self.learned(ids)
        if self.static_proj is not None:
            x = x + self.static_proj(self.static[ids])
        return x


class BoardEmbedding(nn.Module):
    def __init__(self, card_embed: CardEmbedding, d: int):
        super().__init__()
        self.card_embed = card_embed
        sizes = [len(CATEGORICAL_VOCABS[c]) for c in CATEGORICAL_COLUMNS]
        self.register_buffer('offsets', torch.tensor([0] + sizes[:-1]).cumsum(0))
        self.categorical = nn.Embedding(sum(sizes), d)
        self.tools = nn.Linear(d, d, bias=False)
        self.energies = nn.Linear(d, d, bias=False)
        self.pre_evolution = nn.Linear(d, d, bias=False)
        self.numeric = nn.Linear(len(NUMERIC_COLUMNS) + len(ENERGY_COUNT_COLUMNS) + len(MATCHUP_COLUMNS), d)
        self.slot = nn.Embedding(N_SLOTS, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, b):
        embed = self.card_embed
        x = embed(b['input_ids'])
        x = x + self.tools(embed(b['tools']).sum(2))
        x = x + self.energies(embed(b['energies']).sum(2))
        x = x + self.pre_evolution(embed(b['pre_evolution']).sum(2))
        x = x + self.categorical(b['categorical'] + self.offsets).sum(2)
        x = x + self.numeric(torch.cat([b['numeric'], b['energy_counts'], b['matchup']], dim=-1))
        return self.norm(x + self.slot.weight)


class HandEmbedding(nn.Module):
    def __init__(self, card_embed: CardEmbedding, d: int):
        super().__init__()
        self.card_embed = card_embed
        self.position = nn.Embedding(MAX_HAND, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, b):
        return self.norm(self.card_embed(b['hand_ids']) + self.position.weight)


class LookingEmbedding(nn.Module):
    def __init__(self, card_embed: CardEmbedding, d: int):
        super().__init__()
        self.card_embed = card_embed
        self.owner = nn.Embedding(len(OWNERS), d)
        self.position = nn.Embedding(MAX_LOOKING, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, b):
        x = self.card_embed(b['looking_ids']) + self.owner(b['looking_owner']) + self.position.weight
        return self.norm(x)


class PileEmbedding(nn.Module):
    def __init__(self, card_embed: CardEmbedding, d: int):
        super().__init__()
        self.card_embed = card_embed
        self.count = nn.Linear(1, d)
        self.pile = nn.Embedding(len(PILES), d)
        self.norm = nn.LayerNorm(d)

    def forward(self, b):
        x = self.card_embed(b['pile_ids']) + self.count(b['pile_counts'].unsqueeze(-1)) + self.pile.weight[:, None, :]
        return self.norm(x.flatten(1, 2))


class LogEmbedding(nn.Module):
    def __init__(self, card_embed: CardEmbedding, attack_embed: nn.Embedding, ref_embed: nn.Embedding, d: int):
        super().__init__()
        self.card_embed, self.attack_embed, self.ref_embed = card_embed, attack_embed, ref_embed
        self.kind = nn.Embedding(N_LOG_TYPES, d, padding_idx=0)
        self.owner = nn.Embedding(len(OWNERS), d)
        self.card_role = nn.Embedding(3, d)
        self.area = nn.Embedding(N_AREAS, d, padding_idx=0)
        self.area_role = nn.Embedding(2, d)
        self.ref_role = nn.Embedding(2, d)
        self.numeric = nn.Linear(len(LOG_NUMERIC_COLUMNS), d)
        self.position = nn.Embedding(MAX_LOGS, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, b):
        x = self.kind(b['log_type']) + self.owner(b['log_owner']) + self.attack_embed(b['log_attack'])
        x = x + (self.card_embed(b['log_cards']) + self.card_role.weight).sum(2)
        x = x + (self.area(b['log_areas']) + self.area_role.weight).sum(2)
        x = x + (self.ref_embed(b['log_refs']) + self.ref_role.weight).sum(2)
        x = x + self.numeric(b['log_numeric']) + self.position.weight
        return self.norm(x)


class SelectEmbedding(nn.Module):
    def __init__(self, card_embed: CardEmbedding, d: int):
        super().__init__()
        self.card_embed = card_embed
        self.kind = nn.Embedding(N_SELECT_TYPES, d)
        self.context = nn.Embedding(N_SELECT_CONTEXTS, d)
        self.numeric = nn.Linear(len(SELECT_NUMERIC_COLUMNS), d)
        self.norm = nn.LayerNorm(d)

    def forward(self, b):
        question = self.kind(b['select_type']) + self.context(b['select_context'])
        x = question + self.numeric(b['select_numeric']) + self.card_embed(b['select_cards']).sum(1)
        return self.norm(x), question


class OptionEmbedding(nn.Module):
    def __init__(self, card_embed: CardEmbedding, attack_embed: nn.Embedding, d: int):
        super().__init__()
        self.card_embed, self.attack_embed = card_embed, attack_embed
        self.kind = nn.Embedding(N_OPTION_TYPES, d, padding_idx=0)
        self.area = nn.Embedding(N_AREAS, d, padding_idx=0)
        self.owner = nn.Embedding(len(OWNERS), d)
        self.sub_index = nn.Embedding(SUB_INDEX_SIZE, d, padding_idx=0)
        self.condition = nn.Embedding(N_SPECIAL_CONDITIONS, d, padding_idx=0)
        self.picked = nn.Embedding(2, d)
        self.numeric = nn.Linear(len(OPTION_NUMERIC_COLUMNS), d)
        self.src = nn.Linear(d, d, bias=False)
        self.dst = nn.Linear(d, d, bias=False)
        self.norm = nn.LayerNorm(d)

    @staticmethod
    def _gather(memory, index):
        return memory.gather(1, index.unsqueeze(-1).expand(-1, -1, memory.shape[-1]))

    def forward(self, b, ref_memory, question):
        x = self.kind(b['option_type']) + self.area(b['src_area']) + self.owner(b['owner'])
        x = x + self.card_embed(b['card_id']) + self.attack_embed(b['attack_id'])
        x = x + self.sub_index(b['sub_index'].clamp(max=SUB_INDEX_SIZE - 1)) + self.condition(b['special_condition'])
        x = x + self.numeric(b['option_numeric']) + self.picked(b['option_picked'].long())
        x = x + self.src(self._gather(ref_memory, b['src_ref'])) + self.dst(self._gather(ref_memory, b['dst_ref']))
        return self.norm(x + question.unsqueeze(1))


class PolicyNetwork(nn.Module):
    def __init__(self, vocab_size: int, attack_vocab_size: int, card_features=None, config: PolicyConfig | None = None):
        super().__init__()
        config = config or PolicyConfig()
        d = config.d_model
        self.config = config

        self.card_embed = CardEmbedding(vocab_size, d, card_features)
        self.attack_embed = nn.Embedding(attack_vocab_size, d, padding_idx=0)
        self.ref_embed = nn.Embedding(N_REF_TOKENS, d, padding_idx=0)
        self.segment = nn.Embedding(N_SEGMENTS, d)

        self.globals = nn.Linear(len(GLOBAL_COLUMNS), d)
        self.select = SelectEmbedding(self.card_embed, d)
        self.board = BoardEmbedding(self.card_embed, d)
        self.hand = HandEmbedding(self.card_embed, d)
        self.looking = LookingEmbedding(self.card_embed, d)
        self.piles = PileEmbedding(self.card_embed, d)
        self.logs = LogEmbedding(self.card_embed, self.attack_embed, self.ref_embed, d)
        self.options = OptionEmbedding(self.card_embed, self.attack_embed, d)

        layer = dict(d_model=d, nhead=config.n_heads, dim_feedforward=d * config.ffn_mult, dropout=config.dropout,
                     activation='gelu', batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(**layer), config.n_encoder_layers,
                                             norm=nn.LayerNorm(d), enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(**layer), config.n_decoder_layers,
                                             norm=nn.LayerNorm(d))
        self.done_query = nn.Parameter(torch.zeros(d))
        self.done_proj = nn.Linear(2, d)
        self.done_norm = nn.LayerNorm(d)
        self.logit_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))


    @staticmethod
    def _flatten(batch: Mapping[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], tuple[int, ...]]:
        lead = tuple(batch['globals'].shape[:-1])
        flat = {}
        for key, (shape, _) in ALL_SPECS.items():
            x = batch[key]
            flat[key] = x.reshape(-1, *x.shape[x.ndim - len(shape):]) if shape else x.reshape(-1)
        return flat, lead

    def _encode_state(self, b):
        n = b['globals'].shape[0]
        ones = b['globals'].new_ones((n, 1), dtype=torch.bool)
        select_token, question = self.select(b)
        seg = self.segment.weight
        tokens = torch.cat([
            (self.globals(b['globals']) + seg[SEG_GLOBAL]).unsqueeze(1),
            select_token.unsqueeze(1) + seg[SEG_GLOBAL],
            self.board(b) + seg[SEG_BOARD],
            self.hand(b) + seg[SEG_HAND],
            self.looking(b) + seg[SEG_LOOKING],
            self.piles(b) + seg[SEG_PILE],
            self.logs(b) + seg[SEG_LOG],
        ], dim=1)
        mask = torch.cat([ones, ones, b['card_masks'], b['hand_mask'], b['looking_mask'],
                          b['pile_mask'].flatten(1), b['log_mask']], dim=1)
        if n == 1:
            keep = mask[0]
            state = torch.zeros_like(tokens)
            state[0, keep] = self.encoder(tokens[:, keep])[0]
            return state, mask, question
        return self.encoder(tokens, src_key_padding_mask=~mask), mask, question


    def forward(self, *tensors) -> tuple[torch.Tensor, torch.Tensor]:
        batch = tensors[0] if len(tensors) == 1 and isinstance(tensors[0], Mapping) else dict(zip(IN_KEYS, tensors))
        b, lead = self._flatten(batch)

        state, mask, question = self._encode_state(b)
        ref_memory = torch.cat([state.new_zeros(state.shape[0], 1, state.shape[-1]), state[:, BOARD_START:REF_END]], 1)
        options = self.options(b, ref_memory, question)
        stop_allowed = b['action_mask'][:, DONE:].to(options.dtype)
        done = self.done_norm(self.done_query + self.done_proj(torch.cat([stop_allowed, b['picked_count']], -1)) + question)
        queries = torch.cat([options, done.unsqueeze(1)], dim=1)
        query_mask = torch.cat([b['option_mask'], b['option_mask'].new_ones(queries.shape[0], 1)], dim=1)
        hidden = self.decoder(queries, state, tgt_key_padding_mask=~query_mask, memory_key_padding_mask=~mask)

        logits = self.logit_head(hidden).squeeze(-1).masked_fill(~b['action_mask'], float('-inf'))
        value = self.value_head(state[:, 0]).squeeze(-1)
        return logits.reshape(*lead, N_ACTIONS), value.reshape(*lead, 1)

    def value(self, *tensors) -> torch.Tensor:
        batch = tensors[0] if len(tensors) == 1 and isinstance(tensors[0], Mapping) else dict(zip(IN_KEYS, tensors))
        b, lead = self._flatten(batch)
        state, _, _ = self._encode_state(b)
        return self.value_head(state[:, 0]).reshape(*lead, 1)


class _ValueView(nn.Module):
    def __init__(self, network: PolicyNetwork):
        super().__init__()
        self.network = network

    def forward(self, *tensors):
        return self.network.value(*tensors)


class _LogitsView(nn.Module):
    def __init__(self, network: PolicyNetwork):
        super().__init__()
        self.network = network

    def forward(self, *tensors):
        return self.network(*tensors)[0]


def to_tensordict(features: Mapping[str, np.ndarray], device=None):
    from tensordict import TensorDict

    batch_size = [next(iter(features.values())).shape[0]]
    return TensorDict({key: torch.as_tensor(value) for key, value in features.items()}, batch_size=batch_size,
                      device=device)


def make_actor_critic(network: PolicyNetwork):
    from tensordict.nn import TensorDictModule
    from torchrl.modules import MaskedCategorical, ProbabilisticActor

    policy = TensorDictModule(_LogitsView(network), in_keys=list(IN_KEYS), out_keys=['logits'])
    actor = ProbabilisticActor(
        policy,
        in_keys={'logits': 'logits', 'mask': 'action_mask'},
        out_keys=['action'],
        distribution_class=MaskedCategorical,
        return_log_prob=True,
    )
    critic = TensorDictModule(_ValueView(network), in_keys=list(IN_KEYS), out_keys=['state_value'])
    return actor, critic


def build_policy(encoder, config: PolicyConfig | None = None) -> PolicyNetwork:
    return PolicyNetwork(encoder.vocab_size, encoder.attack_vocab_size, encoder.card_features, config)


def save_policy(network: PolicyNetwork, path, **extra) -> None:
    import os
    from dataclasses import asdict

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"state_dict": network.state_dict(), "config": asdict(network.config), **extra}, tmp)
    os.replace(tmp, path)


def load_policy(path, encoder, device="cpu") -> PolicyNetwork:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    network = build_policy(encoder, PolicyConfig(**checkpoint["config"]))
    network.load_state_dict(checkpoint["state_dict"])
    return network.to(device)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from helpers.helper import build_deck
    from env import PokemonTCGEnv

    deck = build_deck("./decks/main_agent.csv")
    env = PokemonTCGEnv(deck, [build_deck("./decks/opponents/dragapult_noir.csv")])
    network = build_policy(env.encoder)
    print(f"parameters: {count_parameters(network):,}  tokens/decision: {N_TOKENS}  actions: {N_ACTIONS}")
    actor, critic = make_actor_critic(network)
    with torch.no_grad():
        rollout = env.rollout(200, policy=actor, break_when_any_done=False)
    legal = rollout["action_mask"].gather(-1, rollout["action"].unsqueeze(-1)).all()
    print({key: tuple(rollout[key].shape) for key in ("logits", "action", "action_log_prob")})
    print("steps:", rollout.shape[0], "| episodes finished:", int(rollout["next", "done"].sum()),
          "| sampled actions all legal:", bool(legal))
    env.close()
