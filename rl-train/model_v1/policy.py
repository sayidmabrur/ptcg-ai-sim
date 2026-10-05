import torch.nn as nn
import torch
from dataclasses import dataclass

@dataclass
class PosMaskIds:
    ACTIVE = 0
    BENCH = 1
    OPP_ACTIVE = 2
    OPP_BENCH = 3
    STADIUM_IN_PLAY = 4
    SUPPORTER_PLAYED = 5
    ENERGY_PLAYED = 6
    STADIUM_PLAYED = 7


class BoardPositionalEncoding(nn.Module):

    def __init__(self, card_embed_dim, max_bench_card=8):
        super().__init__()
        ids = (
            [PosMaskIds.OPP_ACTIVE]
            + [PosMaskIds.OPP_BENCH] * max_bench_card
            + [PosMaskIds.ACTIVE]
            + [PosMaskIds.BENCH] * max_bench_card
            + [
                PosMaskIds.STADIUM_IN_PLAY,
                PosMaskIds.SUPPORTER_PLAYED,
                PosMaskIds.ENERGY_PLAYED,
                PosMaskIds.STADIUM_PLAYED,
            ]
        )
        self.register_buffer('pe', torch.tensor(ids))
        # 8 for pos mask ids length
        self.pe_embed = nn.Embedding(8, card_embed_dim)

    def forward(self):
        return self.pe_embed(self.pe)


class BoardStateEmbedding(nn.Module):

    def __init__(self, card_embed: nn.Embedding, max_bench_card=8):
        super().__init__()
        self.card_embed = card_embed
        self.pos_encoding = BoardPositionalEncoding(card_embed.embedding_dim, max_bench_card)

        #linear layer for prize_count and deck_count
        self.linear1 = nn.Linear(2, card_embed.embedding_dim)
        
    def forward(self, x):

        # since card data indicies are in pos [0, N-1]
        # the last index  is prize count, need to exclude from embedding
        # board input always static, no need for dynamic pos encoding
        # card slots come first; trailing counts (prize/deck) are not card ids, so slice them off
        n_cards = self.pos_encoding.pe.shape[0]
        x_continuous = x[:,n_cards:]
        x = self.card_embed(x[:, :n_cards]) + self.pos_encoding()
        x_continuous = self.linear1(x_continuous)
        return x

class OpponentStateEmbedding(nn.Module):
    pass
class PlayerStateEmbedding(nn.Module):

    # PLAYER_STATE DATA SAMPLE: [HANDS, DISCARD]
    def __init__(self, card_embed: nn.Embedding):
        super().__init__()
        self.card_embed = card_embed

class StateEncoder(nn.Module):
    # data sample:
    # [BOARD_STATE, PLAYER_STATE]
    def __init__(self, card_vocab_size, card_embed_dim):
        super().__init__()
        
        self.card_embed = nn.Embedding(card_vocab_size, card_embed_dim)
        self.board_embedding = BoardStateEmbedding(self.card_embed)
        self.player_embedding = PlayerStateEmbedding(self.card_embed)
    pass
# class PlayerStateEncoder(nn.Module):
#
#     def __init__(self):


from helpers.helper import count_supported_cards, RL_TRAIN_ROOT
# +1 because card ids start at 1 (index 0 = <NONE>)
VOCAB_SIZE = count_supported_cards(RL_TRAIN_ROOT / "engine" / "EN_Card_Data_R2_full.csv") + 1
EMBED_DIM = 512
model = StateEncoder(card_vocab_size=VOCAB_SIZE, card_embed_dim=EMBED_DIM)
print(model)

# dummy board: [B, N_SLOTS] card ids, N_SLOTS = opp active + opp bench + active + bench + 4 board slots
B, MAX_BENCH = 4, 8
N_SLOTS = 2 * (1 + MAX_BENCH) + 4
sample_batch = torch.randint(0, VOCAB_SIZE, (B, N_SLOTS))
print(sample_batch.shape)
print(model.board_embedding(sample_batch).shape)
