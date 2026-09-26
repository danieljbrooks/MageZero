import os
import torch
from torch import nn
import math
import gzip
from enum import Enum

"""
MageZero Neural Network architecture for AlphaZero style MCTS:
2M x 512 embedding table (sparse indices -> ~1300 tokens)
└── 2-layer TransformerEncoder (d=512, nhead=4, ff=1024)
    └── mean pool -> 512D
        ├── MLP(512 -> 256 -> A)    priority_player   (PlayerA priority actions)
        ├── MLP(512 -> 256 -> A)    priority_opponent (PlayerB priority actions)
        ├── MLP(512 -> 256 -> A)    target            (target choices, both players)
        ├── MLP(512 -> 256 -> 2)    binary            (attack/block use decisions)
        └── MLP(512 -> 256 -> 1)    value             (tanh, -1..1)

A, the policy width, is 128 unless an action vocabulary is configured (see action_dim).
"""


GLOBAL_MAX = 2**31-1

# CUDA when available; otherwise train/test fall back to CPU (e.g. to smoke-test on a laptop).
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# States per training/test batch. 512 by default. Long states (a format-wide game encodes
# ~1,700 features per state) need a smaller batch to fit in GPU memory: MZ_TRAIN_BATCH.
TRAIN_BATCH = int(os.environ.get("MZ_TRAIN_BATCH", 512))

def action_dim() -> int:
    """Width of the priority and target heads.

    128 by default, where the JVM hashes every action into 128 slots. When MZ_ACTION_VOCAB points
    at an action vocabulary (the same file the JVM's ActionEncoder reads), the width is its
    `dim` line, and every action listed in it gets a slot of its own. Opt-in: without the
    variable, nothing changes."""
    path = os.environ.get("MZ_ACTION_VOCAB")
    if not path:
        return 128
    with open(path) as f:
        for line in f:
            if line.startswith("dim\t"):
                return int(line.split("\t")[1])
    raise ValueError(f"action vocabulary {path} has no 'dim' line")


ACTION_DIM = action_dim()
PRIORITY_A_MAX = ACTION_DIM
PRIORITY_B_MAX = ACTION_DIM
TARGETS_MAX = ACTION_DIM
BINARY_MAX = 2
# The policy heads' loss weight is ln2/ln(K). K stays 128 when the heads widen, so a wider head
# weighs policy against value the same way a 128-slot head does. MZ_LOSS_K overrides it.
LOSS_K = int(os.environ.get("MZ_LOSS_K", 128))


class ActionType(Enum):
    PRIORITY = 0
    CHOOSE_TARGET = 3
    CHOOSE_USE = 5

def head_weight(K: int) -> float:
    if K <= 1:
        raise ValueError("K must be >= 2 for cross-entropy.")
    return math.log(2.0) / math.log(float(K))

#per head weights
lambda_pA = head_weight(LOSS_K)
lambda_pB = head_weight(LOSS_K)
lambda_t = head_weight(LOSS_K)
lambda_b = head_weight(BINARY_MAX)

class NetTransformer(nn.Module):
    def __init__(self, num_embeddings=2048, policy_size_pA=PRIORITY_A_MAX, policy_size_pB=PRIORITY_B_MAX, policy_size_t=TARGETS_MAX, policy_size_b=BINARY_MAX):
        super().__init__()

        embedding_dim = 512
        hidden_dim_mlp = 256
        self.input_dropout = 0.3


        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            sparse=False,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=4,
            dim_feedforward=1024, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.embedding_dropout = nn.Dropout(p=0.2)

        self.player_priority_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp), nn.ReLU(),
            nn.Linear(hidden_dim_mlp, policy_size_pA),
        )
        self.opponent_priority_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp), nn.ReLU(),
            nn.Linear(hidden_dim_mlp, policy_size_pB),
        )
        self.target_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp), nn.ReLU(),
            nn.Linear(hidden_dim_mlp, policy_size_t),
        )
        self.binary_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp), nn.ReLU(),
            nn.Linear(hidden_dim_mlp, policy_size_b),
        )
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp), nn.ReLU(),
            nn.Linear(hidden_dim_mlp, 1), nn.Tanh(),
        )

    def resize_embedding(self, num_embeddings: int, init_rows=None) -> None:
        """Grow the embedding table to `num_embeddings` rows, keeping existing rows unchanged.
        Used by dense-vocab training when a new generation adds features to the vocab. `init_rows`
        supplies the added rows (vocab.initial_rows draws each from its feature id, so a feature
        starts from the same row whenever it is first seen, whatever generation that is); without
        it they get nn.Embedding's default init."""
        old = self.embedding
        if num_embeddings == old.num_embeddings:
            return
        if num_embeddings < old.num_embeddings:
            raise ValueError("the feature vocab is append-only; the embedding table cannot shrink")
        new = nn.Embedding(num_embeddings, old.embedding_dim, sparse=old.sparse).to(old.weight.device)
        with torch.no_grad():
            new.weight[:old.num_embeddings] = old.weight
            if init_rows is not None:
                added = num_embeddings - old.num_embeddings
                new.weight[old.num_embeddings:] = torch.as_tensor(
                    init_rows, dtype=new.weight.dtype, device=new.weight.device)[:added]
        self.embedding = new

    def forward(self, indices, offsets):
        B = offsets.shape[0]
        ends = torch.cat([offsets[1:], torch.tensor([indices.shape[0]], device=offsets.device)])
        lengths = ends - offsets
        max_len = lengths.max().item()

        # reconstruct padded sequences
        padded = indices.new_zeros(B, max_len)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=indices.device)

        for i in range(B):
            l = lengths[i]
            padded[i, :l] = indices[offsets[i]:offsets[i] + l]
            mask[i, :l] = True

        if self.training and self.input_dropout > 0:
            drop = torch.rand(B, max_len, device=indices.device) < self.input_dropout
            mask = mask & ~drop


        emb = self.embedding(padded)  # (B, max_len, 512)
        emb = self.transformer(emb, src_key_padding_mask=~mask)  # (B, max_len, 512)

        # mean pool over real tokens
        pool_count = mask.sum(1).clamp(min=1).unsqueeze(-1).float()
        emb = (emb * mask.unsqueeze(-1)).sum(1) / pool_count

        emb = self.embedding_dropout(emb)

        return (
            self.player_priority_head(emb),
            self.opponent_priority_head(emb),
            self.target_head(emb),
            self.binary_head(emb),
            self.value_head(emb).squeeze(-1),
        )




def checkpoint_policy_width(state_dict) -> int:
    """Width of a checkpoint's policy heads (the final Linear of the player priority head)."""
    return state_dict["player_priority_head.2.weight"].shape[0]


def require_policy_width(state_dict, source: str) -> None:
    """Refuse a checkpoint whose policy heads don't match this run's width. The JVM indexes the
    logits with its own ActionEncoder, so a mismatch would point every prior at the wrong action."""
    width = checkpoint_policy_width(state_dict)
    if width != ACTION_DIM:
        raise ValueError(f"{source} has {width}-wide policy heads, but this run uses {ACTION_DIM} "
                         f"(MZ_ACTION_VOCAB={os.environ.get('MZ_ACTION_VOCAB') or 'unset'})")


def load_model(path):
    # map to CPU so a checkpoint saved on a GPU also loads where there is none;
    # load_state_dict then copies the weights onto the model's own device
    if path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            return torch.load(f, map_location="cpu")
    return torch.load(path, map_location="cpu")

def normalize_policy_labels(raw: torch.Tensor) -> torch.Tensor:
    total = raw.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return raw / total