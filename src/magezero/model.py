import contextlib
import os
import torch
from torch import nn
import math
import gzip
from enum import Enum

"""
MageZero Neural Network architecture for AlphaZero style MCTS:
2M sparse embedding bag -> 512D embedding layer -> 256D hidden layer -> (3 x 128D policy heads + 2D binary policy head + 1D value head)

Policy heads are for each decision type (disjoint action spaces) they are:
128D PriorityA (priority actions for PlayerA - which is this agent)
128D PriorityB (priority actions for PlayerB - which is the opponent)
128D Choose Target (target choices for both players)
2D Choose Use (binary decisions for either player - this is used for selecting attackers and blockers)
"""


def _action_dim() -> int:
    """Policy head width. Upstream hashes actions into 128 slots; a set vocabulary
    (MZ_ACTION_VOCAB, see experiments/action_vocab/README.md) gives every action in the set its
    own slot and declares the width on its `dim` line. The JVM reads the same file, so both
    sides agree. Checkpoints carry their own width (see build_model_from_checkpoint)."""
    path = os.environ.get("MZ_ACTION_VOCAB")
    if path:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("dim\t"):
                    return int(line.split("\t")[1])
        raise ValueError(f"no dim line in action vocabulary {path}")
    return 128


ACTIONS_MAX = _action_dim()
GLOBAL_MAX = 2000000

# Rows in the dense embedding table. Raw feature ids (0..GLOBAL_MAX) are folded into this
# many rows with a modulo. The 2M default needs ~12 GB for weights + Adam state, so smaller
# machines can set MZ_EMBED_ROWS (e.g. 262144) at the cost of a few % hash collisions.
EMBED_ROWS = int(os.environ.get("MZ_EMBED_ROWS", GLOBAL_MAX))

# States carry ~1-2k tokens before the ignore list prunes them, and attention memory grows
# with batch * tokens^2: 512 fits a large CUDA card, Apple unified memory needs ~32.
BATCH_SIZE = int(os.environ.get("MZ_BATCH_SIZE", 512))

# Round padded sequence length up to a multiple of this. Padding is masked out, so outputs are
# unchanged; on MPS it avoids per-shape kernel rebuilds (~20 s/step -> ~6 s/step measured).
PAD_BUCKET = int(os.environ.get("MZ_PAD_BUCKET", 1))


def pick_device() -> torch.device:
    """MZ_DEVICE overrides; otherwise CUDA, then Apple MPS, then CPU."""
    forced = os.environ.get("MZ_DEVICE")
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()
if DEVICE.type == "mps":
    # nn.TransformerEncoder's eval-mode fast path uses nested-tensor ops MPS lacks; with
    # PYTORCH_ENABLE_MPS_FALLBACK=1 they silently run on CPU, so use the regular path instead.
    torch.backends.mha.set_fastpath_enabled(False)


def autocast():
    """Mixed precision on CUDA only; a no-op elsewhere."""
    if DEVICE.type == "cuda":
        return torch.amp.autocast("cuda")
    return contextlib.nullcontext()



PRIORITY_A_MAX = ACTIONS_MAX
PRIORITY_B_MAX = ACTIONS_MAX
TARGETS_MAX = ACTIONS_MAX
BINARY_MAX = 2

# Loss weights are ln2/ln(K) per head. K stays at the upstream 128 when the heads widen, so a
# vocabulary run weighs policy vs value loss the same way as a 128-slot run (MZ_LOSS_K overrides).
LOSS_K = int(os.environ.get("MZ_LOSS_K", 128))


class ActionType(Enum):
    PRIORITY = 0
    CHOOSE_TARGET = 3
    CHOOSE_USE = 5

def head_weight(K: int) -> float:
    """
    Analytic loss weight to equalize baseline CE scales:
    lambda_K = ln(2) / ln(K)
    """
    if K <= 1:
        raise ValueError("K must be >= 2 for cross-entropy.")
    return math.log(2.0) / math.log(float(K))

#per head weights
lambda_pA = head_weight(LOSS_K)
lambda_pB = head_weight(LOSS_K)
lambda_t = head_weight(LOSS_K)
lambda_b = head_weight(BINARY_MAX)


class Net(nn.Module):
    def __init__(self, num_embeddings, policy_size_A):
        super().__init__()


        embedding_dim = 512  # Output of EmbeddingBag
        hidden_dim_mlp = 256  # Output of the main MLP block
        self.embedding_bag = nn.EmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            mode='sum',
            sparse=True,
            max_norm=1
        )
        self.embedding_bias = nn.Parameter(torch.zeros(embedding_dim))
        self.input_dropout = 0
        #self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.embedding_dropout = nn.Dropout(p=0.5)
        self.l1_penalty = None
        """
        self.fc_after_embedding = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp),  # From 512 to 256
            nn.ReLU(),
        )
        #policy heads (4 x 256->128 + 1 x 256->2)
        self.player_priority_head = nn.Linear(hidden_dim_mlp, policy_size_A)
        self.opponent_priority_head = nn.Linear(hidden_dim_mlp, policy_size_A)
        self.target_head = nn.Linear(hidden_dim_mlp, policy_size_A)
        self.binary_head = nn.Linear(hidden_dim_mlp, 2)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim_mlp, 1),  # From 256 to 1
            nn.Tanh()
        )
        """

        self.player_priority_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp), nn.ReLU(),
            nn.Linear(hidden_dim_mlp, policy_size_A),
        )
        self.opponent_priority_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp), nn.ReLU(),
            nn.Linear(hidden_dim_mlp, policy_size_A),
        )
        self.target_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp), nn.ReLU(),
            nn.Linear(hidden_dim_mlp, policy_size_A),
        )
        self.binary_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp), nn.ReLU(),
            nn.Linear(hidden_dim_mlp, 2),
        )
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp), nn.ReLU(),
            nn.Linear(hidden_dim_mlp, 1), nn.Tanh(),
        )


    def forward(self, indices, offsets):
        input_weights = None
        if self.training and self.input_dropout > 0:
            keep_mask = torch.rand_like(indices, dtype=torch.float32) > self.input_dropout
            keep_mask = keep_mask.to(torch.float32)
            input_weights = keep_mask / (1.0 - self.input_dropout)

        emb = self.embedding_bag(indices, offsets, per_sample_weights=input_weights)

        if self.training:
            self.l1_penalty = emb.abs().sum() * 1e-7

        #emb = emb + self.embedding_bias
        #emb = F.relu(emb)
        #emb = self.embedding_norm(emb)


        emb = self.embedding_dropout(emb)
        return (
            self.player_priority_head(emb),
            self.opponent_priority_head(emb),
            self.target_head(emb),
            self.binary_head(emb),
            self.value_head(emb).squeeze(-1),
        )
        #h = self.fc_after_embedding(emb)
        #return self.player_priority_head(h), self.opponent_priority_head(h), self.target_head(h), self.binary_head(h), self.value_head(h).squeeze(-1)

class NetTransformer(nn.Module):
    def __init__(self, num_embeddings=None, policy_size_A=ACTIONS_MAX):
        super().__init__()
        if num_embeddings is None:
            num_embeddings = EMBED_ROWS
        self.num_embeddings = num_embeddings

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
        """
        self.fc_after_embedding = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp),  # From 512 to 256
            nn.ReLU(),
        )
        
        #policy heads (4 x 256->128 + 1 x 256->2)
        self.player_priority_head = nn.Linear(hidden_dim_mlp, policy_size_A)
        self.opponent_priority_head = nn.Linear(hidden_dim_mlp, policy_size_A)
        self.target_head = nn.Linear(hidden_dim_mlp, policy_size_A)
        self.binary_head = nn.Linear(hidden_dim_mlp, 2)

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim_mlp, 1),  # From 256 to 1
            nn.Tanh()
        )
        """
        self.player_priority_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp), nn.ReLU(),
            nn.Linear(hidden_dim_mlp, policy_size_A),
        )
        self.opponent_priority_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp), nn.ReLU(),
            nn.Linear(hidden_dim_mlp, policy_size_A),
        )
        self.target_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp), nn.ReLU(),
            nn.Linear(hidden_dim_mlp, policy_size_A),
        )
        self.binary_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp), nn.ReLU(),
            nn.Linear(hidden_dim_mlp, 2),
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
        # Keep num_embeddings in sync with the table. forward() does `indices % num_embeddings`,
        # so a stale value silently wraps the newly added rows onto old ones -- the vocab grows
        # but the model cannot reach the new features. It also travels into the checkpoint as
        # embed_rows; build_model_from_checkpoint sizes from the weights regardless, but this is
        # the actual root cause of that mismatch.
        self.num_embeddings = num_embeddings

    def forward(self, indices, offsets):
        indices = indices % self.num_embeddings
        B = offsets.shape[0]
        ends = torch.cat([offsets[1:], torch.tensor([indices.shape[0]], device=offsets.device)])
        lengths = ends - offsets

        if self.training and self.input_dropout > 0:
            # Token dropout by removing tokens rather than masking them: dropped tokens never
            # entered attention or pooling anyway, so this is equivalent but the padded batch is
            # ~30% shorter, which roughly halves attention memory.
            bag = torch.repeat_interleave(torch.arange(B, device=indices.device), lengths)
            keep = torch.rand(indices.shape[0], device=indices.device) >= self.input_dropout
            indices, bag = indices[keep], bag[keep]
            lengths = torch.bincount(bag, minlength=B)
            offsets = torch.cumsum(lengths, 0) - lengths
        max_len = lengths.max().item()
        if PAD_BUCKET > 1:
            # a handful of fixed shapes instead of a new one per batch (MPS recompiles per shape)
            max_len = -(-max_len // PAD_BUCKET) * PAD_BUCKET

        # reconstruct padded sequences
        padded = indices.new_zeros(B, max_len)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=indices.device)

        for i in range(B):
            l = lengths[i]
            padded[i, :l] = indices[offsets[i]:offsets[i] + l]
            mask[i, :l] = True


        emb = self.embedding(padded)  # (B, max_len, 512)
        emb = self.transformer(emb, src_key_padding_mask=~mask)  # (B, max_len, 512)

        # mean pool over real tokens
        #emb = (emb * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1).float()  # (B, 512)
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

        #h = self.fc_after_embedding(emb)
        #return self.player_priority_head(h), self.opponent_priority_head(h), self.target_head(h), self.binary_head(h), self.value_head(h).squeeze(-1)



def load_model(path):
    if path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            return torch.load(f, map_location="cpu")
    return torch.load(path, map_location="cpu")


def build_model_from_checkpoint(ckpt: dict) -> "NetTransformer":
    """Rebuild a NetTransformer sized to match the checkpoint's embedding table."""
    sd = ckpt["model_state_dict"]
    # The saved embedding tensor is the ground truth for its own size. The embed_rows
    # metadata can be stale -- model.num_embeddings is not updated when the embedding table
    # is resized as the feature vocab grows across generations, so a checkpoint can carry
    # embed_rows=10700 while its actual embedding is 14518 rows. Trusting the metadata built
    # a wrong-size model and made every inference-server start past that generation die on
    # load_state_dict (size mismatch). The weights never lie, so size from them.
    rows = sd["embedding.weight"].shape[0]
    # the head's last Linear sets the policy width (128 upstream, the vocab dim otherwise)
    head_keys = sorted((k for k in sd if k.startswith("player_priority_head.") and k.endswith(".weight")),
                       key=lambda k: int(k.split(".")[1]) if k.split(".")[1].isdigit() else 0)
    model = NetTransformer(num_embeddings=rows, policy_size_A=sd[head_keys[-1]].shape[0])
    model.load_state_dict(ckpt["model_state_dict"])
    return model

def policy_width(model) -> int:
    """Width of the priority/opponent/target heads (their final Linear layer)."""
    head = model.player_priority_head
    return head[-1].out_features if isinstance(head, nn.Sequential) else head.out_features


def normalize_policy_labels(raw: torch.Tensor) -> torch.Tensor:
    total = raw.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return raw / total