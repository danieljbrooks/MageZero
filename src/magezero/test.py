import json
import time

import torch
import torch.nn.functional as F
from pyroaring import BitMap
from torch import nn  # optim is not strictly needed for testing if not optimizing
from torch.utils.data import DataLoader

from dataset import H5Indexed, collate_batch, filter_opponent_states
from model import BATCH_SIZE, NetTransformer, load_model, build_model_from_checkpoint, DEVICE, autocast, PRIORITY_A_MAX, PRIORITY_B_MAX, TARGETS_MAX, BINARY_MAX, ActionType, lambda_pA, lambda_pB, lambda_t, lambda_b, normalize_policy_labels
from vocab import FeatureVocab

SHOW_CONFUSION_MATRIX = True

mse = nn.MSELoss()
kld = nn.KLDivLoss(reduction='batchmean')

# head name -> (action type, player filter, logit width, loss weight)
HEADS = {
    "priority_A": (ActionType.PRIORITY.value, True, PRIORITY_A_MAX, lambda_pA),
    "priority_B": (ActionType.PRIORITY.value, False, PRIORITY_B_MAX, lambda_pB),
    "choose_target": (ActionType.CHOOSE_TARGET.value, None, TARGETS_MAX, lambda_t),
    "choose_use": (ActionType.CHOOSE_USE.value, None, BINARY_MAX, lambda_b),
}


def populate_matrix(matrix, actual, predicted):
    for true_action, pred_action in zip(actual.cpu(), predicted.cpu()):
        matrix[true_action, pred_action] += 1

def print_matrix(matrix):
    print("--- Policy Confusion Matrix (True \\ Predicted) ---")
    matrix_size = matrix.shape[0]

    if matrix_size == 2:
        # Print header for predicted actions
        header = "  True   |" + "".join([f"{j: >8}" for j in range(matrix_size)])
        print(header)
        print("-" * len(header))

        # Print each row for true actions
        for r in range(matrix_size):
            row_str = f"{r: >8} |"
            row_str += "".join([f"{matrix[r, c].item(): >8}" for c in range(matrix_size)])
            print(row_str)
    else:
        # Print header for predicted actions
        indices = [i for i in range(matrix_size) if matrix[i].sum() > 0]
        header = "True |" + "".join([f"{j: >4}" for j in indices])
        print(header)
        print("-" * len(header))

        # Print each row for true actions
        for r in indices:
            row_str = f"{r: >4} |"
            row_str += "".join([f"{matrix[r, c].item(): >4}" for c in indices])
            print(row_str)

    print("-" * 60)
def correct_from_matrix(matrix) -> int:
    return int(matrix.diag().sum().item())


def total_from_matrix(matrix) -> int:
    return int(matrix.sum().item())


def _entropy(p: torch.Tensor) -> torch.Tensor:
    """Per-row Shannon entropy (nats) of a probability matrix."""
    return -(p * torch.log(p.clamp(min=1e-12))).sum(dim=1)


def validate(model, dl):
    """Evaluate `model` on `dl`. Prints the legacy summary and returns a metrics dict:
    per-head KL loss / top-1 agreement with MCTS / entropy of MCTS targets vs network,
    plus value MSE, sign accuracy and correlation. `avg_total_loss` is the old return value."""
    total_decision_examples = 0
    total_combined_loss, total_v_loss = 0.0, 0.0
    sums = {h: {"loss": 0.0, "n": 0, "tgt_ent": 0.0, "pred_ent": 0.0, "legal": 0.0} for h in HEADS}
    matrices = {h: torch.zeros(w, w, dtype=torch.long) for h, (_, _, w, _) in HEADS.items()}
    v_pred_all, v_lbl_all = [], []
    model.eval()
    with torch.no_grad():
        for batch_indices, batch_offsets, batch_policy_labels, batch_value_labels, is_players, action_types in dl:
            batch_indices = batch_indices.to(DEVICE)
            batch_offsets = batch_offsets.to(DEVICE)
            batch_policy_labels = batch_policy_labels.to(DEVICE)
            batch_value_labels = batch_value_labels.to(DEVICE)
            is_players = is_players.to(DEVICE).squeeze(-1).to(torch.bool)
            action_types = action_types.to(DEVICE).squeeze(-1).to(torch.long)

            with autocast():
                outs = model(batch_indices, batch_offsets)
            logits = dict(zip(HEADS, outs[:4]))
            value_pred = outs[4].float()

            decision_mask = (batch_policy_labels > 0).sum(dim=1) > 0
            total_decision_examples += decision_mask.sum().item()

            batch_loss = 0.0
            for h, (atype, player, width, lam) in HEADS.items():
                mask = (action_types == atype) & decision_mask
                if player is True:
                    mask &= is_players
                elif player is False:
                    mask &= ~is_players
                raw = batch_policy_labels[mask][:, :width]
                log_probs = F.log_softmax(logits[h][mask][:, :width].float(), dim=1)
                tgt = normalize_policy_labels(raw)
                loss = torch.nan_to_num(kld(log_probs, tgt) * lam)
                s = log_probs.size(0)
                batch_loss = batch_loss + loss
                if s == 0:
                    continue
                sums[h]["loss"] += loss.item() * s
                sums[h]["n"] += s
                sums[h]["tgt_ent"] += _entropy(tgt).sum().item()
                # network entropy restricted to actions MCTS considered legal (label support)
                legal = raw > 0
                pred = torch.softmax(logits[h][mask][:, :width].float().masked_fill(~legal, -1e9), dim=1)
                sums[h]["pred_ent"] += _entropy(pred).sum().item()
                sums[h]["legal"] += legal.sum().item()
                populate_matrix(matrices[h], torch.argmax(tgt, dim=1), torch.argmax(log_probs, dim=1))

            lv = mse(value_pred, batch_value_labels.squeeze(-1))
            total_combined_loss += (batch_loss + lv).item()
            total_v_loss += lv.item()
            v_pred_all.append(value_pred.cpu())
            v_lbl_all.append(batch_value_labels.squeeze(-1).float().cpu())

    n_batches = max(len(dl), 1)
    metrics = {
        "decision_states": int(total_decision_examples),
        "value_loss": total_v_loss / n_batches,
        "avg_total_loss": total_combined_loss / n_batches,
    }
    if v_pred_all:
        vp, vl = torch.cat(v_pred_all), torch.cat(v_lbl_all)
        metrics["value_sign_acc"] = float(((vp > 0) == (vl > 0)).float().mean())
        metrics["value_pred_mean_abs"] = float(vp.abs().mean())
        if vp.numel() > 1 and vp.std() > 0 and vl.std() > 0:
            metrics["value_corr"] = float(torch.corrcoef(torch.stack([vp, vl]))[0, 1])
    for h in HEADS:
        n = sums[h]["n"]
        metrics[f"{h}_n"] = n
        if n == 0:
            continue
        metrics[f"{h}_loss"] = sums[h]["loss"] / n
        metrics[f"{h}_acc"] = correct_from_matrix(matrices[h]) / total_from_matrix(matrices[h])
        metrics[f"{h}_target_entropy"] = sums[h]["tgt_ent"] / n
        metrics[f"{h}_pred_entropy"] = sums[h]["pred_ent"] / n
        metrics[f"{h}_legal_actions"] = sums[h]["legal"] / n

    print(f"Validation loss:  priority_A_loss={metrics.get('priority_A_loss', 0):.3f}  priority_B_loss={metrics.get('priority_B_loss', 0):.3f} "
          f"choose_target_loss={metrics.get('choose_target_loss', 0):.3f} choose_use_loss={metrics.get('choose_use_loss', 0):.3f} "
          f"value_loss={metrics['value_loss']:.3f} avg_total_loss={metrics['avg_total_loss']:.3f} decision_states={total_decision_examples}")
    for h in HEADS:
        if metrics[f"{h}_n"] > 0:
            print(f"Test {h}_accuracy={metrics[f'{h}_acc']:.3f}")
            if SHOW_CONFUSION_MATRIX:
                print_matrix(matrices[h])
        else:
            print(f"No {h} samples in test set to calculate accuracy.")

    return metrics


def append_metrics(path: str | None, record: dict) -> None:
    if not path:
        return
    record = {"ts": time.time(), **record}
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--opponent-head", action="store_true")
    parser.add_argument("--metrics-out", default=None, help="append a JSON line of metrics here")
    parser.add_argument("--gen", type=int, default=None)
    args = parser.parse_args()

    checkpoint_path = f"models/{args.deck}/ver{args.version}/model.pt.gz"
    try:
        checkpoint = load_model(checkpoint_path)
    except FileNotFoundError:
        checkpoint = None
        print(f"ERROR: Checkpoint not found at {checkpoint_path}. Testing with uninitialized model.")

    vocab = None
    if checkpoint is not None and "feature_vocab" in checkpoint:
        vocab = FeatureVocab.from_state_dict(checkpoint["feature_vocab"])
        vocab.require_encoding(GLOBAL_MAX)
        print(f"feature vocab: {len(vocab)} rows")
        ds = H5Indexed(f"data/{args.deck}/ver{args.version}/testing", vocab=vocab)
    else:
        ignore_path = f"models/{args.deck}/ver{args.version}/ignore.roar"
        with open(ignore_path, "rb") as f:
            ignore = BitMap.deserialize(f.read())
        print(f"ignore list size: {len(ignore)}")
        ds = H5Indexed(f"data/{args.deck}/ver{args.version}/testing", set(ignore), fold_bins=GLOBAL_MAX)
    if not args.opponent_head:
        ds = filter_opponent_states(ds, TARGETS_MAX)

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                    collate_fn=collate_batch, pin_memory=DEVICE.type == "cuda", persistent_workers=False)

    try:
        model = build_model_from_checkpoint(load_model(checkpoint_path)).to(DEVICE)
        print(f"Loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"ERROR: Checkpoint not found at {checkpoint_path}. Testing with uninitialized model.")
        model = NetTransformer().to(DEVICE)

    m = validate(model, dl)
    # "prev model on fresh self-play data": how well last gen's network predicts the new search targets
    append_metrics(args.metrics_out, {"kind": "eval_prev_model", "deck": args.deck, "version": args.version,
                                      "gen": args.gen, **m})
