import json
from collections import Counter

import numpy as np
import pandas as pd
import torch

from config import DATA_DIR, PIECES_DIR


def fail(msg, **extra):
    return {
        "mse": float("inf"),
        "mse_first_100": float("inf"),
        "mse_last_100": float("inf"),
        "worst_decile_mse": float("inf"),
        "solved": False,
        "error": msg,
        **extra,
    }


def parse_permutation(raw: str) -> list[int]:
    return [int(x) for x in raw.split(",") if x.strip()]


def _load_data():
    df = pd.read_csv(DATA_DIR / "historical_data.csv")
    meas_cols = [c for c in df.columns if c.startswith("measurement_")]
    if len(meas_cols) != 48:
        raise ValueError(f"Expected 48 measurement columns, found {len(meas_cols)}")
    if "pred" not in df.columns:
        raise ValueError("Missing required 'pred' column in historical_data.csv")
    x = torch.tensor(df[meas_cols].values, dtype=torch.float32)
    pred_true = torch.tensor(df["pred"].values, dtype=torch.float32)
    return x, pred_true


def _load_states() -> list[dict]:
    states = []
    for i in range(97):
        p = PIECES_DIR / f"piece_{i}.pth"
        if not p.exists():
            raise FileNotFoundError(f"Missing piece file: {p}")
        st = torch.load(p, map_location="cpu")
        if "weight" not in st or "bias" not in st:
            raise ValueError(f"piece_{i}.pth missing weight/bias keys")
        states.append(st)
    return states


def _validate_permutation(permutation: list[int], states: list[dict]) -> dict | None:
    if len(permutation) != 97:
        return fail(f"Bad permutation length: {len(permutation)} (expected 97)")

    counts = Counter(permutation)
    duplicates = [k for k, v in counts.items() if v > 1]
    if duplicates:
        return fail("Permutation contains duplicate layer indices", duplicates=duplicates)

    out_of_range = [i for i in permutation if i < 0 or i > 96]
    if out_of_range:
        return fail("Permutation contains out-of-range indices", out_of_range=sorted(set(out_of_range)))

    def w_shape(idx: int):
        return tuple(states[idx]["weight"].shape)

    for b in range(48):
        inp_idx = permutation[2 * b]
        out_idx = permutation[2 * b + 1]
        inp_shape = w_shape(inp_idx)
        out_shape = w_shape(out_idx)
        if inp_shape != (96, 48):
            return fail(
                f"Block {b}: invalid inp piece shape",
                block=b,
                piece=inp_idx,
                got_shape=inp_shape,
                expected_shape=(96, 48),
            )
        if out_shape != (48, 96):
            return fail(
                f"Block {b}: invalid out piece shape",
                block=b,
                piece=out_idx,
                got_shape=out_shape,
                expected_shape=(48, 96),
            )

    final_idx = permutation[96]
    final_shape = w_shape(final_idx)
    if final_shape != (1, 48):
        return fail(
            "Invalid final layer shape",
            piece=final_idx,
            got_shape=final_shape,
            expected_shape=(1, 48),
        )
    return None


def assemble_and_evaluate(permutation: list[int]) -> dict:
    try:
        x, pred_true = _load_data()
    except Exception as e:
        return fail("Failed to read historical_data.csv", exception=repr(e))

    try:
        states = _load_states()
    except Exception as e:
        return fail("Failed while loading piece files", exception=repr(e))

    validation = _validate_permutation(permutation, states)
    if validation is not None:
        return validation

    def get_wb(idx: int):
        st = states[idx]
        return st["weight"].float(), st["bias"].float()

    try:
        with torch.no_grad():
            cur = x
            for b in range(48):
                inp_idx = permutation[2 * b]
                out_idx = permutation[2 * b + 1]
                inp_w, inp_b = get_wb(inp_idx)
                out_w, out_b = get_wb(out_idx)
                hidden = torch.relu(cur @ inp_w.T + inp_b)
                cur = cur + (hidden @ out_w.T + out_b)
            final_w, final_b = get_wb(permutation[96])
            preds = (cur @ final_w.T + final_b).squeeze(-1)
    except Exception as e:
        return fail("Forward pass failed", exception=repr(e))

    if preds.shape != pred_true.shape:
        return fail(
            "Prediction shape mismatch",
            pred_shape=tuple(preds.shape),
            target_shape=tuple(pred_true.shape),
        )

    err = (preds - pred_true) ** 2
    mse_total = float(err.mean())
    mse_first100 = float(err[:100].mean()) if len(err) >= 100 else mse_total
    mse_last100 = float(err[-100:].mean()) if len(err) >= 100 else mse_total

    return {
        "mse": round(mse_total, 8),
        "mse_first_100": round(mse_first100, 8),
        "mse_last_100": round(mse_last100, 8),
        "worst_decile_mse": round(float(np.percentile(err.numpy(), 90)), 8),
        "solved": mse_total < 1e-6,
    }


def evaluate_from_cli_arg(raw: str) -> str:
    try:
        perm = parse_permutation(raw.strip())
        return json.dumps(assemble_and_evaluate(perm))
    except Exception as e:
        return json.dumps(fail("Unhandled exception in evaluate.py", exception=repr(e)))
