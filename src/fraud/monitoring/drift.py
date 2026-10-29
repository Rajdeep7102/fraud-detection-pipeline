"""Population Stability Index (PSI) helpers shared by baseline creation (training) and the drift job.

A feature's baseline is a binning plus the share of baseline rows in each bin:
  * low-cardinality features (<= MAX_CATEGORIES distinct values: flags, small counts) get one bin per
    value plus a trailing 'other' bin for values never seen in the baseline;
  * continuous features get quantile bins.
Quantile bins alone are wrong for flags: every row of a 0/1 feature lands in one bin, so PSI is always 0
and drift on e.g. is_foreign_country would be invisible.

PSI = sum_i (actual_i - expected_i) * ln(actual_i / expected_i); common reading: < 0.1 stable,
0.1-0.2 moderate shift, > 0.2 significant shift.
"""
from __future__ import annotations

import numpy as np

MAX_CATEGORIES = 10
QUANTILE_BINS = 10
EPS = 1e-4   # floor on a bin's share so empty bins do not give log(0)


def describe(v: np.ndarray) -> dict:
    v = np.asarray(v, dtype=np.float64)
    values = np.unique(v)
    if len(values) <= MAX_CATEGORIES:
        spec = {"type": "categorical", "values": [float(x) for x in values]}
    else:
        edges = np.unique(np.quantile(v, np.linspace(0, 1, QUANTILE_BINS + 1)[1:-1]))
        spec = {"type": "quantile", "edges": [float(e) for e in edges]}
    spec["expected"] = [float(s) for s in bin_shares(spec, v)]
    spec["mean"], spec["std"] = float(v.mean()), float(v.std())
    return spec


def bin_shares(spec: dict, v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    if spec["type"] == "categorical":
        values = np.asarray(spec["values"])
        idx = np.searchsorted(values, v)
        idx = np.where((idx < len(values)) & (values[np.minimum(idx, len(values) - 1)] == v), idx, len(values))
        n_bins = len(values) + 1                       # trailing 'other' bin
    else:
        idx = np.searchsorted(np.asarray(spec["edges"]), v, side="right")
        n_bins = len(spec["edges"]) + 1
    return np.bincount(idx, minlength=n_bins) / max(len(v), 1)


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.maximum(np.asarray(expected, dtype=np.float64), EPS)
    a = np.maximum(np.asarray(actual, dtype=np.float64), EPS)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_psi(spec: dict, values: np.ndarray) -> float:
    return psi(spec["expected"], bin_shares(spec, values))
