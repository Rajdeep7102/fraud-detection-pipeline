"""Synthetic card-transaction generator.

Writes ``n_transactions`` rows as Parquet, one file per card bucket
(``card_id % n_buckets``), so every later stage can work one bucket at a time
inside a small memory budget.

Each card has a stable profile (home country, usual hour, spend level, favourite
merchants, devices). Fraud is injected as whole compromise episodes:

* card_testing      -- a burst of tiny charges at many new merchants, then cash-out
* account_takeover  -- new device, often abroad, several large charges within a day
* stealth           -- 1-3 moderately large charges that look almost normal

Legitimate look-alikes (trips abroad, one-off big purchases, new phones) keep the
task from being trivially separable.

Usage:
    python -m fraud.data.generate              # full 30M rows from config.yaml
    python -m fraud.data.generate --scale 0.01 --out data/sample
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from fraud.config import load_config, project_path

COUNTRIES = np.array(["US", "CA", "GB", "DE", "FR", "IN", "MX", "BR", "AU", "JP", "NG", "RU"])
HOME_WEIGHTS = np.array([0.55, 0.15, 0.08, 0.05, 0.04, 0.05, 0.03, 0.02, 0.02, 0.01, 0.0, 0.0])
N_MCC = 20
MAX_FAVS = 12
NEW_DEVICE_BASE = 10**12  # device ids above this were never seen before on any card
EPISODE_TYPES = ("card_testing", "account_takeover", "stealth")


@dataclass
class World:
    home: np.ndarray        # (n_cards,) country index
    hour_mu: np.ndarray     # usual hour of day
    amt_mu: np.ndarray      # log-amount mean
    amt_sigma: np.ndarray   # log-amount spread
    second_device: np.ndarray
    n_favs: np.ndarray
    favs: np.ndarray        # (n_cards, MAX_FAVS) merchant ids
    merchant_cdf: np.ndarray
    merchant_mcc: np.ndarray


def build_world(n_cards: int, n_merchants: int, rng: np.random.Generator) -> World:
    popularity = 1.0 / np.arange(1, n_merchants + 1) ** 1.1
    rng.shuffle(popularity)
    cdf = np.cumsum(popularity / popularity.sum())
    favs = np.searchsorted(cdf, rng.random((n_cards, MAX_FAVS))).astype(np.int32)
    return World(
        home=rng.choice(len(COUNTRIES), n_cards, p=HOME_WEIGHTS / HOME_WEIGHTS.sum()).astype(np.int8),
        hour_mu=np.clip(rng.normal(14, 3.5, n_cards), 7, 22),
        amt_mu=rng.normal(3.3, 0.7, n_cards),
        amt_sigma=rng.uniform(0.4, 0.8, n_cards),
        second_device=rng.random(n_cards) < 0.4,
        n_favs=rng.integers(3, MAX_FAVS + 1, n_cards),
        favs=favs,
        merchant_cdf=cdf,
        merchant_mcc=rng.integers(0, N_MCC, n_merchants).astype(np.int16),
    )


def plan_episodes(cfg: dict, n_cards: int, horizon_s: int, rng: np.random.Generator) -> pd.DataFrame:
    """One row per fraud or travel episode: card, kind, start second, row count."""
    d = cfg["data"]
    n_fraud = round(n_cards * d["compromise_rate"])
    n_travel = round(n_cards * d["travel_rate"])
    cards = rng.choice(n_cards, n_fraud + n_travel, replace=False)
    mix = d["episode_mix"]
    kinds = np.concatenate([
        rng.choice(EPISODE_TYPES, n_fraud, p=[mix[k] for k in EPISODE_TYPES]),
        np.full(n_travel, "travel"),
    ])
    n_rows = np.select(
        [kinds == "card_testing", kinds == "account_takeover", kinds == "stealth"],
        [rng.integers(7, 24, len(kinds)), rng.integers(2, 9, len(kinds)), rng.integers(1, 4, len(kinds))],
        default=rng.integers(3, 8, len(kinds)),
    )
    # Leave a few days of history before any episode, and room after it (trips last up to 7 days).
    start = rng.integers(5 * 86400, horizon_s - 8 * 86400, len(kinds))
    return pd.DataFrame({"card_id": cards, "kind": kinds, "start": start, "n_rows": n_rows})


def popular_merchants(w: World, n: int, rng: np.random.Generator) -> np.ndarray:
    return np.searchsorted(w.merchant_cdf, rng.random(n)).astype(np.int32)


def normal_rows(card_ids: np.ndarray, counts: np.ndarray, w: World, horizon_days: int,
                d: dict, rng: np.random.Generator) -> dict[str, np.ndarray]:
    c = np.repeat(card_ids, counts)
    n = len(c)
    day = rng.integers(0, horizon_days, n)
    hour = np.mod(rng.normal(w.hour_mu[c], 3.0), 24)
    secs = day * 86400 + (hour * 3600).astype(np.int64) + rng.integers(0, 60, n)

    amount = np.exp(rng.normal(w.amt_mu[c], w.amt_sigma[c]))
    big = rng.random(n) < d["big_purchase_rate"]
    amount[big] *= rng.uniform(3, 8, big.sum())

    fav_pick = (rng.random(n) * w.n_favs[c]).astype(np.int64)
    merchant = w.favs[c, fav_pick]
    explore = rng.random(n) < 0.2
    merchant[explore] = popular_merchants(w, explore.sum(), rng)

    device = c.astype(np.int64) * 4
    use_second = w.second_device[c] & (rng.random(n) < 0.35)
    device[use_second] += 1
    new_dev = rng.random(n) < d["legit_new_device_rate"]
    device[new_dev] = NEW_DEVICE_BASE + rng.integers(0, 2**40, new_dev.sum())

    country = w.home[c].copy()
    abroad = rng.random(n) < 0.01
    country[abroad] = rng.integers(0, 10, abroad.sum())

    return dict(card_id=c, secs=secs, amount=amount, merchant_id=merchant, device_id=device,
                country=country, is_fraud=np.zeros(n, np.int8), kind=np.full(n, "normal"))


def episode_rows(ep: pd.DataFrame, w: World, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Rows for every fraud / travel episode in ``ep`` (small: a few hundred per bucket)."""
    out: dict[str, list] = {k: [] for k in
                            ("card_id", "secs", "amount", "merchant_id", "device_id", "country", "is_fraud", "kind")}

    def foreign(card: int) -> int:
        choices = [i for i in range(len(COUNTRIES)) if i != w.home[card]]
        return int(rng.choice(choices))

    for card, kind, start, n in ep[["card_id", "kind", "start", "n_rows"]].itertuples(index=False):
        mu = w.amt_mu[card]
        primary = card * 4
        if kind == "card_testing":
            n_big = int(rng.integers(1, 4))
            n_small = max(n - n_big, 3)
            burst = rng.uniform(1800, 7200)
            t_small = start + np.sort(rng.uniform(0, burst, n_small))
            t_big = t_small[-1] + np.sort(rng.uniform(600, 6 * 3600, n_big))
            secs = np.concatenate([t_small, t_big])
            amount = np.concatenate([rng.uniform(0.5, 9.99, n_small), np.exp(mu + rng.uniform(1.0, 2.5, n_big))])
            merchant = np.concatenate([rng.integers(0, len(w.merchant_mcc), n_small), popular_merchants(w, n_big, rng)])
            device = np.full(len(secs), NEW_DEVICE_BASE + rng.integers(0, 2**40) if rng.random() < 0.7 else primary)
            country = np.full(len(secs), w.home[card] if rng.random() < 0.6 else foreign(card))
        elif kind == "account_takeover":
            secs = start + np.sort(rng.uniform(0, rng.uniform(3600, 86400), n))
            amount = np.exp(mu + rng.uniform(0.8, 2.2, n) + rng.normal(0, 0.3, n))
            merchant = popular_merchants(w, n, rng)
            device = np.full(n, NEW_DEVICE_BASE + rng.integers(0, 2**40) if rng.random() < 0.85 else primary)
            country = np.full(n, foreign(card) if rng.random() < 0.5 else w.home[card])
        elif kind == "stealth":
            day0 = start - start % 86400
            hour = np.mod(rng.normal(w.hour_mu[card], 2.0, n), 24)
            secs = np.sort(day0 + rng.integers(0, 3, n) * 86400 + hour * 3600)
            amount = np.exp(mu + rng.uniform(0.3, 1.2, n))
            merchant = popular_merchants(w, n, rng)
            device = np.full(n, primary if rng.random() < 0.7 else NEW_DEVICE_BASE + rng.integers(0, 2**40))
            country = np.full(n, w.home[card])
        else:  # travel: legitimate, but abroad, new merchants, larger spend
            secs = start + np.sort(rng.uniform(0, rng.uniform(2, 7) * 86400, n))
            amount = np.exp(mu + rng.uniform(0.0, 1.2, n))
            merchant = popular_merchants(w, n, rng)
            device = np.full(n, primary if rng.random() < 0.9 else NEW_DEVICE_BASE + rng.integers(0, 2**40))
            country = np.full(n, foreign(card))

        m = len(secs)
        out["card_id"].append(np.full(m, card))
        out["secs"].append(np.asarray(secs).astype(np.int64))
        out["amount"].append(amount)
        out["merchant_id"].append(merchant)
        out["device_id"].append(device)
        out["country"].append(country)
        out["is_fraud"].append(np.full(m, 0 if kind == "travel" else 1, np.int8))
        out["kind"].append(np.full(m, kind))

    return {k: np.concatenate(v) if v else np.array([]) for k, v in out.items()}


def to_table(cols: dict[str, np.ndarray], bucket: int, w: World, start_ts: pd.Timestamp) -> pa.Table:
    df = pd.DataFrame({
        "card_id": cols["card_id"].astype(np.int32),
        "ts": start_ts + pd.to_timedelta(cols["secs"], unit="s"),
        "amount": np.round(cols["amount"], 2).astype(np.float64),
        "merchant_id": cols["merchant_id"].astype(np.int32),
        "device_id": cols["device_id"].astype(np.int64),
        "country": COUNTRIES[cols["country"].astype(np.int64)],
        "is_fraud": cols["is_fraud"].astype(np.int8),
        "kind": cols["kind"],
    })
    df["mcc"] = w.merchant_mcc[df["merchant_id"].to_numpy()]
    df = df.sort_values(["card_id", "ts"], kind="stable").reset_index(drop=True)
    df.insert(0, "transaction_id", np.int64(bucket) * 100_000_000 + np.arange(len(df), dtype=np.int64))
    df["ts"] = df["ts"].astype("datetime64[us]")
    return pa.Table.from_pandas(df, preserve_index=False)


def generate(cfg: dict, out_dir: Path, scale: float = 1.0) -> dict:
    d = cfg["data"]
    rng = np.random.default_rng(cfg["seed"])
    n_cards = max(int(d["n_cards"] * scale), 1000)
    n_merchants = max(int(d["n_merchants"] * scale), 2000)
    n_total = int(d["n_transactions"] * scale)
    n_buckets = d["n_buckets"]
    horizon_days = d["days"]
    start_ts = pd.Timestamp(d["start_date"])

    w = build_world(n_cards, n_merchants, rng)
    episodes = plan_episodes(cfg, n_cards, horizon_days * 86400, rng)
    n_normal = n_total - int(episodes["n_rows"].sum())
    activity = rng.lognormal(0, 0.8, n_cards)
    counts = rng.multinomial(n_normal, activity / activity.sum())

    if out_dir.exists():
        shutil.rmtree(out_dir)
    tx_dir = out_dir / "transactions"
    tx_dir.mkdir(parents=True)

    pq.write_table(pa.table({"card_id": np.arange(n_cards, dtype=np.int32),
                             "home_country": COUNTRIES[w.home]}), out_dir / "cards.parquet")
    pq.write_table(pa.table({"merchant_id": np.arange(n_merchants, dtype=np.int32),
                             "mcc": w.merchant_mcc}), out_dir / "merchants.parquet")

    card_bucket = np.arange(n_cards) % n_buckets
    total_rows, kind_counts = 0, {}
    t0 = time.perf_counter()
    for b in range(n_buckets):
        brng = np.random.default_rng([cfg["seed"], b])
        cards = np.flatnonzero(card_bucket == b)
        normal = normal_rows(cards, counts[cards], w, horizon_days, d, brng)
        eps = episode_rows(episodes[episodes["card_id"] % n_buckets == b], w, brng)
        cols = {k: np.concatenate([normal[k], eps[k]]) for k in normal}
        table = to_table(cols, b, w, start_ts)
        part = tx_dir / f"bucket={b:02d}"
        part.mkdir()
        pq.write_table(table, part / "part-0.parquet", compression="zstd")
        total_rows += table.num_rows
        for k, v in pd.Series(cols["kind"]).value_counts().items():
            kind_counts[k] = kind_counts.get(k, 0) + int(v)
        print(f"bucket {b:02d}: {table.num_rows:,} rows  ({time.perf_counter() - t0:,.0f}s)", flush=True)

    n_fraud = sum(v for k, v in kind_counts.items() if k in EPISODE_TYPES)
    summary = {
        "rows": total_rows,
        "cards": n_cards,
        "merchants": n_merchants,
        "fraud_rows": n_fraud,
        "fraud_rate": round(n_fraud / total_rows, 5),
        "rows_by_kind": dict(sorted(kind_counts.items())),
        "episodes_by_kind": episodes["kind"].value_counts().sort_index().to_dict(),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    (out_dir / "_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scale", type=float, default=1.0, help="fraction of the configured size (smoke tests)")
    ap.add_argument("--out", default=None, help="output dir (default: data.raw_dir from config)")
    args = ap.parse_args()
    cfg = load_config()
    out = project_path(args.out or cfg["data"]["raw_dir"])
    summary = generate(cfg, out, args.scale)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
