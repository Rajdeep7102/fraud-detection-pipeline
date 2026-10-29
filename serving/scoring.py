"""Model loading, input validation, scoring and audit logging. No web framework in here, so it is easy to test.

Everything the model needs is read from the model directory (SageMaker extracts model.tar.gz to
/opt/ml/model): model.joblib, model_meta.json (threshold, version) and feature_spec.yaml (feature order).
"""
from __future__ import annotations

import json
import math
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import yaml

from audit import sink_from_env

MAX_BATCH = 100


class ValidationError(ValueError):
    """The request is malformed; the caller gets a 400 with this message."""


class Scorer:
    def __init__(self, model_dir: str | Path, sink=None):
        d = Path(model_dir)
        self.model = joblib.load(d / "model.joblib")
        meta = json.loads((d / "model_meta.json").read_text())
        spec = yaml.safe_load((d / "feature_spec.yaml").read_text())
        self.features: list[str] = [f["name"] for f in spec["features"]]
        self.model_version: str = meta["model_version"]
        self.threshold: float = float(meta["threshold"])
        if self.model.n_features_in_ != len(self.features):
            raise RuntimeError("model and feature_spec disagree on the number of features")
        self.sink = sink or sink_from_env(self.model_version)

    # -- validation ----------------------------------------------------------------------------
    def parse(self, payload) -> tuple[list[dict], np.ndarray]:
        if not isinstance(payload, dict) or not isinstance(payload.get("instances"), list):
            raise ValidationError('body must be JSON like {"instances": [{...}, ...]}')
        instances = payload["instances"]
        if not 1 <= len(instances) <= MAX_BATCH:
            raise ValidationError(f"instances must hold 1 to {MAX_BATCH} items, got {len(instances)}")
        x = np.empty((len(instances), len(self.features)), dtype=np.float32)
        for i, inst in enumerate(instances):
            if not isinstance(inst, dict):
                raise ValidationError(f"instances[{i}] must be an object")
            for j, name in enumerate(self.features):
                v = inst.get(name)
                if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                    raise ValidationError(f"instances[{i}].{name} must be a finite number, got {v!r}")
                x[i, j] = v
        return instances, x

    # -- scoring -------------------------------------------------------------------------------
    def score(self, payload, request_id: str | None = None) -> dict:
        t0 = time.perf_counter()
        request_id = request_id or str(uuid.uuid4())
        instances, x = self.parse(payload)
        p = self.model.predict_proba(x)[:, 1]
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        out, records = [], []
        for inst, row, score in zip(instances, x, p):
            score = float(score)
            decision = "FRAUD" if score >= self.threshold else "LEGIT"
            confidence = max(score, 1.0 - score)
            pred = {"transaction_id": inst.get("transaction_id"), "fraud_score": round(score, 6),
                    "confidence": round(confidence, 6), "decision": decision}
            out.append(pred)
            records.append({
                "event": "prediction",
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "request_id": request_id,
                "transaction_id": inst.get("transaction_id"),
                "model_version": self.model_version,
                "threshold": self.threshold,
                "fraud_score": pred["fraud_score"],
                "confidence": pred["confidence"],
                "decision": decision,
                "features": {n: float(v) for n, v in zip(self.features, row)},
                "batch_size": len(instances),
                "latency_ms": latency_ms,
            })
        self.sink.write(records)    # durable before we answer: every prediction is audit-logged or the sink says why not
        return {"model_version": self.model_version, "threshold": self.threshold, "predictions": out}
