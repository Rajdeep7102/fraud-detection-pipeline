"""Flask app implementing the SageMaker bring-your-own-container contract:
GET /ping -> 200 when the model is loaded; POST /invocations -> scores a JSON batch."""
from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

from flask import Flask, jsonify, request
from werkzeug.exceptions import BadRequest

from scoring import Scorer, ValidationError


def create_app(model_dir: str | None = None) -> Flask:
    app = Flask(__name__)
    scorer = Scorer(model_dir or os.environ.get("MODEL_DIR", "/opt/ml/model"))

    @app.get("/ping")
    def ping():
        return jsonify(status="ok", model_version=scorer.model_version), 200

    @app.post("/invocations")
    def invocations():
        request_id = request.headers.get("X-Amzn-SageMaker-Custom-Attributes") or str(uuid.uuid4())
        try:
            payload = request.get_json(force=True)   # raises BadRequest on malformed JSON
            return jsonify(scorer.score(payload, request_id)), 200
        except (ValidationError, BadRequest) as exc:
            reason = exc.description if isinstance(exc, BadRequest) else str(exc)
            scorer.sink.write([{"event": "request_rejected", "request_id": request_id,
                                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                                "model_version": scorer.model_version, "reason": reason[:300]}])
            return jsonify(error=reason[:300]), 400

    return app
