"""Where audit records go.

StdoutSink prints one JSON line per record (local runs and tests).

CloudWatchSink writes a request's records to CloudWatch Logs in ONE put_log_events call *before* the
response is returned. Printing to stdout is not enough on a serverless endpoint: the platform's log
forwarder is best-effort and containers are frozen after responding, which measurably dropped ~9% of lines
under load. If the CloudWatch call fails twice, the records fall back to stdout (so a copy usually still
exists) and an `audit_sink_error` line says so.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid


def _line(record: dict) -> str:
    return json.dumps(record, separators=(",", ":"))


class StdoutSink:
    def write(self, records: list[dict]) -> None:
        for r in records:
            print(_line(r), flush=True)


class CloudWatchSink:
    def __init__(self, group: str, stream_prefix: str, client=None, fallback=None):
        if client is None:
            import boto3
            client = boto3.client("logs", region_name=os.environ.get("AWS_REGION")
                                  or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1")
        self.client, self.group = client, group
        self.stream = f"{stream_prefix}/{uuid.uuid4().hex[:12]}"   # one stream per container process
        self.fallback = fallback or StdoutSink()
        self._ready = False
        self._lock = threading.Lock()

    def _ensure_stream(self) -> None:
        with self._lock:
            if not self._ready:
                try:
                    self.client.create_log_stream(logGroupName=self.group, logStreamName=self.stream)
                except self.client.exceptions.ResourceAlreadyExistsException:
                    pass
                self._ready = True

    def write(self, records: list[dict]) -> None:
        if not records:
            return
        now = int(time.time() * 1000)
        events = [{"timestamp": now, "message": _line(r)} for r in records]   # a request has <= 100 records
        error = None
        for attempt in range(2):
            try:
                self._ensure_stream()
                self.client.put_log_events(logGroupName=self.group, logStreamName=self.stream, logEvents=events)
                return
            except Exception as exc:  # noqa: BLE001 - any failure must reach the fallback, never the caller
                error = exc
                self._ready = False          # the stream may have been deleted; recreate on retry
                time.sleep(0.2)
        print(_line({"event": "audit_sink_error", "error": str(error)[:300], "records": len(records)}), flush=True)
        self.fallback.write(records)


def sink_from_env(model_version: str):
    group = os.environ.get("AUDIT_LOG_GROUP")
    return CloudWatchSink(group, model_version) if group else StdoutSink()
