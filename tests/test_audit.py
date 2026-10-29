import json
import sys

import pytest

from fraud.config import ROOT

sys.path.insert(0, str(ROOT / "serving"))
from audit import CloudWatchSink, StdoutSink  # noqa: E402


class _Exceptions:
    class ResourceAlreadyExistsException(Exception):
        pass


class FakeLogs:
    exceptions = _Exceptions

    def __init__(self, put_failures=0, stream_exists=False):
        self.calls, self.streams = [], []
        self.put_failures, self.stream_exists = put_failures, stream_exists

    def create_log_stream(self, **kw):
        self.streams.append(kw)
        if self.stream_exists:
            raise _Exceptions.ResourceAlreadyExistsException()

    def put_log_events(self, **kw):
        if self.put_failures:
            self.put_failures -= 1
            raise RuntimeError("throttled")
        self.calls.append(kw)


RECORDS = [{"event": "prediction", "transaction_id": i, "fraud_score": 0.1 * i} for i in range(5)]


def test_one_call_per_request_with_every_record():
    fake = FakeLogs()
    sink = CloudWatchSink("/fraud/audit", "rf-1", client=fake)
    sink.write(RECORDS)
    sink.write(RECORDS[:2])
    assert len(fake.calls) == 2 and len(fake.streams) == 1          # stream created once, one call per request
    sent = [json.loads(e["message"]) for e in fake.calls[0]["logEvents"]]
    assert sent == RECORDS
    assert fake.calls[0]["logGroupName"] == "/fraud/audit" and fake.calls[0]["logStreamName"].startswith("rf-1/")


def test_existing_stream_is_fine():
    fake = FakeLogs(stream_exists=True)
    CloudWatchSink("/g", "rf-1", client=fake).write(RECORDS)
    assert len(fake.calls) == 1


def test_one_transient_failure_is_retried():
    fake = FakeLogs(put_failures=1)
    CloudWatchSink("/g", "rf-1", client=fake).write(RECORDS)
    assert len(fake.calls) == 1


def test_persistent_failure_falls_back_to_stdout_and_never_raises(capsys):
    fake = FakeLogs(put_failures=5)
    CloudWatchSink("/g", "rf-1", client=fake).write(RECORDS)
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert lines[0]["event"] == "audit_sink_error"
    assert [ln["transaction_id"] for ln in lines[1:]] == list(range(5))   # all records still emitted


def test_stdout_sink(capsys):
    StdoutSink().write(RECORDS[:2])
    assert len(capsys.readouterr().out.splitlines()) == 2
