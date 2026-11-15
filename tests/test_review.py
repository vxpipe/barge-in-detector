import csv
import json

import pytest

from barge_in_detector.review.store import ReviewStore


@pytest.fixture
def store(tmp_path):
    source = tmp_path / "challenge.csv"
    with source.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "text", "label", "language"])
        writer.writeheader()
        writer.writerow(dict(case_id="en-001", text="Stop", label="barge_in", language="en"))
    return ReviewStore(source, tmp_path / "reviews.json")


def test_blind_cases_and_durable_review(store):
    assert "label" not in store.snapshot()["cases"][0]
    assert store.snapshot()["cases"][0]["review"] is None
    saved = store.save("en-001", "barge_in", "Clear stop request", 0)
    assert saved["revision"] == 1
    reloaded = ReviewStore(store.source, store.destination)
    assert reloaded.snapshot()["cases"][0]["review"]["notes"] == "Clear stop request"
    assert "barge_in" in reloaded.export_csv()


def test_conflicting_or_invalid_write_preserves_saved_review(store):
    store.save("en-001", "backchannel", "Acknowledgment", 0)
    with pytest.raises(ValueError, match="changed"):
        store.save("en-001", "barge_in", "", 0)
    with pytest.raises(ValueError):
        store.save("en-001", "other", "", 1)
    with pytest.raises(ValueError):
        store.save("en-001", "needs_context", "", 1)
    with pytest.raises(ValueError):
        store.save("unknown", "barge_in", "", 0)
    assert store.snapshot()["cases"][0]["review"]["decision"] == "backchannel"


def test_source_change_and_wrong_source_prevent_reusing_annotations(store):
    store.save("en-001", "backchannel", "", 0)
    store.source.write_text(store.source.read_text().replace("Stop", "Wait"))
    with pytest.raises(ValueError, match="dataset"):
        ReviewStore(store.source, store.destination)


def test_atomic_save_does_not_leave_temp_file(store):
    store.save("en-001", "backchannel", "", 0)
    assert (
        json.loads(store.destination.read_text())["reviews"]["en-001"]["decision"] == "backchannel"
    )
    assert list(store.destination.parent.glob("*.tmp")) == []


def test_http_blind_review_and_origin_protection(store):
    import threading
    import urllib.error
    import urllib.request
    from http.server import ThreadingHTTPServer

    from barge_in_detector.review.__main__ import make_handler

    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(store))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/api/cases") as response:
            data = json.load(response)
        assert "label" not in data["cases"][0]
        with urllib.request.urlopen(base + "/") as response:
            assert b"Let the agent continue" in response.read()
        body = json.dumps(
            dict(case_id="en-001", decision="barge_in", notes="", revision=0)
        ).encode()
        request = urllib.request.Request(
            base + "/api/review",
            data=body,
            headers={"Content-Type": "application/json", "Origin": "https://elsewhere.example"},
        )
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(request)
        assert error.value.code == 403
        request.remove_header("Origin")
        with urllib.request.urlopen(request) as response:
            assert json.load(response)["decision"] == "barge_in"
        with urllib.request.urlopen(base + "/export.csv") as response:
            assert b"barge_in" in response.read()
        assert store.source.read_text().count("Stop") == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
