import importlib.util
import pathlib
import threading
import time

_spec = importlib.util.spec_from_file_location(
    "card_info", pathlib.Path(__file__).resolve().parent.parent / "src" / "draw" / "card_info.py")
card_info = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(card_info)
card_id_from_label = card_info.card_id_from_label
YgoprodeckMetadata = card_info.YgoprodeckMetadata
CardDatabase = card_info.CardDatabase
CardInfoEmitter = card_info.CardInfoEmitter


def test_card_id_from_normal_label():
    assert card_id_from_label("89631139") == "89631139"


def test_card_id_from_setcode_label():
    assert card_id_from_label("LOB-EN005-89631139") == "89631139"


def _raw(**over):
    row = {"id": 89631139, "name": "Blue-Eyes White Dragon", "type": "Normal Monster",
           "attribute": "LIGHT", "race": "Dragon", "atk": 3000, "def": 2500,
           "desc": "This legendary dragon...", "extra": "ignored"}
    row.update(over)
    return {"data": [row]}


def test_metadata_normalizes_known_subset(tmp_path):
    md = YgoprodeckMetadata(cache_dir=tmp_path, fetcher=lambda cid: _raw())
    out = md.get("89631139")
    assert out == {"id": 89631139, "name": "Blue-Eyes White Dragon",
                   "type": "Normal Monster", "attribute": "LIGHT", "race": "Dragon",
                   "atk": 3000, "def": 2500, "desc": "This legendary dragon..."}


def test_metadata_caches_and_does_not_refetch(tmp_path):
    calls = {"n": 0}

    def fetch(cid):
        calls["n"] += 1
        return _raw()

    md = YgoprodeckMetadata(cache_dir=tmp_path, fetcher=fetch)
    md.get("89631139")
    md2 = YgoprodeckMetadata(cache_dir=tmp_path, fetcher=fetch)  # fresh mem cache
    md2.get("89631139")
    assert calls["n"] == 1  # second instance read from disk cache


def test_metadata_failure_returns_empty(tmp_path):
    def boom(cid):
        raise RuntimeError("network down")

    md = YgoprodeckMetadata(cache_dir=tmp_path, fetcher=boom)
    assert md.get("89631139") == {}


import json as _json


class FakeResolver:
    def __init__(self, table):
        self.table = table

    def get(self, cid):
        return dict(self.table.get(str(cid), {}))


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _read_json(p):
    return _json.loads(p.read_text(encoding="utf-8"))


def _read_events(p):
    return [_json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def _emitter(tmp_path, clock, out_time=25):
    resolver = FakeResolver({"89631139": {"id": 89631139, "name": "Blue-Eyes White Dragon"}})
    return CardInfoEmitter(str(tmp_path), 1, resolver, out_of_screen_time=out_time, clock=clock)


def test_emit_appeared_writes_snapshot_text_and_event(tmp_path):
    clock = FakeClock()
    em = _emitter(tmp_path, clock)
    em.update(["LOB-EN005-89631139"])
    em.flush()  # let the background metadata resolve, then re-snapshot

    snap = _read_json(tmp_path / "current_p1.json")
    assert snap["channel"] == 1
    assert snap["cards"] == [{"id": 89631139, "name": "Blue-Eyes White Dragon"}]
    assert (tmp_path / "current_card_p1.txt").read_text(encoding="utf-8") == "Blue-Eyes White Dragon"

    events = _read_events(tmp_path / "events_p1.jsonl")
    assert len(events) == 1
    assert events[0]["event"] == "appeared"
    assert events[0]["card"]["id"] == 89631139


def test_refresh_same_label_emits_no_duplicate_event(tmp_path):
    clock = FakeClock()
    em = _emitter(tmp_path, clock)
    em.update(["89631139"])
    clock.t += 1
    em.update(["89631139"])
    assert len(_read_events(tmp_path / "events_p1.jsonl")) == 1  # still just the appear


def test_disappeared_after_hysteresis(tmp_path):
    clock = FakeClock()
    em = _emitter(tmp_path, clock, out_time=25)
    em.update(["89631139"])           # appears at t=1000
    clock.t += 10
    em.update([])                     # absent 10s < 25s: still present, no event
    assert len(_read_events(tmp_path / "events_p1.jsonl")) == 1
    clock.t += 20                     # absent 30s > 25s
    em.update([])
    events = _read_events(tmp_path / "events_p1.jsonl")
    assert [e["event"] for e in events] == ["appeared", "disappeared"]
    assert _read_json(tmp_path / "current_p1.json")["cards"] == []


def test_unknown_card_falls_back_to_id_as_name(tmp_path):
    clock = FakeClock()
    em = CardInfoEmitter(str(tmp_path), 2, FakeResolver({}), clock=clock)
    em.update(["77777777"])
    snap = _read_json(tmp_path / "current_p2.json")
    assert snap["cards"] == [{"id": 77777777, "name": "77777777"}]


def test_from_env_returns_none_without_dir(monkeypatch):
    monkeypatch.delenv("DRAW2_CARD_INFO_DIR", raising=False)
    assert CardInfoEmitter.from_env(1) is None


def test_from_env_builds_emitter_with_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("DRAW2_CARD_INFO_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("DRAW2_CARD_INFO_LANG", "fr")
    monkeypatch.setenv("DRAW2_CARD_INFO_DB", "0")        # no bulk download in tests
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))  # keep the DB under tmp
    em = CardInfoEmitter.from_env(2, out_of_screen_time=10)
    assert isinstance(em, CardInfoEmitter)
    assert em.channel == 2
    assert em.out_of_screen_time == 10
    assert isinstance(em.resolver, CardDatabase)
    assert em.resolver.lang == "fr"


def test_update_swallows_resolver_exception(tmp_path):
    class BoomResolver:
        def get(self, cid):
            raise RuntimeError("resolver down")

    clock = FakeClock()
    em = CardInfoEmitter(str(tmp_path), 1, BoomResolver(), clock=clock)
    em.update(["89631139"])  # must not raise


def test_update_does_not_block_on_slow_resolver(tmp_path):
    # A resolver whose get() blocks until released, simulating a slow network.
    class BlockingResolver:
        def __init__(self):
            self.release = threading.Event()

        def get(self, cid):
            self.release.wait(5)
            return {"id": int(cid), "name": "Resolved"}

    clock = FakeClock()
    resolver = BlockingResolver()
    em = CardInfoEmitter(str(tmp_path), 1, resolver, clock=clock)
    try:
        t0 = time.monotonic()
        em.update(["123"])  # must return immediately despite the slow resolver
        assert time.monotonic() - t0 < 1.0

        # Before the fetch resolves, the card is exported with the passcode as name.
        assert _read_json(tmp_path / "current_p1.json")["cards"] == [{"id": 123, "name": "123"}]
        assert (tmp_path / "current_card_p1.txt").read_text(encoding="utf-8") == "123"

        # Once the fetch completes, flush() re-writes the snapshot, enriched.
        resolver.release.set()
        em.flush()
        assert _read_json(tmp_path / "current_p1.json")["cards"] == [{"id": 123, "name": "Resolved"}]
        assert (tmp_path / "current_card_p1.txt").read_text(encoding="utf-8") == "Resolved"
    finally:
        em.close()


# --- CardDatabase (local SQLite store + version-checked bulk refresh) ---

def _row_blue(**over):
    row = {"id": 89631139, "name": "Blue-Eyes White Dragon", "type": "Normal Monster",
           "attribute": "LIGHT", "race": "Dragon", "atk": 3000, "def": 2500,
           "desc": "This legendary dragon...", "misc": "ignored"}
    row.update(over)
    return row


def _bulk(*rows):
    return {"data": list(rows)}


def _db(tmp_path, **kw):
    kw.setdefault("auto_refresh", False)
    kw.setdefault("version_fetcher", lambda: "v1")
    kw.setdefault("bulk_fetcher", lambda: _bulk(_row_blue()))
    kw.setdefault("card_fetcher", lambda cid: _bulk())
    return CardDatabase(db_path=str(tmp_path / "cards.sqlite"), **kw)


def test_db_serves_from_local_after_refresh_without_single_fetch(tmp_path):
    def no_single(cid):
        raise AssertionError("should not single-fetch a card present in the local DB")

    db = _db(tmp_path, card_fetcher=no_single)
    db.refresh()
    out = db.get("89631139")
    assert out["name"] == "Blue-Eyes White Dragon"
    assert out["atk"] == 3000  # rich fields from the bulk DB


def test_db_refresh_skips_when_version_unchanged(tmp_path):
    calls = {"n": 0}

    def bulk():
        calls["n"] += 1
        return _bulk(_row_blue())

    db = _db(tmp_path, bulk_fetcher=bulk)
    db.refresh()  # downloads
    db.refresh()  # version unchanged + non-empty -> no re-download
    assert calls["n"] == 1


def test_db_redownloads_on_new_version(tmp_path):
    ver = {"v": "v1"}
    calls = {"n": 0}

    def bulk():
        calls["n"] += 1
        return _bulk(_row_blue())

    db = _db(tmp_path, version_fetcher=lambda: ver["v"], bulk_fetcher=bulk)
    db.refresh()
    ver["v"] = "v2"
    db.refresh()
    assert calls["n"] == 2


def test_db_miss_falls_back_to_single_fetch_and_caches(tmp_path):
    calls = {"n": 0}

    def card(cid):
        calls["n"] += 1
        return _bulk(_row_blue(id=int(cid)))

    db = _db(tmp_path, bulk_fetcher=lambda: _bulk(), card_fetcher=card)  # empty bulk
    db.refresh()
    assert db.get("89631139")["name"] == "Blue-Eyes White Dragon"  # single fetch
    assert db.get("89631139")["name"] == "Blue-Eyes White Dragon"  # served locally now
    assert calls["n"] == 1


def test_db_offline_version_check_keeps_local_db(tmp_path):
    db = _db(tmp_path)
    db.refresh()  # populate with v1

    def boom():
        raise RuntimeError("offline")

    def no_bulk():
        raise AssertionError("must not bulk-download when offline")

    db2 = CardDatabase(db_path=str(tmp_path / "cards.sqlite"), auto_refresh=False,
                       version_fetcher=boom, bulk_fetcher=no_bulk, card_fetcher=lambda cid: _bulk())
    db2.refresh()  # version check fails -> no crash, no download
    assert db2.get("89631139")["name"] == "Blue-Eyes White Dragon"  # from the local DB


def test_db_unknown_card_returns_empty(tmp_path):
    db = _db(tmp_path, bulk_fetcher=lambda: _bulk(), card_fetcher=lambda cid: _bulk())
    db.refresh()
    assert db.get("77777777") == {}


def test_db_bulk_url_has_only_supported_params(tmp_path, monkeypatch):
    # YGOPRODeck answers 400 to unknown values (e.g. misc=no), which silently
    # disabled the full download; only `language` may be added.
    urls = []
    db = CardDatabase(db_path=str(tmp_path / "cards.sqlite"), auto_refresh=False)
    monkeypatch.setattr(db, "_request", lambda url, timeout: urls.append(url) or {})
    db._http_bulk()
    db_fr = CardDatabase(lang="fr", db_path=str(tmp_path / "fr.sqlite"), auto_refresh=False)
    monkeypatch.setattr(db_fr, "_request", lambda url, timeout: urls.append(url) or {})
    db_fr._http_bulk()
    assert "misc" not in urls[0] and "misc" not in urls[1]
    assert urls[1].endswith("language=fr")


def test_db_version_reads_database_version_key(tmp_path, monkeypatch):
    db = CardDatabase(db_path=str(tmp_path / "cards.sqlite"), auto_refresh=False)
    monkeypatch.setattr(db, "_request", lambda url, timeout: [{"database_version": "147.37",
                                                               "last_update": "2026-10-16 16:01:51"}])
    assert db._http_version() == "147.37"


def test_first_event_has_name_when_card_is_in_local_db(tmp_path):
    db = _db(tmp_path)
    db.refresh()
    em = CardInfoEmitter(str(tmp_path / "out"), 1, db, clock=lambda: 100.0)
    em.update(["LOB-EN001-89631139"])
    event = _json.loads((tmp_path / "out" / "events_p1.jsonl").read_text().splitlines()[0])
    assert event["card"]["name"] == "Blue-Eyes White Dragon"
    em.close()
