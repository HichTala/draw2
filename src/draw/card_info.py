"""Opt-in export of structured detected-card info to per-channel files.

Active only when DRAW2_CARD_INFO_DIR is set in the environment (the OBS plugin
sets it when the user enables the feature). Otherwise nothing here runs.

Metadata lookups (YGOPRODeck) can be slow on a cold cache, so they run on a
background thread: the detection loop only ever reads the local cache and
writes files, and never blocks on the network. A card shows up immediately
with its passcode as the name and is enriched (atk/def/desc/…) as soon as the
background fetch resolves.
"""

import json
import os
import queue
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
from pathlib import Path

CARDINFO_URL = "https://db.ygoprodeck.com/api/v7/cardinfo.php"
DBVER_URL = "https://db.ygoprodeck.com/api/v7/checkDBVer.php"
_METADATA_KEYS = ("id", "name", "type", "attribute", "race", "atk", "def", "desc")


def _default_metadata_cache_dir(lang):
    base = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")) / "draw2" / "cardinfo" / lang
    return base


def _card_subset(row):
    """Keep only the small, display-relevant metadata fields that are present."""
    return {k: row[k] for k in _METADATA_KEYS if row.get(k) is not None}


def card_id_from_label(label):
    """Extract the YGOPRODeck passcode from a classifier label.

    Labels end in '-<id>' (e.g. 'LOB-EN005-89631139'): the passcode is always
    the last '-'-separated segment, same as the decklist filter in draw.py.
    """
    return str(label).split("-")[-1]


class YgoprodeckMetadata:
    """Resolve a YGOPRODeck passcode to a small metadata dict, cached on disk."""

    def __init__(self, lang="en", cache_dir=None, fetcher=None):
        self.lang = lang or "en"
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _default_metadata_cache_dir(self.lang)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._fetcher = fetcher if fetcher is not None else self._http_fetch
        self._mem = {}

    def get(self, card_id):
        cid = str(card_id)
        if cid in self._mem:
            return self._mem[cid]
        data = self._load_cache(cid)
        if data is None:
            try:
                data = self._normalize(self._fetcher(cid))
            except Exception as exc:  # never break the caller
                print(f"[card_info] metadata fetch failed {cid}: {exc}")
                data = {}
            if data:  # only persist successful lookups; retry empties next session
                self._save_cache(cid, data)
        self._mem[cid] = data
        return data

    def peek(self, card_id):
        """Local-only lookup (memory or disk cache); None on a miss, never fetches."""
        cid = str(card_id)
        if cid in self._mem:
            return self._mem[cid]
        return self._load_cache(cid)

    def _normalize(self, raw):
        rows = (raw or {}).get("data") or []
        if not rows:
            return {}
        row = rows[0]
        return {k: row[k] for k in _METADATA_KEYS if row.get(k) is not None}

    def _http_fetch(self, cid):
        params = {"id": cid}
        if self.lang and self.lang != "en":
            params["language"] = self.lang
        url = CARDINFO_URL + "?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"User-Agent": "draw2-plugin"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _cache_path(self, cid):
        return self.cache_dir / f"{cid}.json"

    def _load_cache(self, cid):
        path = self._cache_path(cid)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _save_cache(self, cid, data):
        path = self._cache_path(cid)
        tmp = str(path) + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False))
            os.replace(tmp, path)
        except Exception as exc:
            print(f"[card_info] metadata cache write failed {cid}: {exc}")


class CardDatabase:
    """Local YGOPRODeck metadata store backed by SQLite.

    Lookups are served from the local DB (offline, instant). The full database
    is (re)downloaded in the background only when YGOPRODeck reports a new
    version (checkDBVer), so a normal start is just a cheap version check. A
    passcode missing from the local DB (e.g. a brand-new card) falls back to a
    single-card fetch and is stored. Drop-in resolver for CardInfoEmitter:
    exposes get(cid) -> metadata dict.

    Fetchers are injectable for testing. Set the env var DRAW2_CARD_INFO_DB=0 to
    skip the full download and resolve lazily, one card at a time.
    """

    def __init__(self, lang="en", db_path=None, *, bulk_fetcher=None, card_fetcher=None,
                 version_fetcher=None, auto_refresh=True):
        self.lang = lang or "en"
        if db_path is None:
            cache_dir = _default_metadata_cache_dir(self.lang)
            cache_dir.mkdir(parents=True, exist_ok=True)
            db_path = cache_dir / "cards.sqlite"
        self.db_path = str(db_path)
        self._bulk_fetcher = bulk_fetcher or self._http_bulk
        self._card_fetcher = card_fetcher or self._http_card
        self._version_fetcher = version_fetcher or self._http_version
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._mem = {}
        self._mem_lock = threading.Lock()
        self._init_db()
        self.refreshed = threading.Event()  # set once the background refresh finishes
        if auto_refresh:
            threading.Thread(target=self._refresh_safe, name=f"card-db-refresh-{self.lang}",
                             daemon=True).start()

    # --- connection / schema ---

    def _conn(self):
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.db_path, timeout=30)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _init_db(self):
        with self._write_lock:
            conn = self._conn()
            conn.execute("CREATE TABLE IF NOT EXISTS cards (id TEXT PRIMARY KEY, meta TEXT NOT NULL)")
            conn.execute("CREATE TABLE IF NOT EXISTS meta_kv (k TEXT PRIMARY KEY, v TEXT)")
            conn.commit()

    def _get_kv(self, key):
        row = self._conn().execute("SELECT v FROM meta_kv WHERE k=?", (key,)).fetchone()
        return row[0] if row else None

    def _count(self):
        return self._conn().execute("SELECT COUNT(*) FROM cards").fetchone()[0]

    # --- lookups (resolver interface) ---

    def get(self, card_id):
        cid = str(card_id)
        with self._mem_lock:
            if cid in self._mem:
                return self._mem[cid]
        data = self._read(cid)
        if data is None:
            data = self._fetch_one(cid)
        with self._mem_lock:
            self._mem[cid] = data
        return data

    def peek(self, card_id):
        """Local-only lookup (memory or SQLite); None on a miss, never fetches."""
        cid = str(card_id)
        with self._mem_lock:
            if cid in self._mem:
                return self._mem[cid]
        return self._read(cid)

    def _read(self, cid):
        row = self._conn().execute("SELECT meta FROM cards WHERE id=?", (cid,)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    def _fetch_one(self, cid):
        try:
            raw = self._card_fetcher(cid)
        except Exception as exc:  # never break the caller
            print(f"[card_info] metadata fetch failed {cid}: {exc}")
            return {}
        rows = (raw or {}).get("data") or []
        if not rows:
            return {}
        sub = _card_subset(rows[0])
        if sub:
            self._store([sub])
        return sub

    # --- bulk refresh ---

    def _refresh_safe(self):
        try:
            self.refresh()
        except Exception as exc:
            print(f"[card_info] DB refresh failed: {exc}")
        finally:
            self.refreshed.set()

    def refresh(self):
        try:
            remote = self._version_fetcher()
        except Exception as exc:  # offline / failed: keep whatever local DB we have
            print(f"[card_info] DB version check failed (using local DB): {exc}")
            return
        remote = str(remote) if remote is not None else None
        if remote and self._get_kv("version") == remote and self._count() > 0:
            return  # already up to date
        raw = self._bulk_fetcher()
        rows = (raw or {}).get("data") or []
        if not rows:
            return
        subs = [s for s in (_card_subset(r) for r in rows) if s]
        self._store(subs, version=remote)
        with self._mem_lock:
            self._mem.clear()

    def _store(self, subsets, version=None):
        with self._write_lock:
            conn = self._conn()
            conn.executemany("INSERT OR REPLACE INTO cards(id, meta) VALUES (?, ?)",
                             [(str(s.get("id")), json.dumps(s, ensure_ascii=False)) for s in subsets])
            if version is not None:
                conn.execute("INSERT OR REPLACE INTO meta_kv(k, v) VALUES ('version', ?)", (version,))
            conn.commit()

    # --- HTTP defaults (overridable for tests) ---

    def _request(self, url, timeout):
        req = urllib.request.Request(url, headers={"User-Agent": "draw2-plugin"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_bulk(self):
        params = {}
        if self.lang and self.lang != "en":
            params["language"] = self.lang
        return self._request(CARDINFO_URL + "?" + urllib.parse.urlencode(params), timeout=120)

    def _http_card(self, cid):
        params = {"id": cid}
        if self.lang and self.lang != "en":
            params["language"] = self.lang
        return self._request(CARDINFO_URL + "?" + urllib.parse.urlencode(params), timeout=20)

    def _http_version(self):
        data = self._request(DBVER_URL, timeout=20)
        # checkDBVer answers [{"database_version": "...", "last_update": "..."}].
        if isinstance(data, list) and data:
            data = data[0]
        if isinstance(data, dict):
            return data.get("database_version") or data.get("version")
        return None


# Sentinel pushed on the fetch queue to stop the worker thread.
_STOP = object()


class CardInfoEmitter:
    """Track currently-detected cards and export them to per-channel files.

    Appearance is emitted on first sight; disappearance only after a label has
    been absent for `out_of_screen_time` seconds (hysteresis avoids flicker).

    Metadata is resolved on a background daemon thread so the detection loop is
    never blocked by a (possibly slow) network lookup. `update()` returns
    immediately; an unresolved card is written with its passcode as the name and
    the snapshot is re-written, enriched, once the lookup completes.
    """

    def __init__(self, out_dir, channel, resolver, out_of_screen_time=25, clock=time.time):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.channel = int(channel)
        self.resolver = resolver
        self.out_of_screen_time = out_of_screen_time
        self._clock = clock
        self._present = {}  # label -> last_seen_ts, insertion-ordered
        suffix = f"_p{self.channel}"
        self._snapshot_path = os.path.join(out_dir, f"current{suffix}.json")
        self._events_path = os.path.join(out_dir, f"events{suffix}.jsonl")
        self._text_path = os.path.join(out_dir, f"current_card{suffix}.txt")

        # Background metadata resolution. The detection thread only reads `_meta`
        # (under `_meta_lock`) and enqueues misses; the worker does the blocking
        # `resolver.get()` and sets `_refresh` so the next `update()` re-writes
        # the snapshot with the enriched data.
        self._meta = {}  # cid -> resolved metadata dict
        self._meta_lock = threading.Lock()
        self._fetch_q = queue.Queue()
        self._queued = set()
        self._refresh = threading.Event()
        self._worker = threading.Thread(
            target=self._resolve_loop, name=f"card-info-meta-p{self.channel}", daemon=True)
        self._worker.start()

    @classmethod
    def from_env(cls, channel, out_of_screen_time=25):
        out_dir = os.environ.get("DRAW2_CARD_INFO_DIR")
        if not out_dir:
            return None
        lang = os.environ.get("DRAW2_CARD_INFO_LANG", "en")
        # Full local DB by default (offline, no per-card requests); set
        # DRAW2_CARD_INFO_DB=0 to skip the download and resolve lazily instead.
        auto_refresh = os.environ.get("DRAW2_CARD_INFO_DB", "1") != "0"
        resolver = CardDatabase(lang=lang, auto_refresh=auto_refresh)
        return cls(out_dir, channel, resolver, out_of_screen_time=out_of_screen_time)

    # --- background metadata worker -------------------------------------------------

    def _resolve_loop(self):
        while True:
            cid = self._fetch_q.get()
            try:
                if cid is _STOP:
                    return
                try:
                    meta = self.resolver.get(cid)
                except Exception as exc:  # never let the worker die
                    print(f"[card_info] metadata fetch failed {cid}: {exc}")
                    meta = {}
                with self._meta_lock:
                    self._meta[cid] = meta or {}
                self._refresh.set()  # ask the next update() to re-write the snapshot
            finally:
                self._fetch_q.task_done()

    def _meta_for(self, cid):
        """Local-only metadata lookup; never blocks. Queues a fetch on a miss."""
        with self._meta_lock:
            if cid in self._meta:
                return self._meta[cid]
        # A resolver with a local store (e.g. the SQLite DB) can answer right
        # away, so even the first "appeared" event carries the card's name.
        peek = getattr(self.resolver, "peek", None)
        if peek is not None:
            try:
                local = peek(cid)
            except Exception:
                local = None
            if local:
                with self._meta_lock:
                    self._meta[cid] = local
                return local
        if cid not in self._queued:
            self._queued.add(cid)
            self._fetch_q.put(cid)
        return {}

    def flush(self):
        """Block until queued metadata fetches finish, then re-write the snapshot.

        For shutdown / tests; the live detection loop never calls this.
        """
        self._fetch_q.join()
        self._refresh.clear()
        if self._present:
            try:
                self._write_snapshot(self._clock())
            except Exception as exc:
                print(f"[card_info] snapshot write failed: {exc}")

    def close(self):
        """Stop the background worker (best-effort; the thread is a daemon)."""
        self._fetch_q.put(_STOP)

    # --- detection-loop entry point -------------------------------------------------

    def update(self, labels):
        now = self._clock()
        seen = list(dict.fromkeys(labels))  # de-dup, keep order
        changed = False

        for label in seen:
            if label not in self._present:
                self._present[label] = now
                self._emit_event("appeared", label, now)
                changed = True
            else:
                self._present[label] = now

        for label in [l for l, ts in self._present.items() if l not in seen and now - ts > self.out_of_screen_time]:
            del self._present[label]
            self._emit_event("disappeared", label, now)
            changed = True

        # A background fetch may have enriched a still-present card; re-write even
        # if the on-screen set itself did not change.
        if self._refresh.is_set():
            self._refresh.clear()
            changed = True

        if changed:
            try:
                self._write_snapshot(now)
            except Exception as exc:
                print(f"[card_info] snapshot write failed: {exc}")

    def _entry(self, label):
        cid = card_id_from_label(label)
        meta = dict(self._meta_for(cid))
        if "id" not in meta:
            meta["id"] = int(cid) if cid.isdigit() else cid
        if "name" not in meta:
            meta["name"] = cid
        return meta

    def _emit_event(self, event, label, now):
        try:
            line = json.dumps({"ts": now, "event": event, "card": self._entry(label)}, ensure_ascii=False)
            with open(self._events_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as exc:
            print(f"[card_info] event write failed: {exc}")

    def _write_snapshot(self, now):
        cards = [self._entry(label) for label in self._present]
        self._write_atomic(self._snapshot_path,
                            json.dumps({"updated_at": now, "channel": self.channel, "cards": cards},
                                       ensure_ascii=False))
        self._write_atomic(self._text_path, "\n".join(c["name"] for c in cards))

    def _write_atomic(self, path, text):
        tmp = path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, path)
        except Exception as exc:
            print(f"[card_info] write failed {path}: {exc}")
