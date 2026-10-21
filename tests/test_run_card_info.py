import importlib
import types

# draw/__init__.py re-exports the run() function as `draw.run`, so fetch the module itself.
run = importlib.import_module("draw.run")


def _patch_draw(monkeypatch):
    # Make constructing DrawSharedMemoryHandler cheap (no model/dataset load).
    monkeypatch.setattr(run, "Draw", lambda **kw: types.SimpleNamespace())
    monkeypatch.setattr(run, "get_deck_list", lambda dl: [])


def test_handler_has_no_emitter_without_env(monkeypatch):
    _patch_draw(monkeypatch)
    monkeypatch.delenv("DRAW2_CARD_INFO_DIR", raising=False)
    h = run.DrawSharedMemoryHandler(channel="1")
    assert h.card_info is None


def test_handler_builds_emitter_with_env(monkeypatch, tmp_path):
    _patch_draw(monkeypatch)
    monkeypatch.setenv("DRAW2_CARD_INFO_DIR", str(tmp_path))
    h = run.DrawSharedMemoryHandler(channel="2")
    from draw.card_info import CardInfoEmitter
    assert isinstance(h.card_info, CardInfoEmitter)
    assert h.card_info.channel == 2
