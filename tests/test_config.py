import json
from pathlib import Path

from jarvis_nlu.config import Config


def test_missing_file_yields_defaults(tmp_path):
    config = Config.load(tmp_path / "absent.json")
    assert config.tick_seconds == 20
    assert config.daily_brief_at == "08:30"
    assert config.persona_path is None


def test_file_overrides_defaults(tmp_path):
    path = tmp_path / "jarvis.config.json"
    path.write_text(json.dumps({"tick_seconds": 5, "daily_brief_at": None}))
    config = Config.load(path)
    assert config.tick_seconds == 5
    assert config.daily_brief_at is None


def test_paths_are_resolved_to_path_objects(tmp_path):
    path = tmp_path / "jarvis.config.json"
    path.write_text(json.dumps({"db_path": "data/custom.db"}))
    assert isinstance(Config.load(path).db_path, Path)


def test_explicit_null_tick_seconds_falls_back_to_default(tmp_path):
    path = tmp_path / "jarvis.config.json"
    path.write_text(json.dumps({"tick_seconds": None}))
    config = Config.load(path)
    assert config.tick_seconds == 20


def test_explicit_null_db_path_falls_back_to_default(tmp_path):
    path = tmp_path / "jarvis.config.json"
    path.write_text(json.dumps({"db_path": None}))
    config = Config.load(path)
    assert config.db_path == Path("data/jarvis.db")


def test_explicit_null_model_dir_falls_back_to_default(tmp_path):
    path = tmp_path / "jarvis.config.json"
    path.write_text(json.dumps({"model_dir": None}))
    config = Config.load(path)
    assert config.model_dir == Path("models")
