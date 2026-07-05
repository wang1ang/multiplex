import importlib.util
import sys
from pathlib import Path


def _load_registry():
    path = Path(__file__).parent / "multiplex" / "registry.py"
    spec = importlib.util.spec_from_file_location("registry_under_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


registry = _load_registry()


def test_missing_named_model_downloads_from_hf(monkeypatch, tmp_path):
    calls = []

    def fake_download(repo_id, root=registry.DEFAULT_ROOT):
        calls.append((repo_id, root))
        return registry.ModelEntry(name="org--repo", path=str(tmp_path / "org--repo"))

    monkeypatch.setattr(registry, "download_model", fake_download)

    entry = registry.select("org/repo", root=str(tmp_path))

    assert entry.name == "org--repo"
    assert calls == [("org/repo", str(tmp_path))]


def test_existing_local_name_wins_over_hf_download(monkeypatch, tmp_path):
    model_dir = tmp_path / "local-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    def fail_download(*_args, **_kwargs):
        raise AssertionError("should not download an existing local model")

    monkeypatch.setattr(registry, "download_model", fail_download)

    entry = registry.select("local-model", root=str(tmp_path))

    assert entry.name == "local-model"
    assert entry.path == str(model_dir)


def test_existing_hf_style_local_dir_wins_over_download(monkeypatch, tmp_path):
    model_dir = tmp_path / "org--repo"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    def fail_download(*_args, **_kwargs):
        raise AssertionError("should not download an existing local model")

    monkeypatch.setattr(registry, "download_model", fail_download)

    entry = registry.select("org/repo", root=str(tmp_path))

    assert entry.name == "org--repo"
    assert entry.path == str(model_dir)


def test_huggingface_url_downloads_repo_id(monkeypatch, tmp_path):
    calls = []

    def fake_download(model, root=registry.DEFAULT_ROOT):
        calls.append((model, root))
        return registry.ModelEntry(name="org--repo", path=str(tmp_path / "org--repo"))

    monkeypatch.setattr(registry, "download_model", fake_download)

    entry = registry.select("https://huggingface.co/org/repo/tree/main", root=str(tmp_path))

    assert entry.name == "org--repo"
    assert calls == [
        ("https://huggingface.co/org/repo/tree/main", str(tmp_path))
    ]


def test_huggingface_url_reuses_existing_local_dir(monkeypatch, tmp_path):
    model_dir = tmp_path / "org--repo"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")

    def fail_download(*_args, **_kwargs):
        raise AssertionError("should not download an existing local model")

    monkeypatch.setattr(registry, "download_model", fail_download)

    entry = registry.select("https://huggingface.co/org/repo", root=str(tmp_path))

    assert entry.name == "org--repo"
    assert entry.path == str(model_dir)
