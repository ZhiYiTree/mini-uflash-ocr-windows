from pathlib import Path

import pytest

from webapp import config
from webapp.model_manager import ModelManager
from webapp.mini_uflash_core.checkpoint import load_checkpoint


def test_model_directory_validation(tmp_path):
    model = tmp_path / "Unlimited OCR"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"x")
    (model / "modeling_unlimitedocr.py").write_text("# local", encoding="utf-8")
    ModelManager._validate_model_dir(model)


def test_model_directory_rejects_incomplete(tmp_path):
    with pytest.raises(FileNotFoundError):
        ModelManager._validate_model_dir(tmp_path)


def test_checkpoint_priority_and_never_last(tmp_path, monkeypatch):
    (tmp_path / "drafter_last.pt").write_bytes(b"last")
    (tmp_path / "drafter_v2_30k.pt").write_bytes(b"base")
    (tmp_path / "drafter_v2_stage11b_best.pt").write_bytes(b"best")
    monkeypatch.setattr(config, "_WEIGHT_SEARCH_ROOTS", [tmp_path])
    assert config.discover_weight().name == "drafter_v2_stage11b_best.pt"


def test_real_checkpoint_can_move_as_a_module():
    path = config.discover_weight()
    if path is None:
        pytest.skip("Stage 11B checkpoint not present")
    model, checkpoint = load_checkpoint(path, device="cpu")
    model = model.to("cpu").eval()
    assert int(checkpoint.get("step", -1)) >= 0
    assert next(model.parameters()).device.type == "cpu"
