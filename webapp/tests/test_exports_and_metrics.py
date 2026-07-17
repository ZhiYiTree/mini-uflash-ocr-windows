import json

from webapp import config
from webapp.export_utils import build_result_json, create_run_dir, save_result
from webapp.metrics import format_metrics_report
from webapp.mini_uflash_core.stage9_common import first_mismatch
from webapp.unlimited_ocr_engine import clean_markdown, _markdown_to_plain


def test_output_directory_and_json_export(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "输出")
    run_dir = create_run_dir()
    payload = build_result_json("stable", "image", "# 标题", "标题", 1.25, 8)
    files = save_result(
        run_dir,
        markdown="# 标题",
        plain_text="标题",
        result_json=payload,
        metrics_json={"generated_tokens": 8},
    )
    loaded = json.loads(files["result.json"].read_text(encoding="utf-8"))
    assert loaded["platform"] == "Windows"
    assert loaded["markdown"] == "# 标题"
    assert files["metrics.json"].is_file()


def test_markdown_cleaning_and_plain_text():
    raw = "# 标题\n**正文**<｜end▁of▁sentence｜>"
    assert clean_markdown(raw).endswith("**正文**")
    assert _markdown_to_plain(clean_markdown(raw)) == "标题\n正文"


def test_metrics_format_and_exactness_mismatch():
    assert first_mismatch([1, 2, 9], [1, 2, 3]) == 2
    report = format_metrics_report(
        {
            "checkpoint_path": "stage11b.pt",
            "checkpoint_step": 49100,
            "final_token_exactness": False,
            "first_mismatch": 2,
            "per_position_accuracy": [1.0, 0.5],
            "note": "理论估算，不是当前端到端实际加速",
        },
        mode="mini_uflash_precise",
    )
    assert "Exactness: ❌ 失败" in report
    assert "首次不匹配位置: 2" in report


def test_dflash_metrics_include_tier_label():
    report = format_metrics_report(
        {
            "tier": "fast",
            "pages": 1,
            "elapsed_seconds": 3.2,
            "generated_tokens": 40,
            "speculative_rounds": 10,
            "direct_block_commits": 4,
            "full_block_commits": 1,
            "direct_committed_tokens": 12,
            "full_block_ratio": 0.1,
            "mean_accepted_draft": 1.2,
            "resync_count": 0,
            "pure_b1_rounds": 2,
            "target_forward_reduction": 1.1,
            "fallback_pages": [],
        },
        mode="mini_uflash_stable_dflash",
    )
    assert "快速" in report
    assert "加速档位" in report

