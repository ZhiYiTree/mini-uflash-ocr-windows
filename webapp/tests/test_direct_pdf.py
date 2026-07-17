from webapp.app import _aggregate_dflash_metrics


def test_dflash_pdf_metrics_include_commits_and_fallbacks():
    pages = [
        {
            "mode": "mini_uflash_stable_dflash",
            "tier": "fast",
            "generated_tokens": 80,
            "speculative_rounds": 12,
            "direct_block_commits": 12,
            "full_block_commits": 5,
            "fallback_rounds": 7,
            "direct_committed_tokens": 40,
            "target_decode_forwards": 30,
            "resync_count": 2,
            "pure_b1_rounds": 3,
        },
        {
            "mode": "stable_fallback",
            "generated_tokens": 20,
            "elapsed_seconds": 2.0,
        },
    ]

    result = _aggregate_dflash_metrics(pages, 8.5, [2], tier="fast")

    assert result["mode"] == "mini_uflash_stable_dflash_pdf"
    assert result["tier"] == "fast"
    assert result["pages"] == 2
    assert result["generated_tokens"] == 100
    assert result["full_block_commits"] == 5
    assert result["direct_block_commits"] == 12
    assert result["direct_committed_tokens"] == 40
    assert result["resync_count"] == 2
    assert result["fallback_pages"] == [2]
    assert result["non_lossless"] is True


def test_single_dflash_page_keeps_mode():
    result = _aggregate_dflash_metrics(
        [{
            "mode": "mini_uflash_stable_dflash",
            "generated_tokens": 8,
            "speculative_rounds": 1,
            "full_block_commits": 1,
            "target_decode_forwards": 1,
            "resync_count": 0,
        }],
        1.0,
        [],
        tier="lossless",
    )

    assert result["mode"] == "mini_uflash_stable_dflash"
    assert result["tier"] == "lossless"
    assert result["non_lossless"] is False
    assert result["target_forward_reduction"] == 8.0
