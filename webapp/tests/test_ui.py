from webapp.app import on_run, on_toggle_pause
from webapp.ui import _empty_result_markup, _progress_markup


def test_result_and_progress_have_visible_states():
    assert "文字会显示在这里" in _empty_result_markup()
    assert 'data-state="running"' in _progress_markup(True)
    assert 'role="status"' in _progress_markup(True)
    assert 'data-state="paused"' in _progress_markup(True, paused=True)


def test_pause_toggle_preserves_a_resumable_state():
    _, paused = on_toggle_pause()
    _, resumed = on_toggle_pause()
    assert "识别已暂停" in paused
    assert "正在识别文档" in resumed


def test_missing_file_error_uses_action_status_slot():
    outputs = next(
        on_run(None, "", "稳定模式", "gundam", 4096, 256, "fast", progress=None)
    )
    assert len(outputs) == 11
    assert "请先上传" in outputs[5]
    assert "status-bar" in outputs[10]


def test_mode_controls_show_tier_for_accel():
    from webapp.ui import _toggle_mode_controls

    probe, tier = _toggle_mode_controls("Mini UFlash 精确模式")
    assert probe["visible"] is False
    assert tier["visible"] is True
    probe2, tier2 = _toggle_mode_controls("稳定模式")
    assert tier2["visible"] is False
