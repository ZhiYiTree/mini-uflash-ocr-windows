from webapp.app import on_run, on_toggle_pause, on_stop_job, _CANCEL_EVENT, _PAUSE_EVENT
from webapp.ui import _empty_result_markup, _progress_markup, _show_progress, _hide_progress


def test_result_and_progress_have_visible_states():
    assert "文字会显示在这里" in _empty_result_markup()
    assert 'data-state="running"' in _progress_markup(True)
    assert 'role="status"' in _progress_markup(True)
    assert 'data-state="paused"' in _progress_markup(True, paused=True)


def test_pause_toggle_preserves_a_resumable_state():
    _CANCEL_EVENT.clear()
    _PAUSE_EVENT.clear()
    _, paused = on_toggle_pause()
    _, resumed = on_toggle_pause()
    assert "识别已暂停" in paused
    assert "正在识别文档" in resumed


def test_stop_requests_cancel_and_unblocks_pause():
    _CANCEL_EVENT.clear()
    _PAUSE_EVENT.set()
    pause_upd, stop_upd, progress, status = on_stop_job()
    assert _CANCEL_EVENT.is_set()
    assert not _PAUSE_EVENT.is_set()
    assert "停止" in status
    assert 'data-state="idle"' in progress or 'data-state="paused"' not in progress
    _CANCEL_EVENT.clear()


def test_missing_file_error_uses_action_status_slot():
    _CANCEL_EVENT.clear()
    outputs = next(
        on_run(None, "", "稳定模式", "gundam", 4096, 256, "balanced", progress=None)
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


def test_progress_controls_include_stop_button_state():
    vis, pause, stop = _show_progress()
    assert 'data-state="running"' in vis
    assert pause.get("interactive") is True
    assert stop.get("interactive") is True
    vis2, pause2, stop2 = _hide_progress()
    assert 'data-state="idle"' in vis2
    assert pause2.get("interactive") is False
    assert stop2.get("interactive") is False
