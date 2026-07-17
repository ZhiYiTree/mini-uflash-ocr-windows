from pathlib import Path, PureWindowsPath

from webapp import path_utils


def test_windows_drive_path_shape():
    path = PureWindowsPath(r"D:\AI\mini uflash\模型")
    assert path.drive == "D:"
    assert "mini uflash" in path.parts


def test_space_and_chinese_paths(tmp_path):
    target = tmp_path / "有 空格" / "结果"
    resolved = path_utils.ensure_dir(target)
    assert resolved.is_dir()
    assert path_utils.contains_chinese(resolved)


def test_long_path_warning(tmp_path):
    target = tmp_path / ("a" * 220)
    is_long, length = path_utils.check_long_path(target)
    assert is_long == (length >= path_utils.LONG_PATH_WARN)

