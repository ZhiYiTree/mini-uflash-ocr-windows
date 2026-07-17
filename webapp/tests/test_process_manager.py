from webapp import process_manager


def test_pid_file_lifecycle(tmp_path):
    path = process_manager.write_pid(tmp_path, 12345)
    assert path.is_file()
    assert process_manager.read_pid(tmp_path) == 12345
    process_manager.remove_pid(tmp_path)
    assert process_manager.read_pid(tmp_path) is None


def test_pid_command_line_guard():
    assert process_manager._cmdline_belongs_to_project(
        ["python.exe", "-m", "webapp.app"]
    )
    assert not process_manager._cmdline_belongs_to_project(
        ["python.exe", "unrelated.py"]
    )
