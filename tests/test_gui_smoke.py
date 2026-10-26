from cv_midterm.gui import main


def test_gui_smoke_mode() -> None:
    assert main(["--smoke-test"]) == 0
