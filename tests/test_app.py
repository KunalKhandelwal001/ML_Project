from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_shell_renders_without_exception():
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    app.run()

    assert not app.exception
    assert len(app.title) >= 1
