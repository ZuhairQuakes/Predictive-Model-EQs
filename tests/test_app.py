from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def test_streamlit_app_starts_without_exception() -> None:
    app = AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=30).run()

    assert not app.exception
    assert app.title or app.markdown
