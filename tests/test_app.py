"""Exercise actual Streamlit rendering and reruns against the committed report."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "artifacts/summary.json").exists(), reason="Train or restore report artifacts first")
def test_dashboard_navigation_and_chart_controls():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.metric) == 4
    app.toggle[0].set_value(True).run()
    app.selectbox[0].select_index(0).run()
    assert not app.exception
    app.radio[0].set_value("Model audit").run()
    assert not app.exception
    assert len(app.dataframe) == 2
    app.radio[0].set_value("Forecast lab").run()
    assert not app.exception
    if (ROOT / "artifacts/model.joblib").exists():
        app.number_input[0].set_value(350)
        app.button[0].click().run()
        assert not app.exception
        assert len(app.metric) == 3
