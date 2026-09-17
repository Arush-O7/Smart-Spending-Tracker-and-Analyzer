"""Runs app.py in-process so a broken tab fails here instead of in the browser."""

import pandas as pd
import pytest

from spending_analyzer.config import PROJECT_ROOT

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = PROJECT_ROOT / "app.py"
SAMPLE = PROJECT_ROOT / "data" / "for_testing.csv"


def run(state=None):
    app = AppTest.from_file(str(APP), default_timeout=120)
    for k, v in (state or {}).items():
        app.session_state[k] = v
    return app.run()


def test_renders_with_no_data():
    app = run()
    assert not app.exception
    assert len(app.tabs) == 6


def test_renders_every_tab_with_data():
    if not SAMPLE.exists():
        pytest.skip("sample data missing")
    app = run({"df_main": pd.read_csv(SAMPLE)})
    assert not app.exception, app.exception
    assert not app.error, [e.value for e in app.error]
    assert any("Total spend" in m.label for m in app.metric)


def test_a_bad_file_shows_an_error_instead_of_crashing():
    app = run({"df_main": pd.DataFrame({"foo": [1], "bar": [2]})})
    assert not app.exception
    assert any("Could not detect required column" in e.value for e in app.error)


def test_the_anomaly_button_produces_results():
    if not SAMPLE.exists():
        pytest.skip("sample data missing")
    app = run({"df_main": pd.read_csv(SAMPLE)})
    button = next(b for b in app.button if "anomaly detection" in b.label.lower())
    app = button.click().run()
    assert not app.exception, app.exception
    assert app.session_state["anomaly_result"].total_anomalies > 0


def test_budget_inputs_feed_the_summary():
    if not SAMPLE.exists():
        pytest.skip("sample data missing")
    app = run({"df_main": pd.read_csv(SAMPLE)})
    key = next(k for k in app.session_state.filtered_state if k.startswith("budget_"))
    app.session_state[key] = 5000.0
    app = app.run()
    assert not app.exception, app.exception
    assert any("Budget" in m.label for m in app.metric)
