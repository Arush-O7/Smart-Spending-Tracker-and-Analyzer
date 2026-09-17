import pandas as pd
import pytest

from spending_analyzer import cli
from spending_analyzer.config import PROJECT_ROOT

SAMPLE = PROJECT_ROOT / "data" / "for_testing.csv"


@pytest.fixture(autouse=True)
def skip_without_sample():
    if not SAMPLE.exists():
        pytest.skip("sample data missing")


def test_predict_writes_a_csv(tmp_path, capsys):
    out = tmp_path / "predictions.csv"
    assert cli.main(["predict", str(SAMPLE), "--output", str(out)]) == 0
    written = pd.read_csv(out)
    assert "pred_category" in written.columns
    assert len(written) > 0
    assert "Agreement with provided labels" in capsys.readouterr().err


def test_risk_writes_alerts(tmp_path):
    out = tmp_path / "alerts.csv"
    assert cli.main(["risk", str(SAMPLE), "--output", str(out), "--percentile", "99"]) == 0
    assert "risk_score" in pd.read_csv(out).columns


def test_fit_bins_writes_the_edges(tmp_path, monkeypatch):
    target = tmp_path / "amount_bins.json"
    monkeypatch.setattr(cli, "save_amount_binning",
                        lambda edges: (target.write_text(str(list(edges))), target)[1])
    assert cli.main(["fit-bins", str(SAMPLE), "--bins", "5"]) == 0
    assert target.exists()


def test_show_config(capsys):
    assert cli.main(["show-config"]) == 0
    assert "amount bins" in capsys.readouterr().out


def test_missing_file(tmp_path):
    with pytest.raises(SystemExit) as e:
        cli.main(["predict", str(tmp_path / "nope.csv")])
    assert "not found" in str(e.value)


def test_bad_schema_exits_non_zero(tmp_path, capsys):
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n")
    assert cli.main(["predict", str(bad)]) == 1
    assert "error:" in capsys.readouterr().err
