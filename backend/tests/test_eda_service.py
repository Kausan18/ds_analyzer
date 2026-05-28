import pandas as pd
import numpy as np
import io
import pytest
from services.eda_service import run_eda, get_missing, get_outliers, get_duplicates


def make_csv(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


def test_run_eda_returns_session_and_report():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    session_id, report = run_eda(make_csv(df), "test.csv")
    assert isinstance(session_id, str)
    assert len(session_id) == 8
    assert report["shape"]["rows"] == 3
    assert report["shape"]["cols"] == 2


def test_missing_detected():
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
    result = get_missing(df)
    assert "a" in result
    assert result["a"]["count"] == 1
    assert "b" not in result


def test_no_missing_returns_empty():
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert get_missing(df) == {}


def test_duplicates_detected():
    df = pd.DataFrame({"a": [1, 1, 2]})
    result = get_duplicates(df)
    assert result["count"] == 1


def test_outliers_detected():
    # Z-score > 3 outlier
    normal = [10.0] * 50
    outlier = [10000.0]
    df = pd.DataFrame({"x": normal + outlier})
    result = get_outliers(df)
    assert "x" in result
    assert result["x"]["count"] >= 1
