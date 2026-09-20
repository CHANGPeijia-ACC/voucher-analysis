from pathlib import Path

import pandas as pd
import pytest

from gl_builder import make_gl, voucher
from voucher_analysis import evaluate as ev, jet
from voucher_analysis.loaders import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_config(REPO_ROOT / "config.yaml")


def flags(*pairs):
    """Flag rows: (test_id, voucher_no)."""
    return pd.DataFrame([{"test_id": t, "voucher_no": v, "line_no": 1,
                          "amount": 100.0, "reason": "because"} for t, v in pairs],
                        columns=jet.OUTPUT_COLUMNS)


def truth(*pairs):
    """Ground truth rows: (voucher_no, anomaly_type)."""
    return pd.DataFrame([{"voucher_no": v, "anomaly_type": a} for v, a in pairs],
                        columns=["voucher_no", "anomaly_type"])


def test_ratio_returns_none_instead_of_dividing_by_zero():
    assert ev.ratio(1, 4) == 0.25
    assert ev.ratio(0, 0) is None


def test_score_counts_hits_and_misses():
    row = ev.score("JET01", "unbalanced", {"JV000001", "JV000002"}, {"JV000001", "JV000003"})

    assert row["true_positives"] == 1     # JV000001 is flagged and injected
    assert row["false_positives"] == 1    # JV000002 is flagged but normal
    assert row["false_negatives"] == 1    # JV000003 is injected but missed
    assert row["precision"] == 0.5
    assert row["recall"] == 0.5


def test_evaluate_scores_each_test_and_overall():
    table = ev.evaluate(
        flags(("JET01", "JV000001"), ("JET03", "JV000002"), ("JET03", "JV000009")),
        truth(("JV000001", "unbalanced"), ("JV000002", "weekend_holiday"),
              ("JV000003", "weekend_holiday")),
    ).set_index("test_id")

    assert table.loc["JET01", "recall"] == 1.0
    assert table.loc["JET03", "precision"] == 0.5      # JV000009 is a false alarm
    assert table.loc["JET03", "recall"] == 0.5         # JV000003 was missed
    assert table.loc["overall", "true_positives"] == 2
    assert table.loc["overall", "injected"] == 3


def test_evaluate_reports_na_for_a_test_without_ground_truth():
    table = ev.evaluate(flags(("JET00", "JV000001")),
                        truth(("JV000001", "unbalanced"))).set_index("test_id")

    assert table.loc["JET00", "anomaly_type"] == "n/a"
    assert pd.isna(table.loc["JET00", "recall"])


def test_evaluate_lists_every_test_even_with_no_flags():
    table = ev.evaluate(flags(), truth(("JV000001", "unbalanced")))
    assert len(table) == len(ev.TEST_ANOMALY) + 1


def test_compare_outlier_scales_scores_both_variants(config):
    rows = []
    for i in range(40):                      # normal lines, 1,000 to 1,195
        rows += voucher(f"JV{i + 1:06d}", 1000 + i * 5)
    rows += voucher("JV000041", 50000)       # the injected extreme amount
    gl = make_gl(*rows)
    ground_truth = truth(("JV000041", "extreme_amount"))

    table = ev.compare_outlier_scales(gl, ground_truth, config).set_index("test_id")

    assert list(table.index) == ["JET10 raw scale", "JET10 log scale"]
    assert table.loc["JET10 log scale", "recall"] == 1.0
    assert config["jet"]["JET10_outliers"]["log_scale"] is True   # config not changed
