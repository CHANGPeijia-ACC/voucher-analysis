"""Compare test flags with the injected ground truth (precision and recall).

This only works on synthetic data, where the anomalies are known. On real
data nobody knows the full list, which is why the generator writes one.

precision = caught / everything flagged. Low precision means the reviewer
spends time on normal entries.
recall = caught / everything injected. Low recall means real anomalies are
missed. Loosening a threshold usually raises recall and lowers precision.
"""

import copy

import pandas as pd

from voucher_analysis import jet

# The anomaly each test is meant to find. JET00 has no counterpart: the
# generator writes clean records, so its correctness is covered by pytest.
TEST_ANOMALY = {
    "JET00": None,
    "JET01": "unbalanced",
    "JET02": "duplicate_entry",
    "JET03": "weekend_holiday",
    "JET04": "late_night",
    "JET05": "after_close",
    "JET06": "round_amount",
    "JET07": "same_preparer_approver",
    "JET08": "split_payment",
    "JET09": "suspicious_keyword",
    "JET10": "extreme_amount",
}

EVALUATION_COLUMNS = ["test_id", "anomaly_type", "injected", "flagged", "true_positives",
                      "false_positives", "false_negatives", "precision", "recall"]


def ratio(part, whole):
    """part divided by whole, or None when whole is zero."""
    if whole == 0:
        return None
    return round(part / whole, 3)


def score(test_id, anomaly_type, flagged, injected):
    """Count hits and misses for one test. Both arguments are sets of voucher numbers."""
    caught = flagged & injected
    return {
        "test_id": test_id,
        "anomaly_type": anomaly_type or "n/a",
        "injected": len(injected),
        "flagged": len(flagged),
        "true_positives": len(caught),
        "false_positives": len(flagged - injected),
        "false_negatives": len(injected - flagged),
        "precision": ratio(len(caught), len(flagged)),
        "recall": ratio(len(caught), len(injected)),
    }


def injected_vouchers(ground_truth, anomaly_type):
    """Vouchers carrying one anomaly type, or none when the test has no counterpart."""
    if anomaly_type is None:
        return set()
    return set(ground_truth.loc[ground_truth["anomaly_type"] == anomaly_type, "voucher_no"])


def evaluate(flags, ground_truth):
    """One row per test plus an overall row, comparing flags with the ground truth.

    A test catches a voucher when it flags any line of it, because the
    ground truth and the reviewer both work at voucher level.
    """
    flags = flags.reindex(columns=jet.OUTPUT_COLUMNS)  # also works when nothing was flagged
    rows = []
    for test_id, anomaly_type in TEST_ANOMALY.items():
        flagged = set(flags.loc[flags["test_id"] == test_id, "voucher_no"])
        rows.append(score(test_id, anomaly_type, flagged,
                          injected_vouchers(ground_truth, anomaly_type)))

    rows.append(score("overall", "any", set(flags["voucher_no"]),
                      set(ground_truth["voucher_no"])))
    return pd.DataFrame(rows, columns=EVALUATION_COLUMNS)


def compare_outlier_scales(gl, ground_truth, config):
    """Score JET10 twice, on the raw amounts and on log(amount).

    Shows what the log scale is worth. Accounting amounts are right-skewed:
    many small lines and a few large ones. On the raw scale a large but
    normal line is often far enough from the median to be flagged.
    """
    injected = injected_vouchers(ground_truth, TEST_ANOMALY["JET10"])
    rows = []
    for log_scale in [False, True]:
        variant = copy.deepcopy(config)
        variant["jet"]["JET10_outliers"]["log_scale"] = log_scale
        flags = jet.jet10_outliers(gl, variant)
        row = score("JET10 log scale" if log_scale else "JET10 raw scale",
                    TEST_ANOMALY["JET10"], set(flags["voucher_no"]), injected)
        rows.append(row)
    return pd.DataFrame(rows, columns=EVALUATION_COLUMNS)
