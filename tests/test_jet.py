from pathlib import Path

import pytest

from gl_builder import make_gl, voucher
from voucher_analysis import jet
from voucher_analysis.loaders import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_config(REPO_ROOT / "config.yaml")


def flagged_vouchers(result):
    return set(result["voucher_no"])


def test_flag_whole_vouchers_flags_every_line(config):
    gl = make_gl(*voucher("JV000001", 100), *voucher("JV000002", 200))
    mask = gl["account_code"] == "6601"
    mask[gl["voucher_no"] == "JV000002"] = False
    result = jet.flag_whole_vouchers(gl, mask, "TEST", gl["account_code"])

    assert list(result.columns) == jet.OUTPUT_COLUMNS
    assert list(result["line_no"]) == [1, 2]
    assert set(result["reason"]) == {"6601"}


def test_jet00_accepts_clean_data(config):
    gl = make_gl(*voucher("JV000001", 100))
    assert jet.jet00_validation(gl, config).empty


def test_jet00_reports_each_problem(config):
    gl = make_gl(
        {"voucher_no": "X12"},
        {"voucher_no": "JV000002", "prepared_by": None},
        {"voucher_no": "JV000003", "debit": 50, "credit": 50},
        {"voucher_no": "JV000004", "debit": None, "credit": None},
        {"voucher_no": "JV000005", "debit": -10},
        {"voucher_no": "JV000006"},
        {"voucher_no": "JV000006"},
    )
    reasons = jet.jet00_validation(gl, config).groupby("voucher_no")["reason"].first()

    assert reasons["X12"] == "wrong voucher number format"
    assert reasons["JV000002"] == "missing prepared_by"
    assert reasons["JV000003"] == "both debit and credit on one line"
    assert reasons["JV000004"] == "no debit or credit amount"
    assert reasons["JV000005"] == "negative amount"
    assert reasons["JV000006"] == "voucher_no and line_no used twice"


def test_jet01_flags_only_unbalanced_vouchers(config):
    gl = make_gl(
        *voucher("JV000001", 100),
        {"voucher_no": "JV000002", "line_no": 1, "debit": 100.00},
        {"voucher_no": "JV000002", "line_no": 2, "debit": None, "credit": 90.00},
    )
    result = jet.jet01_unbalanced(gl, config)

    assert flagged_vouchers(result) == {"JV000002"}
    assert len(result) == 2
    assert result["reason"].iloc[0] == "Debit 100.00 vs credit 90.00"


def test_jet01_catches_one_cent(config):
    gl = make_gl(
        {"voucher_no": "JV000001", "line_no": 1, "debit": 100.01},
        {"voucher_no": "JV000001", "line_no": 2, "debit": None, "credit": 100.00},
    )
    assert flagged_vouchers(jet.jet01_unbalanced(gl, config)) == {"JV000001"}


def test_jet02_flags_same_invoice_within_window(config):
    gl = make_gl(
        *voucher("JV000001", 5000, posting_date="2025-03-03"),
        *voucher("JV000002", 5000, posting_date="2025-03-06"),
        *voucher("JV000003", 7000, posting_date="2025-03-03"),
        *voucher("JV000004", 7000, posting_date="2025-03-17"),
    )
    result = jet.jet02_duplicates(gl, config)

    assert flagged_vouchers(result) == {"JV000001", "JV000002"}
    assert "3 days apart" in result["reason"].iloc[0]


def test_jet02_ignores_reversal_on_opposite_side(config):
    gl = make_gl(
        *voucher("JV000001", 5000, debit_account="6605", credit_account="2241"),
        *voucher("JV000002", 5000, debit_account="2241", credit_account="6605",
                 posting_date="2025-03-04"),
    )
    assert jet.jet02_duplicates(gl, config).empty


def test_jet03_flags_weekend_and_holiday(config):
    gl = make_gl(
        *voucher("JV000001", 100, posting_date="2025-03-03"),
        *voucher("JV000002", 100, posting_date="2025-03-08"),
        *voucher("JV000003", 100, posting_date="2025-10-01"),
    )
    reasons = jet.jet03_weekend_holiday(gl, config).groupby("voucher_no")["reason"].first()

    assert set(reasons.index) == {"JV000002", "JV000003"}
    assert reasons["JV000002"] == "Posted on Saturday 2025-03-08"
    assert reasons["JV000003"].startswith("Posted on public holiday 2025-10-01")


def test_jet04_uses_working_hours_boundaries(config):
    times = {"JV000001": "07:59", "JV000002": "08:00", "JV000003": "19:59",
             "JV000004": "20:00", "JV000005": "23:30"}
    rows = []
    for voucher_no, clock in times.items():
        rows += voucher(voucher_no, 100, entry_time=f"2025-03-03 {clock}:00")
    result = jet.jet04_outside_hours(make_gl(*rows), config)

    assert flagged_vouchers(result) == {"JV000001", "JV000004", "JV000005"}


def test_jet05_flags_entry_after_close_day(config):
    gl = make_gl(
        *voucher("JV000001", 100, posting_date="2025-03-10", entry_time="2025-04-05 10:00:00"),
        *voucher("JV000002", 100, posting_date="2025-03-10", entry_time="2025-04-06 10:00:00"),
    )
    result = jet.jet05_after_close(gl, config)

    assert flagged_vouchers(result) == {"JV000002"}
    assert result["reason"].iloc[0] == (
        "Entered 2025-04-06, after close deadline 2025-04-05 for period 2025-03")


def test_month_end_summary_shares(config):
    gl = make_gl(
        *voucher("JV000001", 100, posting_date="2025-03-10"),
        *voucher("JV000002", 300, posting_date="2025-03-30"),
    )
    row = jet.month_end_summary(gl, config).iloc[0]

    assert row["period"] == "2025-03"
    assert row["voucher_share"] == 0.5
    assert row["debit_share"] == 0.75


def test_jet06_flags_round_amounts_above_floor(config):
    gl = make_gl(
        *voucher("JV000001", 60000),
        *voucher("JV000002", 12000),
        *voucher("JV000003", 9000),
        *voucher("JV000004", 10000.50),
    )
    reasons = jet.jet06_round_amounts(gl, config).groupby("voucher_no")["reason"].first()

    assert set(reasons.index) == {"JV000001", "JV000002"}
    assert reasons["JV000001"] == "Round amount, multiple of 10,000"
    assert reasons["JV000002"] == "Round amount, multiple of 1,000"
