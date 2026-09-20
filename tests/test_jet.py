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
