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
