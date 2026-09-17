from pathlib import Path

import pandas as pd
import pytest

from voucher_analysis.loaders import load_config, read_bank, read_gl

REPO_ROOT = Path(__file__).resolve().parents[1]

GL_HEADER = (
    "voucher_no,line_no,posting_date,entry_time,account_code,account_name,"
    "debit,credit,department,supplier,prepared_by,approved_by,description"
)


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_read_gl_parses_types(tmp_path):
    path = write(tmp_path, "gl.csv", GL_HEADER + "\n"
                 "JV000001,1,2025-01-06,2025-01-06 10:15:00,6601,"
                 "Travel expense 差旅费,1200.50,,Sales,,user01,user02,Trip\n"
                 "JV000001,2,2025-01-06,2025-01-06 10:15:00,1002,"
                 "Bank deposits 银行存款,,1200.50,Sales,,user01,user02,Trip\n")
    gl = read_gl(path)

    assert len(gl) == 2
    assert gl.loc[0, "debit"] == 1200.50
    assert pd.isna(gl.loc[0, "credit"])
    assert str(gl["posting_date"].dtype).startswith("datetime64")
    assert gl.loc[1, "account_code"] == "1002"


def test_read_gl_keeps_bad_values_as_missing(tmp_path):
    path = write(tmp_path, "gl.csv", GL_HEADER + "\n"
                 "JV000002,1,not-a-date,,6601,Travel,abc,,,,u1,u2,\n")
    gl = read_gl(path)

    assert len(gl) == 1
    assert gl["posting_date"].isna().all()
    assert gl["debit"].isna().all()


def test_read_gl_missing_column_raises(tmp_path):
    path = write(tmp_path, "gl.csv", "voucher_no,line_no\nJV000001,1\n")
    with pytest.raises(ValueError, match="posting_date"):
        read_gl(path)


def test_read_bank_parses_types(tmp_path):
    path = write(tmp_path, "bank.csv",
                 "bank_date,amount,counterparty,bank_ref\n"
                 "2025-01-07,-1200.50,Shanghai Travel Co,BR0001\n")
    bank = read_bank(path)

    assert bank.loc[0, "amount"] == -1200.50
    assert str(bank["bank_date"].dtype).startswith("datetime64")


def test_read_bank_missing_column_raises(tmp_path):
    path = write(tmp_path, "bank.csv", "bank_date,amount\n2025-01-07,10\n")
    with pytest.raises(ValueError, match="counterparty"):
        read_bank(path)


def test_load_repo_config():
    config = load_config(REPO_ROOT / "config.yaml")

    assert config["jet"]["JET03_weekend_holiday"]["country"] == "CN"
    assert config["reconcile"]["date_window_days"] == 3
    assert "调整" in config["jet"]["JET09_keywords"]["keywords"]


def test_load_config_missing_section_raises(tmp_path):
    path = write(tmp_path, "config.yaml", "jet: {}\n")
    with pytest.raises(ValueError, match="reconcile"):
        load_config(path)
