from datetime import date

import pandas as pd
import pytest

from voucher_analysis import generate as g
from voucher_analysis.loaders import BANK_COLUMNS, GL_COLUMNS, read_bank, read_gl


@pytest.fixture(scope="module")
def tables():
    """Generate once and share the result between tests, since it takes a few seconds."""
    return g.generate_all()


def voucher_totals(gl):
    return gl.groupby("voucher_no")[["debit", "credit"]].sum()


def test_same_seed_gives_same_data(tables):
    again = g.generate_all()
    for name in tables:
        pd.testing.assert_frame_equal(tables[name], again[name])


def test_gl_has_expected_columns_and_size(tables):
    gl = tables["gl"]
    assert list(gl.columns) == GL_COLUMNS
    assert 18000 <= len(gl) <= 22000


def test_every_anomaly_type_is_injected(tables):
    types = set(tables["ground_truth"]["anomaly_type"])
    assert len(types) == 10


def test_ground_truth_vouchers_exist_in_gl(tables):
    assert set(tables["ground_truth"]["voucher_no"]) <= set(tables["gl"]["voucher_no"])


def test_only_injected_vouchers_are_unbalanced(tables):
    totals = voucher_totals(tables["gl"])
    unbalanced = set(totals.index[(totals["debit"] - totals["credit"]).abs() > 0.005])
    truth = tables["ground_truth"]
    expected = set(truth.loc[truth["anomaly_type"] == "unbalanced", "voucher_no"])
    assert unbalanced == expected


def test_normal_postings_are_on_working_days(tables):
    gl, truth = tables["gl"], tables["ground_truth"]
    injected = set(truth.loc[truth["anomaly_type"] == "weekend_holiday", "voucher_no"])
    hols = g.make_holidays(g.YEAR)
    normal_dates = gl.loc[~gl["voucher_no"].isin(injected), "posting_date"].unique()
    assert all(g.is_working_day(date.fromisoformat(d), hols) for d in normal_dates)


def test_normal_preparer_differs_from_approver(tables):
    gl, truth = tables["gl"], tables["ground_truth"]
    same = set(gl.loc[gl["prepared_by"] == gl["approved_by"], "voucher_no"])
    expected = set(truth.loc[truth["anomaly_type"] == "same_preparer_approver", "voucher_no"])
    assert same == expected


def test_bank_statement_has_no_voucher_numbers(tables):
    assert list(tables["bank"].columns) == BANK_COLUMNS


def test_bank_truth_refers_to_real_items(tables):
    truth = tables["bank_ground_truth"]
    refs = set(truth.loc[truth["bank_ref"] != "", "bank_ref"])
    vouchers = set(truth.loc[truth["voucher_no"] != "", "voucher_no"])
    assert refs == set(tables["bank"]["bank_ref"])
    assert vouchers <= set(tables["gl"]["voucher_no"])


def test_book_and_bank_differ_only_by_reconciling_items(tables):
    """Bank total minus book total must equal bank-only items, less book-only
    items, less the transfer fees deducted in amount differences."""
    gl, bank, truth = tables["gl"], tables["bank"], tables["bank_ground_truth"]
    bank_lines = gl[gl["account_code"] == "1002"]
    book_total = bank_lines["debit"].sum() - bank_lines["credit"].sum()

    book_only = truth.loc[truth["bank_ref"] == "", "voucher_no"]
    book_only_lines = bank_lines[bank_lines["voucher_no"].isin(book_only)]
    book_only_total = book_only_lines["debit"].sum() - book_only_lines["credit"].sum()

    bank_only_refs = truth.loc[truth["voucher_no"] == "", "bank_ref"]
    bank_only_total = bank.loc[bank["bank_ref"].isin(bank_only_refs), "amount"].sum()

    fees = (truth["match_type"] == "amount_difference").sum() * g.TRANSFER_FEE
    expected = bank_only_total - book_only_total - fees
    assert bank["amount"].sum() - book_total == pytest.approx(expected, abs=0.01)


def test_written_files_can_be_loaded(tables, tmp_path):
    g.write_outputs(tables, tmp_path)
    assert len(read_gl(tmp_path / "gl.csv")) == len(tables["gl"])
    assert len(read_bank(tmp_path / "bank.csv")) == len(tables["bank"])
