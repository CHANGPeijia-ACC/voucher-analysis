from pathlib import Path

import pytest

from gl_builder import make_bank, make_gl, voucher
from voucher_analysis import reconcile as rec
from voucher_analysis.loaders import load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    return load_config(REPO_ROOT / "config.yaml")


def payment(voucher_no, amount, day, supplier="Supplier A"):
    """A book payment: debit payables, credit bank."""
    return voucher(voucher_no, amount, debit_account="2202", credit_account="1002",
                   supplier=supplier, posting_date=day)


def receipt(voucher_no, amount, day):
    """A book receipt: debit bank, credit receivables."""
    return voucher(voucher_no, amount, debit_account="1002", credit_account="1122",
                   supplier=None, posting_date=day)


def test_book_items_are_one_row_per_voucher_with_net_amount(config):
    gl = make_gl(
        *payment("JV000001", 1000, "2025-03-03"),
        *receipt("JV000002", 2500, "2025-03-04"),
    )
    items = rec.book_bank_items(gl, config).set_index("voucher_no")

    assert len(items) == 2
    assert items.loc["JV000001", "amount"] == -1000.00      # payment is negative
    assert items.loc["JV000002", "amount"] == 2500.00       # receipt is positive
    assert items.loc["JV000001", "cents"] == -100000
    assert items.loc["JV000001", "counterparty"] == "Supplier A"
    assert items.loc["JV000002", "counterparty"] == ""


def test_book_items_skip_vouchers_that_do_not_touch_the_bank(config):
    gl = make_gl(*voucher("JV000001", 500, debit_account="6601", credit_account="2202"))
    assert rec.book_bank_items(gl, config).empty


def test_pass1_matches_same_amount_within_window(config):
    book = rec.book_bank_items(make_gl(*payment("JV000001", 1000, "2025-03-03")), config)
    bank = rec.bank_items(make_bank({"bank_date": "2025-03-05", "amount": -1000}))
    matches, used_refs, used_vouchers = rec.match_one_to_one(book, bank, 3)

    assert len(matches) == 1
    assert matches[0]["match_type"] == "one_to_one"
    assert matches[0]["difference"] == 0.0
    assert used_refs == {"BR000001"}
    assert used_vouchers == {"JV000001"}


def test_pass1_ignores_dates_outside_the_window(config):
    book = rec.book_bank_items(make_gl(*payment("JV000001", 1000, "2025-03-03")), config)
    bank = rec.bank_items(make_bank({"bank_date": "2025-03-07", "amount": -1000}))
    matches, _, _ = rec.match_one_to_one(book, bank, 3)
    assert matches == []


def test_pass1_takes_the_closest_date(config):
    book = rec.book_bank_items(make_gl(*payment("JV000001", 1000, "2025-03-03")), config)
    bank = rec.bank_items(make_bank(
        {"bank_date": "2025-03-06", "amount": -1000, "bank_ref": "BR000001"},
        {"bank_date": "2025-03-04", "amount": -1000, "bank_ref": "BR000002"},
    ))
    matches, _, _ = rec.match_one_to_one(book, bank, 3)
    assert matches[0]["bank_refs"] == "BR000002"


def test_pass1_never_uses_a_bank_line_twice(config):
    book = rec.book_bank_items(make_gl(
        *payment("JV000001", 1000, "2025-03-03"),
        *payment("JV000002", 1000, "2025-03-04"),
    ), config)
    bank = rec.bank_items(make_bank({"bank_date": "2025-03-04", "amount": -1000}))
    matches, _, used_vouchers = rec.match_one_to_one(book, bank, 3)

    assert len(matches) == 1
    assert used_vouchers == {"JV000001"}  # the older voucher is matched first
