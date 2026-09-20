"""Build small hand-made general ledgers and bank statements for tests."""

import pandas as pd

from voucher_analysis.loaders import BANK_COLUMNS, GL_COLUMNS, prepare_bank, prepare_gl

DEFAULT_LINE = {
    "voucher_no": "JV000001",
    "line_no": 1,
    "posting_date": "2025-03-03",          # a Monday
    "entry_time": "2025-03-03 10:00:00",
    "account_code": "6601",
    "account_name": "Travel expense",
    "debit": 100.00,
    "credit": None,
    "department": "Admin",
    "supplier": "Supplier A",
    "prepared_by": "prep01",
    "approved_by": "appr01",
    "description": "Business travel",
}


def make_gl(*rows):
    """Build a ledger from partial rows. Fields not given use DEFAULT_LINE.

    Values are turned into text first and parsed with prepare_gl, the same
    way a CSV file is read.
    """
    full_rows = []
    for row in rows:
        values = {**DEFAULT_LINE, **row}
        full_rows.append({col: "" if values[col] is None else str(values[col])
                          for col in GL_COLUMNS})
    return prepare_gl(pd.DataFrame(full_rows, columns=GL_COLUMNS))


def voucher(voucher_no, amount, debit_account="6601", credit_account="1002", **fields):
    """Two balanced lines: debit one account, credit another. Extra fields apply to both."""
    return [
        {"voucher_no": voucher_no, "line_no": 1, "account_code": debit_account,
         "debit": amount, "credit": None, **fields},
        {"voucher_no": voucher_no, "line_no": 2, "account_code": credit_account,
         "debit": None, "credit": amount, **fields},
    ]


DEFAULT_BANK_LINE = {
    "bank_date": "2025-03-03",
    "amount": -100.00,
    "counterparty": "Supplier A",
    "bank_ref": None,                      # numbered BR000001, BR000002, ... when left out
}


def make_bank(*rows):
    """Build a bank statement from partial rows, parsed like a CSV file."""
    full_rows = []
    for number, row in enumerate(rows, start=1):
        values = {**DEFAULT_BANK_LINE, "bank_ref": f"BR{number:06d}", **row}
        full_rows.append({col: "" if values[col] is None else str(values[col])
                          for col in BANK_COLUMNS})
    return prepare_bank(pd.DataFrame(full_rows, columns=BANK_COLUMNS))
