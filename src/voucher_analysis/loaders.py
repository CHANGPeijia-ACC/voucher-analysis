"""Read input files and check that they have the expected columns.

Loading only checks the file layout. It does not drop bad rows: finding
bad values is the job of JET00, so the rows must reach the tests.
"""

import pandas as pd
import yaml

ENCODING = "utf-8-sig"  # handles Chinese text and files saved by Excel

GL_COLUMNS = [
    "voucher_no", "line_no", "posting_date", "entry_time",
    "account_code", "account_name", "debit", "credit",
    "department", "supplier", "prepared_by", "approved_by", "description",
]

BANK_COLUMNS = ["bank_date", "amount", "counterparty", "bank_ref"]

CONFIG_SECTIONS = ["jet", "reconcile"]


def check_columns(df, required, source):
    """Raise ValueError if any required column is missing.

    A clear message here is better than a KeyError deep inside a test.
    """
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing columns: {', '.join(missing)}")


def read_gl(path):
    """Read the general ledger CSV into a DataFrame with proper types.

    Codes are kept as text so leading zeros survive. Values that cannot be
    parsed become empty (NaN or NaT) so JET00 can report them.
    """
    df = pd.read_csv(path, encoding=ENCODING, dtype=str, keep_default_na=False)
    check_columns(df, GL_COLUMNS, "General ledger")

    df = df.replace("", pd.NA)
    df["line_no"] = pd.to_numeric(df["line_no"], errors="coerce").astype("Int64")
    df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df["debit"] = pd.to_numeric(df["debit"], errors="coerce")
    df["credit"] = pd.to_numeric(df["credit"], errors="coerce")
    return df


def read_bank(path):
    """Read the bank statement CSV into a DataFrame with proper types."""
    df = pd.read_csv(path, encoding=ENCODING, dtype=str, keep_default_na=False)
    check_columns(df, BANK_COLUMNS, "Bank statement")

    df = df.replace("", pd.NA)
    df["bank_date"] = pd.to_datetime(df["bank_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df


def load_config(path):
    """Read config.yaml and check that the main sections are present."""
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"Config file {path} is empty or not a mapping")

    missing = [s for s in CONFIG_SECTIONS if s not in config]
    if missing:
        raise ValueError(f"Config is missing sections: {', '.join(missing)}")
    return config
