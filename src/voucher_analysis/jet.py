"""Journal entry tests (JET).

Each test takes the general ledger and the config, and returns the lines it
flags with the same five columns: test_id, voucher_no, line_no, amount and
reason. Results from different tests can then be stacked into one table.

Some tests judge a whole voucher, for example whether it balances. Those
flag every line of the voucher, so the reviewer sees the complete entry.
"""

import holidays
import numpy as np
import pandas as pd

OUTPUT_COLUMNS = ["test_id", "voucher_no", "line_no", "amount", "reason"]


# ============================================================
# Helpers
# ============================================================

def settings(config, name):
    """Thresholds for one test, from the jet section of the config."""
    return config["jet"][name]


def money_text(value):
    """Format an amount as 12,345.67 for use in reasons."""
    return f"{value:,.2f}"


def line_amount(gl):
    """Amount on each line: the debit if there is one, otherwise the credit."""
    return gl["debit"].fillna(gl["credit"])


def flag_lines(lines, test_id, reasons):
    """Build the standard output table for the given GL lines.

    `reasons` is either one text for all lines or a Series with one text
    per line, indexed like `lines`.
    """
    if isinstance(reasons, pd.Series):
        reasons = reasons.reindex(lines.index)
    table = pd.DataFrame({
        "test_id": test_id,
        "voucher_no": lines["voucher_no"],
        "line_no": lines["line_no"],
        "amount": line_amount(lines),
        "reason": reasons,
    }, columns=OUTPUT_COLUMNS)
    return table.reset_index(drop=True)


def flag_whole_vouchers(gl, mask, test_id, reasons):
    """Flag every line of each voucher where at least one line matches `mask`.

    All lines of a voucher show the reason of its first matching line.
    """
    mask = mask.fillna(False).astype(bool)
    first_reason = reasons[mask].groupby(gl.loc[mask, "voucher_no"]).first()
    lines = gl[gl["voucher_no"].isin(first_reason.index)]
    return flag_lines(lines, test_id, lines["voucher_no"].map(first_reason))
