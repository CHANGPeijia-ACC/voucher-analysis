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


# ============================================================
# JET00 Input validation
# ============================================================

REQUIRED_FIELDS = ["voucher_no", "line_no", "posting_date", "entry_time",
                   "account_code", "prepared_by", "approved_by"]


def jet00_validation(gl, config):
    """JET00 Input validation: data problems that make the other tests unreliable.

    Checks for missing required fields, a wrong voucher number format, lines
    with no amount or with both debit and credit, negative amounts, and the
    same voucher_no and line_no used twice. One line can have several problems.
    """
    pattern = settings(config, "JET00_validation")["voucher_no_pattern"]
    has_debit = gl["debit"].notna()
    has_credit = gl["credit"].notna()
    good_format = gl["voucher_no"].fillna("").str.fullmatch(pattern)

    checks = [(gl[field].isna(), f"missing {field}") for field in REQUIRED_FIELDS]
    checks += [
        (gl["voucher_no"].notna() & ~good_format, "wrong voucher number format"),
        (~has_debit & ~has_credit, "no debit or credit amount"),
        (has_debit & has_credit, "both debit and credit on one line"),
        ((gl["debit"] < 0) | (gl["credit"] < 0), "negative amount"),
        (gl["voucher_no"].notna() & gl.duplicated(["voucher_no", "line_no"], keep=False),
         "voucher_no and line_no used twice"),
    ]

    reasons = pd.Series("", index=gl.index)
    for mask, text in checks:
        mask = mask.fillna(False).astype(bool)
        reasons[mask] = reasons[mask] + "; " + text
    flagged = reasons != ""
    return flag_lines(gl[flagged], "JET00", reasons[flagged].str.removeprefix("; "))


# ============================================================
# JET01 Unbalanced vouchers
# ============================================================

def jet01_unbalanced(gl, config):
    """JET01 Unbalanced vouchers: total debit is not equal to total credit.

    In double entry every voucher must balance. A difference points to a
    keying error or to an entry that went around system controls.
    """
    tolerance = settings(config, "JET01_unbalanced")["tolerance"]
    totals = gl.groupby("voucher_no")[["debit", "credit"]].sum()
    difference = totals["debit"] - totals["credit"]
    unbalanced = totals[difference.abs() >= tolerance]

    texts = pd.Series(
        [f"Debit {money_text(d)} vs credit {money_text(c)}"
         for d, c in zip(unbalanced["debit"], unbalanced["credit"])],
        index=unbalanced.index, dtype=object)
    mask = gl["voucher_no"].isin(unbalanced.index)
    return flag_whole_vouchers(gl, mask, "JET01", gl["voucher_no"].map(texts))


# ============================================================
# JET02 Possible duplicates
# ============================================================

def jet02_duplicates(gl, config):
    """JET02 Possible duplicates: same account, supplier, side and amount in
    different vouchers posted within N days of each other.

    A duplicated invoice can lead to paying a supplier twice. The side
    (debit or credit) is part of the match, so an accrual and its reversal
    are not reported as duplicates.
    """
    window = settings(config, "JET02_duplicates")["window_days"]
    lines = gl[gl["supplier"].notna() & gl["posting_date"].notna()].copy()
    lines["side"] = np.where(lines["debit"].notna(), "debit", "credit")
    lines["cents"] = (line_amount(lines) * 100).round()  # compare whole cents, not floats
    keys = ["account_code", "supplier", "side", "cents"]
    lines = lines[lines["cents"].notna() & lines.duplicated(keys, keep=False)]

    reasons = {}
    for _, group in lines.sort_values("posting_date").groupby(keys):
        rows = list(group.itertuples())
        for i, first in enumerate(rows):
            for second in rows[i + 1:]:
                days = (second.posting_date - first.posting_date).days
                if days > window:
                    break
                if first.voucher_no != second.voucher_no:
                    text = "Same account, supplier and amount as {}, {} days apart"
                    reasons.setdefault(first.Index, text.format(second.voucher_no, days))
                    reasons.setdefault(second.Index, text.format(first.voucher_no, days))

    reasons = pd.Series(reasons, dtype=object)
    return flag_lines(gl.loc[reasons.index], "JET02", reasons)
