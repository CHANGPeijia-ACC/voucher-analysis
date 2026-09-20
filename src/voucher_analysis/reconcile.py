"""Match bank statement lines to book entries and build a bank reconciliation.

Amounts are compared as whole cents, so no comparison depends on float
equality. Receipts are positive and payments are negative on both sides.

The book only knows the counterparty for payments (the supplier column), so
counterparty is used for payments only.
"""

from itertools import combinations

import pandas as pd

BOOK_COLUMNS = ["voucher_no", "posting_date", "amount", "cents", "counterparty"]

MATCH_COLUMNS = ["match_id", "match_type", "bank_refs", "voucher_nos",
                 "bank_date", "posting_date", "bank_amount", "book_amount", "difference"]

UNMATCHED_COLUMNS = ["item_type", "bank_ref", "voucher_no", "date", "amount", "counterparty"]


def settings(config):
    """The reconcile section of the config."""
    return config["reconcile"]


def to_cents(amounts):
    """Amounts as whole cents, so that equal amounts compare exactly."""
    return (amounts * 100).round().astype("int64")


def book_bank_items(gl, config):
    """One row per voucher that touches the bank account.

    The amount is the net effect of the voucher on the bank account:
    receipts positive, payments negative. This is what the bank would see,
    even when the voucher has several other lines.
    """
    account = str(settings(config)["bank_account"])
    lines = gl[(gl["account_code"] == account) & gl["posting_date"].notna()].copy()
    lines["amount"] = lines["debit"].fillna(0) - lines["credit"].fillna(0)

    items = lines.groupby("voucher_no").agg(
        posting_date=("posting_date", "min"),
        amount=("amount", "sum"),
        counterparty=("supplier", "first")).reset_index()
    items["amount"] = items["amount"].round(2)
    items["cents"] = to_cents(items["amount"])
    items["counterparty"] = items["counterparty"].fillna("")
    return items[items["cents"] != 0][BOOK_COLUMNS].reset_index(drop=True)


def bank_items(bank):
    """Bank statement lines with their amounts in whole cents."""
    items = bank.copy()
    items["cents"] = to_cents(items["amount"])
    return items


def days_apart(bank_row, book_row):
    """Calendar days between a bank date and a posting date, always positive."""
    return abs((bank_row.bank_date - book_row.posting_date).days)


def build_match(match_type, bank_rows, book_rows):
    """One row describing a match between bank lines and book vouchers."""
    bank_amount = round(sum(row.amount for row in bank_rows), 2)
    book_amount = round(sum(row.amount for row in book_rows), 2)
    return {
        "match_type": match_type,
        "bank_refs": ", ".join(row.bank_ref for row in bank_rows),
        "voucher_nos": ", ".join(row.voucher_no for row in book_rows),
        "bank_date": min(row.bank_date for row in bank_rows),
        "posting_date": min(row.posting_date for row in book_rows),
        "bank_amount": bank_amount,
        "book_amount": book_amount,
        "difference": round(bank_amount - book_amount, 2),
    }


# ============================================================
# Pass 1: one to one
# ============================================================

def match_one_to_one(book, bank, window_days):
    """Match a voucher to a bank line with the same amount within the date window.

    Vouchers are handled oldest first. When several bank lines have the same
    amount, the one with the closest date is taken. Nothing is matched twice.
    Returns the matches, the bank refs used and the voucher numbers used.
    """
    by_amount = {}
    for row in bank.itertuples():
        by_amount.setdefault(row.cents, []).append(row)

    matches = []
    used_refs = set()
    used_vouchers = set()
    for item in book.sort_values(["posting_date", "voucher_no"]).itertuples():
        candidates = [row for row in by_amount.get(item.cents, [])
                      if row.bank_ref not in used_refs
                      and days_apart(row, item) <= window_days]
        if not candidates:
            continue
        best = min(candidates, key=lambda row: (days_apart(row, item), row.bank_ref))
        used_refs.add(best.bank_ref)
        used_vouchers.add(item.voucher_no)
        matches.append(build_match("one_to_one", [best], [item]))
    return matches, used_refs, used_vouchers


# ============================================================
# Pass 2: one to many
# ============================================================

def match_one_to_many(book, bank, window_days, max_group_size, max_candidates=12):
    """Match one bank line to several vouchers for the same counterparty.

    A bank transfer can settle several invoices at once. The function tries
    combinations of two up to max_group_size vouchers whose amounts add up
    to the bank amount exactly. Only the max_candidates vouchers closest in
    date are considered, to keep the number of combinations small.
    """
    matches = []
    used_refs = set()
    used_vouchers = set()

    for bank_row in bank.sort_values(["bank_date", "bank_ref"]).itertuples():
        candidates = [item for item in book.itertuples()
                      if item.voucher_no not in used_vouchers
                      and item.counterparty == bank_row.counterparty
                      and days_apart(bank_row, item) <= window_days]
        candidates.sort(key=lambda item: (days_apart(bank_row, item), item.voucher_no))
        candidates = candidates[:max_candidates]

        group = find_group(bank_row.cents, candidates, max_group_size)
        if group:
            used_refs.add(bank_row.bank_ref)
            used_vouchers.update(item.voucher_no for item in group)
            matches.append(build_match("one_to_many", [bank_row], list(group)))
    return matches, used_refs, used_vouchers


def find_group(target_cents, candidates, max_group_size):
    """The first combination of 2 or more vouchers that adds up to the target."""
    for size in range(2, max_group_size + 1):
        for group in combinations(candidates, size):
            if sum(item.cents for item in group) == target_cents:
                return group
    return None


# ============================================================
# Pass 3: classify what is left
# ============================================================

def match_amount_differences(book, bank, window_days, max_relative_difference):
    """Pair items that look like the same payment recorded with different amounts.

    Same counterparty, date within the window, and a difference small
    against the book amount, for example a transfer fee the bank deducted.
    These are not timing differences: somebody has to find out which side is
    right, so they are reported as matches with a difference.
    """
    matches = []
    used_refs = set()
    used_vouchers = set()

    for bank_row in bank.sort_values(["bank_date", "bank_ref"]).itertuples():
        candidates = [item for item in book.itertuples()
                      if item.voucher_no not in used_vouchers
                      and item.counterparty != ""
                      and item.counterparty == bank_row.counterparty
                      and days_apart(bank_row, item) <= window_days
                      and abs(bank_row.cents - item.cents)
                      <= abs(item.cents) * max_relative_difference]
        if not candidates:
            continue
        best = min(candidates, key=lambda item: (abs(bank_row.cents - item.cents),
                                                 item.voucher_no))
        used_refs.add(bank_row.bank_ref)
        used_vouchers.add(best.voucher_no)
        matches.append(build_match("amount_difference", [bank_row], [best]))
    return matches, used_refs, used_vouchers


def classify_unmatched(book, bank):
    """Label the items that no pass could match.

    A book receipt the bank has not credited yet is a deposit in transit
    (在途存款). A book payment the bank has not taken yet is an outstanding
    payment (未兑付付款). A bank line with nothing in the books is bank only
    (银行单边), for example a bank charge or a direct debit.
    """
    rows = []
    for item in book.itertuples():
        rows.append({
            "item_type": "deposit_in_transit" if item.amount > 0 else "outstanding_payment",
            "bank_ref": "", "voucher_no": item.voucher_no, "date": item.posting_date,
            "amount": item.amount, "counterparty": item.counterparty,
        })
    for row in bank.itertuples():
        rows.append({
            "item_type": "bank_only", "bank_ref": row.bank_ref, "voucher_no": "",
            "date": row.bank_date, "amount": row.amount, "counterparty": row.counterparty,
        })
    return pd.DataFrame(rows, columns=UNMATCHED_COLUMNS)
