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


# ============================================================
# JET03 Weekend and public holiday postings
# ============================================================

def jet03_weekend_holiday(gl, config):
    """JET03 Postings dated on a weekend or a public holiday.

    Few staff work on these days, so entries made then get less review.
    The holidays package does not know weekend days that are made working
    days to make up for a holiday, so postings on those days are flagged too.
    """
    country = settings(config, "JET03_weekend_holiday")["country"]
    dates = gl["posting_date"]
    years = sorted(int(year) for year in dates.dt.year.dropna().unique())
    public_holidays = holidays.country_holidays(country, years=years, language="en_US")

    holiday_name = dates.map(lambda d: public_holidays.get(d.date()) if pd.notna(d) else None)
    is_holiday = holiday_name.notna()
    is_weekend = dates.dt.dayofweek >= 5

    reasons = pd.Series("", index=gl.index)
    day_text = dates.dt.strftime("%Y-%m-%d")
    reasons[is_weekend] = "Posted on " + dates[is_weekend].dt.day_name() + " " + day_text[is_weekend]
    reasons[is_holiday] = ("Posted on public holiday " + day_text[is_holiday]
                           + " (" + holiday_name[is_holiday] + ")")
    return flag_whole_vouchers(gl, is_weekend | is_holiday, "JET03", reasons)


# ============================================================
# JET04 Entries outside working hours
# ============================================================

def minutes_of_day(clock_text):
    """Turn "08:30" into 510, the minutes since midnight."""
    hours, minutes = clock_text.split(":")
    return int(hours) * 60 + int(minutes)


def jet04_outside_hours(gl, config):
    """JET04 Entries made before work_start or at or after work_end.

    Late-night entries are less likely to be supervised and can point to
    shared passwords or entries made to avoid review.
    """
    s = settings(config, "JET04_outside_hours")
    start = minutes_of_day(s["work_start"])
    end = minutes_of_day(s["work_end"])

    entry = gl["entry_time"]
    minutes = entry.dt.hour * 60 + entry.dt.minute
    mask = (minutes < start) | (minutes >= end)
    reasons = ("Entered at " + entry.dt.strftime("%H:%M")
               + f", outside {s['work_start']}-{s['work_end']}")
    return flag_whole_vouchers(gl, mask, "JET04", reasons)


# ============================================================
# JET05 Postings after period close
# ============================================================

def close_deadlines(posting_dates, close_day):
    """Last allowed entry date for each posting: its month end plus close_day days."""
    return posting_dates + pd.offsets.MonthEnd(0) + pd.Timedelta(days=close_day)


def jet05_after_close(gl, config):
    """JET05 Entries made after the books for their period were closed.

    Late entries change figures that may already have been reported, and
    are a common way to adjust results after the fact.
    """
    close_day = settings(config, "JET05_period_close")["close_day"]
    deadline = close_deadlines(gl["posting_date"], close_day)
    entry_date = gl["entry_time"].dt.normalize()

    mask = entry_date > deadline
    reasons = ("Entered " + entry_date.dt.strftime("%Y-%m-%d")
               + ", after close deadline " + deadline.dt.strftime("%Y-%m-%d")
               + " for period " + gl["posting_date"].dt.strftime("%Y-%m"))
    return flag_whole_vouchers(gl, mask, "JET05", reasons)


def month_end_summary(gl, config):
    """JET05 companion table: share of vouchers and debit amount posted in the
    last N days of each month.

    Month-end work is normal, so single entries are not flagged. A share that
    jumps in one month can signal pressure to meet targets.
    """
    days = settings(config, "JET05_period_close")["month_end_days"]
    lines = gl[gl["posting_date"].notna()]
    dates = lines["posting_date"]
    frame = pd.DataFrame({
        "period": dates.dt.strftime("%Y-%m"),
        "voucher_no": lines["voucher_no"],
        "debit": lines["debit"].fillna(0),
        "month_end": dates.dt.day > dates.dt.days_in_month - days,
    })

    table = frame.groupby("period").agg(
        vouchers=("voucher_no", "nunique"), debit=("debit", "sum"))
    month_end = frame[frame["month_end"]].groupby("period").agg(
        month_end_vouchers=("voucher_no", "nunique"), month_end_debit=("debit", "sum"))
    table = table.join(month_end).fillna(0)
    table["voucher_share"] = (table["month_end_vouchers"] / table["vouchers"]).round(3)
    table["debit_share"] = (table["month_end_debit"] / table["debit"]).round(3)
    columns = ["vouchers", "month_end_vouchers", "voucher_share",
               "debit", "month_end_debit", "debit_share"]
    return table[columns].reset_index()


# ============================================================
# JET06 Round amounts
# ============================================================

def jet06_round_amounts(gl, config):
    """JET06 Round amounts at or above a floor, such as 30,000.00.

    Real invoices usually have odd amounts. Estimates and made-up entries
    are more often round numbers.
    """
    s = settings(config, "JET06_round_amounts")
    amount = line_amount(gl)
    cents = (amount * 100).round()  # whole cents avoid float remainders

    reasons = pd.Series("", index=gl.index)
    for multiple in sorted(s["multiples"]):  # a larger multiple overwrites a smaller one
        mask = (amount >= s["min_amount"]) & (cents % (multiple * 100) == 0)
        reasons[mask] = f"Round amount, multiple of {multiple:,}"
    flagged = reasons != ""
    return flag_lines(gl[flagged], "JET06", reasons[flagged])


# ============================================================
# JET07 Segregation of duties
# ============================================================

def jet07_same_preparer_approver(gl, config):
    """JET07 Segregation of duties: the same person prepared and approved the voucher.

    Approval is a control only when a second person checks the work. If one
    person does both, errors and fraud are not caught.
    """
    mask = gl["prepared_by"].notna() & (gl["prepared_by"] == gl["approved_by"])
    reasons = "Prepared and approved by " + gl["prepared_by"].fillna("")
    return flag_whole_vouchers(gl, mask, "JET07", reasons)


# ============================================================
# JET08 Split payments
# ============================================================

def jet08_split_payments(gl, config):
    """JET08 Split payments: several payments to one supplier within N days,
    each below the approval limit but together above it.

    Splitting one payment into smaller ones avoids the approval that the
    full amount would need. Payments are credit lines on the bank account.
    A window starts at each payment and covers the next N days.
    """
    s = settings(config, "JET08_split_payments")
    limit = s["approval_limit"]
    window = s["window_days"]

    payments = gl[(gl["account_code"] == str(s["bank_account"]))
                  & (gl["credit"] > 0) & (gl["credit"] < limit)
                  & gl["supplier"].notna() & gl["posting_date"].notna()]

    reasons = {}
    for supplier, group in payments.sort_values("posting_date").groupby("supplier"):
        dates = group["posting_date"].tolist()
        amounts = group["credit"].tolist()
        for start in range(len(group)):
            end = start
            while end + 1 < len(group) and (dates[end + 1] - dates[start]).days <= window:
                end += 1
            count = end - start + 1
            total = sum(amounts[start:end + 1])
            if count >= s["min_payments"] and total > limit:
                text = (f"{count} payments to {supplier} within {window} days, "
                        f"total {money_text(total)}, each below {money_text(limit)}")
                for index in group.index[start:end + 1]:
                    reasons.setdefault(index, text)

    reasons = pd.Series(reasons, dtype=object)
    return flag_lines(gl.loc[reasons.index], "JET08", reasons)


# ============================================================
# JET09 Description keywords
# ============================================================

def jet09_keywords(gl, config):
    """JET09 Descriptions that contain words often used for manual adjustments.

    Matching ignores upper and lower case. When several keywords match, the
    reason shows the first one in the config list.
    """
    keywords = settings(config, "JET09_keywords")["keywords"]
    text = gl["description"].fillna("")

    reasons = pd.Series("", index=gl.index)
    for word in keywords:
        mask = (reasons == "") & text.str.contains(word, case=False, regex=False)
        reasons[mask] = f"Description contains '{word}'"
    return flag_whole_vouchers(gl, reasons != "", "JET09", reasons)


# ============================================================
# JET10 Amount outliers
# ============================================================

def robust_z_scores(values):
    """Robust z-score: 0.6745 * (value - median) / MAD.

    MAD, the median absolute deviation, is the median distance of the
    values from their median. Median and MAD barely move when a few extreme
    values are added, unlike mean and standard deviation. The factor 0.6745
    puts the score on the same scale as an ordinary z-score. Returns None
    when MAD is zero, for example when almost all amounts are identical.
    """
    median = values.median()
    mad = (values - median).abs().median()
    if mad == 0:
        return None
    return 0.6745 * (values - median) / mad


def jet10_outliers(gl, config):
    """JET10 Amounts that are unusually large for their account.

    Scores are computed per account and side (debit or credit), because an
    invoice and a payment on the same account are different kinds of line.
    Accounts with fewer than min_lines lines are skipped, as are the accounts
    in exclude_accounts: a bank line can settle many invoices at once, and
    those invoices are already scored on their own accounts. Only large
    amounts are flagged, since overstated or unauthorised amounts are the risk.
    With log_scale the score uses log(amount): amounts are right-skewed,
    with many small lines and a few large ones, and the log makes the
    distribution closer to symmetric.
    """
    s = settings(config, "JET10_outliers")
    excluded = [str(account) for account in s.get("exclude_accounts", [])]
    lines = gl[(line_amount(gl) > 0) & ~gl["account_code"].isin(excluded)].copy()
    lines["side"] = np.where(lines["debit"].notna(), "debit", "credit")
    lines["value"] = line_amount(lines)

    reasons = {}
    for (account, side), group in lines.groupby(["account_code", "side"]):
        if len(group) < s["min_lines"]:
            continue
        values = np.log(group["value"]) if s["log_scale"] else group["value"]
        scores = robust_z_scores(values)
        if scores is None:
            continue
        median = money_text(group["value"].median())
        for index, score in scores[scores > s["z_threshold"]].items():
            reasons[index] = (f"Robust z-score {score:.1f} among {len(group)} {side} lines "
                              f"on account {account}, median {median}")

    reasons = pd.Series(reasons, dtype=object)
    return flag_lines(gl.loc[reasons.index], "JET10", reasons)


# ============================================================
# Run all tests
# ============================================================

ALL_TESTS = [
    jet00_validation,
    jet01_unbalanced,
    jet02_duplicates,
    jet03_weekend_holiday,
    jet04_outside_hours,
    jet05_after_close,
    jet06_round_amounts,
    jet07_same_preparer_approver,
    jet08_split_payments,
    jet09_keywords,
    jet10_outliers,
]


def run_all(gl, config):
    """Run every test and stack the flagged lines into one table."""
    results = [test(gl, config) for test in ALL_TESTS]
    results = [result for result in results if not result.empty]
    if not results:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    return pd.concat(results, ignore_index=True)
