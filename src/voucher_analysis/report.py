"""Write test and reconciliation results to one Excel workbook.

The layout follows an audit workpaper: a summary that says how much was
flagged, one sheet per test with the detail behind it, and the bank
reconciliation. Formatting stays plain: bold headers, frozen panes, number
formats, no colours.
"""

import pandas as pd
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

SYNTHETIC_NOTE = "All data in this workbook is synthetic. It is not real company data."

TEST_DESCRIPTIONS = {
    "JET00": "Input validation: missing fields, bad voucher numbers, negative amounts",
    "JET01": "Unbalanced vouchers: total debit is not equal to total credit",
    "JET02": "Possible duplicates: same account, supplier, side and amount within N days",
    "JET03": "Postings dated on a weekend or a public holiday",
    "JET04": "Entries made outside working hours",
    "JET05": "Entries made after the period was closed",
    "JET06": "Round amounts at or above the floor",
    "JET07": "Segregation of duties: preparer is also approver",
    "JET08": "Split payments: several payments just under the approval limit",
    "JET09": "Descriptions containing keywords used for manual adjustments",
    "JET10": "Amounts that are unusually large for their account",
}

DETAIL_COLUMNS = ["voucher_no", "line_no", "posting_date", "entry_time",
                  "account_code", "account_name", "debit", "credit", "department",
                  "supplier", "prepared_by", "approved_by", "description", "reason"]

UNMATCHED_SHEET_COLUMNS = ["item_type", "bank_ref", "voucher_no", "date",
                           "amount", "counterparty", "difference"]

MONEY_FORMAT = "#,##0.00"
DATE_FORMAT = "yyyy-mm-dd"
TIME_FORMAT = "yyyy-mm-dd hh:mm"
MAX_COLUMN_WIDTH = 46


def summary_table(flags, gl):
    """One row per test: lines flagged, vouchers flagged and amount flagged.

    The amount is the total debit of the flagged vouchers, each counted
    once. Only one side is taken because a voucher moves the same money as
    debit and as credit. Whole vouchers are taken because some tests flag a
    single line, such as the bank line of a payment, and the size of the
    entry is what a reviewer wants to see.
    """
    voucher_debit = gl.groupby("voucher_no")["debit"].sum()
    rows = []
    for test_id, description in TEST_DESCRIPTIONS.items():
        part = flags[flags["test_id"] == test_id]
        vouchers = part["voucher_no"].unique()
        rows.append({
            "test_id": test_id,
            "description": description,
            "lines_flagged": len(part),
            "vouchers_flagged": len(vouchers),
            "voucher_debit_flagged": round(voucher_debit.reindex(vouchers).fillna(0).sum(), 2),
        })
    return pd.DataFrame(rows)


def test_detail(flags, gl, test_id):
    """Flagged lines of one test, with the ledger fields a reviewer needs."""
    part = flags[flags["test_id"] == test_id]
    gl_columns = [column for column in DETAIL_COLUMNS if column in gl.columns]
    detail = part.merge(gl[gl_columns], on=["voucher_no", "line_no"], how="left")
    return detail[DETAIL_COLUMNS]


def unmatched_table(result):
    """Everything from the reconciliation that still needs attention.

    One-sided items first, then pairs where the two sides show different
    amounts. Those are not timing differences and have to be investigated.
    """
    unmatched = result["unmatched"].copy()
    unmatched["difference"] = pd.NA

    differences = result["matches"]
    differences = differences[differences["match_type"] == "amount_difference"].rename(
        columns={"match_type": "item_type", "bank_refs": "bank_ref",
                 "voucher_nos": "voucher_no", "bank_date": "date", "bank_amount": "amount"})
    differences = differences.reindex(columns=UNMATCHED_SHEET_COLUMNS)
    differences["counterparty"] = ""

    table = pd.concat([unmatched.reindex(columns=UNMATCHED_SHEET_COLUMNS), differences],
                      ignore_index=True)
    return table[UNMATCHED_SHEET_COLUMNS]


def column_format(values):
    """The number format for a column, based on what it holds."""
    if pd.api.types.is_datetime64_any_dtype(values):
        has_time = values.dt.hour.fillna(0).ne(0).any() or values.dt.minute.fillna(0).ne(0).any()
        return TIME_FORMAT if has_time else DATE_FORMAT
    if pd.api.types.is_float_dtype(values):
        return MONEY_FORMAT
    return None


def format_sheet(worksheet, table, header_row):
    """Bold header, frozen panes, filter, column widths and number formats."""
    for cell in worksheet[header_row]:
        cell.font = Font(bold=True)
    worksheet.freeze_panes = worksheet.cell(row=header_row + 1, column=1).coordinate
    if len(table.columns):
        last_column = get_column_letter(len(table.columns))
        worksheet.auto_filter.ref = f"A{header_row}:{last_column}{header_row + len(table)}"

    for number, name in enumerate(table.columns, start=1):
        letter = get_column_letter(number)
        sample = table[name].head(200).astype(str).fillna("")  # missing values stay NaN
        width = max([len(str(name))] + [len(value) for value in sample]) + 2
        worksheet.column_dimensions[letter].width = min(width, MAX_COLUMN_WIDTH)

        number_format = column_format(table[name])
        if number_format:
            for row in range(header_row + 1, header_row + len(table) + 1):
                worksheet.cell(row=row, column=number).number_format = number_format


def add_sheet(writer, name, table, note=None):
    """Write one table to a sheet, with an optional note above the header."""
    start_row = 2 if note else 0
    table.to_excel(writer, sheet_name=name, index=False, startrow=start_row)
    worksheet = writer.sheets[name]
    if note:
        worksheet["A1"] = note
    format_sheet(worksheet, table, header_row=start_row + 1)


def write_workbook(path, gl, flags, month_end, rec_result, evaluation=None):
    """Write the whole workbook and return the path.

    Sheets: Summary, one per test, MonthEnd, BankRec, Unmatched, and
    Evaluation when ground truth was available.
    """
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        add_sheet(writer, "Summary", summary_table(flags, gl), note=SYNTHETIC_NOTE)
        for test_id in TEST_DESCRIPTIONS:
            add_sheet(writer, test_id, test_detail(flags, gl, test_id))
        add_sheet(writer, "MonthEnd", month_end)
        add_sheet(writer, "BankRec", rec_result["statement"], note=SYNTHETIC_NOTE)
        add_sheet(writer, "Unmatched", unmatched_table(rec_result))
        if evaluation is not None:
            add_sheet(writer, "Evaluation", evaluation)
    return path
