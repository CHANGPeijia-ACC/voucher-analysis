"""Generate a synthetic general ledger and bank statement.

The data is made up. Company, supplier and customer names are invented.
Known anomalies are injected on purpose and written to ground truth files,
so the tests can be measured against them.
"""

from datetime import date, datetime, time, timedelta
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

from voucher_analysis.loaders import ENCODING

# ============================================================
# Reference lists
# ============================================================

SEED = 42
YEAR = 2025
COUNTRY = "CN"
CLOSE_DAY = 5  # books for a month close on this day of the next month

# Simplified chart of accounts. Codes follow the Chinese numbering style,
# but the expense accounts are flattened for readability.
ACCOUNTS = {
    "1002": "Bank deposits 银行存款",
    "1122": "Accounts receivable 应收账款",
    "1405": "Inventory 库存商品",
    "1601": "Fixed assets 固定资产",
    "1602": "Accumulated depreciation 累计折旧",
    "2202": "Accounts payable 应付账款",
    "2211": "Salaries payable 应付职工薪酬",
    "222101": "Input VAT 应交增值税-进项税额",
    "222102": "Output VAT 应交增值税-销项税额",
    "2241": "Other payables 其他应付款",
    "6001": "Sales revenue 主营业务收入",
    "6401": "Cost of sales 主营业务成本",
    "6601": "Travel expense 差旅费",
    "6602": "Office supplies 办公费",
    "6603": "Entertainment 业务招待费",
    "6604": "Rent expense 租赁费",
    "6605": "Utilities 水电费",
    "6606": "Consulting fees 咨询费",
    "6607": "Salaries expense 工资费用",
    "6608": "Depreciation 折旧费",
    "6609": "Repairs and maintenance 修理费",
    "6610": "Bank charges 银行手续费",
    "6611": "Advertising 广告宣传费",
}

# Suppliers grouped by the account their invoices are posted to.
SUPPLIERS = {
    "1405": [
        "Tianrui Materials 天瑞物资", "Hongyu Electronics 鸿宇电子",
        "Jiahe Packaging 嘉禾包装", "Ruifeng Components 瑞丰元件",
        "Baocheng Plastics 宝成塑料", "Kangda Metals 康达金属",
        "Yongsheng Chemicals 永盛化工", "Lianchuang Parts 联创配件",
        "Dingxin Steel 鼎鑫钢材", "Huamei Textiles 华美纺织",
    ],
    "1601": ["Haotian Machinery 昊天机械", "Ruida Office Furniture 瑞达办公家具"],
    "6601": ["Jietu Travel Service 捷途旅行社", "Yunfan Ticketing 云帆票务"],
    "6602": ["Huaxin Office Supplies 华信办公用品", "Jiayi Printing 嘉艺印务"],
    "6603": ["Jinyue Restaurant 金悦酒楼", "Songhe Tea House 松鹤茶楼"],
    "6604": ["Hengtai Property 恒泰物业"],
    "6605": ["Donghu Water and Power 东湖水电"],
    "6606": ["Zhiyuan Consulting 智远咨询", "Qianfeng Tax Advisory 千峰税务咨询"],
    "6609": ["Anjie Equipment Repair 安捷设备维修", "Wanli Facility Services 万里设施服务"],
    "6611": ["Xingchen Advertising 星辰广告", "Fangzhou Exhibition 方舟会展"],
}

CUSTOMERS = [
    "Beichen Trading 北辰贸易", "Tongda Retail 通达零售", "Jinhai Industrial 金海实业",
    "Yuanhang Technology 远航科技", "Shunfa Wholesale 顺发批发",
    "Minghui Manufacturing 明辉制造", "Fuyuan Supermarkets 富源超市",
    "Xinhe Distribution 信和分销", "Guangyao Electric 光耀电器",
    "Longsheng Home 隆盛家居", "Qinghe Foods 清禾食品", "Anping Hardware 安平五金",
]

PREPARERS = ["prep01", "prep02", "prep03", "prep04", "prep05"]
APPROVERS = ["appr01", "appr02", "appr03"]
DEPARTMENTS = ["Sales", "Production", "Admin", "Finance", "IT"]

# Monthly salary cost per department, before small random changes.
SALARY_BASE = {
    "Sales": 180000, "Production": 150000, "Admin": 90000,
    "Finance": 70000, "IT": 110000,
}

# Normal net amount per account: (median, spread). Amounts follow a
# lognormal distribution: most are near the median, a few are much larger.
AMOUNT_PROFILE = {
    "1405": (15000, 0.7),
    "1601": (300000, 0.4),
    "6001": (48000, 0.7),
    "6601": (2500, 0.6),
    "6602": (800, 0.5),
    "6603": (1800, 0.5),
    "6606": (20000, 0.5),
    "6609": (3500, 0.6),
    "6611": (25000, 0.6),
}

VAT_RATE = {"1405": 0.13, "1601": 0.13, "6001": 0.13,
            "6606": 0.06, "6609": 0.13, "6611": 0.06}


# ============================================================
# Calendar helpers
# ============================================================

def make_holidays(year, country=COUNTRY):
    """Public holidays for the fiscal year and the next year (for late entries)."""
    return holidays.country_holidays(country, years=[year, year + 1])


def is_working_day(day, hols):
    """A working day is Monday to Friday and not a public holiday."""
    return day.weekday() < 5 and day not in hols


def next_working_day(day, hols):
    """The first working day strictly after `day`."""
    day += timedelta(days=1)
    while not is_working_day(day, hols):
        day += timedelta(days=1)
    return day


def working_day_on_or_before(day, hols):
    """`day` itself if it is a working day, otherwise the working day before it."""
    while not is_working_day(day, hols):
        day -= timedelta(days=1)
    return day


def working_day_on_or_after(day, hols):
    """`day` itself if it is a working day, otherwise the working day after it."""
    while not is_working_day(day, hols):
        day += timedelta(days=1)
    return day


def working_days_in_year(year, hols):
    """All working days of the year, in order."""
    day = date(year, 1, 1)
    days = []
    while day.year == year:
        if is_working_day(day, hols):
            days.append(day)
        day += timedelta(days=1)
    return days


def month_end(year, month):
    """Last calendar day of a month."""
    if month == 12:
        return date(year, 12, 31)
    return date(year, month + 1, 1) - timedelta(days=1)


def close_deadline(posting_date, close_day=CLOSE_DAY):
    """Last day an entry for this posting month may be made."""
    return month_end(posting_date.year, posting_date.month) + timedelta(days=close_day)


# ============================================================
# Building vouchers
# ============================================================
# A voucher is a dict with header fields and a list of lines.
# Each line is a dict with account_code, debit and credit.

def money(value):
    """Round to cents."""
    return round(float(value), 2)


def draw_amount(rng, account):
    """Draw a normal net amount for an account from its lognormal profile.

    Amounts are capped at five times the median so that the normal data
    has no extreme values; extreme values are injected separately.
    """
    median, spread = AMOUNT_PROFILE[account]
    value = rng.lognormal(mean=np.log(median), sigma=spread)
    return money(min(max(value, 50), median * 5))


def line(account, debit=0.0, credit=0.0, department=None):
    """One voucher line. Department is only set when it differs by line."""
    return {"account_code": account, "debit": money(debit),
            "credit": money(credit), "department": department}


def voucher(kind, posting_date, lines, description,
            department="", supplier="", counterparty=""):
    """Create a voucher. People, entry time and number are added later."""
    return {
        "kind": kind, "posting_date": posting_date, "lines": lines,
        "description": description, "department": department,
        "supplier": supplier, "counterparty": counterparty,
        "entry_time": None, "prepared_by": "", "approved_by": "",
        "voucher_no": "", "anomaly": None, "bank_group": None,
    }


def invoice_lines(account, net, rate):
    """Lines for a supplier invoice: expense or asset, input VAT, payable."""
    vat = money(net * rate)
    return [
        line(account, debit=net),
        line("222101", debit=vat),
        line("2202", credit=money(net + vat)),
    ]


def invoice_gross(v):
    """Total payable on a supplier invoice voucher."""
    return money(sum(ln["credit"] for ln in v["lines"] if ln["account_code"] == "2202"))


def bank_amount(v):
    """Net effect of a voucher on the bank account: receipts positive."""
    return money(sum(ln["debit"] - ln["credit"]
                     for ln in v["lines"] if ln["account_code"] == "1002"))


# ============================================================
# Normal business
# ============================================================

EXPENSE_TEXT = {
    "6601": "Business travel", "6602": "Office supplies", "6603": "Client meal",
    "6606": "Consulting services", "6609": "Repair services", "6611": "Advertising services",
}
EXPENSE_DEPARTMENT = {"6606": "Finance", "6609": "Production", "6611": "Sales"}


def purchase_voucher(day, rng, account="1405"):
    """Supplier invoice for materials or equipment, paid later through accounts payable."""
    supplier = str(rng.choice(SUPPLIERS[account]))
    net = draw_amount(rng, account)
    text = "Purchase of materials" if account == "1405" else "Purchase of equipment"
    department = "Production" if account == "1405" else "Admin"
    return voucher("purchase", day, invoice_lines(account, net, VAT_RATE[account]),
                   text, department, supplier, supplier)


def expense_invoice_voucher(day, rng):
    """Service invoice (consulting, repairs, advertising), paid later through payables."""
    account = str(rng.choice(["6606", "6609", "6611"], p=[0.3, 0.45, 0.25]))
    supplier = str(rng.choice(SUPPLIERS[account]))
    net = draw_amount(rng, account)
    return voucher("expense_invoice", day, invoice_lines(account, net, VAT_RATE[account]),
                   EXPENSE_TEXT[account], EXPENSE_DEPARTMENT[account], supplier, supplier)


def direct_expense_voucher(day, rng):
    """Small expense (travel, office supplies, meals) paid straight from the bank."""
    account = str(rng.choice(["6601", "6602", "6603"], p=[0.45, 0.25, 0.3]))
    supplier = str(rng.choice(SUPPLIERS[account]))
    amount = draw_amount(rng, account)
    lines = [line(account, debit=amount), line("1002", credit=amount)]
    return voucher("direct_expense", day, lines, EXPENSE_TEXT[account],
                   str(rng.choice(DEPARTMENTS)), supplier, supplier)


def sale_voucher(day, rng):
    """Sales invoice with output VAT and the matching cost of sales."""
    customer = str(rng.choice(CUSTOMERS))
    net = draw_amount(rng, "6001")
    vat = money(net * VAT_RATE["6001"])
    cost = money(net * rng.uniform(0.55, 0.70))
    lines = [
        line("1122", debit=money(net + vat)),
        line("6001", credit=net),
        line("222102", credit=vat),
        line("6401", debit=cost),
        line("1405", credit=cost),
    ]
    v = voucher("sale", day, lines, "Sales invoice", "Sales", counterparty=customer)
    v["customer"] = customer
    return v


def daily_vouchers(day, rng):
    """Vouchers for one working day. The counts vary randomly around a daily average."""
    vouchers = []
    for _ in range(rng.poisson(9)):
        vouchers.append(purchase_voucher(day, rng))
    for _ in range(rng.poisson(2.5)):
        vouchers.append(expense_invoice_voucher(day, rng))
    for _ in range(rng.poisson(3)):
        vouchers.append(direct_expense_voucher(day, rng))
    for _ in range(rng.poisson(4)):
        vouchers.append(sale_voucher(day, rng))
    return vouchers


def receipt_vouchers(sales, year, hols, rng):
    """Customers pay each sales invoice 30 to 60 days later, if still in the year."""
    vouchers = []
    for sale in sales:
        days = int(rng.choice([30, 45, 60])) + int(rng.integers(-3, 8))
        day = working_day_on_or_after(sale["posting_date"] + timedelta(days=days), hols)
        if day.year != year:
            continue
        amount = sale["lines"][0]["debit"]
        lines = [line("1002", debit=amount), line("1122", credit=amount)]
        vouchers.append(voucher("receipt", day, lines, "Receipt from " + sale["customer"],
                                "Sales", counterparty=sale["customer"]))
    return vouchers


def payment_voucher(day, supplier, invoices, text):
    """Pay one or more invoices: one payable line per invoice, one bank line."""
    lines = [line("2202", debit=invoice_gross(inv)) for inv in invoices]
    total = money(sum(ln["debit"] for ln in lines))
    lines.append(line("1002", credit=total))
    return voucher("payment", day, lines, text, "Finance", supplier, supplier)


def payment_batches(invoices, year, hols):
    """Group invoices into monthly payment runs (around the 25th) by supplier.

    An invoice is paid in the first run that is at least 30 days after it.
    Returns a list of (run_day, supplier, invoices).
    """
    open_invoices = sorted(invoices, key=lambda v: v["posting_date"])
    batches = []
    for month in range(1, 13):
        run_day = working_day_on_or_before(date(year, month, 25), hols)
        cutoff = run_day - timedelta(days=30)
        due = [v for v in open_invoices if v["posting_date"] <= cutoff]
        open_invoices = [v for v in open_invoices if v["posting_date"] > cutoff]
        by_supplier = {}
        for inv in due:
            by_supplier.setdefault(inv["supplier"], []).append(inv)
        for supplier in sorted(by_supplier):
            batches.append((run_day, supplier, by_supplier[supplier]))
    return batches


def payment_run_vouchers(invoices, year, hols, rng, split_groups=3):
    """One payment voucher per supplier per run.

    For a few mid-year runs, two to four invoices are paid as separate
    vouchers that the bank settles in one transfer. This gives the bank
    reconciliation a one-to-many case.
    """
    batches = payment_batches(invoices, year, hols)
    candidates = [i for i, (day, _, invs) in enumerate(batches)
                  if 3 <= day.month <= 10 and len(invs) >= 2]
    split_ids = set(int(i) for i in rng.choice(candidates, size=split_groups, replace=False))

    vouchers = []
    for i, (run_day, supplier, invs) in enumerate(batches):
        text = f"Payment run {run_day:%Y-%m}"
        if i in split_ids:
            size = min(len(invs), int(rng.integers(2, 5)))
            for inv in invs[:size]:
                v = payment_voucher(run_day, supplier, [inv], text)
                v["bank_group"] = f"G{i}"
                vouchers.append(v)
            invs = invs[size:]
        if invs:
            vouchers.append(payment_voucher(run_day, supplier, invs, text))
    return vouchers


def payroll_vouchers(year, month, hols, rng):
    """Accrue salaries by department at month end; pay them on the 10th of next month."""
    last_day = working_day_on_or_before(month_end(year, month), hols)
    label = f"{year}-{month:02d}"
    lines = [line("6607", debit=SALARY_BASE[d] * rng.uniform(0.97, 1.03), department=d)
             for d in DEPARTMENTS]
    total = money(sum(ln["debit"] for ln in lines))
    lines.append(line("2211", credit=total))
    vouchers = [voucher("payroll", last_day, lines, f"Payroll accrual {label}", "Finance")]

    if month < 12:  # December salaries are paid next year
        pay_day = working_day_on_or_after(date(year, month + 1, 10), hols)
        lines = [line("2211", debit=total), line("1002", credit=total)]
        vouchers.append(voucher("salary_payment", pay_day, lines, f"Salary payment {label}",
                                "Finance", counterparty="Payroll batch 工资代发"))
    return vouchers


def utilities_vouchers(year, month, hols, rng):
    """Pay the utilities bill mid-month, accrue an estimate at month end,
    and reverse that accrual on the first working day of the next month."""
    label = f"{year}-{month:02d}"
    utility = SUPPLIERS["6605"][0]

    bill = money(rng.uniform(9000, 15000))
    bill_day = working_day_on_or_after(date(year, month, 15), hols)
    lines = [line("6605", debit=bill), line("1002", credit=bill)]
    vouchers = [voucher("utilities", bill_day, lines, f"Utilities bill {label}",
                        "Admin", utility, utility)]

    estimate = money(rng.uniform(10000, 14000))
    last_day = working_day_on_or_before(month_end(year, month), hols)
    lines = [line("6605", debit=estimate), line("2241", credit=estimate)]
    vouchers.append(voucher("accrual", last_day, lines, f"Utilities accrual {label}",
                            "Admin", utility))

    if month < 12:
        first_day = working_day_on_or_after(date(year, month + 1, 1), hols)
        lines = [line("2241", debit=estimate), line("6605", credit=estimate)]
        vouchers.append(voucher("accrual", first_day, lines,
                                f"Reversal of utilities accrual {label}", "Admin", utility))
    return vouchers


def monthly_vouchers(year, hols, rng):
    """Recurring entries: payroll, depreciation, rent, utilities and bank charges."""
    vouchers = []
    for month in range(1, 13):
        last_day = working_day_on_or_before(month_end(year, month), hols)
        label = f"{year}-{month:02d}"

        vouchers.extend(payroll_vouchers(year, month, hols, rng))
        vouchers.extend(utilities_vouchers(year, month, hols, rng))

        lines = [line("6608", debit=43583.33), line("1602", credit=43583.33)]
        vouchers.append(voucher("depreciation", last_day, lines, f"Depreciation {label}", "Finance"))

        rent_day = working_day_on_or_after(date(year, month, 5), hols)
        landlord = SUPPLIERS["6604"][0]
        lines = [line("6604", debit=60000), line("1002", credit=60000)]
        vouchers.append(voucher("rent", rent_day, lines, f"Office rent {label}",
                                "Admin", landlord, landlord))

        if month < 12:  # December charges are not booked before year end
            fee = money(rng.uniform(80, 400))
            lines = [line("6610", debit=fee), line("1002", credit=fee)]
            vouchers.append(voucher("bank_fee", last_day, lines, f"Bank charges {label}",
                                    "Finance", counterparty="Bank service charge 银行手续费"))
    return vouchers


def normal_vouchers(year, hols, rng):
    """All normal business for the year, before any anomaly is injected."""
    workdays = working_days_in_year(year, hols)
    vouchers = []
    for day in workdays:
        vouchers.extend(daily_vouchers(day, rng))
    for i in sorted(rng.choice(len(workdays), size=6, replace=False)):
        vouchers.append(purchase_voucher(workdays[i], rng, account="1601"))

    sales = [v for v in vouchers if v["kind"] == "sale"]
    invoices = [v for v in vouchers if v["kind"] in ("purchase", "expense_invoice")]
    vouchers.extend(receipt_vouchers(sales, year, hols, rng))
    vouchers.extend(payment_run_vouchers(invoices, year, hols, rng))
    vouchers.extend(monthly_vouchers(year, hols, rng))
    return vouchers


# ============================================================
# People, entry times and voucher numbers
# ============================================================

def normal_entry_time(posting_date, hols, rng):
    """Entered in office hours on the posting day or up to two working days later,
    never after the period close deadline."""
    day = posting_date
    for _ in range(int(rng.choice([0, 1, 2], p=[0.7, 0.2, 0.1]))):
        day = next_working_day(day, hols)
    day = min(day, close_deadline(posting_date))
    minutes = int(rng.integers(9 * 60, 18 * 60 + 30))
    return datetime.combine(day, time(minutes // 60, minutes % 60, int(rng.integers(0, 60))))


def assign_people_and_times(vouchers, hols, rng):
    """Give each voucher a preparer, an approver from a separate pool, and an entry time."""
    for v in vouchers:
        v["prepared_by"] = str(rng.choice(PREPARERS))
        v["approved_by"] = str(rng.choice(APPROVERS))
        v["entry_time"] = normal_entry_time(v["posting_date"], hols, rng)


def assign_voucher_numbers(vouchers):
    """Number vouchers JV000001, JV000002, ... in order of posting date and entry time."""
    vouchers.sort(key=lambda v: (v["posting_date"], v["entry_time"]))
    for i, v in enumerate(vouchers, start=1):
        v["voucher_no"] = f"JV{i:06d}"


def vouchers_to_gl(vouchers):
    """Flatten vouchers into one GL row per line. Zero amounts are left empty."""
    rows = []
    for v in vouchers:
        for line_no, ln in enumerate(v["lines"], start=1):
            rows.append({
                "voucher_no": v["voucher_no"],
                "line_no": line_no,
                "posting_date": v["posting_date"].isoformat(),
                "entry_time": v["entry_time"].strftime("%Y-%m-%d %H:%M:%S"),
                "account_code": ln["account_code"],
                "account_name": ACCOUNTS[ln["account_code"]],
                "debit": ln["debit"] or None,
                "credit": ln["credit"] or None,
                "department": ln["department"] or v["department"],
                "supplier": v["supplier"],
                "prepared_by": v["prepared_by"],
                "approved_by": v["approved_by"],
                "description": v["description"],
            })
    return pd.DataFrame(rows)


# ============================================================
# Injected anomalies
# ============================================================
# Each function changes or adds vouchers and tags them with the anomaly
# type. Only vouchers without a tag are picked, so each tagged voucher has
# exactly one known anomaly. Except for split payments, anomalies go into
# vouchers that do not touch the bank, so the bank statement stays clean.

ANOMALY_COUNT = 15        # vouchers per anomaly type
DUPLICATE_PAIRS = 8       # each pair tags the original and the copy
SPLIT_GROUPS = 5          # each group has three payments
APPROVAL_LIMIT = 50000    # matches the config default

NON_BANK_KINDS = ("purchase", "expense_invoice", "sale")

KEYWORD_TEXTS = [
    "Manual adjustment per management request",
    "Correction of prior month posting",
    "Reversal of duplicate invoice",
    "调整上月入库差异",
    "冲回多计费用",
    "暂估入库",
]


def pick(vouchers, rng, count, kinds=NON_BANK_KINDS, condition=None):
    """Randomly pick untagged vouchers of the given kinds."""
    eligible = [v for v in vouchers
                if v["anomaly"] is None and v["kind"] in kinds
                and (condition is None or condition(v))]
    chosen = rng.choice(len(eligible), size=count, replace=False)
    return [eligible[i] for i in chosen]


def office_hours_time(day, rng):
    """A random time between 09:00 and 18:30 on `day`."""
    minutes = int(rng.integers(9 * 60, 18 * 60 + 30))
    return datetime.combine(day, time(minutes // 60, minutes % 60))


def set_invoice_net(v, net):
    """Change the net amount of a supplier invoice and recalculate VAT and payable."""
    account = v["lines"][0]["account_code"]
    v["lines"] = invoice_lines(account, money(net), VAT_RATE[account])


def transpose_digits(amount, rng):
    """Swap two neighbouring digits, a common typing error (1250.00 becomes 1520.00)."""
    whole, cents = f"{amount:.2f}".split(".")
    positions = [i for i in range(len(whole) - 1) if whole[i] != whole[i + 1]]
    if not positions:
        return money(amount + 100)
    i = int(rng.choice(positions))
    swapped = whole[:i] + whole[i + 1] + whole[i] + whole[i + 2:]
    return money(f"{swapped}.{cents}")


def inject_unbalanced(vouchers, hols, rng):
    """Mistype the first debit line so debit no longer equals credit."""
    for v in pick(vouchers, rng, ANOMALY_COUNT):
        first = v["lines"][0]
        first["debit"] = transpose_digits(first["debit"], rng)
        v["anomaly"] = "unbalanced"


def inject_duplicates(vouchers, hols, rng):
    """Post the same supplier invoice again one to five days later."""
    originals = pick(vouchers, rng, DUPLICATE_PAIRS, kinds=("purchase", "expense_invoice"),
                     condition=lambda v: v["posting_date"].month < 12)
    for original in originals:
        copy = voucher(original["kind"], original["posting_date"],
                       [dict(ln) for ln in original["lines"]], original["description"],
                       original["department"], original["supplier"], original["counterparty"])
        later = original["posting_date"] + timedelta(days=int(rng.integers(1, 6)))
        copy["posting_date"] = working_day_on_or_after(later, hols)
        copy["prepared_by"] = str(rng.choice(PREPARERS))
        copy["approved_by"] = str(rng.choice(APPROVERS))
        copy["entry_time"] = normal_entry_time(copy["posting_date"], hols, rng)
        original["anomaly"] = copy["anomaly"] = "duplicate_entry"
        vouchers.append(copy)


def inject_weekend_holiday(vouchers, hols, rng):
    """Move the posting date to a weekend or public holiday in the same month."""
    for v in pick(vouchers, rng, ANOMALY_COUNT):
        day = v["posting_date"]
        off_days = [date(day.year, day.month, d)
                    for d in range(1, month_end(day.year, day.month).day + 1)
                    if not is_working_day(date(day.year, day.month, d), hols)]
        new_day = off_days[int(rng.integers(len(off_days)))]
        v["posting_date"] = new_day
        v["entry_time"] = office_hours_time(new_day, rng)
        v["anomaly"] = "weekend_holiday"


def inject_late_night(vouchers, hols, rng):
    """Keep the entry date but change the entry time to between 22:00 and 05:59."""
    for v in pick(vouchers, rng, ANOMALY_COUNT):
        hour = int(rng.choice([22, 23, 0, 1, 2, 3, 4, 5]))
        v["entry_time"] = datetime.combine(v["entry_time"].date(),
                                           time(hour, int(rng.integers(0, 60))))
        v["anomaly"] = "late_night"


def inject_after_close(vouchers, hols, rng):
    """Enter the voucher 3 to 40 days after the period close deadline."""
    for v in pick(vouchers, rng, ANOMALY_COUNT):
        late = close_deadline(v["posting_date"]) + timedelta(days=int(rng.integers(3, 41)))
        v["entry_time"] = office_hours_time(working_day_on_or_after(late, hols), rng)
        v["anomaly"] = "after_close"


def inject_round_amounts(vouchers, hols, rng):
    """Set a service invoice to a round net amount between 10,000 and 40,000."""
    for v in pick(vouchers, rng, ANOMALY_COUNT, kinds=("expense_invoice",),
                  condition=lambda v: v["lines"][0]["account_code"] in ("6606", "6611")):
        if rng.random() < 0.5:
            net = 10000 * int(rng.integers(1, 5))
        else:
            net = 1000 * int(rng.integers(11, 40))
        set_invoice_net(v, net)
        v["anomaly"] = "round_amount"


def inject_same_preparer_approver(vouchers, hols, rng):
    """The person who prepared the voucher also approved it."""
    for v in pick(vouchers, rng, ANOMALY_COUNT):
        v["approved_by"] = v["prepared_by"]
        v["anomaly"] = "same_preparer_approver"


def inject_split_payments(vouchers, hols, rng):
    """Three payments to one supplier on consecutive working days, each just
    under the approval limit, together well above it."""
    workdays = [d for d in working_days_in_year(YEAR, hols) if 2 <= d.month <= 11]
    for _ in range(SPLIT_GROUPS):
        supplier = str(rng.choice(SUPPLIERS["1405"]))
        day = workdays[int(rng.integers(len(workdays)))]
        for _ in range(3):
            amount = money(rng.uniform(0.84, 0.998) * APPROVAL_LIMIT)
            lines = [line("2202", debit=amount), line("1002", credit=amount)]
            v = voucher("payment", day, lines, "Payment to supplier", "Finance",
                        supplier, supplier)
            v["prepared_by"] = str(rng.choice(PREPARERS))
            v["approved_by"] = str(rng.choice(APPROVERS))
            v["entry_time"] = normal_entry_time(day, hols, rng)
            v["anomaly"] = "split_payment"
            vouchers.append(v)
            day = next_working_day(day, hols)


def inject_keywords(vouchers, hols, rng):
    """Replace the description with wording that often hides manual adjustments."""
    for v in pick(vouchers, rng, ANOMALY_COUNT):
        v["description"] = KEYWORD_TEXTS[int(rng.integers(len(KEYWORD_TEXTS)))]
        v["anomaly"] = "suspicious_keyword"


def inject_extreme_amounts(vouchers, hols, rng):
    """Multiply a normal invoice by 15 to 30 times."""
    for v in pick(vouchers, rng, ANOMALY_COUNT, kinds=("purchase", "expense_invoice"),
                  condition=lambda v: v["lines"][0]["account_code"] != "1601"):
        set_invoice_net(v, v["lines"][0]["debit"] * rng.uniform(15, 30))
        v["anomaly"] = "extreme_amount"


INJECTORS = [
    inject_unbalanced,
    inject_duplicates,
    inject_weekend_holiday,
    inject_late_night,
    inject_after_close,
    inject_round_amounts,
    inject_same_preparer_approver,
    inject_split_payments,
    inject_keywords,
    inject_extreme_amounts,
]


def inject_anomalies(vouchers, hols, rng):
    """Run every injector once, in a fixed order so the seed gives the same result."""
    for inject in INJECTORS:
        inject(vouchers, hols, rng)


def ground_truth(vouchers):
    """One row per voucher with an injected anomaly."""
    rows = [{"voucher_no": v["voucher_no"], "anomaly_type": v["anomaly"]}
            for v in vouchers if v["anomaly"]]
    return pd.DataFrame(rows, columns=["voucher_no", "anomaly_type"])


def generate_gl(seed=SEED, year=YEAR):
    """Build the general ledger and its ground truth.

    Returns (gl, ground_truth, vouchers). The voucher list is used to
    build the bank statement.
    """
    rng = np.random.default_rng(seed)
    hols = make_holidays(year)
    vouchers = normal_vouchers(year, hols, rng)
    assign_people_and_times(vouchers, hols, rng)
    inject_anomalies(vouchers, hols, rng)
    assign_voucher_numbers(vouchers)
    return vouchers_to_gl(vouchers), ground_truth(vouchers), vouchers


# ============================================================
# Bank statement
# ============================================================
# The bank statement is built from the vouchers that touch account 1002.
# It has no voucher numbers. Online transfers settle on any calendar day,
# so bank dates are not moved to working days.

OUTSTANDING_PAYMENTS = 3   # paid in late December, cleared by the bank in January
DEPOSITS_IN_TRANSIT = 2    # received in late December, credited by the bank in January
AMOUNT_DIFFERENCES = 3     # bank deducted a transfer fee from the payment
TRANSFER_FEE = 25.00

BANK_FEE_PARTY = "Bank service charge 银行手续费"
DIRECT_DEBITS = [
    ("Donghu Telecom 东湖通信", 3000, 6000),
    ("Social insurance direct debit 社保代扣", 20000, 30000),
]


def bank_lag(amount, kind, rng):
    """Days from book date to bank date.

    Payments reach the bank 0 to 3 days after booking. Receipts are often
    seen by the bank first, so the bank date is 0 or 1 day earlier.
    """
    if kind == "bank_fee":
        return 0
    if amount < 0:
        return int(rng.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1]))
    return -int(rng.choice([0, 1], p=[0.6, 0.4]))


def pick_special_cases(vouchers, year, rng):
    """Choose the vouchers that become outstanding items or amount differences.

    Returns a dict from voucher_no to the special case name.
    """
    def candidates(kinds, condition):
        return [v for v in vouchers
                if v["kind"] in kinds and v["anomaly"] is None
                and v["bank_group"] is None and condition(v)]

    late_december = lambda v: v["posting_date"] >= date(year, 12, 20)
    mid_year = lambda v: 3 <= v["posting_date"].month <= 10

    groups = [
        ("outstanding_payment", OUTSTANDING_PAYMENTS,
         candidates(("payment", "direct_expense", "utilities"), late_december)),
        ("deposit_in_transit", DEPOSITS_IN_TRANSIT, candidates(("receipt",), late_december)),
        ("amount_difference", AMOUNT_DIFFERENCES, candidates(("payment",), mid_year)),
    ]
    special = {}
    for name, count, pool in groups:
        for i in rng.choice(len(pool), size=count, replace=False):
            special[pool[int(i)]["voucher_no"]] = name
    return special


def book_to_bank_items(vouchers, year, rng):
    """Turn bank vouchers into bank items: one item per voucher, or one per bank group.

    Each item is a dict with bank_date, amount, counterparty, the voucher
    numbers it covers and its match type.
    """
    special = pick_special_cases(vouchers, year, rng)
    items = []
    groups = {}

    for v in vouchers:
        amount = bank_amount(v)
        if amount == 0:
            continue
        if v["bank_group"]:
            groups.setdefault(v["bank_group"], []).append(v)
            continue

        case = special.get(v["voucher_no"], "one_to_one")
        bank_date = v["posting_date"] + timedelta(days=bank_lag(amount, v["kind"], rng))
        if case in ("outstanding_payment", "deposit_in_transit"):
            bank_date = v["posting_date"] + timedelta(days=int(rng.integers(12, 20)))
        if case == "amount_difference":
            amount = money(amount - TRANSFER_FEE)
        items.append({"bank_date": max(bank_date, date(year, 1, 1)), "amount": amount,
                      "counterparty": v["counterparty"],
                      "voucher_nos": [v["voucher_no"]], "match_type": case})

    for members in groups.values():
        first = members[0]
        lag = bank_lag(-1, first["kind"], rng)
        items.append({"bank_date": first["posting_date"] + timedelta(days=lag),
                      "amount": money(sum(bank_amount(v) for v in members)),
                      "counterparty": first["counterparty"],
                      "voucher_nos": [v["voucher_no"] for v in members],
                      "match_type": "one_to_many"})
    return items


def bank_only_items(year, hols, rng):
    """Items the bank recorded but the books did not: December charges and direct debits."""
    last_day = working_day_on_or_before(date(year, 12, 31), hols)
    items = [{"bank_date": last_day, "amount": -money(rng.uniform(80, 400)),
              "counterparty": BANK_FEE_PARTY, "voucher_nos": [], "match_type": "bank_fee"}]
    for party, low, high in DIRECT_DEBITS:
        day = working_day_on_or_after(date(year, 12, int(rng.integers(15, 27))), hols)
        items.append({"bank_date": day, "amount": -money(rng.uniform(low, high)),
                      "counterparty": party, "voucher_nos": [], "match_type": "direct_debit"})
    return items


def build_bank_statement(vouchers, year, hols, rng):
    """Build the bank statement for the year and the bank-side ground truth.

    Book items whose bank date falls after year end do not appear on the
    statement. They are book-only items, whether chosen on purpose or
    caused by a normal date lag.
    """
    items = book_to_bank_items(vouchers, year, rng) + bank_only_items(year, hols, rng)
    for item in items:
        if item["bank_date"].year > year:
            item["match_type"] = ("outstanding_payment" if item["amount"] < 0
                                  else "deposit_in_transit")

    on_statement = sorted((i for i in items if i["bank_date"].year == year),
                          key=lambda i: (i["bank_date"], i["amount"]))
    bank_rows = []
    truth_rows = []
    for n, item in enumerate(on_statement, start=1):
        item["bank_ref"] = f"BR{n:06d}"
        bank_rows.append({"bank_date": item["bank_date"].isoformat(), "amount": item["amount"],
                          "counterparty": item["counterparty"], "bank_ref": item["bank_ref"]})

    for item in items:
        ref = item.get("bank_ref", "")
        for voucher_no in item["voucher_nos"] or [""]:
            truth_rows.append({"bank_ref": ref, "voucher_no": voucher_no,
                               "match_type": item["match_type"]})

    bank = pd.DataFrame(bank_rows)
    truth = pd.DataFrame(truth_rows).sort_values(["bank_ref", "voucher_no"], ignore_index=True)
    return bank, truth


# ============================================================
# Putting it together
# ============================================================

def generate_all(seed=SEED, year=YEAR):
    """Generate every table. Returns a dict from file name to DataFrame."""
    gl, truth, vouchers = generate_gl(seed, year)
    rng = np.random.default_rng(seed + 1)
    bank, bank_truth = build_bank_statement(vouchers, year, make_holidays(year), rng)
    return {"gl": gl, "ground_truth": truth, "bank": bank, "bank_ground_truth": bank_truth}


def write_outputs(tables, out_dir):
    """Write each table to <out_dir>/<name>.csv."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(out / f"{name}.csv", index=False, encoding=ENCODING, float_format="%.2f")


if __name__ == "__main__":
    write_outputs(generate_all(), "data/sample")
