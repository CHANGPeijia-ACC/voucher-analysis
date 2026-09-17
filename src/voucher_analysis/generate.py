"""Generate a synthetic general ledger and bank statement.

The data is made up. Company, supplier and customer names are invented.
Known anomalies are injected on purpose and written to ground truth files,
so the tests can be measured against them.
"""

from datetime import date, datetime, time, timedelta

import holidays
import numpy as np
import pandas as pd

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
