"""
凭证数据分析工具

功能:
    1. 数据校验     —— 检查凭证号格式、金额有效性
    2. 月度汇总     —— 按月统计笔数、合计、平均      → monthly.csv
    3. 交叉汇总     —— 按月份 × 科目двумерная 汇总    → cross.csv
    4. 异常检测     —— 按科目分别识别超额记录        → outliers.csv
    5. 银行对账     —— 与银行流水逐笔比对            → diff.csv

用法:
    把 data.csv 和 bank.csv 放在同一目录,运行本文件即可。
"""

import csv

# ============================================================
# 配置
# ============================================================

DATA_FILE = "data.csv"
BANK_FILE = "bank.csv"
OUTLIER_TIMES = 3          # 异常阈值:超过本科目均值的几倍
ENCODING = "utf-8-sig"     # 写给 Excel 看的编码


# ============================================================
# 工具函数
# ============================================================

def is_valid_code(code):
    """凭证号必须是 V 开头 + 三位数字,共 4 位。"""
    if len(code) != 4:
        return False
    if not code.startswith("V"):
        return False
    return code[1:].isdigit()


def is_number(s):
    """判断字符串能否转成数字。空值、文字都返回 False。"""
    try:
        float(s)
        return True
    except ValueError:
        return False


def read_csv(path):
    """读 CSV,返回列表套字典。"""
    with open(path, "r", encoding=ENCODING) as f:
        return list(csv.DictReader(f))


def write_csv(path, headers, rows):
    """写 CSV:表头一行,数据若干行。"""
    with open(path, "w", encoding=ENCODING, newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"  已生成 {path}")


# ============================================================
# 1. 数据校验
# ============================================================

def validate(rows):
    """
    分拣数据。
    返回 (good, problems):
        good     —— 校验通过的记录,金额已转成 float
        problems —— [(凭证号, 问题描述), ...]
    """
    good = []
    problems = []

    for row in rows:
        errors = []

        if not is_valid_code(row["凭证号"]):
            errors.append("凭证号不合规")
        if not is_number(row["金额"]):
            errors.append("金额缺失或非数字")

        if errors:
            problems.append((row["凭证号"], "、".join(errors)))
        else:
            row["金额"] = float(row["金额"])
            good.append(row)

    return good, problems


def report_validation(rows, good, problems):
    """把校验结果打印出来。"""
    print("=" * 50)
    print("一、数据校验")
    print("=" * 50)

    if problems:
        for code, msg in problems:
            print(f"  [问题] {code}:{msg}")
    else:
        print("  未发现格式问题")

    negatives = [r for r in good if r["金额"] < 0]
    for r in negatives:
        print(f"  [提示] {r['凭证号']} 金额为负 {r['金额']:,.0f},需人工确认")

    print(f"\n  共 {len(rows)} 条,通过 {len(good)} 条,问题 {len(problems)} 条")


# ============================================================
# 2. 月度汇总
# ============================================================

def summarize_by_month(good):
    """按月份汇总,返回 {月份: {笔数, 合计}}。"""
    monthly = {}

    for row in good:
        month = row["日期"][:7]          # "2025-01-05" → "2025-01"
        if month not in monthly:
            monthly[month] = {"笔数": 0, "合计": 0}
        monthly[month]["笔数"] += 1
        monthly[month]["合计"] += row["金额"]

    return monthly


def report_monthly(monthly):
    print("\n" + "=" * 50)
    print("二、月度汇总")
    print("=" * 50)

    rows = []
    for month in sorted(monthly):
        d = monthly[month]
        avg = d["合计"] / d["笔数"]
        print(f"  {month}  {d['笔数']} 笔  合计 {d['合计']:>10,.2f}  平均 {avg:>10,.2f}")
        rows.append([month, d["笔数"], d["合计"], round(avg, 2)])

    write_csv("monthly.csv", ["月份", "笔数", "合计金额", "平均金额"], rows)


# ============================================================
# 3. 交叉汇总(月份 × 科目)
# ============================================================

def cross_summarize(good):
    """
    双维度汇总。
    返回 (cross, subjects, months):
        cross    —— {(科目, 月份): 金额}
        subjects —— 出现过的科目,按首次出现排序
        months   —— 出现过的月份,已排序
    """
    cross = {}
    subjects = []
    months = []

    for row in good:
        month = row["日期"][:7]
        subject = row["科目"]

        key = (subject, month)           # 元组当键,一键携带两个维度
        cross[key] = cross.get(key, 0) + row["金额"]

        if subject not in subjects:
            subjects.append(subject)
        if month not in months:
            months.append(month)

    months.sort()
    return cross, subjects, months


def report_cross(cross, subjects, months):
    print("\n" + "=" * 50)
    print("三、月份 × 科目 交叉汇总")
    print("=" * 50)

    # 屏幕打印
    print(f"  {'科目':<10}", end="")
    for m in months:
        print(f"{m:>12}", end="")
    print()

    for s in subjects:
        print(f"  {s:<10}", end="")
        for m in months:
            print(f"{cross.get((s, m), 0):>12,.0f}", end="")
        print()

    # 写文件
    headers = ["科目"] + months + ["合计"]
    rows = []
    for s in subjects:
        line = [s]
        total = 0
        for m in months:
            amount = cross.get((s, m), 0)
            line.append(amount)
            total += amount
        line.append(total)
        rows.append(line)

    write_csv("cross.csv", headers, rows)


# ============================================================
# 4. 异常检测(按科目分别设阈值)
# ============================================================

def find_outliers(good, times=OUTLIER_TIMES):
    """
    按科目分别计算均值,标记超过 times 倍的记录。
    返回 (outliers, averages)。
    """
    # 第一遍:按科目累计
    totals = {}
    for row in good:
        s = row["科目"]
        if s not in totals:
            totals[s] = {"合计": 0, "笔数": 0}
        totals[s]["合计"] += row["金额"]
        totals[s]["笔数"] += 1

    averages = {}
    for s, d in totals.items():
        averages[s] = d["合计"] / d["笔数"]

    # 第二遍:逐条比对
    outliers = []
    for row in good:
        s = row["科目"]
        avg = averages[s]

        # 均值非正时,倍数比较没有意义,跳过
        if avg <= 0:
            continue

        if row["金额"] > avg * times:
            outliers.append({
                "凭证号": row["凭证号"],
                "科目": s,
                "金额": row["金额"],
                "科目均值": avg,
                "超出倍数": row["金额"] / avg,
            })

    return outliers, averages


def report_outliers(outliers, averages, times=OUTLIER_TIMES):
    print("\n" + "=" * 50)
    print(f"四、异常检测(阈值:本科目均值的 {times} 倍)")
    print("=" * 50)

    print("  各科目均值:")
    for s, avg in averages.items():
        print(f"    {s}:{avg:,.2f}")

    print()
    if outliers:
        for o in outliers:
            print(f"  [异常] {o['凭证号']} {o['科目']} {o['金额']:,.0f}"
                  f"(均值 {o['科目均值']:,.0f},超出 {o['超出倍数']:.1f} 倍)")
    else:
        print("  未发现异常")

    rows = [
        [o["凭证号"], o["科目"], o["金额"],
         round(o["科目均值"], 2), round(o["超出倍数"], 2)]
        for o in outliers
    ]
    write_csv("outliers.csv",
              ["凭证号", "科目", "金额", "科目均值", "超出倍数"],
              rows)


# ============================================================
# 5. 银行对账
# ============================================================

def reconcile(good, bank_rows):
    """
    与银行流水逐笔比对。
    返回差异列表,每项为 [凭证号, 差异类型, 账簿金额, 银行金额, 差额]。
    """
    # 两边都转成 {凭证号: 金额},配对与顺序、笔数无关
    bank = {}
    for row in bank_rows:
        bank[row["凭证号"]] = float(row["金额"])

    book = {}
    for row in good:
        book[row["凭证号"]] = row["金额"]

    diffs = []

    # 第一遍:以账簿为准
    for code in book:
        if code not in bank:
            diffs.append([code, "只在账簿有", book[code], "", ""])
        elif book[code] != bank[code]:
            diffs.append([code, "金额不一致", book[code], bank[code],
                          book[code] - bank[code]])

    # 第二遍:找银行独有的
    for code in bank:
        if code not in book:
            diffs.append([code, "只在银行有", "", bank[code], ""])

    return diffs


def report_reconcile(diffs):
    print("\n" + "=" * 50)
    print("五、银行对账")
    print("=" * 50)

    if diffs:
        for d in diffs:
            print(f"  [{d[1]}] {d[0]}  账簿 {d[2]}  银行 {d[3]}")
    else:
        print("  账实相符,无差异")

    print(f"\n  共 {len(diffs)} 处差异")

    write_csv("diff.csv",
              ["凭证号", "差异类型", "账簿金额", "银行金额", "差额"],
              diffs)


# ============================================================
# 主流程
# ============================================================

def main():
    # 读文件——只读一次
    rows = read_csv(DATA_FILE)
    bank_rows = read_csv(BANK_FILE)

    # 1. 校验,拿到干净数据
    good, problems = validate(rows)
    report_validation(rows, good, problems)

    # 后面所有分析都基于 good
    monthly = summarize_by_month(good)
    report_monthly(monthly)

    cross, subjects, months = cross_summarize(good)
    report_cross(cross, subjects, months)

    outliers, averages = find_outliers(good)
    report_outliers(outliers, averages)

    diffs = reconcile(good, bank_rows)
    report_reconcile(diffs)

    print("\n" + "=" * 50)
    print("全部报表已生成")
    print("=" * 50)


if __name__ == "__main__":
    main()
