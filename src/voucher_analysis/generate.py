"""Generate a synthetic general ledger and bank statement.

The data is made up. Company, supplier and customer names are invented.
Known anomalies are injected on purpose and written to ground truth files,
so the tests can be measured against them.
"""

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
