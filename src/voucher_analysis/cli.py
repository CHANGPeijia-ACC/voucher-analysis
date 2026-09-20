"""Command line entry point.

Two commands: `generate` writes a synthetic data set, `run` tests a ledger
and reconciles it with a bank statement, producing one Excel workbook.
"""

import argparse
from pathlib import Path

import pandas as pd

from voucher_analysis import evaluate as ev
from voucher_analysis import generate, jet, report
from voucher_analysis import reconcile as rec
from voucher_analysis.loaders import load_config, read_bank, read_gl

WORKBOOK_NAME = "workpaper.xlsx"


def build_parser():
    """Describe the commands and their options."""
    parser = argparse.ArgumentParser(
        prog="python -m voucher_analysis",
        description="Journal entry testing and bank reconciliation on synthetic data.")
    commands = parser.add_subparsers(dest="command", required=True)

    maker = commands.add_parser("generate", help="write a synthetic ledger and bank statement")
    maker.add_argument("--out", default="data/sample", help="folder for the CSV files")
    maker.add_argument("--seed", type=int, default=generate.SEED,
                       help="random seed; the same seed gives the same data")
    maker.add_argument("--year", type=int, default=generate.YEAR, help="fiscal year")

    runner = commands.add_parser("run", help="run the tests and the reconciliation")
    runner.add_argument("--gl", required=True, help="general ledger CSV")
    runner.add_argument("--bank", required=True, help="bank statement CSV")
    runner.add_argument("--config", default="config.yaml", help="thresholds")
    runner.add_argument("--out", default="outputs", help="folder for the workbook")
    runner.add_argument("--ground-truth", dest="ground_truth",
                        help="optional list of known anomalies; adds the Evaluation sheet")
    return parser


def run_generate(args):
    """Write the synthetic files and report what was written."""
    tables = generate.generate_all(seed=args.seed, year=args.year)
    generate.write_outputs(tables, args.out)

    gl = tables["gl"]
    print(f"Wrote {len(tables)} files to {args.out}")
    print(f"  gl.csv                  {len(gl)} lines, {gl['voucher_no'].nunique()} vouchers")
    print(f"  bank.csv                {len(tables['bank'])} lines")
    print(f"  ground_truth.csv        {len(tables['ground_truth'])} vouchers with an anomaly")
    print(f"  bank_ground_truth.csv   {len(tables['bank_ground_truth'])} expected matches")
    return 0


def run_analysis(args):
    """Run every test, reconcile the bank, write the workbook.

    Returns 1 when the reconciliation statement does not tie. That is a
    self-check: every item must end up in exactly one place, so a
    difference means the result cannot be relied on.
    """
    config = load_config(args.config)
    gl = read_gl(args.gl)
    bank = read_bank(args.bank)

    flags = jet.run_all(gl, config)
    month_end = jet.month_end_summary(gl, config)
    result = rec.reconcile(gl, bank, config)
    evaluation = None
    if args.ground_truth:
        evaluation = ev.evaluate(flags, pd.read_csv(args.ground_truth))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = report.write_workbook(out_dir / WORKBOOK_NAME, gl, flags,
                                 month_end, result, evaluation)

    print_report(gl, bank, flags, result, evaluation)
    print(f"\nWorkbook: {path}")
    if not rec.statement_ties(result["statement"]):
        print("The bank reconciliation does not tie. Do not rely on this workbook.")
        return 1
    return 0


def print_report(gl, bank, flags, result, evaluation):
    """Print the same figures the workbook shows, for a quick look."""
    print(f"General ledger: {len(gl)} lines, {gl['voucher_no'].nunique()} vouchers")
    print(f"Bank statement: {len(bank)} lines\n")

    summary = report.summary_table(flags, gl)
    print(summary[["test_id", "lines_flagged", "vouchers_flagged",
                   "voucher_debit_flagged"]].to_string(index=False))

    print("\nBank reconciliation")
    for match_type, count in result["matches"]["match_type"].value_counts().items():
        print(f"  {match_type:<20} {count}")
    for item_type, count in result["unmatched"]["item_type"].value_counts().items():
        print(f"  {item_type:<20} {count}")
    ties = "yes" if rec.statement_ties(result["statement"]) else "no"
    print(f"  statement ties       {ties}")

    if evaluation is not None:
        overall = evaluation[evaluation["test_id"] == "overall"].iloc[0]
        print(f"\nAgainst ground truth: precision {overall['precision']}, "
              f"recall {overall['recall']}")


def main(argv=None):
    """Parse the arguments and run the chosen command. Returns the exit code."""
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        return run_generate(args)
    return run_analysis(args)
