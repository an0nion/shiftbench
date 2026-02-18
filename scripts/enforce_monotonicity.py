"""
Tau Monotonicity Enforcement (Downward Closure)
================================================

Post-processes certification results to enforce logical nesting:
  If cohort certified at tau_high, auto-certify all tau_low < tau_high.

Rationale: PPV >= 0.9 implies PPV >= 0.8 (monotonicity).
Non-monotone certifications are artifacts of discrete testing
and should be corrected as a protocol invariant.

Usage:
  python scripts/enforce_monotonicity.py results/some_results.csv
  python scripts/enforce_monotonicity.py results/ --recursive
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def enforce_tau_monotonicity(
    df: pd.DataFrame,
    cohort_col: str = "cohort_id",
    tau_col: str = "tau",
    decision_col: str = "decision",
) -> tuple:
    """
    Enforce downward closure on certification decisions.

    For each cohort:
      1. Find max tau where CERTIFY
      2. Auto-certify all tau < tau_max

    Returns: (corrected_df, n_corrections)
    """
    result = df.copy()
    n_corrections = 0

    if cohort_col not in result.columns or tau_col not in result.columns:
        print(f"  Warning: Required columns '{cohort_col}' or '{tau_col}' not found. Skipping.")
        return result, 0

    if decision_col not in result.columns:
        print(f"  Warning: Decision column '{decision_col}' not found. Skipping.")
        return result, 0

    for cohort in result[cohort_col].unique():
        cohort_mask = result[cohort_col] == cohort
        certified_mask = cohort_mask & (result[decision_col] == "CERTIFY")
        certified_taus = result.loc[certified_mask, tau_col]

        if len(certified_taus) == 0:
            continue

        tau_max = certified_taus.max()

        # Find rows that should be certified but aren't (downward closure)
        lower_mask = cohort_mask & (result[tau_col] < tau_max) & (result[decision_col] != "CERTIFY")
        n_fixes = lower_mask.sum()

        if n_fixes > 0:
            result.loc[lower_mask, decision_col] = "CERTIFY"
            n_corrections += n_fixes

    return result, n_corrections


def check_monotonicity_violations(
    df: pd.DataFrame,
    cohort_col: str = "cohort_id",
    tau_col: str = "tau",
    decision_col: str = "decision",
) -> pd.DataFrame:
    """
    Check for monotonicity violations without fixing them.

    Returns DataFrame of violations (certified at tau_high but not tau_low).
    """
    violations = []

    if cohort_col not in df.columns or tau_col not in df.columns:
        return pd.DataFrame()

    for cohort in df[cohort_col].unique():
        cohort_mask = df[cohort_col] == cohort
        cohort_df = df[cohort_mask].sort_values(tau_col)

        certified_taus = cohort_df.loc[cohort_df[decision_col] == "CERTIFY", tau_col].values
        not_certified_taus = cohort_df.loc[cohort_df[decision_col] != "CERTIFY", tau_col].values

        if len(certified_taus) == 0:
            continue

        tau_max = certified_taus.max()

        for tau_low in not_certified_taus:
            if tau_low < tau_max:
                violations.append({
                    "cohort_id": cohort,
                    "tau_not_certified": tau_low,
                    "tau_certified_above": tau_max,
                    "violation": f"Certified at tau={tau_max:.2f} but NOT at tau={tau_low:.2f}",
                })

    return pd.DataFrame(violations)


def process_file(filepath: str, dry_run: bool = False) -> dict:
    """
    Process a single CSV file for monotonicity violations.

    Returns dict with file stats.
    """
    path = Path(filepath)
    if not path.exists() or path.suffix != ".csv":
        return {"file": str(path), "status": "skipped", "reason": "not a CSV"}

    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return {"file": str(path), "status": "error", "reason": str(e)}

    # Try common column name patterns
    cohort_col = None
    tau_col = None
    decision_col = None

    for col in df.columns:
        cl = col.lower()
        if "cohort" in cl and "id" in cl:
            cohort_col = col
        elif cl == "tau":
            tau_col = col
        elif "decision" in cl:
            decision_col = col

    if not all([cohort_col, tau_col, decision_col]):
        return {"file": str(path), "status": "skipped",
                "reason": f"missing columns (need cohort_id, tau, decision)"}

    # Check violations
    violations = check_monotonicity_violations(df, cohort_col, tau_col, decision_col)
    n_violations = len(violations)

    if n_violations == 0:
        return {"file": str(path), "status": "clean", "violations": 0, "corrections": 0}

    if dry_run:
        return {"file": str(path), "status": "violations_found",
                "violations": n_violations, "corrections": 0}

    # Fix violations
    fixed_df, n_corrections = enforce_tau_monotonicity(df, cohort_col, tau_col, decision_col)

    # Save corrected file
    backup_path = str(path) + ".bak"
    df.to_csv(backup_path, index=False)
    fixed_df.to_csv(filepath, index=False)

    return {"file": str(path), "status": "fixed",
            "violations": n_violations, "corrections": n_corrections,
            "backup": backup_path}


def main():
    parser = argparse.ArgumentParser(
        description="Enforce tau monotonicity (downward closure) on certification results"
    )
    parser.add_argument("path", help="CSV file or directory to process")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively process all CSV files in directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Check for violations without fixing")
    parser.add_argument("--report", default=None,
                        help="Save violation report to file")
    args = parser.parse_args()

    path = Path(args.path)

    if path.is_file():
        files = [str(path)]
    elif path.is_dir():
        if args.recursive:
            files = sorted(str(f) for f in path.rglob("*.csv"))
        else:
            files = sorted(str(f) for f in path.glob("*.csv"))
    else:
        print(f"Error: {path} does not exist")
        sys.exit(1)

    print(f"Processing {len(files)} file(s)...")
    if args.dry_run:
        print("(DRY RUN - no changes will be made)")
    print()

    results = []
    total_violations = 0
    total_corrections = 0

    for filepath in files:
        result = process_file(filepath, dry_run=args.dry_run)
        results.append(result)

        v = result.get("violations", 0)
        c = result.get("corrections", 0)
        total_violations += v
        total_corrections += c

        status = result["status"]
        if status == "clean":
            print(f"  [OK] {filepath}: no violations")
        elif status == "fixed":
            print(f"  [FIXED] {filepath}: {v} violations -> {c} corrections (backup saved)")
        elif status == "violations_found":
            print(f"  [WARN] {filepath}: {v} violations found (dry run)")
        elif status == "skipped":
            print(f"  [SKIP] {filepath}: {result.get('reason', '')}")
        elif status == "error":
            print(f"  [ERR] {filepath}: {result.get('reason', '')}")

    print(f"\n{'='*50}")
    print(f"Total files processed: {len(results)}")
    print(f"Total violations found: {total_violations}")
    print(f"Total corrections made: {total_corrections}")

    if args.dry_run and total_violations > 0:
        print(f"\nRun without --dry-run to apply fixes.")

    if args.report:
        report_df = pd.DataFrame(results)
        report_df.to_csv(args.report, index=False)
        print(f"\nReport saved to {args.report}")


if __name__ == "__main__":
    main()
