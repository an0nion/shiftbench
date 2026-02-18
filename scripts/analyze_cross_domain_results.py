#!/usr/bin/env python3
"""
Quick analysis script for cross-domain benchmark results.

Prints key findings to console in a readable format.

Usage:
    python scripts/analyze_cross_domain_results.py results/cross_domain/
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def load_results(input_dir: Path) -> dict:
    """Load all result CSVs."""
    files = {
        "raw": "cross_domain_raw_results.csv",
        "summary": "cross_domain_summary.csv",
        "by_dataset": "cross_domain_by_dataset.csv",
        "by_method": "cross_domain_by_method.csv",
        "difficulty": "cross_domain_difficulty.csv",
        "analysis": "cross_domain_statistical_analysis.csv",
    }

    results = {}
    for key, filename in files.items():
        filepath = input_dir / filename
        if filepath.exists():
            results[key] = pd.read_csv(filepath)
        else:
            print(f"Warning: {filename} not found")

    return results


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_subheader(title: str):
    """Print formatted subsection header."""
    print(f"\n{title}")
    print("-" * len(title))


def analyze_domain_difficulty(df_difficulty: pd.DataFrame):
    """Analyze and print domain difficulty ranking."""
    print_header("FINDING 1: Domain Difficulty Hierarchy")

    if df_difficulty is None or len(df_difficulty) == 0:
        print("No data available.")
        return

    # Sort by certification rate (ascending = hardest first)
    df = df_difficulty.reset_index()
    df = df.sort_values("cert_rate")

    print("\nDomains ranked by difficulty (hardest → easiest):")
    print()

    for i, row in df.iterrows():
        domain = row["domain"]
        cert_rate = row["cert_rate"] * 100
        n_decisions = row["n_decisions"]

        difficulty = "Very Hard" if cert_rate < 10 else \
                    "Hard" if cert_rate < 30 else \
                    "Moderate" if cert_rate < 60 else \
                    "Easy"

        print(f"  {i+1}. {domain.upper():<12} {cert_rate:>5.1f}% certified  "
              f"({n_decisions} decisions)  [{difficulty}]")

    print("\nInterpretation:")
    hardest = df.iloc[0]["domain"]
    easiest = df.iloc[-1]["domain"]
    print(f"  - Hardest domain: {hardest} (lowest certification rate)")
    print(f"  - Easiest domain: {easiest} (highest certification rate)")


def analyze_method_performance(df_summary: pd.DataFrame):
    """Analyze method performance across domains."""
    print_header("FINDING 2: Method Performance Across Domains")

    if df_summary is None or len(df_summary) == 0:
        print("No data available.")
        return

    df = df_summary.reset_index()

    print("\nCertification rates by domain and method:")
    print()

    # Pivot table
    pivot = df.pivot(index="method", columns="domain", values="cert_rate_%")
    print(pivot.to_string())

    print("\n\nMethod ranking by domain:")
    for domain in pivot.columns:
        sorted_methods = pivot[domain].sort_values(ascending=False)
        print(f"\n  {domain.upper()}:")
        for i, (method, rate) in enumerate(sorted_methods.items(), 1):
            print(f"    {i}. {method:<20} {rate:>6.1f}%")


def analyze_ravel_gating(df_raw: pd.DataFrame):
    """Analyze RAVEL's abstention behavior."""
    print_header("FINDING 3: RAVEL's Gating Advantage")

    if df_raw is None or len(df_raw) == 0:
        print("No data available.")
        return

    # Filter for RAVEL
    df_ravel = df_raw[df_raw["method"] == "ravel"]

    if len(df_ravel) == 0:
        print("RAVEL not found in results (may not be installed).")
        return

    print("\nRAVEL abstention rates by domain:")
    print()

    for domain in df_ravel["domain"].unique():
        df_domain = df_ravel[df_ravel["domain"] == domain]

        n_total = len(df_domain)
        n_certify = (df_domain["decision"] == "CERTIFY").sum()
        n_abstain = (df_domain["decision"] == "ABSTAIN").sum()
        n_no_guarantee = (df_domain["decision"] == "NO-GUARANTEE").sum()

        cert_rate = n_certify / n_total * 100
        abstain_rate = n_abstain / n_total * 100
        no_guarantee_rate = n_no_guarantee / n_total * 100

        print(f"  {domain.upper()}:")
        print(f"    CERTIFY:       {n_certify:>4} / {n_total:<4} ({cert_rate:>5.1f}%)")
        print(f"    ABSTAIN:       {n_abstain:>4} / {n_total:<4} ({abstain_rate:>5.1f}%)")
        print(f"    NO-GUARANTEE:  {n_no_guarantee:>4} / {n_total:<4} ({no_guarantee_rate:>5.1f}%)")
        print()

    print("Interpretation:")
    print("  - High NO-GUARANTEE rate → Weights unstable (PSIS/ESS gates failing)")
    print("  - High ABSTAIN rate → Insufficient evidence (low effective sample size)")
    print("  - High CERTIFY rate → Stable weights, sufficient evidence")


def analyze_shift_severity(df_raw: pd.DataFrame):
    """Analyze shift severity indicators."""
    print_header("FINDING 4: Shift Severity Indicators")

    if df_raw is None or len(df_raw) == 0:
        print("No data available.")
        return

    # Remove invalid n_eff
    df = df_raw[df_raw["n_eff"] > 0]

    print("\nEffective Sample Size (ESS) by domain:")
    print()

    ess_stats = df.groupby("domain")["n_eff"].agg(["mean", "median", "std", "min", "max"])

    print(ess_stats.to_string())

    print("\n\nInterpretation:")
    print("  - Higher ESS → Less severe shift (weights closer to uniform)")
    print("  - Lower ESS → More severe shift (weights have high variance)")
    print("  - ESS ≈ N → Minimal shift")
    print("  - ESS << N → Severe shift")


def analyze_hardest_datasets(df_by_dataset: pd.DataFrame):
    """Find hardest and easiest datasets."""
    print_header("FINDING 5: Hardest/Easiest Datasets")

    if df_by_dataset is None or len(df_by_dataset) == 0:
        print("No data available.")
        return

    df = df_by_dataset.reset_index()

    # Average across methods
    dataset_avg = df.groupby(["dataset", "domain"])["cert_rate_%"].mean().reset_index()
    dataset_avg = dataset_avg.sort_values("cert_rate_%")

    print("\nHARDEST datasets (lowest certification rate):")
    print()
    for i, row in dataset_avg.head(5).iterrows():
        print(f"  {i+1}. {row['dataset']:<20} ({row['domain']:<10}) {row['cert_rate_%']:>5.1f}%")

    print("\n\nEASIEST datasets (highest certification rate):")
    print()
    for i, row in dataset_avg.tail(5).iterrows():
        print(f"  {i+1}. {row['dataset']:<20} ({row['domain']:<10}) {row['cert_rate_%']:>5.1f}%")


def analyze_statistical_tests(df_analysis: pd.DataFrame):
    """Summarize statistical test results."""
    print_header("Statistical Test Results")

    if df_analysis is None or len(df_analysis) == 0:
        print("No statistical analysis available.")
        return

    print()
    for _, row in df_analysis.iterrows():
        test_name = row["test"]
        p_value = row.get("p_value", None)
        significant = row.get("significant", None)
        interpretation = row.get("interpretation", "")

        print(f"\nTest: {test_name}")

        if p_value is not None and not pd.isna(p_value):
            print(f"  p-value: {p_value:.4f}")
            print(f"  Significant: {'Yes (p < 0.05)' if significant else 'No'}")

        if interpretation:
            print(f"  Result: {interpretation}")


def analyze_runtime(df_summary: pd.DataFrame):
    """Analyze runtime by domain and method."""
    print_header("Runtime Analysis")

    if df_summary is None or len(df_summary) == 0:
        print("No data available.")
        return

    df = df_summary.reset_index()

    print("\nMean runtime (seconds) by domain and method:")
    print()

    pivot = df.pivot(index="method", columns="domain", values="mean_runtime_sec")
    print(pivot.to_string())

    print("\n\nInterpretation:")
    print("  - Higher runtime in molecular domain suggests more cohorts/complexity")
    print("  - RAVEL typically slower than uLSIF/KLIEP due to gating overhead")


def print_summary_statistics(results: dict):
    """Print overall summary statistics."""
    print_header("Overall Summary Statistics")

    df_raw = results.get("raw")

    if df_raw is None:
        print("No raw results available.")
        return

    n_datasets = df_raw["dataset"].nunique()
    n_methods = df_raw["method"].nunique()
    n_cohorts = df_raw["cohort_id"].nunique()
    n_total = len(df_raw)

    n_certify = (df_raw["decision"] == "CERTIFY").sum()
    n_abstain = (df_raw["decision"] == "ABSTAIN").sum()
    n_no_guarantee = (df_raw["decision"] == "NO-GUARANTEE").sum()

    print(f"\nDatasets evaluated:     {n_datasets}")
    print(f"Methods compared:       {n_methods}")
    print(f"Unique cohorts:         {n_cohorts}")
    print(f"Total decisions:        {n_total}")
    print()
    print(f"Decision breakdown:")
    print(f"  CERTIFY:       {n_certify:>6} ({n_certify/n_total*100:>5.1f}%)")
    print(f"  ABSTAIN:       {n_abstain:>6} ({n_abstain/n_total*100:>5.1f}%)")
    print(f"  NO-GUARANTEE:  {n_no_guarantee:>6} ({n_no_guarantee/n_total*100:>5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cross-domain benchmark results"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing cross-domain results"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        sys.exit(1)

    print(f"Loading results from: {input_dir}")

    results = load_results(input_dir)

    if not results:
        print("Error: No results found!")
        sys.exit(1)

    # Print analyses
    print_summary_statistics(results)

    if "difficulty" in results:
        analyze_domain_difficulty(results["difficulty"])

    if "summary" in results:
        analyze_method_performance(results["summary"])

    if "raw" in results:
        analyze_ravel_gating(results["raw"])

    if "raw" in results:
        analyze_shift_severity(results["raw"])

    if "by_dataset" in results:
        analyze_hardest_datasets(results["by_dataset"])

    if "analysis" in results:
        analyze_statistical_tests(results["analysis"])

    if "summary" in results:
        analyze_runtime(results["summary"])

    # Final summary
    print_header("Next Steps")
    print("\n1. Review findings above and compare to predictions in docs/CROSS_DOMAIN_ANALYSIS.md")
    print("2. Update analysis document with actual results")
    print("3. Generate plots: python scripts/plot_cross_domain.py --input", input_dir)
    print("4. Write discussion for paper/report")
    print()


if __name__ == "__main__":
    main()
