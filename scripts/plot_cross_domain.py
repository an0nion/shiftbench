#!/usr/bin/env python3
"""
Cross-Domain Visualization Script for ShiftBench

Generates publication-quality figures for cross-domain analysis:
1. Certification rate by domain (bar chart)
2. Method ranking by domain (heatmap)
3. Runtime by domain (scatter/bar)
4. Decision distribution (stacked bar)
5. Domain difficulty comparison

Usage:
    python scripts/plot_cross_domain.py --input results/cross_domain/
    python scripts/plot_cross_domain.py --input results/cross_domain/ --format pdf
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication-quality defaults
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

# Color palette
COLORS = {
    "molecular": "#1f77b4",  # blue
    "text": "#ff7f0e",       # orange
    "tabular": "#2ca02c",    # green
}

METHOD_COLORS = {
    "ulsif": "#d62728",      # red
    "kliep": "#9467bd",      # purple
    "kmm": "#8c564b",        # brown
    "rulsif": "#e377c2",     # pink
    "ravel": "#7f7f7f",      # gray
    "weighted_conformal": "#bcbd22",  # olive
}


def load_results(input_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load all result CSVs from input directory."""
    results = {}

    files = {
        "raw": "cross_domain_raw_results.csv",
        "summary": "cross_domain_summary.csv",
        "by_dataset": "cross_domain_by_dataset.csv",
        "by_method": "cross_domain_by_method.csv",
        "difficulty": "cross_domain_difficulty.csv",
        "runtime": "cross_domain_runtime.csv",
        "decision_dist": "cross_domain_decision_distribution.csv",
    }

    for key, filename in files.items():
        filepath = input_dir / filename
        if filepath.exists():
            results[key] = pd.read_csv(filepath)
        else:
            print(f"Warning: {filename} not found in {input_dir}")

    return results


def plot_certification_rate_by_domain(
    df_summary: pd.DataFrame,
    output_file: Path
):
    """
    Bar chart: Certification rate by domain and method.

    X-axis: Domain
    Y-axis: Certification rate (%)
    Bars: Methods (grouped)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Reset index to get domain and method as columns
    df = df_summary.reset_index()

    # Pivot for grouped bar chart
    pivot = df.pivot(index="domain", columns="method", values="cert_rate_%")

    # Plot
    pivot.plot(kind="bar", ax=ax, width=0.8, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Domain", fontweight="bold")
    ax.set_ylabel("Certification Rate (%)", fontweight="bold")
    ax.set_title("Certification Rate by Domain and Method", fontweight="bold")
    ax.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_file}")


def plot_method_ranking_heatmap(
    df_by_method: pd.DataFrame,
    output_file: Path
):
    """
    Heatmap: Method ranking by domain.

    X-axis: Domain
    Y-axis: Method
    Color: Certification rate (%)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Reset index and pivot
    df = df_by_method.reset_index()
    pivot = df.pivot(index="method", columns="domain", values="cert_rate_%")

    # Plot heatmap
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="YlGnBu",
        cbar_kws={"label": "Certification Rate (%)"},
        linewidths=0.5,
        linecolor="gray",
        ax=ax
    )

    ax.set_xlabel("Domain", fontweight="bold")
    ax.set_ylabel("Method", fontweight="bold")
    ax.set_title("Method Performance Across Domains (Certification Rate %)", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_file}")


def plot_runtime_by_domain(
    df_by_method: pd.DataFrame,
    output_file: Path
):
    """
    Bar chart: Runtime by domain and method.

    X-axis: Domain
    Y-axis: Mean runtime (seconds)
    Bars: Methods (grouped)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Reset index and pivot
    df = df_by_method.reset_index()
    pivot = df.pivot(index="domain", columns="method", values="mean_runtime_sec")

    # Plot
    pivot.plot(kind="bar", ax=ax, width=0.8, edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Domain", fontweight="bold")
    ax.set_ylabel("Mean Runtime (seconds)", fontweight="bold")
    ax.set_title("Runtime by Domain and Method", fontweight="bold")
    ax.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_yscale("log")  # Log scale for better visibility

    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_file}")


def plot_decision_distribution(
    df_raw: pd.DataFrame,
    output_file: Path
):
    """
    Stacked bar chart: Decision distribution by domain.

    X-axis: Domain
    Y-axis: Percentage of decisions
    Stacks: CERTIFY, ABSTAIN, NO-GUARANTEE
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Count decisions by domain
    decision_counts = pd.crosstab(
        df_raw["domain"],
        df_raw["decision"],
        normalize="index"
    ) * 100  # Convert to percentage

    # Ensure all decision types exist
    for decision in ["CERTIFY", "ABSTAIN", "NO-GUARANTEE"]:
        if decision not in decision_counts.columns:
            decision_counts[decision] = 0

    # Reorder columns
    decision_counts = decision_counts[["CERTIFY", "ABSTAIN", "NO-GUARANTEE"]]

    # Plot
    decision_counts.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=["#2ca02c", "#ff7f0e", "#d62728"],  # green, orange, red
        edgecolor="black",
        linewidth=0.5
    )

    ax.set_xlabel("Domain", fontweight="bold")
    ax.set_ylabel("Percentage of Decisions (%)", fontweight="bold")
    ax.set_title("Decision Distribution by Domain", fontweight="bold")
    ax.legend(title="Decision", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Rotate x-axis labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_file}")


def plot_domain_difficulty(
    df_difficulty: pd.DataFrame,
    output_file: Path
):
    """
    Bar chart: Domain difficulty (by certification rate).

    X-axis: Domain (sorted by difficulty)
    Y-axis: Certification rate (%)
    Error bars: Standard deviation
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Reset index
    df = df_difficulty.reset_index()

    # Sort by certification rate (ascending = hardest first)
    df = df.sort_values("cert_rate")

    # Convert to percentage
    df["cert_rate_pct"] = df["cert_rate"] * 100

    # Plot
    ax.bar(
        df["domain"],
        df["cert_rate_pct"],
        color=[COLORS.get(d, "#cccccc") for d in df["domain"]],
        edgecolor="black",
        linewidth=1.5
    )

    ax.set_xlabel("Domain (sorted by difficulty)", fontweight="bold")
    ax.set_ylabel("Certification Rate (%)", fontweight="bold")
    ax.set_title("Domain Difficulty Ranking", fontweight="bold")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    # Add value labels on bars
    for i, (domain, rate) in enumerate(zip(df["domain"], df["cert_rate_pct"])):
        ax.text(i, rate + 1, f"{rate:.1f}%", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_file}")


def plot_effective_sample_size(
    df_raw: pd.DataFrame,
    output_file: Path
):
    """
    Box plot: Effective sample size distribution by domain.

    X-axis: Domain
    Y-axis: Effective sample size (log scale)
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Remove invalid values
    df = df_raw[df_raw["n_eff"] > 0].copy()

    # Box plot
    domains = df["domain"].unique()
    data = [df[df["domain"] == d]["n_eff"].values for d in domains]

    bp = ax.boxplot(
        data,
        labels=domains,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="red", linewidth=2)
    )

    # Color boxes by domain
    for patch, domain in zip(bp["boxes"], domains):
        patch.set_facecolor(COLORS.get(domain, "#cccccc"))
        patch.set_alpha(0.7)

    ax.set_xlabel("Domain", fontweight="bold")
    ax.set_ylabel("Effective Sample Size (log scale)", fontweight="bold")
    ax.set_title("Effective Sample Size Distribution by Domain", fontweight="bold")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_file}")


def plot_method_comparison_scatter(
    df_raw: pd.DataFrame,
    output_file: Path
):
    """
    Scatter plot: Method comparison (mu_hat vs lower_bound).

    X-axis: mu_hat (point estimate)
    Y-axis: lower_bound
    Color: Method
    Facets: Domain
    """
    # Filter certified decisions only
    df = df_raw[df_raw["decision"] == "CERTIFY"].copy()

    if len(df) == 0:
        print("Warning: No certified decisions found - skipping scatter plot")
        return

    # Create faceted plot
    domains = df["domain"].unique()
    n_domains = len(domains)

    fig, axes = plt.subplots(1, n_domains, figsize=(6 * n_domains, 5), sharey=True)

    if n_domains == 1:
        axes = [axes]

    for ax, domain in zip(axes, domains):
        df_domain = df[df["domain"] == domain]

        for method in df_domain["method"].unique():
            df_method = df_domain[df_domain["method"] == method]

            ax.scatter(
                df_method["mu_hat"],
                df_method["lower_bound"],
                label=method,
                color=METHOD_COLORS.get(method, "#cccccc"),
                alpha=0.6,
                s=50,
                edgecolors="black",
                linewidths=0.5
            )

        # Plot diagonal line (mu_hat = lower_bound)
        lim = [0, 1]
        ax.plot(lim, lim, 'k--', alpha=0.3, zorder=0)

        ax.set_xlabel("Point Estimate (mu_hat)", fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Lower Bound", fontweight="bold")
        ax.set_title(f"Domain: {domain}", fontweight="bold")
        ax.legend()
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_file}")


def generate_all_plots(input_dir: Path, output_dir: Path, fmt: str = "png"):
    """Generate all cross-domain visualization plots."""
    print(f"Loading results from {input_dir}...")
    results = load_results(input_dir)

    if not results:
        print("Error: No results found!")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating plots (format: {fmt})...")

    # 1. Certification rate by domain
    if "summary" in results:
        plot_certification_rate_by_domain(
            results["summary"],
            output_dir / f"cert_rate_by_domain.{fmt}"
        )

    # 2. Method ranking heatmap
    if "by_method" in results:
        plot_method_ranking_heatmap(
            results["by_method"],
            output_dir / f"method_ranking_heatmap.{fmt}"
        )

    # 3. Runtime by domain
    if "by_method" in results:
        plot_runtime_by_domain(
            results["by_method"],
            output_dir / f"runtime_by_domain.{fmt}"
        )

    # 4. Decision distribution
    if "raw" in results:
        plot_decision_distribution(
            results["raw"],
            output_dir / f"decision_distribution.{fmt}"
        )

    # 5. Domain difficulty
    if "difficulty" in results:
        plot_domain_difficulty(
            results["difficulty"],
            output_dir / f"domain_difficulty.{fmt}"
        )

    # 6. Effective sample size
    if "raw" in results:
        plot_effective_sample_size(
            results["raw"],
            output_dir / f"effective_sample_size.{fmt}"
        )

    # 7. Method comparison scatter
    if "raw" in results:
        plot_method_comparison_scatter(
            results["raw"],
            output_dir / f"method_comparison_scatter.{fmt}"
        )

    print(f"\nAll plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate cross-domain visualizations for ShiftBench",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input directory containing cross-domain results"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for plots (default: <input>/plots/)"
    )

    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format for plots"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else input_dir / "plots"

    generate_all_plots(input_dir, output_dir, args.format)


if __name__ == "__main__":
    main()
