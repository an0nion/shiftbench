#!/bin/bash
# Demo: Quick cross-domain evaluation on subset of datasets
# This runs a smaller benchmark for testing (takes ~5-10 minutes)

set -e

echo "=========================================="
echo "Cross-Domain Benchmark - Quick Demo"
echo "=========================================="
echo ""

# Configuration
OUTPUT_DIR="results/cross_domain_demo"
METHODS="ulsif,kliep"
DOMAINS="molecular,text,tabular"

# Note: We'll test on smaller subset for speed
# Molecular: bace, bbbp (2 datasets)
# Text: test_dataset (synthetic for speed)
# Tabular: adult (1 dataset)

echo "Configuration:"
echo "  Methods: $METHODS"
echo "  Domains: $DOMAINS"
echo "  Output: $OUTPUT_DIR"
echo ""

# Run benchmark
echo "Step 1: Running cross-domain benchmark..."
python scripts/run_cross_domain_benchmark.py \
    --methods "$METHODS" \
    --domains "$DOMAINS" \
    --output "$OUTPUT_DIR" \
    --tau 0.5,0.7,0.9 \
    --alpha 0.05

echo ""
echo "Step 2: Generating visualizations..."
python scripts/plot_cross_domain.py \
    --input "$OUTPUT_DIR" \
    --output "$OUTPUT_DIR/plots" \
    --format png

echo ""
echo "=========================================="
echo "Demo Complete!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - cross_domain_raw_results.csv"
echo "  - cross_domain_summary.csv"
echo "  - cross_domain_by_dataset.csv"
echo "  - cross_domain_by_method.csv"
echo "  - cross_domain_statistical_analysis.csv"
echo ""
echo "Plots saved to: $OUTPUT_DIR/plots/"
echo ""
