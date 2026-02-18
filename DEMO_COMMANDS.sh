#!/bin/bash
# ShiftBench Evaluation Harness - Demo Commands
# These commands demonstrate the complete functionality

echo "=========================================="
echo "ShiftBench Evaluation Harness Demo"
echo "=========================================="
echo ""

# Navigate to source directory
cd src

echo "1. List available datasets:"
echo "   python -m shiftbench.evaluate --method ulsif --dataset list"
echo ""

echo "2. Evaluate uLSIF on test dataset (synthetic):"
echo "   python -m shiftbench.evaluate --method ulsif --dataset test_dataset --output ../results/demo/"
echo ""

echo "3. Evaluate uLSIF on BACE dataset (real molecular data):"
echo "   python -m shiftbench.evaluate --method ulsif --dataset bace --output ../results/demo/"
echo ""

echo "4. Custom tau grid:"
echo "   python -m shiftbench.evaluate --method ulsif --dataset bace --tau 0.5,0.7,0.9"
echo ""

echo "5. Verbose logging:"
echo "   python -m shiftbench.evaluate --method ulsif --dataset bace --verbose"
echo ""

echo "6. Batch processing (Python API):"
echo "   python -c \""
echo "   from shiftbench.evaluate import evaluate_batch"
echo "   from pathlib import Path"
echo "   results, metadata = evaluate_batch("
echo "       dataset_names=['test_dataset', 'bace', 'bbbp'],"
echo "       method_names=['ulsif'],"
echo "       output_dir=Path('../results/batch'),"
echo "       continue_on_error=True"
echo "   )"
echo "   \""
echo ""

echo "7. View results:"
echo "   cat ../results/demo/ulsif_bace_results.csv | head -10"
echo "   cat ../results/demo/aggregated_summary.csv"
echo ""

echo "=========================================="
echo "Output Files Generated:"
echo "=========================================="
echo "- {method}_{dataset}_results.csv - Per-cohort results"
echo "- all_results.csv - Combined results"
echo "- all_metadata.csv - Run metadata"
echo "- aggregated_summary.csv - Summary statistics"
echo ""

echo "=========================================="
echo "Documentation:"
echo "=========================================="
echo "- QUICK_START.md - Quick reference"
echo "- EVALUATION_HARNESS_SUMMARY.md - Full documentation"
echo "- IMPLEMENTATION_REPORT.md - Implementation details"
echo "- FINAL_SUMMARY.txt - Summary of all features"
echo ""
