"""
Extended cross-domain benchmark: uLSIF on all 24 registered datasets.

Extends run_cross_domain_benchmark.py with muv, molhiv, student_performance.

Output: results/cross_domain_extended/
"""
import sys
from pathlib import Path

shift_bench = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(shift_bench / "src"))
sys.path.insert(0, str(shift_bench / "scripts"))

import run_cross_domain_benchmark as rb

# Extend DOMAIN_DATASETS with new datasets
rb.DOMAIN_DATASETS["molecular"] = [
    "bace", "bbbp", "clintox", "esol", "freesolv", "lipophilicity",
    "sider", "tox21", "toxcast", "molhiv", "muv",
]
rb.DOMAIN_DATASETS["tabular"] = [
    "adult", "compas", "bank", "german_credit",
    "heart_disease", "diabetes", "student_performance",
]
rb.DOMAIN_DATASETS["text"] = [
    "imdb", "yelp", "amazon", "civil_comments", "twitter",
]
rb.NAN_LABEL_DATASETS = {"tox21", "toxcast", "muv"}

# Inject argv so argparse picks up our args
sys.argv = [
    "run_extended_benchmark.py",
    "--methods", "ulsif",
    "--domains", "molecular,tabular,text",
    "--output", "results/cross_domain_extended",
    "--tau", "0.5,0.6,0.7,0.8,0.9",
    "--alpha", "0.05",
]

if __name__ == "__main__":
    rb.main()
