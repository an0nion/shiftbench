"""
Extended cross-domain benchmark: all registered methods on all 35+ datasets.

Extends run_cross_domain_benchmark.py with all processed datasets and
all 10 baselines (ulsif, kliep, kmm, rulsif, weighted_conformal,
split_conformal, cvplus, group_dro, bbse, ravel).

Session 9 adds 12 new datasets:
  text:     ag_news, dbpedia, imdb_genre, sst2
  tabular:  wine_quality, online_shoppers, communities_crime, mushroom
  molecular: hiv, qm7, delaney, sampl

Usage:
    python scripts/run_extended_benchmark.py
    python scripts/run_extended_benchmark.py --methods ulsif,cvplus,bbse
    python scripts/run_extended_benchmark.py --output results/my_run/

Output: results/cross_domain_extended/ (default)
"""
import sys
from pathlib import Path

shift_bench = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(shift_bench / "src"))
sys.path.insert(0, str(shift_bench / "scripts"))

import run_cross_domain_benchmark as rb

# All known datasets (35 total, excluding test_dataset/camelyon17/waterbirds)
rb.DOMAIN_DATASETS["molecular"] = [
    # Original
    "bace", "bbbp", "clintox", "esol", "freesolv", "lipophilicity",
    "sider", "tox21", "toxcast", "molhiv", "muv",
    # New (Session 9)
    "hiv", "qm7", "delaney", "sampl",
]
rb.DOMAIN_DATASETS["tabular"] = [
    # Original
    "adult", "compas", "bank", "german_credit",
    "heart_disease", "diabetes", "student_performance",
    # Session 9
    "wine_quality", "online_shoppers", "communities_crime", "mushroom",
    # Session 10
    "credit_default",
]
rb.DOMAIN_DATASETS["text"] = [
    # Original
    "imdb", "yelp", "amazon", "civil_comments", "twitter",
    # Session 9
    "ag_news", "dbpedia", "imdb_genre", "sst2",
    # Session 10
    "hate_speech", "rotten_tomatoes",
]
rb.NAN_LABEL_DATASETS = {"tox21", "toxcast", "muv"}

# If called directly without extra args, use defaults; otherwise forward CLI args
if len(sys.argv) > 1:
    # Forward external args but ensure script name is set correctly
    sys.argv[0] = "run_extended_benchmark.py"
else:
    # Default: run all 10 methods
    sys.argv = [
        "run_extended_benchmark.py",
        "--methods", "ulsif,kliep,kmm,rulsif,weighted_conformal,split_conformal,cvplus,group_dro,bbse,ravel",
        "--domains", "molecular,tabular,text",
        "--output", "results/full_method_matrix",
        "--tau", "0.5,0.6,0.7,0.8,0.9",
        "--alpha", "0.05",
    ]

if __name__ == "__main__":
    rb.main()
