"""
H3 Molecular Intervention: Raise n_eff via Dimensionality Reduction
=====================================================================
Molecular datasets have n_eff~1 because 217-dim fingerprint features
create extreme covariate shift between scaffold cohorts. This experiment
tests whether PCA/random projection on features (for ratio estimation only)
can raise n_eff without breaking validity.

Intervention: reduce fingerprint features to k dimensions before weight
estimation, but keep original features for everything else.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

shift_bench_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(shift_bench_dir / "src"))
sys.path.insert(0, str(shift_bench_dir.parent / "ravel" / "src"))

from shiftbench.data import load_dataset
from shiftbench.baselines.ulsif import uLSIFBaseline


def run_molecular_intervention():
    out_dir = shift_bench_dir / "results" / "h3_molecular_intervention"
    os.makedirs(out_dir, exist_ok=True)

    datasets = ["bace", "bbbp"]
    pca_dims = [5, 10, 20, 50, 100, None]  # None = original (217-dim)
    alpha = 0.05
    tau_grid = [0.5, 0.6, 0.7, 0.8, 0.9]

    all_results = []

    for ds_name in datasets:
        print(f"\n{'='*70}")
        print(f"Dataset: {ds_name}")
        print(f"{'='*70}")

        X, y, cohorts, splits = load_dataset(ds_name)
        cal_mask = (splits["split"] == "cal").values
        test_mask = (splits["split"] == "test").values

        X_cal = X[cal_mask]
        y_cal = y[cal_mask].astype(int)
        cohorts_cal = cohorts[cal_mask]
        X_test = X[test_mask]

        # Load real predictions
        import json
        mapping_path = shift_bench_dir / "models" / "prediction_mapping.json"
        if mapping_path.exists():
            with open(mapping_path) as f:
                mapping = json.load(f)
            if ds_name in mapping:
                preds_cal = np.load(shift_bench_dir / mapping[ds_name]["cal_binary"]).astype(int)
            else:
                preds_cal = y_cal.copy()
        else:
            preds_cal = y_cal.copy()

        print(f"  X_cal: {X_cal.shape}, X_test: {X_test.shape}")
        print(f"  n_cohorts: {len(np.unique(cohorts_cal))}")

        for n_pca in pca_dims:
            label = f"PCA-{n_pca}" if n_pca else "original"
            print(f"\n  {label}:")

            if n_pca is not None and n_pca < X_cal.shape[1]:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=n_pca, random_state=42)
                X_cal_reduced = pca.fit_transform(X_cal)
                X_test_reduced = pca.transform(X_test)
                explained_var = pca.explained_variance_ratio_.sum()
                print(f"    Explained variance: {explained_var:.3f}")
            else:
                X_cal_reduced = X_cal
                X_test_reduced = X_test
                n_pca_actual = X_cal.shape[1]
                explained_var = 1.0

            # Estimate weights using reduced features
            n_basis = min(100, len(X_cal_reduced))
            try:
                method = uLSIFBaseline(n_basis=n_basis, sigma=None, lambda_=0.1,
                                       random_state=42)
                weights = method.estimate_weights(X_cal_reduced, X_test_reduced)
            except Exception as e:
                print(f"    Weight estimation failed: {e}")
                continue

            # Compute per-cohort n_eff
            cohort_ids = np.unique(cohorts_cal)
            neffs = []
            for cid in cohort_ids:
                cmask = cohorts_cal == cid
                pmask = cmask & (preds_cal == 1)
                w_c = weights[pmask]
                if len(w_c) > 0 and w_c.sum() > 0:
                    n_eff = (w_c.sum() ** 2) / (w_c ** 2).sum()
                    neffs.append(n_eff)

            # Run certification
            try:
                decisions = method.estimate_bounds(
                    y_cal, preds_cal, cohorts_cal, weights, tau_grid, alpha
                )
                n_cert = sum(1 for d in decisions
                             if (d.decision if isinstance(d.decision, str)
                                 else d.decision.value) == "CERTIFY")
                n_total = len(decisions)
                cert_rate = n_cert / n_total if n_total > 0 else 0
            except Exception as e:
                print(f"    Certification failed: {e}")
                n_cert = 0
                n_total = 0
                cert_rate = 0

            mean_neff = np.mean(neffs) if neffs else 0
            median_neff = np.median(neffs) if neffs else 0

            result = {
                "dataset": ds_name,
                "pca_dims": n_pca if n_pca else X_cal.shape[1],
                "label": label,
                "explained_variance": explained_var,
                "mean_n_eff": mean_neff,
                "median_n_eff": median_neff,
                "cert_rate": cert_rate,
                "n_certified": n_cert,
                "n_total": n_total,
                "n_cohorts": len(cohort_ids),
                "mean_weight": weights.mean(),
                "std_weight": weights.std(),
                "max_weight": weights.max(),
            }
            all_results.append(result)

            print(f"    mean_n_eff={mean_neff:.2f}, median_n_eff={median_neff:.2f}")
            print(f"    cert_rate={cert_rate:.4f} ({n_cert}/{n_total})")

    df = pd.DataFrame(all_results)
    df.to_csv(out_dir / "molecular_intervention.csv", index=False)

    print(f"\n{'='*70}")
    print("MOLECULAR INTERVENTION RESULTS")
    print(f"{'='*70}")
    print(f"{'Dataset':<8} {'Dims':<12} {'ExplVar':>8} {'MeanNeff':>9} {'MedNeff':>8} {'Cert%':>7}")
    print("-" * 56)
    for _, row in df.iterrows():
        print(f"{row['dataset']:<8} {row['label']:<12} {row['explained_variance']:>7.3f} "
              f"{row['mean_n_eff']:>9.2f} {row['median_n_eff']:>8.2f} "
              f"{row['cert_rate']*100:>6.2f}%")

    print(f"\nSaved to {out_dir}/")


if __name__ == "__main__":
    run_molecular_intervention()
