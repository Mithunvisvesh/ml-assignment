# AI Tool Used: Claude Sonnet 5
# Purpose: Performance comparison harness for Lab 04 - A3. Benchmarks the
#          student-written K-means (bl_sc_u4cse24230_ml_lab_3.py) against the
#          AI-assisted K-means (bl_sc_u4cse24230_ml_lab_4.py) on the real
#          marketing_campaign dataset. Does not modify either implementation;
#          it only imports and times/instruments the functions as written.
"""
Performance comparison: Lab 03 (student) K-means vs Lab 04 (AI-assisted) K-means.

Metrics collected, on the real 'marketing_campaign' dataset:
  1. Wall-clock runtime per run (black-box call to each module's kmeans())
  2. Iterations to convergence (instrumented re-run of the same sub-functions)
  3. Clustering quality: final SSE / inertia (sum of squared distances to
     each point's assigned centroid)
  4. Cluster size balance
  5. Runtime scalability as dataset size grows (200 -> full 2240 rows)

Run with:
    python perf_comparison.py
"""

import importlib.util
import sys
import time
from unittest import mock

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATA_PATH = "Lab Session Data.xlsx"
SHEET_NAME = "marketing_campaign"
N_TRIALS = 30          # repeated runs for timing/quality statistics
K = 3                  # matches the K used in both original scripts
SCALE_SIZES = [200, 500, 1000, 1500, 2240]


# ============================================================
# Import both lab scripts as modules (using the REAL dataset),
# silencing their top-level prints/plots/CSV writes.
# ============================================================

def load_lab_module(module_path: str, module_name: str, real_df: pd.DataFrame):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    with mock.patch("pandas.read_excel", return_value=real_df), \
         mock.patch("builtins.print"), \
         mock.patch("matplotlib.pyplot.savefig"), \
         mock.patch.object(pd.DataFrame, "to_csv"):
        spec.loader.exec_module(module)

    return module


def sse(X, centroids, labels):
    """Sum of squared Euclidean distances from each point to its centroid (inertia)."""
    total = 0.0
    for i, c in enumerate(centroids):
        pts = X[labels == i]
        if len(pts) == 0:
            continue
        total += np.sum((pts - c) ** 2)
    return total


def instrumented_kmeans(module, X, k, max_iter=100, seed=None):
    """
    Re-runs the SAME sub-functions the module's own kmeans() calls
    (initialize_centroids / assign_clusters / recompute_centroids), just to
    additionally count iterations. This does not alter or replace the
    module's kmeans() logic -- it mirrors it exactly, one level up, purely
    for measurement purposes.
    """
    import random
    if seed is not None:
        random.seed(seed)

    centroids = module.initialize_centroids(X, k)
    n_iter = 0
    for _ in range(max_iter):
        n_iter += 1
        labels = module.assign_clusters(X, centroids)
        new_centroids = module.recompute_centroids(X, labels, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, labels, n_iter


def main():
    print("Loading real dataset and both lab modules...")
    real_df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)

    lab3 = load_lab_module("bl_sc_u4cse24230_ml_lab_3.py", "lab03_module", real_df)
    lab4 = load_lab_module("bl_sc_u4cse24230_ml_lab_4.py", "lab04_module", real_df)

    # Both modules build their own preprocessed numeric matrix X at import
    # time using their own label_encode/one_hot_encode functions. Confirm
    # they match before comparing, then use one shared matrix so both
    # k-means versions are benchmarked on identical input data.
    assert lab3.X.shape == lab4.X.shape, "Preprocessed matrices differ in shape!"
    np.testing.assert_allclose(lab3.X, lab4.X, rtol=1e-8)
    X = lab3.X
    print(f"Dataset ready: {X.shape[0]} rows x {X.shape[1]} features\n")

    # --------------------------------------------------------
    # 1) Black-box timing + quality over N_TRIALS repeated runs
    # --------------------------------------------------------
    print(f"Running {N_TRIALS} trials of each k-means (k={K})...")
    results = {"lab03": [], "lab04": []}

    for trial in range(N_TRIALS):
        for name, module in [("lab03", lab3), ("lab04", lab4)]:
            import random
            random.seed(trial)  # same seed sequence offered to both versions

            start = time.perf_counter()
            centroids, labels = module.kmeans(X, K)
            elapsed = time.perf_counter() - start

            quality = sse(X, centroids, labels)
            counts = pd.Series(labels).value_counts().sort_index().to_dict()

            results[name].append({
                "trial": trial,
                "runtime_sec": elapsed,
                "sse": quality,
                "cluster_counts": counts,
            })

    df3 = pd.DataFrame(results["lab03"])
    df4 = pd.DataFrame(results["lab04"])

    # --------------------------------------------------------
    # 2) Iterations to convergence (instrumented, same sub-functions)
    # --------------------------------------------------------
    print("Measuring iterations to convergence...")
    iters3, iters4 = [], []
    for trial in range(N_TRIALS):
        _, _, n3 = instrumented_kmeans(lab3, X, K, seed=trial)
        _, _, n4 = instrumented_kmeans(lab4, X, K, seed=trial)
        iters3.append(n3)
        iters4.append(n4)
    df3["iterations"] = iters3
    df4["iterations"] = iters4

    # --------------------------------------------------------
    # 3) Summary statistics
    # --------------------------------------------------------
    def summarize(df, label):
        return {
            "version": label,
            "runtime_mean_s": df.runtime_sec.mean(),
            "runtime_std_s": df.runtime_sec.std(),
            "runtime_min_s": df.runtime_sec.min(),
            "runtime_max_s": df.runtime_sec.max(),
            "sse_mean": df.sse.mean(),
            "sse_std": df.sse.std(),
            "iterations_mean": df.iterations.mean(),
            "iterations_std": df.iterations.std(),
        }

    summary = pd.DataFrame([
        summarize(df3, "Lab03 (student)"),
        summarize(df4, "Lab04 (AI-assisted)"),
    ])
    print("\n=== Summary over {} trials (k={}) ===".format(N_TRIALS, K))
    print(summary.to_string(index=False))

    # Paired comparison: same trial index/seed used for both -> paired test
    from scipy import stats
    t_runtime, p_runtime = stats.ttest_rel(df3.runtime_sec, df4.runtime_sec)
    t_sse, p_sse = stats.ttest_rel(df3.sse, df4.sse)
    print(f"\nPaired t-test (runtime): t={t_runtime:.3f}, p={p_runtime:.4f}")
    print(f"Paired t-test (SSE)    : t={t_sse:.3f}, p={p_sse:.4f}")

    # --------------------------------------------------------
    # 4) Scalability: runtime vs dataset size
    # --------------------------------------------------------
    print("\nMeasuring runtime scalability with dataset size...")
    scale_results = []
    rng = np.random.default_rng(0)
    for n in SCALE_SIZES:
        idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
        X_sub = X[idx]
        for name, module in [("lab03", lab3), ("lab04", lab4)]:
            import random
            random.seed(0)
            start = time.perf_counter()
            module.kmeans(X_sub, K)
            elapsed = time.perf_counter() - start
            scale_results.append({"n_rows": n, "version": name, "runtime_sec": elapsed})

    scale_df = pd.DataFrame(scale_results)
    print(scale_df.pivot(index="n_rows", columns="version", values="runtime_sec"))

    # --------------------------------------------------------
    # 5) Plots
    # --------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    axes[0].boxplot([df3.runtime_sec, df4.runtime_sec], tick_labels=["Lab03", "Lab04"])
    axes[0].set_title("Runtime per run (s)")
    axes[0].set_ylabel("seconds")

    axes[1].boxplot([df3.sse, df4.sse], tick_labels=["Lab03", "Lab04"])
    axes[1].set_title("Clustering quality (SSE)")
    axes[1].set_ylabel("SSE")

    for name, group in scale_df.groupby("version"):
        axes[2].plot(group.n_rows, group.runtime_sec, marker="o", label=name)
    axes[2].set_title("Runtime vs dataset size")
    axes[2].set_xlabel("Number of rows")
    axes[2].set_ylabel("seconds")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("kmeans_performance_comparison.png", dpi=150)
    plt.close()
    print("\nSaved plot: kmeans_performance_comparison.png")

    # Save raw results for the report
    df3.assign(version="lab03").to_csv("kmeans_trials_lab03.csv", index=False)
    df4.assign(version="lab04").to_csv("kmeans_trials_lab04.csv", index=False)
    scale_df.to_csv("kmeans_scalability.csv", index=False)
    summary.to_csv("kmeans_summary.csv", index=False)
    print("Saved CSVs: kmeans_trials_lab03.csv, kmeans_trials_lab04.csv, "
          "kmeans_scalability.csv, kmeans_summary.csv")


if __name__ == "__main__":
    main()
