# AI Tool Used: Claude Sonnet 5
# Purpose: Unit test suite generated to validate the modular functions written
#          by the student in Lab 03 (bl_sc_u4cse24230_ml_lab_3.py) and the
#          AI-assisted Lab 04 version (bl_sc_u4cse24230_ml_lab_4.py).
# Function definitions under test, modularization, and program design: student's own work.
# Test cases: generated with AI assistance, reviewed and integrated by the student.
"""
Unit tests for the modular functions defined in:
  - bl_sc_u4cse24230_ml_lab_3.py  (Lab 03, student-written)
  - bl_sc_u4cse24230_ml_lab_4.py  (Lab 04, AI-assisted)

Both scripts run top-level code on import (loading the marketing_campaign
dataset, generating plots, running K-means, writing a CSV). To keep the
tests fast, deterministic, and independent of the real Excel file, this
suite stubs out pandas.read_excel with a small synthetic dataset that has
the same columns the scripts expect, and silences print/plot/file-write
side effects during import. The *actual* functions from each script are
then imported and tested directly -- nothing about the functions
themselves is re-implemented here.

Run with:
    python -m unittest test_lab_functions.py -v
"""

import importlib.util
import inspect
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless backend so plotting never blocks/opens windows


# ============================================================
# Helpers: build a synthetic dataset & import a lab script safely
# ============================================================

def _make_synthetic_marketing_df(n_rows: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Build a small synthetic DataFrame with the same columns the lab scripts
    expect from the 'marketing_campaign' sheet, so the scripts can be
    imported and exercised without needing the real Excel file.
    """
    rng = np.random.default_rng(seed)

    education_options = ["Basic", "Graduation", "Master", "PhD"]
    marital_options = ["Single", "Married", "Divorced"]

    data = {
        "ID": np.arange(1, n_rows + 1),
        "Year_Birth": rng.integers(1945, 2000, n_rows),
        "Education": rng.choice(education_options, n_rows),
        "Marital_Status": rng.choice(marital_options, n_rows),
        "Income": rng.uniform(20000, 90000, n_rows).round(2),
        "Kidhome": rng.integers(0, 3, n_rows),
        "Teenhome": rng.integers(0, 3, n_rows),
        "Dt_Customer": pd.date_range("2012-01-01", periods=n_rows, freq="D"),
        "Recency": rng.integers(0, 100, n_rows),
        "MntWines": rng.integers(0, 1500, n_rows),
        "MntFruits": rng.integers(0, 200, n_rows),
        "MntMeatProducts": rng.integers(0, 1800, n_rows),
        "MntFishProducts": rng.integers(0, 300, n_rows),
        "MntSweetProducts": rng.integers(0, 300, n_rows),
        "MntGoldProds": rng.integers(0, 300, n_rows),
        "NumDealsPurchases": rng.integers(0, 15, n_rows),
        "NumWebPurchases": rng.integers(0, 15, n_rows),
        "NumCatalogPurchases": rng.integers(0, 15, n_rows),
        "NumStorePurchases": rng.integers(0, 15, n_rows),
        "NumWebVisitsMonth": rng.integers(0, 15, n_rows),
        "AcceptedCmp3": rng.integers(0, 2, n_rows),
        "AcceptedCmp4": rng.integers(0, 2, n_rows),
        "AcceptedCmp5": rng.integers(0, 2, n_rows),
        "AcceptedCmp1": rng.integers(0, 2, n_rows),
        "AcceptedCmp2": rng.integers(0, 2, n_rows),
        "Complain": rng.integers(0, 2, n_rows),
        "Z_CostContact": np.full(n_rows, 3),
        "Z_Revenue": np.full(n_rows, 11),
        "Response": rng.integers(0, 2, n_rows),
    }
    return pd.DataFrame(data)


def load_lab_module(module_path: str, module_name: str):
    """
    Import a lab script as a standalone module while stubbing out the parts
    that would otherwise require the real dataset file or produce visible
    side effects (prints, plot windows, saved files):

      - pandas.read_excel  -> returns a synthetic marketing_campaign frame
      - builtins.print      -> silenced
      - plt.savefig         -> no-op
      - DataFrame.to_csv    -> no-op

    Returns the imported module so its functions can be called and tested
    directly, exactly as the student wrote them.
    """
    synthetic_df = _make_synthetic_marketing_df()

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module

    with mock.patch("pandas.read_excel", return_value=synthetic_df), \
         mock.patch("builtins.print"), \
         mock.patch("matplotlib.pyplot.savefig"), \
         mock.patch.object(pd.DataFrame, "to_csv"):
        spec.loader.exec_module(module)

    return module


# ============================================================
# Test suite (module-agnostic -- reused for Lab 03 and Lab 04)
# ============================================================

class LabFunctionsTestMixin:
    """
    Mixin containing all test cases. A concrete TestCase subclass must set
    the class attribute `module` to the imported lab script before the
    tests are run. The same test bodies therefore validate both the Lab 03
    and Lab 04 implementations without duplicating test logic.
    """

    module = None  # set by subclasses

    # ---------- A2/A3: Encoding functions ----------

    def test_label_encode_mapping_and_values(self):
        series = pd.Series(["b", "a", "c", "a", None])
        encoded, mapping = self.module.label_encode(series)

        self.assertEqual(mapping, {"a": 0, "b": 1, "c": 2})
        self.assertEqual(encoded.iloc[0], 1)  # "b" -> 1
        self.assertEqual(encoded.iloc[1], 0)  # "a" -> 0
        self.assertEqual(encoded.iloc[2], 2)  # "c" -> 2
        self.assertTrue(pd.isna(encoded.iloc[4]))  # None stays NaN

    def test_one_hot_encode_columns_and_row_sums(self):
        df = pd.DataFrame({
            "Category": ["x", "y", "x", "z"],
            "Value": [1, 2, 3, 4],
        })
        out = self.module.one_hot_encode(df, ["Category"])

        # original column removed, dummy columns added
        self.assertNotIn("Category", out.columns)
        for col in ["Category_x", "Category_y", "Category_z"]:
            self.assertIn(col, out.columns)

        # exactly one dummy column should be 1 per row
        dummy_cols = ["Category_x", "Category_y", "Category_z"]
        row_sums = out[dummy_cols].sum(axis=1)
        self.assertTrue((row_sums == 1).all())

        # non-categorical column untouched
        self.assertListEqual(list(out["Value"]), [1, 2, 3, 4])

    # ---------- A4/A6: Minkowski distance ----------

    def test_minkowski_distance_manhattan(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        self.assertAlmostEqual(self.module.minkowski_distance(a, b, p=1), 7.0)

    def test_minkowski_distance_euclidean(self):
        a = np.array([0.0, 0.0])
        b = np.array([3.0, 4.0])
        self.assertAlmostEqual(self.module.minkowski_distance(a, b, p=2), 5.0)

    def test_minkowski_distance_zero_for_identical_vectors(self):
        a = np.array([1.5, -2.0, 3.25])
        self.assertAlmostEqual(self.module.minkowski_distance(a, a, p=3), 0.0)

    def test_minkowski_distance_matches_scipy(self):
        from scipy.spatial.distance import minkowski as scipy_minkowski
        rng = np.random.default_rng(0)
        a, b = rng.uniform(-10, 10, 8), rng.uniform(-10, 10, 8)
        for p in range(1, 6):
            custom = self.module.minkowski_distance(a, b, p)
            reference = scipy_minkowski(a, b, p)
            self.assertAlmostEqual(custom, reference, places=6)

    # ---------- A7: Dot product & norm ----------

    def test_dot_product_matches_numpy(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([4.0, -5.0, 6.0])
        self.assertAlmostEqual(self.module.dot_product(a, b), np.dot(a, b))

    def test_euclidean_norm_matches_numpy(self):
        a = np.array([3.0, 4.0])
        self.assertAlmostEqual(self.module.euclidean_norm(a), np.linalg.norm(a))
        self.assertAlmostEqual(self.module.euclidean_norm(a), 5.0)

    # ---------- A8/A9: Mean, variance, std ----------

    def test_my_mean_matches_numpy(self):
        x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        self.assertAlmostEqual(self.module.my_mean(x), np.mean(x))

    def test_my_variance_matches_numpy(self):
        x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        self.assertAlmostEqual(self.module.my_variance(x), np.var(x))

    def test_my_std_matches_numpy(self):
        x = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        self.assertAlmostEqual(self.module.my_std(x), np.std(x))

    def test_matrix_statistics_matches_numpy_axiswise(self):
        rng = np.random.default_rng(1)
        matrix = rng.uniform(-5, 5, size=(20, 4))

        means, variances, stds = self.module.matrix_statistics(matrix)

        np.testing.assert_allclose(means, np.mean(matrix, axis=0), rtol=1e-8)
        np.testing.assert_allclose(variances, np.var(matrix, axis=0), rtol=1e-8)
        np.testing.assert_allclose(stds, np.std(matrix, axis=0), rtol=1e-8)

    # ---------- A11: K-means building blocks ----------

    def test_initialize_centroids_shape_and_membership(self):
        X = np.arange(20).reshape(10, 2).astype(float)
        k = 3
        centroids = self.module.initialize_centroids(X, k)

        self.assertEqual(centroids.shape, (k, X.shape[1]))
        # every returned centroid must be an actual row from X
        for c in centroids:
            self.assertTrue(any(np.array_equal(c, row) for row in X))

    def test_initialize_centroids_reproducible_when_supported(self):
        # Lab 04's version accepts an optional random_state; Lab 03's does not.
        # Only run this check for implementations that support it, so the
        # same test file works unmodified against both scripts.
        sig = inspect.signature(self.module.initialize_centroids)
        if "random_state" not in sig.parameters:
            self.skipTest("random_state not supported by this implementation")

        X = np.arange(40).reshape(20, 2).astype(float)
        c1 = self.module.initialize_centroids(X, 3, random_state=7)
        c2 = self.module.initialize_centroids(X, 3, random_state=7)
        np.testing.assert_array_equal(c1, c2)

    def test_assign_clusters_picks_nearest_centroid(self):
        X = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
        centroids = np.array([[0.0, 0.0], [10.0, 10.0]])

        labels = self.module.assign_clusters(X, centroids)

        self.assertEqual(len(labels), len(X))
        self.assertEqual(labels[0], 0)
        self.assertEqual(labels[1], 0)
        self.assertEqual(labels[2], 1)
        self.assertEqual(labels[3], 1)

    def test_recompute_centroids_computes_correct_means(self):
        X = np.array([[0.0, 0.0], [2.0, 0.0], [10.0, 10.0], [12.0, 10.0]])
        labels = np.array([0, 0, 1, 1])

        centroids = self.module.recompute_centroids(X, labels, k=2)

        np.testing.assert_allclose(centroids[0], [1.0, 0.0])
        np.testing.assert_allclose(centroids[1], [11.0, 10.0])

    def test_recompute_centroids_handles_empty_cluster(self):
        # No point is labeled '2', forcing the empty-cluster fallback path.
        X = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0]])
        labels = np.array([0, 0, 1])

        centroids = self.module.recompute_centroids(X, labels, k=3)

        self.assertEqual(centroids.shape, (3, 2))
        # the fallback centroid for the empty cluster must be one of X's rows
        self.assertTrue(any(np.array_equal(centroids[2], row) for row in X))

    def test_kmeans_output_shapes_and_label_range(self):
        rng = np.random.default_rng(2)
        cluster_a = rng.normal(loc=[0, 0], scale=0.3, size=(15, 2))
        cluster_b = rng.normal(loc=[10, 10], scale=0.3, size=(15, 2))
        X = np.vstack([cluster_a, cluster_b])
        k = 2

        centroids, labels = self.module.kmeans(X, k)

        self.assertEqual(centroids.shape, (k, 2))
        self.assertEqual(len(labels), len(X))
        self.assertTrue(set(np.unique(labels)).issubset(set(range(k))))

    def test_kmeans_separates_well_defined_clusters(self):
        # With two clearly separated blobs, every point in a blob should
        # end up in the same cluster.
        rng = np.random.default_rng(3)
        cluster_a = rng.normal(loc=[0, 0], scale=0.2, size=(10, 2))
        cluster_b = rng.normal(loc=[20, 20], scale=0.2, size=(10, 2))
        X = np.vstack([cluster_a, cluster_b])

        _, labels = self.module.kmeans(X, k=2, random_state=0) \
            if "random_state" in inspect.signature(self.module.kmeans).parameters \
            else self.module.kmeans(X, k=2)

        labels_a = labels[:10]
        labels_b = labels[10:]
        self.assertEqual(len(set(labels_a)), 1)
        self.assertEqual(len(set(labels_b)), 1)
        self.assertNotEqual(labels_a[0], labels_b[0])


# ============================================================
# Concrete TestCase classes: one per lab script
# ============================================================

class TestLab03Functions(LabFunctionsTestMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_lab_module(
            "bl_sc_u4cse24230_ml_lab_3.py", "lab03_module"
        )


class TestLab04Functions(LabFunctionsTestMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_lab_module(
            "bl_sc_u4cse24230_ml_lab_4.py", "lab04_module"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)