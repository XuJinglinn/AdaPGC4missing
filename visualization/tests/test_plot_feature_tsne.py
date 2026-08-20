import csv
import argparse
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from visualization.plot_feature_tsne import (
    ExperimentRecords,
    _common_balanced_names,
    plot_corruption,
    plot_alignment,
    plot_recovery,
)


class FakeTorch(types.SimpleNamespace):
    @staticmethod
    def load(path, map_location=None, weights_only=False):
        del map_location, weights_only
        with Path(path).open("rb") as handle:
            return pickle.load(handle)


def write_pickle(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class ExperimentRecordsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.exp = Path(self.temp_dir.name) / "exp"
        self.records = self.exp / "recovered_features_records"
        self.corruption = "missing_a_0.70"
        self.names = ["s0", "s1", "s2"]
        write_csv(
            self.exp / "predictions.csv",
            ["corruption", "severity", "epoch", "sample_name", "logits", "true_label"],
            [
                {"corruption": self.corruption, "severity": 5, "epoch": 0,
                 "sample_name": name, "logits": "[]", "true_label": index % 2}
                for index, name in enumerate(self.names)
            ],
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_expected_recovery_uses_alpha_weighted_conditional_means(self):
        alpha = np.asarray([[0.25, 0.75], [0.60, 0.40]], dtype=np.float32)
        means = np.asarray(
            [
                [[1.0, 0.0, 2.0], [3.0, 4.0, 0.0]],
                [[2.0, 1.0, 0.0], [0.0, 3.0, 5.0]],
            ],
            dtype=np.float32,
        )
        recovery_rel = f"{self.corruption}/batch_00000_x2f_v.pt"
        warmup_rel = f"{self.corruption}/batch_00001_x2f_v.pt"
        write_pickle(
            self.records / recovery_rel,
            {"sample_names": self.names[:2], "source": "v", "warmup_fallback": False,
             "alpha": alpha, "cond_means": means},
        )
        write_pickle(
            self.records / warmup_rel,
            {"sample_names": self.names[2:], "source": "v", "warmup_fallback": True,
             "alpha": None, "cond_means": None},
        )
        write_csv(
            self.records / "index.csv",
            ["corruption", "batch_index", "record_type", "source", "num_samples",
             "warmup_fallback", "file"],
            [
                {"corruption": self.corruption, "batch_index": 0, "record_type": "predict_x2f",
                 "source": "v", "num_samples": 2, "warmup_fallback": False,
                 "file": recovery_rel},
                {"corruption": self.corruption, "batch_index": 1, "record_type": "predict_x2f",
                 "source": "v", "num_samples": 1, "warmup_fallback": True,
                 "file": warmup_rel},
            ],
        )
        with patch.dict(sys.modules, {"torch": FakeTorch()}):
            table = ExperimentRecords(self.exp, severity=5).load_recovered(self.corruption, "v")
        expected = np.einsum("bk,bkd->bd", alpha, means)
        np.testing.assert_allclose(table.features, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(table.sample_names, self.names[:2])
        self.assertEqual(table.metadata["warmup_samples_skipped"], 1)

    def test_forward_schema_and_full_modality_alignment(self):
        rel = f"{self.corruption}/batch_00000_forward_results.pt"
        payload = {
            "sample_names": self.names,
            "feat": np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),
            "ca": np.asarray([[10, 11, 12], [13, 14, 15]], dtype=np.float32),
            "cv": np.asarray([[20, 21, 22], [23, 24, 25]], dtype=np.float32),
            "full_sample_names": ["s0", "s2"],
            "mask": {
                "full": np.asarray([True, False, True]),
                "audio_only": np.asarray([False, False, False]),
                "video_only": np.asarray([False, True, False]),
                "both_missing": np.asarray([False, False, False]),
            },
        }
        write_pickle(self.records / rel, payload)
        write_csv(
            self.records / "index.csv",
            ["corruption", "batch_index", "record_type", "source", "num_samples",
             "warmup_fallback", "file"],
            [{"corruption": self.corruption, "batch_index": 0, "record_type": "forward_results",
              "source": "all", "num_samples": 3, "warmup_fallback": "", "file": rel}],
        )
        with patch.dict(sys.modules, {"torch": FakeTorch()}):
            store = ExperimentRecords(self.exp)
            forward = store.load_forward(self.corruption)
            full = store.load_full_modalities(self.corruption)
        np.testing.assert_allclose(forward.features, payload["feat"])
        self.assertEqual(full["Audio"].sample_names, ["s0", "s2"])
        np.testing.assert_allclose(full["Audio"].features, payload["ca"])
        np.testing.assert_allclose(full["Video"].features, payload["cv"])
        np.testing.assert_allclose(full["Fused"].features, payload["feat"][[0, 2]])

    def test_class_selection_is_support_based_and_balanced(self):
        rel = f"{self.corruption}/batch_00000_forward_results.pt"
        names = [f"c0_{i}" for i in range(5)] + [f"c1_{i}" for i in range(7)]
        labels = [0] * 5 + [1] * 7
        write_csv(
            self.exp / "predictions.csv",
            ["corruption", "severity", "epoch", "sample_name", "logits", "true_label"],
            [{"corruption": self.corruption, "severity": 5, "epoch": 0,
              "sample_name": name, "logits": "[]", "true_label": label}
             for name, label in zip(names, labels)],
        )
        payload = {
            "sample_names": names,
            "feat": np.arange(len(names) * 3, dtype=np.float32).reshape(len(names), 3),
            "ca": np.arange(len(names) * 3, dtype=np.float32).reshape(len(names), 3),
            "cv": np.arange(len(names) * 3, dtype=np.float32).reshape(len(names), 3),
            "full_sample_names": names,
            "mask": {"full": np.ones(len(names), bool), "audio_only": np.zeros(len(names), bool),
                     "video_only": np.zeros(len(names), bool), "both_missing": np.zeros(len(names), bool)},
        }
        write_pickle(self.records / rel, payload)
        write_csv(
            self.records / "index.csv",
            ["corruption", "batch_index", "record_type", "source", "num_samples",
             "warmup_fallback", "file"],
            [{"corruption": self.corruption, "batch_index": 0, "record_type": "forward_results",
              "source": "all", "num_samples": len(names), "warmup_fallback": "", "file": rel}],
        )
        with patch.dict(sys.modules, {"torch": FakeTorch()}):
            table = ExperimentRecords(self.exp).load_forward(self.corruption)
        selected_names, classes, counts = _common_balanced_names(
            [table], None, n_classes=2, min_per_class=3, max_per_class=4, seed=111
        )
        self.assertEqual(classes, [1, 0])
        self.assertEqual(counts, {1: 4, 0: 4})
        self.assertEqual(len(selected_names), 8)


class EndToEndRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import matplotlib  # noqa: F401
            import sklearn  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError as error:
            raise unittest.SkipTest(f"optional render dependencies unavailable: {error}")

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.output = self.root / "figures"
        self.rng = np.random.default_rng(111)
        self.n_classes = 4
        self.per_class = 16
        self.dimension = 12
        self.labels = np.repeat(np.arange(self.n_classes), self.per_class)
        self.names = [f"class{label}_sample{index:03d}" for index, label in enumerate(self.labels)]
        centers = self.rng.normal(size=(self.n_classes, self.dimension)) * 2.5
        self.clean_features = centers[self.labels] + self.rng.normal(
            scale=0.35, size=(len(self.labels), self.dimension)
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_predictions(self, exp, corruption, severity):
        write_csv(
            exp / "predictions.csv",
            ["corruption", "severity", "epoch", "sample_name", "logits", "true_label"],
            [{"corruption": corruption, "severity": severity, "epoch": 0,
              "sample_name": name, "logits": "[]", "true_label": int(label)}
             for name, label in zip(self.names, self.labels)],
        )

    def _write_forward_exp(self, exp, corruption, features, severity):
        records = exp / "recovered_features_records"
        rel = f"{corruption}/batch_00000_forward_results.pt"
        mask = np.ones(len(self.names), dtype=bool)
        payload = {
            "sample_names": self.names,
            "feat": np.asarray(features, dtype=np.float32),
            "ca": np.asarray(features + 0.05, dtype=np.float32),
            "cv": np.asarray(features - 0.05, dtype=np.float32),
            "full_sample_names": self.names,
            "mask": {"full": mask, "audio_only": ~mask, "video_only": ~mask,
                     "both_missing": ~mask},
        }
        write_pickle(records / rel, payload)
        write_csv(
            records / "index.csv",
            ["corruption", "batch_index", "record_type", "source", "num_samples",
             "warmup_fallback", "file"],
            [{"corruption": corruption, "batch_index": 0, "record_type": "forward_results",
              "source": "all", "num_samples": len(self.names), "warmup_fallback": "",
              "file": rel}],
        )
        self._write_predictions(exp, corruption, severity)

    def _common_args(self):
        return dict(
            classes=None, n_classes=4, min_per_class=10, max_per_class=14,
            seed=111, pca_dim=10, perplexity=20.0, label_csv=None,
            output_dir=str(self.output), title=None,
        )

    def test_recovery_and_corruption_figures_render_with_audits(self):
        clean_exp = self.root / "clean"
        source_exp = self.root / "source"
        adapted_exp = self.root / "adapted"
        missing_exp = self.root / "missing"
        self._write_forward_exp(clean_exp, "clean", self.clean_features, 0)
        source_features = self.clean_features + self.rng.normal(
            scale=1.5, size=self.clean_features.shape
        )
        adapted_features = self.clean_features + self.rng.normal(
            scale=0.55, size=self.clean_features.shape
        )
        self._write_forward_exp(source_exp, "gaussian_noise", source_features, 5)
        self._write_forward_exp(adapted_exp, "gaussian_noise", adapted_features, 5)

        records = missing_exp / "recovered_features_records"
        corruption = "missing_a_0.70"
        rel = f"{corruption}/batch_00000_x2f_v.pt"
        alpha = np.full((len(self.names), self.n_classes), 0.02, dtype=np.float32)
        alpha[np.arange(len(self.names)), self.labels] = 0.94
        cond_means = np.repeat(self.clean_features[:, None, :], self.n_classes, axis=1)
        cond_means += self.rng.normal(scale=0.18, size=cond_means.shape)
        write_pickle(
            records / rel,
            {"sample_names": self.names, "source": "v", "warmup_fallback": False,
             "alpha": alpha, "cond_means": cond_means.astype(np.float32)},
        )
        write_csv(
            records / "index.csv",
            ["corruption", "batch_index", "record_type", "source", "num_samples",
             "warmup_fallback", "file"],
            [{"corruption": corruption, "batch_index": 0, "record_type": "predict_x2f",
              "source": "v", "num_samples": len(self.names), "warmup_fallback": False,
              "file": rel}],
        )
        self._write_predictions(missing_exp, corruption, 5)

        with patch.dict(sys.modules, {"torch": FakeTorch()}):
            recovery_args = argparse.Namespace(
                **self._common_args(), clean_exp=str(clean_exp), missing_exp=str(missing_exp),
                clean_corruption="clean", missing_corruption=corruption, source=None,
                clean_severity=0, missing_severity=5, overlay_pairs_per_class=6,
                name="qa_recovery",
            )
            recovery_result = plot_recovery(recovery_args)
            corruption_args = argparse.Namespace(
                **self._common_args(),
                condition=[f"Source={source_exp}", f"AdaPGC={adapted_exp}"],
                corruption="gaussian_noise", severity=5, name="qa_corruption",
            )
            corruption_result = plot_corruption(corruption_args)
            alignment_args = argparse.Namespace(
                **self._common_args(), exp=str(clean_exp), corruption="clean", severity=0,
                name="qa_alignment",
            )
            alignment_result = plot_alignment(alignment_args)

        from PIL import Image
        for result, name in ((recovery_result, "qa_recovery"),
                             (corruption_result, "qa_corruption"),
                             (alignment_result, "qa_alignment")):
            self.assertTrue(Path(result["png"]).is_file())
            self.assertTrue(Path(result["pdf"]).is_file())
            self.assertTrue((self.output / f"{name}_points.csv").is_file())
            self.assertTrue((self.output / f"{name}_metrics.csv").is_file())
            self.assertTrue((self.output / f"{name}_manifest.json").is_file())
            with Image.open(result["png"]) as image:
                self.assertGreaterEqual(image.width, 2000)
                self.assertGreaterEqual(image.height, 900)


if __name__ == "__main__":
    unittest.main()
