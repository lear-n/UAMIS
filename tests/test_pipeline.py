import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jsd_screening import analyze_tissue, js_divergence, normalize_relative, run_screening
from summarize_mc_predictions import CATEGORIES, Z95, summarize_folder
from run_pipeline import run


class CoreTests(unittest.TestCase):
    def test_relative_normalization_uses_delta(self):
        x = np.array([0.0, 1.0, 0.0, 0.0, 0.0])
        q = normalize_relative(x, 1e-12)
        self.assertAlmostEqual(q.sum(), 1.0)
        self.assertTrue((q > 0).all())

    def test_jsd_symmetry_and_identity(self):
        self.assertAlmostEqual(js_divergence([1, 0], [1, 0]), 0.0, places=12)
        self.assertAlmostEqual(js_divergence([1, 0], [0, 1]), js_divergence([0, 1], [1, 0]), places=12)
        self.assertGreater(js_divergence([1, 0], [0, 1]), 0.99)

    def test_mc_prediction_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            pd.DataFrame({"Protein Name": ["G1", "G1"], "Protein Id": ["P1", "P1"]}).to_csv(folder / "metadata.csv", index=False)
            arr = np.zeros((30, 2, 5), dtype=float)
            arr[:, :, 0] = np.arange(30)[:, None] / 100.0
            np.save(folder / "mc_predictions.npy", arr)
            out = summarize_folder(folder, folder / "summary.csv", expected_passes=30)
            self.assertEqual(len(out), 2)
            expected_sd = np.std(np.arange(30) / 100.0, ddof=0)
            self.assertAlmostEqual(out["SD_Cytoplasm"].iloc[0], expected_sd)
            self.assertAlmostEqual(out["CI95_width_Cytoplasm"].iloc[0], 2 * Z95 * expected_sd / np.sqrt(30))

    def test_average_images_before_normalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cols = [f"Mean_{c}" for c in CATEGORIES]
            normal = pd.DataFrame([
                ["G1", "P1", .9, .1, 0, 0, 0],
                ["G1", "P1", .3, .3, .3, .3, .3],
            ], columns=["Protein Name", "Protein Id", *cols])
            tumor = pd.DataFrame([["G1", "P1", .3, .8, .1, .1, .1]], columns=["Protein Name", "Protein Id", *cols])
            normal.to_csv(root / "n.csv", index=False)
            tumor.to_csv(root / "t.csv", index=False)
            result, _ = analyze_tissue(root / "n.csv", root / "t.csv", "Breast")
            expected = normalize_relative(normal[cols].mean().to_numpy())
            observed = result[[f"Normal_{c.replace('_', ' ')}_relative" for c in CATEGORIES]].iloc[0].to_numpy(float)
            np.testing.assert_allclose(expected, observed)

    def test_end_to_end_public_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            for state, vector in [("normal", [0.9, .1, .1, .1, .1]), ("tumor", [.1, .9, .1, .1, .1])]:
                folder = inputs / "Breast" / state
                folder.mkdir(parents=True)
                pd.DataFrame({"Protein Name": ["G1"], "Protein Id": ["P1"]}).to_csv(folder / "metadata.csv", index=False)
                arr = np.tile(np.array(vector, dtype=float), (30, 1, 1))
                np.save(folder / "mc_predictions.npy", arr)
            cfg_dir = root / "configs"
            cfg_dir.mkdir()
            (cfg_dir / "pipeline.json").write_text(json.dumps({
                "input_root": "../inputs",
                "output_root": "../results",
                "tissues": ["Breast"],
                "passes": 30,
                "delta": 1e-12,
                "jsd_cutoff": 0.1,
            }))
            run(cfg_dir / "pipeline.json")
            candidates = pd.read_csv(root / "results/screening/candidates.csv")
            self.assertEqual(len(candidates), 1)
            sensitivity = pd.read_csv(root / "results/screening/threshold_sensitivity.csv")
            self.assertTrue(((sensitivity["Scope"] == "Overall") & (sensitivity["JSD_cutoff"] == 0.1)).any())


if __name__ == "__main__":
    unittest.main()
