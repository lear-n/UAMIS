import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from biological_evidence import export_candidates, import_david, summarize_literature, summarize_tcga


class BiologicalEvidenceTests(unittest.TestCase):
    def test_biological_evidence_interfaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidates = pd.DataFrame({
                "Tissue": ["Breast", "Breast"],
                "Protein Id": ["P1", "P2"],
                "Protein Name": ["G1", "G2"],
            })
            candidates.to_csv(root / "candidates.csv", index=False)
            export_candidates(root / "candidates.csv", root / "export")

            evidence = pd.DataFrame({
                "Tissue": ["Breast"],
                "Protein Id": ["P1"],
                "Evidence category": ["same_cancer"],
                "Reference DOI or PMID": ["PMID:123"],
                "Evidence note": ["example"],
            })
            evidence.to_csv(root / "evidence.csv", index=False)
            summarize_literature(root / "candidates.csv", root / "evidence.csv", root / "literature")
            lit = pd.read_csv(root / "literature/candidate_literature.csv")
            self.assertEqual(lit["Evidence category"].tolist(), ["same_cancer", "unreviewed"])

            david = pd.DataFrame({
                "Category": ["GOTERM_BP_DIRECT", "GOTERM_BP_DIRECT"],
                "Term": ["term1", "term2"],
                "Count": [2, 2],
                "PValue": [0.01, 0.08],
                "Genes": ["G1,G2", "G1,G2"],
                "Fold Enrichment": [2.0, 1.5],
                "Bonferroni": [0.2, 0.8],
                "Benjamini": [0.1, 0.6],
                "FDR": [0.1, 0.6],
            })
            david.to_csv(root / "david.tsv", sep="\t", index=False)
            import_david(root / "david.tsv", "Breast", root / "david_out")
            self.assertEqual(len(pd.read_csv(root / "david_out/david_complete_output.csv")), 2)
            self.assertEqual(len(pd.read_csv(root / "david_out/david_main_text_terms.csv")), 1)

            z = pd.DataFrame({"Sample Id": ["S1", "S2", "S3"], "G1": [-1.2, 0.0, 1.3]})
            z.to_csv(root / "tcga.csv", index=False)
            summarize_tcga(root / "candidates.csv", root / "tcga.csv", "Breast", root / "tcga_out")
            tcga = pd.read_csv(root / "tcga_out/tcga_expression_summary.csv")
            self.assertEqual(tcga.loc[tcga["Gene"].eq("G1"), "z_gt_threshold"].iloc[0], 1)
            self.assertEqual(tcga.loc[tcga["Gene"].eq("G1"), "z_lt_negative_threshold"].iloc[0], 1)


if __name__ == "__main__":
    unittest.main()
