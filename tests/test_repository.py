from pathlib import Path
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


class RepositoryContractTests(unittest.TestCase):
    def test_database_schema(self):
        data = pd.read_excel(ROOT / "data" / "frcc_database.xlsx", sheet_name="Sheet1")
        expected = {"FA/C", "W/B", "S/B", "SP/B", "Vf(%)", "If",
                    "CX(MPa)", "UTX(Mpa)", "G(KJm3)", "MiniSF(cm)"}
        self.assertTrue(expected.issubset(data.columns))
        self.assertGreaterEqual(len(data), 300)

    def test_parametric_workbook(self):
        book = pd.ExcelFile(ROOT / "data" / "index_analysis_predictions.xlsx")
        self.assertTrue({"Fig1", "Fig2", "Fig3", "Vari", "Exp"}.issubset(book.sheet_names))
        self.assertEqual(pd.read_excel(book, "Fig1").shape[0], 15)
        self.assertEqual(pd.read_excel(book, "Fig2").shape[0], 5)
        self.assertEqual(pd.read_excel(book, "Fig3").shape[0], 10)

    def test_optimization_outputs(self):
        for fiber in ("PE", "PVA"):
            path = ROOT / "data" / "optimization" / fiber / "topsis_ranking.csv"
            data = pd.read_csv(path)
            self.assertEqual(len(data), 100)
            self.assertEqual(int(data["TOPSIS Rank"].min()), 1)

    def test_model_files_present(self):
        self.assertEqual(len(list((ROOT / "models" / "mc_dropout").glob("*.pth"))), 4)
        self.assertEqual(len(list((ROOT / "models" / "bbb").glob("*.pth"))), 1)


if __name__ == "__main__":
    unittest.main()
