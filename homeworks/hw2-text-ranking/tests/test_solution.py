"""Independent regression tests; no competition data are needed."""

import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SOLUTION = Path(__file__).resolve().parents[1] / "solution.py"


def import_solution():
    spec = importlib.util.spec_from_file_location("hw2_solution_under_test", SOLUTION)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_fixture(directory):
    """Small reranking corpus with empty fields, Russian text and OOV queries."""
    files = {
        "vkmarco-docs.tsv": (
            "d1\thttps://example.org/cats\tкошки питание\tкошки питание корм\n"
            "d2\thttps://example.org/cars\tавтомобили ремонт\tавтомобили ремонт мотор\n"
            "d3\thttps://example.org/empty\t\t\n"
            "d4\thttps://example.org/mixed\tкошка\tавтомобили мотор кошки\n"
            "d5\thttps://example.org/other\tсады\tрастения цветы\n"
        ),
        "vkmarco-doctrain-queries.tsv": "".join(
            f"{i}\t{'кошки питание' if i % 2 else 'автомобили ремонт'}\n"
            for i in range(1, 13)
        ),
        "vkmarco-doctrain-qrels.tsv": "".join(
            f"{i} 0 {docid} {label}\n"
            for i in range(1, 13)
            for docid, label in (
                ("d1", 3 if i % 2 else 0),
                ("d2", 0 if i % 2 else 3),
                ("d3", 0),
                ("d4", 1),
            )
        ),
        "vkmarco-doceval-queries.tsv": (
            "101\tкошки питание\n102\tавтомобили ремонт\n"
            "103\tнесуществующийтермин\n104\t\n"
        ),
        "objects.csv": (
            "ObjectId,QueryId,DocumentId\n"
            "9,101,d2\n3,101,d1\n11,101,d3\n"
            "5,102,d1\n2,102,d2\n4,102,d4\n"
            "20,103,d1\n21,103,d3\n22,104,d2\n23,104,d3\n"
            "24,101,d1\n"
        ),
    }
    for name, content in files.items():
        (directory / name).write_text(content, encoding="utf-8")


class CommandLineTests(unittest.TestCase):
    def run_solution(self, directory, destination, *options):
        return subprocess.run(
            [sys.executable, str(SOLUTION), "--submission_file=" + str(destination),
             *map(str, options), str(directory)],
            capture_output=True, text=True, timeout=60,
        )

    def test_required_invocation_is_finite_ordered_and_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            write_fixture(data)
            first = data / "first.csv"
            second = data / "second.csv"
            for destination in (first, second):
                completed = subprocess.run(
                    [sys.executable, str(SOLUTION),
                     "--submission_file=" + str(destination), str(data)],
                    capture_output=True, text=True, timeout=60,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with first.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ["ObjectId", "Score"])
                rows = list(reader)
            expected_ids = ["9", "3", "11", "5", "2", "4", "20", "21", "22", "23", "24"]
            self.assertEqual([row["ObjectId"] for row in rows], expected_ids)
            scores = {row["ObjectId"]: float(row["Score"]) for row in rows}
            self.assertTrue(all(math.isfinite(value) for value in scores.values()))
            self.assertGreater(scores["3"], scores["9"])
            self.assertGreater(scores["2"], scores["5"])
            self.assertEqual(scores["3"], scores["24"])

    def test_report_parameters_reproduce_scores_without_retuning(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            write_fixture(data)
            first, replay, report = data / "first.csv", data / "replay.csv", data / "report.json"
            completed = self.run_solution(data, first, "--report_file", report)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            metrics = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(metrics["gain"], "exponential")
            self.assertEqual(metrics["fit_query_count"] + metrics["validation_query_count"], 12)
            self.assertGreater(metrics["validation_query_count"], 0)
            self.assertEqual(metrics["test_object_count"], 11)
            # A frozen formula's inference must be independent of training labels.
            qrels = data / "vkmarco-doctrain-qrels.tsv"
            qrels.write_text("\n".join(
                line.rsplit(" ", 1)[0] + " 0" for line in qrels.read_text().splitlines()
            ) + "\n", encoding="utf-8")
            completed = self.run_solution(data, replay, "--parameters", report)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(first.read_bytes(), replay.read_bytes())
            linear_report = data / "linear.json"
            completed = self.run_solution(
                data, replay, "--parameters", report, "--gain", "linear", "--report_file", linear_report,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(linear_report.read_text())["gain"], "linear")
            self.assertEqual(first.read_bytes(), replay.read_bytes())

    def test_missing_candidate_document_fails_with_actionable_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            write_fixture(data)
            objects = data / "objects.csv"
            objects.write_text(objects.read_text().replace("9,101,d2", "9,101,missing"))
            output = data / "submission.csv"
            completed = self.run_solution(data, output)
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("missing", completed.stderr)
            self.assertFalse(output.exists())

    def test_duplicate_object_identifier_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            write_fixture(data)
            objects = data / "objects.csv"
            objects.write_text(objects.read_text().replace("3,101,d1", "9,101,d1"))
            completed = self.run_solution(data, data / "submission.csv")
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("duplicate ObjectId", completed.stderr)

    def test_train_and_eval_query_ids_may_overlap_without_changing_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            write_fixture(data)
            original, overlapping = data / "original.csv", data / "overlapping.csv"
            report_before, report_after = data / "before.json", data / "after.json"
            completed = self.run_solution(data, original, "--report_file", report_before)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            # Train ID 2 is a car query; eval ID 101 is a cat query. Train ID 1
            # is a cat query; eval ID 102 is a car query. Only eval IDs change.
            eval_file = data / "vkmarco-doceval-queries.tsv"
            eval_file.write_text(
                eval_file.read_text(encoding="utf-8")
                .replace("101\t", "2\t").replace("102\t", "1\t"),
                encoding="utf-8",
            )
            objects = data / "objects.csv"
            objects.write_text(
                objects.read_text(encoding="utf-8")
                .replace(",101,", ",2,").replace(",102,", ",1,"),
                encoding="utf-8",
            )
            completed = self.run_solution(data, overlapping, "--report_file", report_after)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(original.read_bytes(), overlapping.read_bytes())
            before = json.loads(report_before.read_text())
            after = json.loads(report_after.read_text())
            before.pop("elapsed_seconds")
            after.pop("elapsed_seconds")
            self.assertEqual(before, after)


class MetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solution = import_solution()

    def test_stress_and_decomposed_letters_normalize_without_changing_raw_baseline(self):
        self.assertEqual(
            self.solution.tokenize("Камбо\u0301джа и\u0306од е\u0308лка"),
            self.solution.tokenize("Камбоджа йод елка"),
        )
        self.assertEqual(self.solution.normalize_text("и\u0306од е\u0308лка"), "йод елка")
        self.assertNotEqual(self.solution.tokenize("йод"), self.solution.tokenize("иод"))
        # The exact TF-IDF baseline keeps the original sklearn word boundaries.
        self.assertEqual(
            self.solution.raw_tokens("Камбо\u0301джа йод ёлка"),
            ["камбо", "джа", "йод", "ёлка"],
        )

    def test_linear_dcg_and_exponential_gain(self):
        labels = [3, 1, 0, 2]
        expected = 3 + 1 / math.log2(3) + 2 / math.log2(5)
        self.assertAlmostEqual(self.solution.dcg(labels), expected)
        self.assertAlmostEqual(
            self.solution.dcg(labels, gain="exponential"),
            7 + 1 / math.log2(3) + 3 / math.log2(5),
        )

    def test_ndcg_ideal_reverse_and_zero_relevance(self):
        self.assertAlmostEqual(self.solution.ndcg([3, 2, 0], [3, 2, 1]), 1.0)
        actual = self.solution.ndcg([3, 2, 0], [1, 2, 3])
        expected = (2 / math.log2(3) + 3 / math.log2(4)) / (3 + 2 / math.log2(3))
        self.assertAlmostEqual(actual, expected)
        self.assertEqual(self.solution.ndcg([0, 0], [5, 1]), 0.0)

    def test_tie_averaging_crossing_top_k_boundary(self):
        # First two candidates tie for rank 1: their expected top-1 gain is 1.5.
        actual = self.solution.ndcg([3, 0, 1], [5, 5, 1], k=1)
        self.assertAlmostEqual(actual, 0.5)

    def test_all_tied_scores_do_not_depend_on_candidate_order(self):
        self.assertAlmostEqual(
            self.solution.ndcg([3, 0, 1, 2], [0, 0, 0, 0], k=3),
            self.solution.ndcg([1, 3, 2, 0], [0, 0, 0, 0], k=3),
        )

    def test_bm25f_combines_fields_before_saturation(self):
        params = self.solution.Parameters(
            k1=1.2, b_title=0.5, b_body=0.75, title_weight=3,
        )
        # Title normalization = 1.5; body normalization = 0.625.
        pseudo_tf = 3 * 2 / 1.5 + 4 / 0.625
        expected = 2.5 * 2.2 * pseudo_tf / (1.2 + pseudo_tf)
        self.assertAlmostEqual(
            self.solution.bm25f_score([2, 0], [4, 0], 20, 50, 10, 100, [2.5, 7], params),
            expected,
        )
        self.assertEqual(
            self.solution.bm25f_score([0], [0], 0, 0, 0, 0, [2.5], params),
            0.0,
        )

    def test_tfidf_baseline_uses_full_document_norm_and_raw_query_tf(self):
        with tempfile.TemporaryDirectory() as tmp:
            corpus = Path(tmp) / "docs.tsv"
            corpus.write_text(
                "d1\thttps://example.org\talpha\tbeta beta gamma\n"
                "d2\thttps://example.org\talpha alpha\tbeta delta\n"
                "d3\thttps://example.org\tepsilon\tepsilon\n",
                encoding="utf-8",
            )
            features = self.solution.build_features(
                corpus, {"q": "alpha alpha beta missing"}, {"q": {"d1", "d2"}},
            )
            common_idf = math.log(4 / 3) + 1
            rare_idf = math.log(4 / 2) + 1
            doc_norm = math.sqrt(5 * common_idf ** 2 + rare_idf ** 2)
            query_norm = math.sqrt(5 * common_idf ** 2)
            self.assertAlmostEqual(
                features.pairs["q", "d1"].baseline,
                4 * common_idf ** 2 / (doc_norm * query_norm),
            )
            self.assertAlmostEqual(
                features.pairs["q", "d2"].baseline,
                5 * common_idf ** 2 / (doc_norm * query_norm),
            )
            self.assertEqual(features.document_count, 3)


if __name__ == "__main__":
    unittest.main()
