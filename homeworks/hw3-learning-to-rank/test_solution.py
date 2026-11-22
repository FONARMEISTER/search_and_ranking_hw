import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from solution import features, predict, query_scores, train


class RankingTests(unittest.TestCase):
    def test_ndcg_uses_top_ten_and_weights_queries_equally(self):
        labels = [3] + [0] * 10 + [1]
        scores = list(range(11)) + [0]
        self.assertAlmostEqual(query_scores(labels, scores, ["a"] * 11 + ["b"]), 0.5)
        self.assertAlmostEqual(query_scores([1, 3], [2, 1], ["a", "a"]),
                               (1 + 7 / np.log2(3)) / (7 + 1 / np.log2(3)))

    def test_ranks_use_only_siblings_and_average_ties(self):
        frame = pd.DataFrame({"QueryId": ["a", "b", "a", "a"], "x": [2, 1, 2, 0]})
        ranked = features(frame, ["x"], True)
        np.testing.assert_allclose(ranked["query_rank:x"], [2.5 / 3, 1, 2.5 / 3, 1 / 3])

    def test_model_round_trip_preserves_object_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            rows = pd.DataFrame({"QueryId": np.tile(np.arange(10), 12),
                                 "x": np.repeat(np.arange(12), 10)})
            rows.assign(Label=rows.x // 3).to_csv(directory / "train.csv", index=False)
            test = rows.iloc[[20, 11, 90]].assign(ObjectId=["01", "02", "03"])
            test.to_csv(directory / "test.csv", index=False)
            model = directory / "model.bin"
            out = directory / "submission.csv"
            train(directory, model)
            predict(directory, model, out)
            result = pd.read_csv(out, dtype={"ObjectId": str})
            self.assertEqual(list(result), ["ObjectId", "Score"])
            self.assertEqual(list(result.ObjectId), list(test.ObjectId))
            self.assertTrue(np.isfinite(result.Score).all())


if __name__ == "__main__":
    unittest.main()
