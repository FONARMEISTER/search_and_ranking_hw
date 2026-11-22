#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

SEED = 42
THREADS = 4
CHECKPOINT = 100


def features(frame, columns, relative):
    values = frame[columns].astype("float32").replace([np.inf, -np.inf], np.nan)
    if relative:
        ranks = values.groupby(frame["QueryId"], sort=False).rank(pct=True)
        values = pd.concat([values, ranks.astype("float32").add_prefix("query_rank:")], axis=1)
    return values


def query_scores(labels, scores, groups):
    rows = pd.DataFrame({"query": groups, "label": labels, "score": scores})
    discounts = 1 / np.log2(np.arange(2, 12))

    def dcg(column):
        ordered = rows.sort_values(column, ascending=False, kind="stable")
        top = ordered.groupby("query", sort=False).head(10).copy()
        position = top.groupby("query", sort=False).cumcount()
        top["gain"] = np.expm1(np.log(2) * top["label"]) * discounts[position]
        return top.groupby("query")["gain"].sum()

    ideal = dcg("label")
    return (dcg("score") / ideal.where(ideal > 0)).fillna(0).mean()


def ranker(iterations, relative):
    options = {"depth": 6, "learning_rate": 0.06} if relative else {}
    return CatBoostRanker(iterations=iterations, loss_function="YetiRank",
                          random_seed=SEED, thread_count=THREADS,
                          allow_writing_files=False, verbose=False, **options)


def pool(frame, columns, relative):
    return Pool(features(frame, columns, relative), label=frame["Label"],
                group_id=frame["QueryId"])


def train(directory, model_file):
    data = pd.read_csv(directory / "train.csv", dtype={"QueryId": str})
    data = data.sort_values("QueryId", kind="stable").reset_index(drop=True)
    columns = data.columns.drop(["QueryId", "Label"]).tolist()
    groups = data["QueryId"].unique()
    if len(groups) < 2:
        raise ValueError("Training requires at least two distinct queries")
    held = np.random.default_rng(SEED).permutation(groups)[:max(1, len(groups) // 5)]
    mask = data["QueryId"].isin(held)
    valid = data.loc[mask]
    best_score = -np.inf
    for relative, iterations in [(False, 200), (True, 600)]:
        candidate = ranker(iterations, relative)
        candidate.fit(pool(data.loc[~mask], columns, relative))
        validation = features(valid, columns, relative)
        for step, scores in enumerate(candidate.staged_predict(validation, eval_period=CHECKPOINT,
                                                               thread_count=THREADS), 1):
            trees = step * CHECKPOINT
            if not relative and trees != iterations:
                continue
            score = query_scores(valid["Label"].to_numpy(), scores, valid["QueryId"].to_numpy())
            print(f"query_ranks={relative} trees={trees} validation_nDCG@10={score:.6f}", flush=True)
            if score > best_score:
                best_score, best_relative, best_trees = score, relative, trees
    model = ranker(best_trees, best_relative)
    model.fit(pool(data, columns, best_relative))
    model.save_model(str(model_file))
    print(f"Selected query_ranks={best_relative}, trees={best_trees}")


def predict(directory, model_file, submission_file):
    model = CatBoostRanker().load_model(str(model_file))
    data = pd.read_csv(directory / "test.csv", dtype={"ObjectId": str, "QueryId": str},
                       keep_default_na=False, na_values={"QueryId": [""]})
    columns = data.columns.drop(["ObjectId", "QueryId"]).tolist()
    raw = [name for name in model.feature_names_ if name in columns]
    if set(raw) != set(columns):
        raise ValueError("Test features differ from training features")
    data[raw] = data[raw].apply(pd.to_numeric, errors="coerce")
    values = features(data, raw, len(raw) != len(model.feature_names_))
    if list(values) != model.feature_names_:
        raise ValueError("Missing or incompatible test features")
    scores = model.predict(values, thread_count=THREADS)
    if not np.isfinite(scores).all() or data["ObjectId"].duplicated().any():
        raise ValueError("Expected one finite score per unique ObjectId")
    pd.DataFrame({"ObjectId": data["ObjectId"], "Score": scores}).to_csv(submission_file, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--model_file", type=Path, required=True)
    parser.add_argument("--submission_file", type=Path, default=Path("submission.csv"))
    parser.add_argument("data", type=Path)
    args = parser.parse_args()
    if args.train:
        train(args.data, args.model_file)
    else:
        predict(args.data, args.model_file, args.submission_file)


if __name__ == "__main__":
    main()
