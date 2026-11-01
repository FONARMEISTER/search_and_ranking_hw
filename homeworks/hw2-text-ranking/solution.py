#!/usr/bin/env python3
"""Own BM25F, lexical features and exact TF-IDF cosine; train-only grid search.

No downloaded linguistic resources or pretrained ranking models are used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from time import perf_counter
import unicodedata
from urllib.parse import unquote, urlsplit

import snowballstemmer

TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")
STEMMER = snowballstemmer.stemmer("russian")


def raw_tokens(text):
    """The default sklearn word analyzer, without stop-word filtering."""
    return TOKEN_RE.findall(text.lower())


def normalize_text(text):
    # Remove stress accents before word boundaries are detected. Preserve the
    # breve in й by removing only acute/grave marks and recomposing to NFC.
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", decomposed.replace("\u0300", "").replace("\u0301", "")).replace("ё", "е").replace("Ё", "Е")


@lru_cache(maxsize=500_000)
def stem(token):
    text = normalize_text(token)
    return STEMMER.stemWord(text) if any("а" <= ch <= "я" for ch in text) else text


def needs_normalization(text):
    return any(mark in text for mark in "\u0300\u0301\u0306\u0308")


def tokenize(text):
    if needs_normalization(text):
        text = normalize_text(text)
    return [stem(token) for token in raw_tokens(text)]


def read_queries(path):
    queries = {}
    with open(path, encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            fields = line.rstrip("\r\n").split("\t", 1)
            if len(fields) != 2 or not fields[0]:
                raise ValueError(f"{path}:{number}: expected QueryId<TAB>QueryText")
            if fields[0] in queries:
                raise ValueError(f"{path}:{number}: duplicate query {fields[0]}")
            queries[fields[0]] = fields[1]
    return queries


def read_qrels(path):
    qrels = defaultdict(dict)
    with open(path, encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(f"{path}:{number}: expected four qrels fields")
            query, _, doc, label = fields
            label = int(label)
            if not 0 <= label <= 3:
                raise ValueError(f"{path}:{number}: relevance must be in [0,3]")
            if doc in qrels[query] and qrels[query][doc] != label:
                raise ValueError(f"{path}:{number}: conflicting qrels")
            qrels[query][doc] = label
    return dict(qrels)


def read_objects(path):
    objects, seen = [], set()
    with open(path, encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not {"ObjectId", "QueryId", "DocumentId"}.issubset(reader.fieldnames or []):
            raise ValueError(f"{path}: expected ObjectId,QueryId,DocumentId columns")
        for row in reader:
            item = tuple(row[key] for key in ("ObjectId", "QueryId", "DocumentId"))
            if not all(item) or item[0] in seen:
                raise ValueError(f"{path}: empty identifier or duplicate ObjectId")
            seen.add(item[0])
            objects.append(item)
    return objects


def iter_docs(path):
    with open(path, encoding="utf-8-sig") as source:
        for number, line in enumerate(source, 1):
            fields = line.rstrip("\r\n").split("\t", 3)
            if len(fields) != 4 or not fields[0]:
                raise ValueError(f"{path}:{number}: expected DocumentId,URL,Title,Body TSV fields")
            yield fields


def dcg(labels, k=10, gain="linear"):
    transform = (lambda x: 2.0 ** x - 1) if gain == "exponential" else float
    return sum(transform(label) / math.log2(rank + 2)
               for rank, label in enumerate(labels[:k]))


def ndcg(labels, scores, k=10, gain="linear"):
    """Expected DCG inside tied-score groups, as sklearn.ndcg_score defaults."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have equal lengths")
    ideal = dcg(sorted(labels, reverse=True), k, gain)
    if not ideal:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: -scores[i])
    transform = (lambda x: 2.0 ** x - 1) if gain == "exponential" else float
    result, start = 0.0, 0
    while start < min(k, len(order)):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        mean_gain = sum(transform(labels[order[i]]) for i in range(start, end)) / (end - start)
        result += mean_gain * sum(1 / math.log2(i + 2) for i in range(start, min(end, k)))
        start = end
    return result / ideal


@dataclass(frozen=True)
class Parameters:
    k1: float = 1.2
    b_title: float = 0.3
    b_body: float = 0.6
    title_weight: float = 4.0
    url_weight: float = 0.15
    coverage_weight: float = 0.3
    phrase_weight: float = 0.15
    baseline_weight: float = 0.0

    def validate(self):
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("all parameters must be finite")
        if self.k1 <= 0 or not 0 <= self.b_title <= 1 or not 0 <= self.b_body <= 1:
            raise ValueError("k1 must be positive and field b must be in [0,1]")
        if any(value < 0 for key, value in asdict(self).items() if key.endswith("weight")):
            raise ValueError("feature weights must be nonnegative")
        return self


def bm25f_score(title_tf, body_tf, title_length, body_length, avg_title, avg_body, idfs, params):
    """Normalize fields, combine weighted TF, then saturate exactly once."""
    nt = max(1e-12, 1 - params.b_title + params.b_title * title_length / max(avg_title, 1e-12))
    nb = max(1e-12, 1 - params.b_body + params.b_body * body_length / max(avg_body, 1e-12))
    score = 0.0
    for title_count, body_count, idf in zip(title_tf, body_tf, idfs):
        tf = params.title_weight * title_count / nt + body_count / nb
        score += idf * (params.k1 + 1) * tf / (params.k1 + tf)
    return score


@dataclass
class Pair:
    title_tf: tuple
    body_tf: tuple
    title_length: int
    body_length: int
    url: float
    phrase: float
    coverage: float = 0.0
    baseline: float = 0.0
    raw_tf: tuple = ()


@dataclass
class Features:
    pairs: dict
    terms: dict
    idfs: dict
    avg_title: float
    avg_body: float
    document_count: int


def build_features(docs_path, queries, candidates):
    """Stream corpus twice. Keep only candidate/query counts, never full texts."""
    terms = {qid: tuple(sorted(set(tokenize(text)))) for qid, text in queries.items()}
    query_sequences = {qid: tokenize(text) for qid, text in queries.items()}
    raw_queries = {qid: Counter(raw_tokens(text)) for qid, text in queries.items()}
    vocabulary = set(term for values in terms.values() for term in values)
    wanted = defaultdict(set)
    for qid, docs in candidates.items():
        for doc in docs:
            wanted[doc].add(qid)
    pairs, raw_df, normalized_df = {}, Counter(), Counter()
    document_count = title_total = body_total = 0
    seen = set()
    print("Pass 1/2: corpus statistics and candidate lexical features", file=sys.stderr)
    for doc_id, url, title, body in iter_docs(docs_path):
        if doc_id in seen:
            raise ValueError(f"duplicate document {doc_id}")
        seen.add(doc_id)
        title_raw, body_raw = raw_tokens(title), raw_tokens(body)
        raw_df.update(set(title_raw).union(body_raw))
        title_terms = (tokenize(title) if needs_normalization(title) else
                       [stem(token) for token in title_raw])
        body_terms = (tokenize(body) if needs_normalization(body) else
                      [stem(token) for token in body_raw])
        title_total += len(title_terms)
        body_total += len(body_terms)
        document_count += 1
        title_counts, body_counts = Counter(title_terms), Counter(body_terms)
        normalized_df.update((set(title_counts) | set(body_counts)) & vocabulary)
        if doc_id not in wanted:
            continue
        raw_counts = Counter(title_raw)
        raw_counts.update(body_raw)
        try:
            parsed_url = urlsplit(url)
            searchable_url = parsed_url.netloc + " " + parsed_url.path
        except ValueError:
            searchable_url = url  # malformed URLs should still have usable lexical evidence
        url_terms = set(tokenize(unquote(searchable_url).replace("_", " ")))
        title_bigrams = set(zip(title_terms, title_terms[1:]))
        body_bigrams = set(zip(body_terms, body_terms[1:]))
        for qid in wanted[doc_id]:
            query_terms = terms[qid]
            query_bigrams = set(zip(query_sequences[qid], query_sequences[qid][1:]))
            phrase = (sum(2 * (bigram in title_bigrams) + (bigram in body_bigrams)
                          for bigram in query_bigrams) / (3 * len(query_bigrams))) if query_bigrams else 0.0
            pairs[qid, doc_id] = Pair(
                tuple(title_counts[term] for term in query_terms),
                tuple(body_counts[term] for term in query_terms), len(title_terms), len(body_terms),
                len(set(query_terms) & url_terms) / max(1, len(query_terms)), phrase,
                raw_tf=tuple(raw_counts[term] for term in raw_queries[qid]))
    missing = set(wanted) - seen
    if missing:
        raise ValueError(f"{len(missing)} candidate document(s) missing from corpus: {sorted(missing)[:5]}")
    if not document_count:
        raise ValueError("document corpus is empty")
    idfs = {qid: tuple(math.log1p((document_count - normalized_df[term] + 0.5) /
                                 (normalized_df[term] + 0.5)) for term in query_terms)
            for qid, query_terms in terms.items()}
    for (qid, _), pair in pairs.items():
        denominator = sum(idfs[qid]) or 1.0
        pair.coverage = sum(idf for idf, t, b in zip(idfs[qid], pair.title_tf, pair.body_tf)
                            if t or b) / denominator
    # Smooth IDF and raw TF are sklearn TfidfVectorizer defaults. Its document
    # L2 norm includes ALL terms, including terms absent from every query.
    raw_idf = {term: math.log((document_count + 1) / (frequency + 1)) + 1
               for term, frequency in raw_df.items()}
    del raw_df
    query_weights = {qid: tuple(count * raw_idf.get(term, 0.0) for term, count in counts.items())
                     for qid, counts in raw_queries.items()}
    query_norms = {qid: math.sqrt(sum(value * value for value in values))
                   for qid, values in query_weights.items()}
    print("Pass 2/2: exact full-vocabulary TF-IDF document norms", file=sys.stderr)
    for doc_id, _, title, body in iter_docs(docs_path):
        if doc_id not in wanted:
            continue
        counts = Counter(raw_tokens(title))
        counts.update(raw_tokens(body))
        doc_norm = math.sqrt(sum((count * raw_idf[term]) ** 2 for term, count in counts.items()))
        for qid in wanted[doc_id]:
            pair = pairs[qid, doc_id]
            numerator = sum(tf * raw_idf.get(term, 0.0) * weight
                            for tf, term, weight in zip(pair.raw_tf, raw_queries[qid], query_weights[qid]))
            norm = doc_norm * query_norms[qid]
            pair.baseline = numerator / norm if norm else 0.0
            pair.raw_tf = ()
    return Features(pairs, terms, idfs, title_total / document_count,
                    body_total / document_count, document_count)


def score_candidates(features, qid, docs, params):
    base = [features.pairs[qid, doc].baseline for doc in docs]
    low, high = min(base, default=0.0), max(base, default=0.0)
    idfs = features.idfs[qid]
    denominator = sum(idfs) or 1.0
    result = []
    for doc, cosine in zip(docs, base):
        pair = features.pairs[qid, doc]
        bm25 = bm25f_score(pair.title_tf, pair.body_tf, pair.title_length,
                          pair.body_length, features.avg_title, features.avg_body, idfs, params)
        normalized_cosine = (cosine - low) / (high - low) if high > low else 0.0
        result.append(bm25 / denominator + params.url_weight * pair.url +
                      params.coverage_weight * pair.coverage + params.phrase_weight * pair.phrase +
                      params.baseline_weight * normalized_cosine)
    return result


def evaluate(features, qrels, qids, params=None, gain="linear"):
    values = []
    for qid in qids:
        docs = sorted(qrels[qid])
        labels = [qrels[qid][doc] for doc in docs]
        scores = ([features.pairs[qid, doc].baseline for doc in docs] if params is None else
                  score_candidates(features, qid, docs, params))
        values.append(ndcg(labels, scores, gain=gain))
    return sum(values) / len(values) if values else None


def split_queries(qids):
    """Stable query-level 80/20 holdout independent of filesystem/input order."""
    qids = sorted(qids, key=lambda qid: hashlib.sha256(qid.encode()).hexdigest())
    if len(qids) < 5:
        return sorted(qids), []
    count = max(1, len(qids) // 5)
    return sorted(qids[count:]), sorted(qids[:count])


def tune(features, qrels, qids, gain="linear", initial=Parameters()):
    """Deterministic coordinate grid search, permitted formula coefficients only."""
    params = initial
    best = evaluate(features, qrels, qids, params, gain)
    # Two sweeps let length normalization and field weights adapt together.
    grids = {
        "k1": (0.4, 0.8, 1.2, 2.0, 3.0),
        "b_body": (0.0, 0.3, 0.6, 0.85, 1.0),
        "title_weight": (1.0, 2.0, 4.0, 8.0, 12.0),
        "b_title": (0.0, 0.3, 0.7, 1.0),
        "coverage_weight": (0.0, 0.15, 0.3, 0.6, 1.0),
        "phrase_weight": (0.0, 0.15, 0.3, 0.6),
        "url_weight": (0.0, 0.15, 0.3, 0.6),
        "baseline_weight": (0.0, 0.25, 0.5, 1.0, 2.0),
    }
    if best is None:
        return params
    for sweep in range(2):
        for key, values in grids.items():
            for value in values:
                candidate = Parameters(**(asdict(params) | {key: value}))
                score = evaluate(features, qrels, qids, candidate, gain)
                if score > best + 1e-12:
                    params, best = candidate, score
        print(f"Grid sweep {sweep + 1}: fit nDCG@10={best:.6f}", file=sys.stderr)
    return params


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission_file", required=True, type=Path)
    parser.add_argument("--report_file", type=Path, help="write metrics and chosen parameters as JSON")
    parser.add_argument("--parameters", type=Path, help="reuse a JSON parameter object (or report file)")
    parser.add_argument("--no-tune", action="store_true", help="use fixed default/loaded parameters")
    parser.add_argument("--gain", choices=("linear", "exponential"), default="exponential",
                        help="gain for nDCG: default 2**label-1; linear is optional")
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args(argv)
    start = perf_counter()
    try:
        test_queries = read_queries(args.data_dir / "vkmarco-doceval-queries.tsv")
        objects = read_objects(args.data_dir / "objects.csv")
        train_queries = read_queries(args.data_dir / "vkmarco-doctrain-queries.tsv")
        qrels = read_qrels(args.data_dir / "vkmarco-doctrain-qrels.tsv")
        unknown = set(qrels) - set(train_queries)
        if unknown:
            raise ValueError(f"qrels contain unknown training queries: {sorted(unknown)[:5]}")
        unknown = {query for _, query, _ in objects} - set(test_queries)
        if unknown:
            raise ValueError(f"objects contain unknown evaluation queries: {sorted(unknown)[:5]}")
        # Query IDs are local to each input split; preserve distinct texts even
        # when both files number their queries from zero. Choose a namespace
        # that cannot collide with any original training identifier.
        test_prefix = "__evaluation__:"
        while any(test_prefix + qid in train_queries for qid in test_queries):
            test_prefix = "_" + test_prefix
        test_queries = {test_prefix + qid: text for qid, text in test_queries.items()}
        objects = [(oid, test_prefix + qid, doc) for oid, qid, doc in objects]
        queries = train_queries | test_queries
        candidates = {qid: set(docs) for qid, docs in qrels.items()}
        for _, qid, doc in objects:
            candidates.setdefault(qid, set()).add(doc)
        features = build_features(args.data_dir / "vkmarco-docs.tsv", queries, candidates)
        params = Parameters()
        if args.parameters:
            loaded = json.loads(args.parameters.read_text(encoding="utf-8"))
            params = Parameters(**loaded.get("parameters", loaded)).validate()
        fit, validation = split_queries(qrels)
        if not args.no_tune and not args.parameters and fit:
            params = tune(features, qrels, fit, args.gain, params)
        report = {
            "parameters": asdict(params), "gain": args.gain, "metric": "nDCG@10",
            "tie_policy": "expected gain within tied score groups",
            "document_count": features.document_count, "fit_query_count": len(fit),
            "validation_query_count": len(validation), "test_object_count": len(objects),
            "fit_baseline": evaluate(features, qrels, fit, gain=args.gain),
            "fit_model": evaluate(features, qrels, fit, params, args.gain),
            "validation_baseline": evaluate(features, qrels, validation, gain=args.gain),
            "validation_model": evaluate(features, qrels, validation, params, args.gain),
            "train_baseline": evaluate(features, qrels, sorted(qrels), gain=args.gain),
            "train_model": evaluate(features, qrels, sorted(qrels), params, args.gain),
        }
        test_scores = {}
        for qid in sorted({query for _, query, _ in objects}):
            docs = sorted(candidates[qid])
            test_scores.update({(qid, doc): score for doc, score in
                                zip(docs, score_candidates(features, qid, docs, params))})
        with args.submission_file.open("w", encoding="utf-8", newline="") as output:
            writer = csv.writer(output)
            writer.writerow(["ObjectId", "Score"])
            for object_id, qid, doc in objects:
                writer.writerow([object_id, format(test_scores[qid, doc], ".17g")])
        report["elapsed_seconds"] = round(perf_counter() - start, 3)
        if args.report_file:
            args.report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
