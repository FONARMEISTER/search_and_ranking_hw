# HW2: implementation and reproduction

Author: Ilya Sarantsev. Kaggle nickname: `ilyasarantsev` (as recorded in the
existing HW1 README). Public leaderboard score: pending submission.

The assignment is in [README.md](README.md). The implementation uses only the
competition files and permitted lexical ranking methods. All ranking formulas,
the metric, and the parameter search are implemented in `solution.py`.

## Run

From the repository root, create an isolated environment (Python 3.10 or later;
verified with Python 3.12.13):

```bash
python3 -m venv .venvs/hw2-text-ranking
source .venvs/hw2-text-ranking/bin/activate
pip install -r homeworks/hw2-text-ranking/requirements.txt
cd homeworks/hw2-text-ranking
./solution.py --submission_file=submission.csv --report_file=report.json /path/to/hw2-data
```

The data directory must contain `vkmarco-docs.tsv`, both query TSV files,
`vkmarco-doctrain-qrels.tsv`, and `objects.csv`, extracted from the **HW2**
competition archive. HW1's similarly named files are a different corpus.

The required invocation also works without `--report_file`:

```bash
./solution.py --submission_file=submission.csv /path/to/hw2-data
```

When using the standalone hand-in archive, its files are already at the top
level. Create/activate a virtual environment there, run `pip install -r
requirements.txt`, then use the same `./solution.py` command. In the standalone
archive, run checks with `python -m unittest discover -s tests -v`.

The output preserves every ObjectId and the input row order, with header
`ObjectId,Score`. Identifiers are treated as strings. Repeated query/document
pairs receive the same score. Missing candidates, conflicting labels, duplicate
identifiers, and unknown query references produce an error.

## Formula

Unicode word tokens are lowercased; the BM25F branch normalizes Russian accents
and `ё`, then applies Snowball Russian stemming. No external dictionaries or
pretrained resources are downloaded. The cosine baseline uses unstemmed tokens
of at least two characters, raw term frequency, smoothed IDF
`1 + log((N+1)/(df+1))`, and full-document L2 normalization, matching the default
word analyzer and weighting of `TfidfVectorizer`.

For each stemmed query term, BM25F first combines title and body frequencies:

```text
f = title_weight * title_tf / (1 - b_title + b_title * title_len / avg_title_len)
    + body_tf / (1 - b_body + b_body * body_len / avg_body_len)
idf = log(1 + (N - df + 0.5) / (df + 0.5))
BM25F = sum(idf * (k1 + 1) * f / (k1 + f))
```

The final score combines BM25F divided by the query's total IDF, IDF-weighted
query coverage, adjacent query-bigram matches (title and body weights 2:1), URL
token coverage, and candidate-wise min-max normalized TF-IDF cosine. Empty and
out-of-vocabulary queries receive finite scores. URL tokens affect their own
coverage feature, not title/body corpus statistics.

The initial coefficients are `k1=1.2`, `b_title=0.3`, `b_body=0.6`,
`title_weight=4`, `url_weight=0.15`, `coverage_weight=0.3`, `phrase_weight=0.15`,
and `baseline_weight=0`. These are starting values, not measured optimal values.
By default, two deterministic coordinate-grid sweeps tune these coefficients
using **only the fit portion of the training labels**. No classifier, learned
ranker, embeddings, or ready-made scoring library is used.

## Evaluation and parameter replay

SHA-256 of QueryId defines a fixed query-level 80/20 fit/validation split (for
fewer than five labeled queries, all queries are used for fitting). The validation
labels do not select parameters. The report records fit, validation, and combined
training nDCG@10 for both the baseline and the resulting formula, together with
parameters, query counts, elapsed time, and corpus size. Combined training
metrics include fitting queries and should not be treated as held-out estimates.

The CLI default metric uses **exponential relevance gains**, `2**label - 1`, and
averages expected DCG over tied scores. All-zero relevance queries score zero.
This gain convention is inferred from the assignment: expected random-order
nDCG@10 on the actual 1,000 training queries is 0.579770 with exponential gains,
matching its stated approximate 0.58; linear gains give 0.698481. The binary
relevance data in the course seminar cannot distinguish these conventions.
`--gain linear` is available for comparison; the live competition's evaluation
page was not accessible for direct verification.

To reproduce scores without repeating parameter search:

```bash
./solution.py --parameters=report.json --submission_file=replayed.csv /path/to/hw2-data
```

`--parameters` accepts either a saved report or a plain JSON parameter object and
automatically skips tuning. `--no-tune` uses the initial coefficients. All corpus
statistics still come from the supplied complete document corpus. Two streaming
passes retain candidate features and corpus term frequencies rather than all
document texts; stemming uses a bounded cache.

## Checks and competition status

```bash
python -m unittest discover -s homeworks/hw2-text-ranking/tests -v
```

Run that test command from the repository root. Tests cover hand-calculated
BM25F, TF-IDF norms and query term frequency, linear/exponential nDCG, ties across
the cutoff, empty/OOV queries, submission row ordering, and deterministic replay.
The requirements file matches the clean Python 3.12 verification environment.

Measured on the supplied HW2 competition archive, with exponential gains:

| Split | Queries | TF-IDF cosine | Tuned formula |
|---|---:|---:|---:|
| Fit | 800 | 0.606343 | 0.623437 |
| Held-out validation | 200 | 0.617381 | 0.624715 |
| Combined training | 1,000 | 0.608550 | 0.623693 |

Chosen parameters: `k1=2`, `b_title=0.3`, `b_body=0.6`, `title_weight=8`,
`url_weight=0`, `coverage_weight=0.3`, `phrase_weight=0.3`,
`baseline_weight=0.25`. They are also saved in `report.json`.

The full run processed 74,848 documents (981 MB of TSV text), tuned parameters,
and wrote all 50,353 test scores in **91.96 seconds**, with peak resident memory
**708,345,856 bytes** (0.71 GB), measured using `resource.getrusage` on macOS
ARM64 with CPython 3.12.13. Machine-dependent measurements are in `runtime.json`.
These are below the assignment's 30-minute and 8-GB limits on this machine;
Linux execution was not available for a separate measurement.

A second full-corpus run using the saved parameters reproduced `submission.csv`
byte for byte. Its SHA-256 is
`3fb5d097407893eb0af2e3c2959a57c61486a370d28c53ed407132a508fbaed2`.

**Public leaderboard verification is pending.** The Kaggle CLI download was
attempted, but Kaggle authentication is not configured on this machine. The
archive was then supplied manually. The local held-out improvement is measured;
the public/private leaderboard scores have not been measured. Upload
`submission.csv` to the competition and record the resulting public score in
this README before final hand-in. The hand-in archive omits the dataset and the
large prediction CSV; it includes the source, pinned environment, report, tests,
and this guide as `README.md`.
