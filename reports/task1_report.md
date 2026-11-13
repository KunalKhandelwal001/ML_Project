# Task 1 report draft — Introduction, dataset, evaluation, reproducibility

> **Status:** draft. The numbers below come from the verified real-data run of
> `python main.py --save-splits --smoke-train` on 14 November 2026.
> The experiment sections that compare clean FedAvg, attack, and defense are
> owned by Tasks 2–4 and should be merged into this draft later.

## 1. Introduction

Federated learning (FL) lets several participants train a shared model without
pooling their raw data. Each client trains locally and sends a model update to
a central server, which combines the updates into a global model. This is
attractive for healthcare data, where privacy rules and institutional
boundaries make centralised data collection difficult.

The same distribution makes FL vulnerable to **model poisoning**. A malicious
client can submit an update that looks valid but pushes the global model toward
poor predictions. Because the server never sees the client's data, it cannot
inspect the update directly. A single compromised participant can therefore
degrade a model that is shared by everyone.

FedDef studies this problem on a concrete clinical task: predicting ten-year
coronary heart disease (CHD) risk from the Framingham Heart Study dataset. The
project simulates 20 clients, of which a configurable fraction are malicious,
and compares three settings:

1. **Clean FedAvg** — no attackers, the reference behaviour.
2. **Attacked FedAvg** — malicious clients submit corrupted updates and the
   server aggregates them without protection.
3. **Defended FedAvg** — the server scores client updates on a held-out
   validation set, accumulates suspicion over rounds, and down-weights or
   excludes anomalous clients before aggregation.

The contribution of this work is a small, reproducible end-to-end study rather
than a new algorithm: every step from raw CSV to metric table is scripted, the
defense is evaluated against the same splits as the baseline, and all
hyperparameters live in one configuration object.

## 2. Dataset

The Framingham Heart Study dataset is a binary classification dataset with
4,240 records and 15 predictor variables. The target is `TenYearCHD`:

| Target | Meaning | Count |
|---|---|---:|
| `0` | No ten-year CHD | 3,596 |
| `1` | Ten-year CHD | 644 |
| **Total** | | **4,240** |

The positive rate is **15.19%**, so the dataset is imbalanced. A model that
always predicts `0` already reaches about 84.8% accuracy, which is why accuracy
alone is not used to judge a model.

Features cover demographics, smoking, blood pressure, cholesterol, diabetes,
and other clinical measurements:

```text
male, age, education, currentSmoker, cigsPerDay, BPMeds,
prevalentStroke, prevalentHyp, diabetes, totChol, sysBP,
diaBP, BMI, heartRate, glucose
```

Missing values are common in the raw data (645 cells in the distributed CSV).
The pipeline handles them as follows:

- rows with a missing target are dropped;
- numeric features are median-imputed;
- categorical features are most-frequent-imputed and one-hot encoded;
- imputation and encoding statistics are fitted on the **training split only**;
- features are standardised with a `StandardScaler` fitted on the **training
  split only**.

The resulting split is stratified and seeded:

| Split | Rows | Positive rate |
|---|---:|---:|
| Train | 2,968 | 15.20% |
| Validation | 424 | 15.09% |
| Test | 848 | 15.21% |

The test split is reserved for final evaluation. In particular, the defense in
Task 3 must score clients on the validation split, never on the test split.

## 3. Evaluation methodology

The model predicts a binary label, so the project reports a standard set of
classification metrics for the positive class (`CHD = 1`):

- **Accuracy** — overall fraction correct; reported for completeness only.
- **Precision** — of the patients flagged as high risk, how many truly develop
  CHD within ten years.
- **Recall (sensitivity)** — of the patients who develop CHD, how many the model
  flags.
- **F1** — harmonic mean of precision and recall; the primary summary metric on
  this imbalanced dataset.
- **ROC-AUC** — threshold-independent ranking quality.
- **Confusion matrix** — raw counts, so the trade-off between false positives
  and false negatives is visible.

Each experiment is repeated across seeds 42, 43, and 44, and results are
reported as **mean ± standard deviation** rather than as a single run. This
applies to the core matrix in `team_plan.md` (malicious fractions 0.0, 0.1,
0.2, 0.3, and 0.5).

The metric code lives in `src/evaluate.py`. It validates that both label
vectors are binary, rejects mismatched lengths, handles one-class inputs
without crashing, and saves metrics as JSON plus an optional confusion-matrix
PNG.

### Pipeline smoke check

As a sanity check that the data and evaluation code agree, a centralized
`LogisticRegression` (`class_weight="balanced"`, `max_iter=2000`) was trained
on the training split and evaluated on the validation split:

| Metric | Validation value |
|---|---:|
| Accuracy | 0.6392 |
| Precision | 0.2428 |
| Recall | 0.6562 |
| F1 | 0.3544 |
| ROC-AUC | 0.7120 |

This is **not** the Task 2 FedAvg baseline. It only demonstrates that the
pipeline learns a non-trivial signal: ROC-AUC is clearly above chance and
recall is high enough that the positive class is not ignored. The low precision
is expected, because the positive class is only 15% of the data.

## 4. Reproducibility

Every run is controlled by a single validated configuration object
(`src/config.py`) and a seed. The default setup is:

| Parameter | Value |
|---|---|
| Seed | 42 |
| Train / validation / test | 70% / 10% / 20% |
| Clients | 20 |
| Malicious fraction | 0.2 |
| Rounds / epochs | 50 |
| Learning rate | 0.2 |
| Attack type | random noise |
| Defense threshold `theta` | 0.25 |
| Anomaly threshold `z0` | 1.0 |

Reproduction steps:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# place framingham.csv at data/framingham.csv (see data/README.md)

python main.py --save-splits --smoke-train
pytest
streamlit run app.py
```

`main.py` writes `results/run_config.json` and
`results/data_summary.json`; the latter records the seed, split sizes,
positive rates, feature names, imputation values, and the number of dropped
rows. The prepared splits are written to `results/splits/` only when
`--save-splits` is passed.

Reproducibility rules followed by the project:

- one `Config` object per experiment, serialised next to its results;
- `numpy.random.default_rng(seed)` for new random operations;
- the same seeded, stratified split for every setting being compared;
- no test-set access during training, attack tuning, or defense tuning;
- no metric offsets (the prototype added `+0.2` to some scores) and no R² on
  binary labels;
- mean ± standard deviation across seeds in the final tables.

### Verification status

- `55` automated tests pass, including data leakage, determinism, metric
  correctness, CLI execution, and Streamlit app-shell rendering.
- The real Framingham CSV was loaded successfully: 4,240 rows, 15 features,
  645 missing values imputed, three disjoint stratified splits, and the
  validation smoke check above.
- The raw CSV is **not** committed to the repository. `data/*.csv` is ignored
  so that patient-level records are not published accidentally.
