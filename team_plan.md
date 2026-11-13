# FedDef — Team Plan

**Project:** FedDef: Robust Aggregation Against Model Poisoning Attacks in Federated Learning  
**Repository:** https://github.com/maskeensingh/FEDDEF  
**Plan date:** 14 November 2026  
**Status:** Prototype → working, reproducible experiment

---

## 1. Team and ownership

| Order | Member | Roll number | Primary workstream |
|---|---|---|---|
| **1** | **Kunal Khandelwal (you)** | IIT2024162 | Foundation, data pipeline, evaluation harness |
| 2 | Dewansh Khandelwal | IIT2024173 | Correct FedAvg baseline + centralized reference |
| 3 | Kunal V Kurundwadkar | IIT2024177 | Defense mechanism + validation protocol |
| 4 | Pratik Rakhecha | IIT2024176 | Attack simulation + experiment orchestration |

> **Note:** Task 1 is intentionally assigned to Kunal Khandelwal, as requested. Tasks 2–4 are split equally among the other three members.  
> The division is based on **dependency order**, not strictly on the original proposal roles. The final report's contribution section should describe the work actually done.

---

## 2. Definition of done

The project is complete when all of the following are true:

- [ ] A fresh clone + `pip install -r requirements.txt` + dataset runs the baseline with one command.
- [ ] The same data split and seeds are used for all three settings.
- [ ] The three settings produce comparable convergence curves:
  1. Clean FedAvg (no attackers)
  2. FedAvg under attack (no defense)
  3. FedAvg under attack with the defense
- [ ] The defense improves the attacked model without materially degrading the clean model.
- [ ] Results are saved as CSV/PNG files and are reproducible from a single experiment command.
- [ ] `streamlit run app.py` works and can show baseline, attack, and defense results.
- [ ] The report contains methodology, threat model, results, limitations, and per-person contributions.
- [ ] Every member has visible commits/PRs on GitHub.

---

## 3. Shared architecture (freeze before coding)

Everyone must code against the same interfaces to avoid merge conflicts and shape bugs.

```text
src/
  data.py          # load, clean, split, scale
  federated.py     # local training, FedAvg, server loop
  evaluate.py      # metrics and plots
  attacks.py       # poisoning attacks
  defense.py       # anomaly scoring and robust aggregation
  experiments.py   # config-driven experiment runner
  config.py        # shared config + seeds
app.py             # Streamlit
tests/
results/
```

**Shared parameter format**

```python
Params = {
    "coef": np.ndarray,       # shape (n_features,), not (1, n_features)
    "intercept": float,
}
```

Rules:

- Always use `np.random.default_rng(seed)`; never global `np.random`.
- The test set is opened only for final evaluation. The defense uses a separate validation set.
- Every result file must record the seed, setting, malicious fraction, attack type, and epoch.
- No hard-coded `+0.2` offsets or R²-on-binary-labels. Use real classification metrics.

---

## 4. Workstreams

### Task 1 — Kunal Khandelwal (you): Foundation, data pipeline, evaluation harness

**Goal:** Make the repository runnable and create the shared data/evaluation contract that everyone else uses.

**Why first:** Every other task depends on a single, correct data split and a trustworthy metric definition.

**Deliverables**

- [x] Add `requirements.txt` with pinned versions of `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `streamlit`, `pytest`.
- [x] Add `.gitignore`, README skeleton, and dataset instructions (`data/README.md`). Do not commit the raw dataset if its license is unclear.
- [x] Add `src/config.py` with a `Config` dataclass: seed, split sizes, `n_clients`, `mal_fraction`, `epochs`, `lr`, `theta`, `attack_type`, `attack_strength`.
- [x] Add `src/data.py`:
  - load `framingham.csv`
  - drop irrelevant/ID columns
  - handle missing values (median/mode imputation; document the choice)
  - stratified train/validation/test split, e.g. 70/10/20
  - fit `StandardScaler` on the training set only
  - return feature names, arrays, and class balance
- [x] Add `src/evaluate.py`: accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and `save_metrics()`.
- [x] Fix `main.py`: remove the missing-import crash, dead code, and duplicate `train_model`; use one split only.
- [x] Add `tests/test_data.py`: no train/test leakage, correct shapes, deterministic seed, class balance sanity check.
- [x] Add the Streamlit app shell (`app.py`) and archive the old prototype.

**Task 1 status:** implementation complete and verified on the real dataset.
`python main.py --save-splits --smoke-train` loaded 4,240 rows, 15 features,
and 645 missing values; it produced a 2,968 / 424 / 848 train/validation/test
split with positive rates of 15.20% / 15.09% / 15.21%. The full suite passes
(`59 tests` including optional real-data checks; `55` without the CSV).
A report draft for the Task 1 sections is in
[`reports/task1_report.md`](reports/task1_report.md).

**Acceptance criteria**

- [x] Fresh clone + dataset + install + `python main.py` runs to completion (verified with the synthetic fixture; real CSV pending).
- [x] `results/data_summary.json` reports split sizes and positive rate.
- [x] The test set is never touched during training or defense tuning.
- [x] All other members can import `src.data` and `src.evaluate` without editing them.

**Estimated effort:** 18–20 hours  
**Depends on:** nothing  
**Hand-off:** frozen data/eval interfaces + generated split files

---

### Task 2 — Dewansh Khandelwal: Correct FedAvg baseline + centralized reference

**Goal:** Implement a clean, no-attack FedAvg baseline and prove it is close to a centralized logistic regression.

**Deliverables**

- [ ] Add `src/federated.py`:
  - partition the training set into `n_clients` with a seeded random split
  - train local `LogisticRegression` with `max_iter=1000`, `class_weight="balanced"`
  - implement `fedavg(updates, weights)` weighted by client sample count
  - implement the server loop
- [ ] Decide and document the FL protocol:
  - **Preferred:** true FedAvg — clients start each round from the global parameters and perform local steps.
  - If time is short, use one-shot averaging but name it correctly in the report.
- [ ] Train a centralized logistic regression on the full training set as a reference.
- [ ] Save the baseline convergence curve and metrics under `results/baseline/`.
- [ ] Add `tests/test_federated.py`: weighted-average correctness, client partition covers all rows, identical models average to themselves, same seed reproduces the same result.

**Acceptance criteria**

- Baseline accuracy, F1, and ROC-AUC are reported on the held-out test set.
- FedAvg is within a documented tolerance of the centralized model (suggested: ±2–3% accuracy).
- The baseline curve is reproducible from one command.
- No attacker code is involved in this task.

**Estimated effort:** 18–20 hours  
**Depends on:** Task 1 interfaces; can start with a temporary data loader

---

### Task 3 — Kunal V Kurundwadkar: Defense mechanism + validation protocol

**Goal:** Implement the z-score anomaly defense correctly and show that it reduces attack damage without hurting the clean case.

**Deliverables**

- [ ] Add `src/defense.py` with `score_clients(updates, reference_data, state, config) -> weights`.
- [ ] Use a robust z-score:
  - per reference sample, compute each client's prediction
  - use median and MAD instead of mean/std
  - `z = 0.6745 * (x - median) / MAD`; handle `MAD = 0`
- [ ] Implement client reputation:
  - accumulate anomaly above a threshold `z0`
  - optionally decay old evidence each round
  - exclude/down-weight clients above `theta`
- [ ] Fix the aggregation math:
  - renormalize weights after zeroing suspicious clients
  - handle the all-clients-flagged case (skip round or fall back to median update)
  - avoid division by zero
- [ ] Use a server-side **validation set**, never the test set, for anomaly scoring.
- [ ] Integrate defense on/off into the server loop and save per-client scores.
- [ ] Add `tests/test_defense.py`: clean defense ≈ FedAvg, attack without defense degrades, defense recovers, θ sweep works.

**Acceptance criteria**

- Clean-case accuracy drop is < 1% versus the undefended baseline.
- Attack-case accuracy/F1 improves versus the undefended attacked model.
- Malicious clients receive the highest anomaly scores in the logs.
- Results are saved under `results/defense/`.

**Estimated effort:** 18–20 hours  
**Depends on:** Tasks 1–2 interfaces; can be developed against mock updates

---

### Task 4 — Pratik Rakhecha: Attack simulation + experiment orchestration

**Goal:** Implement realistic attacks and a reproducible experiment matrix that produces the final tables and plots.

**Deliverables**

- [ ] Add `src/attacks.py`:
  - random noise attack
  - sign-flip attack
  - scaling attack
  - label-flipping attack
  - colluding attackers
- [ ] Add a bounded `attack_strength` parameter. Remove the arbitrary `beta = sigma / (lr * n)` scaling and the 100,000 intercept.
- [ ] Generate per-feature noise with `rng.normal(0, strength, size=coef.shape)`.
- [ ] Select malicious clients randomly with a seed, not always the first `k`.
- [ ] Add `src/experiments.py`:
  - loop over seeds, settings, malicious fractions, and attack types
  - save tidy CSV: `seed, setting, mal_fraction, attack_type, attack_strength, epoch, accuracy, precision, recall, f1, roc_auc`
  - aggregate mean ± std across seeds
- [ ] Add `run_experiments.py` as a single entry point.
- [ ] Produce plots: convergence curves, final metric vs malicious fraction, defense gain.
- [ ] Validate the attack first: undefended performance must drop. If it does not, fix the attack before tuning the defense.

**Acceptance criteria**

- One command runs the core matrix and writes `results/summary.csv`.
- Same seed/config produces identical results.
- The attack degrades the undefended model by a clear margin.
- Plots are ready to paste into the report.

**Estimated effort:** 18–20 hours  
**Depends on:** Tasks 1–3 interfaces; attack module can start early

---

## 5. Equal workload check

| Member | Primary workstream | Shared work | Total |
|---|---|---|---|
| Kunal Khandelwal | 18–20h foundation/data/eval | 3–4h app shell + report intro/data/eval | 21–24h |
| Dewansh Khandelwal | 18–20h baseline/FedAvg | 3–4h baseline app tab + report FL/FedAvg | 21–24h |
| Kunal V Kurundwadkar | 18–20h defense/tests | 3–4h defense app tab + report defense | 21–24h |
| Pratik Rakhecha | 18–20h attacks/experiments | 3–4h attack app tab + report results | 21–24h |

**Rule:** shared work is capped at 4 hours per person. If an app/report item grows beyond that, the owner raises it at the weekly check-in and we rebalance.

---

## 6. Milestones

Dates are suggestions; adjust to your actual deadline.

| Week | Focus | Owners | Exit criteria |
|---|---|---|---|
| Week 1 | Freeze interfaces; foundation; baseline design | Kunal K, Dewansh | Data/eval contract works; baseline skeleton runs |
| Week 2 | Baseline complete; attack v1; defense v1 | Dewansh, Pratik, Kunal V | Clean baseline curve; attack degrades model; defense reduces damage |
| Week 3 | Full experiments + app integration + report draft | All | `results/summary.csv`; app shows three settings; report draft |
| Week 4 | Final runs, report polish, demo | All | Reproducible final results; report and demo ready |

---

## 7. Experiment matrix

Start with the **core matrix**; add the extended matrix only after the core works.

**Core matrix**

| Dimension | Values |
|---|---|
| Settings | clean, attack, attack + defense |
| Malicious fraction | 0.0, 0.1, 0.2, 0.3, 0.5 |
| Seeds | 42, 43, 44 |
| Attack type | random noise |
| Epochs | 50 or until convergence |

**Extended matrix**

| Dimension | Values |
|---|---|
| Attack types | random, sign flip, scaling, label flip, collusion |
| Malicious fraction | 0.2 |
| Seeds | 42, 43, 44 |

Report mean ± standard deviation across seeds, not a single run.

---

## 8. Streamlit and report responsibilities

### Streamlit

| Member | Responsibility |
|---|---|
| Kunal K | App shell, upload, config panel, baseline tab; fix the current one-row `LogisticRegression.fit` startup crash |
| Dewansh | Baseline metrics and convergence plot |
| Kunal V | Defense toggle, threshold control, per-client anomaly table |
| Pratik | Attack controls, comparison curves, results download |

**App acceptance:** `streamlit run app.py` works from a clean environment; the user can compare baseline, attack, and defense; the app does not retrain on every widget change (`st.cache_data`/`st.session_state`).

### Report

| Member | Section |
|---|---|
| Kunal K | Introduction, dataset, evaluation methodology, reproducibility |
| Dewansh | Federated learning background, FedAvg, baseline results |
| Kunal V | Threat model, defense method, defense results |
| Pratik | Attack design, experiment protocol, results and plots |
| All | Discussion, limitations, per-person contributions |

A nominated editor compiles the final draft; everyone reviews it once.

---

## 9. Collaboration rules

- Branch names:
  - `feat/kunal-k-foundation`
  - `feat/dewansh-baseline`
  - `feat/kunal-v-defense`
  - `feat/pratik-attacks`
- No direct pushes to `main`. Open a PR for each workstream.
- Each PR must include: what changed, how it was tested, and a sample command/output.
- One reviewer per PR, rotating.
- Weekly 30-minute check-in: what is done, what is blocked, what is next.
- Commit the plan and the proposal PDF. Keep `results/summary.csv` in git; keep large raw outputs out.
- If a task slips by more than two days, tell the team immediately and rebalance.

---

## 10. Known risks and mitigations

| Risk | Mitigation |
|---|---|
| Dataset missing or license unclear | Add download instructions; do not commit raw data if restricted |
| Class imbalance makes accuracy misleading | Report F1 and ROC-AUC; use `class_weight="balanced"` |
| A client has only one class | Guard local training; merge/skip that client for the round |
| Test-set leakage in defense | Use a separate validation set; code review the split |
| Attack is too weak or too extreme | Validate attack before defense tuning; use bounded strength |
| Defense destroys clean accuracy | Compare clean defense vs clean FedAvg; tune θ |
| Experiments take too long | Run core matrix first; cache results; reduce seeds if needed |
| Merge conflicts | Freeze interfaces in Week 1; small PRs |
| Unequal workload | Weekly check-in; shared work capped at 4h/person |

---

## 11. First actions (this week)

1. **Kunal K:** ✅ Task 1 complete — `requirements.txt`, `.gitignore`, `src/config.py`, `src/data.py`, `src/evaluate.py`, `main.py`, `app.py`, and the test suite are in place. Interfaces are published.
2. **Dewansh:** write `src/federated.py` against the interfaces; produce the first clean baseline curve.
3. **Kunal V:** design `src/defense.py` and the validation-set protocol; test against mock updates.
4. **Pratik:** design `src/attacks.py` and `src/experiments.py`; verify the attack shape/strength on mock updates.
5. **All:** review the frozen `src.data` / `src.evaluate` interfaces and start feature PRs.
