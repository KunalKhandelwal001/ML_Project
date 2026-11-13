# FedDef

**Robust Aggregation Against Model Poisoning Attacks in Federated Learning**

FedDef simulates multiple clients jointly training a logistic-regression model
for ten-year coronary heart disease risk prediction, without sharing raw
patient records. A configurable fraction of clients behave maliciously and
submit corrupted model updates. The project compares:

1. clean FedAvg (no attackers)
2. FedAvg under attack, with no defense
3. FedAvg under attack, with an anomaly-aware defense

The defense scores each client's contribution using a robust z-score over
validation-set predictions, accumulates a suspicion score, and down-weights or
excludes suspicious clients before aggregation.

## Current status

**Task 1 (foundation) is implemented.** This includes:

- reproducible data loading, cleaning, stratified splitting, and scaling
- train-only imputation/encoding/scaling with no validation/test leakage
- a shared evaluation harness with accuracy, precision, recall, F1, ROC-AUC,
  and confusion matrices
- a CLI smoke check and a Streamlit app shell
- automated tests using a synthetic Framingham-shaped dataset

**Still to come:**

- Task 2: clean FedAvg baseline and centralized reference
- Task 3: z-score defense and validation protocol
- Task 4: attack simulation and the full experiment matrix

See [`team_plan.md`](team_plan.md) for the workstreams and owners.

## Repository layout

```text
main.py                 CLI: data summary + optional smoke check
app.py                  Streamlit app shell
src/
  config.py             shared, validated experiment configuration
  data.py               loading, cleaning, splitting, scaling
  evaluate.py           metrics, JSON saving, confusion-matrix plots
tests/
  conftest.py           synthetic Framingham-shaped fixture
  test_data.py
  test_evaluate.py
  test_config.py
  test_main.py
  test_app.py
data/README.md          dataset instructions (raw CSV is gitignored)
archive/                original prototype, kept for reference only
results/                generated summaries, splits, metrics, plots
team_plan.md            four-person task plan
```

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The project was verified with Python 3.14 and the versions in
`requirements.txt`.

## Data and evaluation API

Tasks 2–4 should import the frozen interfaces instead of reimplementing data
handling or metrics:

```python
from src.config import Config
from src.data import load_data, prepare_data
from src.evaluate import evaluate_model, save_metrics

config = Config()
bundle = load_data(
    config.data_path,
    target_col=config.target_col,
    test_size=config.test_size,
    val_size=config.val_size,
    random_state=config.seed,
)

# bundle.X_train, bundle.X_val, bundle.X_test are scaled DataFrames.
# bundle.y_train, bundle.y_val, bundle.y_test are aligned Series.
# bundle.scaler, bundle.preprocessor, bundle.feature_names, bundle.metadata
# are available for inspection and saving.

metrics = evaluate_model(model, bundle.X_val, bundle.y_val)
save_metrics(metrics, config.results_dir / "metrics.json")
```

For an uploaded DataFrame (for example in Streamlit), call
`prepare_data(frame, ...)` instead of `load_data(path, ...)`.

`src.data` guarantees:

- a stratified, seeded train/validation/test split
- binary 0/1 target values
- median/mode imputation and one-hot encoding fitted on train only
- `StandardScaler` fitted on train only
- aligned, finite feature matrices with no missing values

## Dataset

Place the Framingham CSV at `data/framingham.csv`. See
[`data/README.md`](data/README.md) for the expected columns and preprocessing
details.

The raw CSV is **not** committed to the repository.

## Run the foundation check

```bash
python main.py
```

This creates:

- `results/run_config.json`
- `results/data_summary.json`

To also save the prepared splits and run a centralized logistic-regression
sanity check on the validation split:

```bash
python main.py --save-splits --smoke-train
```

The smoke check is **not** the Task 2 FedAvg baseline. It only verifies that
the data and evaluation pipeline works.

To run a final held-out test evaluation (use sparingly):

```bash
python main.py --smoke-train --evaluate-test
```

Useful options:

```text
--data PATH             explicit CSV path
--target COLUMN         target column name (default: TenYearCHD)
--seed INT              split/model seed (default: 42)
--test-size FLOAT       test fraction (default: 0.2)
--val-size FLOAT        validation fraction (default: 0.1)
--results-dir PATH      output directory (default: results)
--save-splits           save prepared train/val/test CSVs
--smoke-train           run the centralized sanity check
--evaluate-test         also evaluate that smoke model on test
```

## Run the app shell

```bash
streamlit run app.py
```

The app currently provides:

- CSV upload and default `data/framingham.csv` loading
- configurable seed, test fraction, and validation fraction
- split sizes and positive rates
- prepared training-feature preview and metadata
- a validation-only centralized smoke check

The Baseline, Attack, and Defense tabs are placeholders for Tasks 2–4.

The old `streamlit.py` prototype was moved to
[`archive/streamlit_prototype.py`](archive/streamlit_prototype.py). It is not
maintained and contains known bugs; use `app.py`.

## Tests

```bash
pytest
```

The test suite does not require the real Framingham CSV. It generates a
synthetic dataset with the same schema and verifies:

- split sizes, disjointness, and seed determinism
- train-only preprocessing (validation/test values cannot change the scaler)
- missing-value imputation and categorical one-hot encoding
- target validation and missing-target handling
- metric correctness, JSON round-tripping, and plot creation
- end-to-end CLI execution
- Streamlit app-shell rendering

## Reproducibility rules

- Use `Config` (`src/config.py`) for every experiment.
- Use `np.random.default_rng(seed)` for new random operations.
- Keep the test split untouched until final evaluation.
- The defense must use the validation split, not the test split.
- Report mean ± standard deviation across seeds, not a single lucky run.
- No metric offsets such as `+0.2` and no R² on binary labels.

## Team

See [`team_plan.md`](team_plan.md) for the full four-person plan.

| Member | Primary workstream |
|---|---|
| Kunal Khandelwal | Foundation, data pipeline, evaluation harness |
| Dewansh Khandelwal | Clean FedAvg baseline + centralized reference |
| Kunal V Kurundwadkar | Defense mechanism + validation protocol |
| Pratik Rakhecha | Attack simulation + experiment orchestration |
