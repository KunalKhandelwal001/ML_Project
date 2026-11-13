# Dataset instructions

FedDef uses the **Framingham Heart Study** dataset for ten-year coronary heart
disease (CHD) risk prediction.

## Expected file

Place the CSV at:

```text
data/framingham.csv
```

The loader also accepts `framingham.csv` in the repository root, but
`data/framingham.csv` is the recommended location.

## Expected columns

The target column must be named `TenYearCHD` and contain binary values `0` and
`1`. The standard Framingham feature columns are:

```text
male, age, education, currentSmoker, cigsPerDay, BPMeds,
prevalentStroke, prevalentHyp, diabetes, totChol, sysBP,
diaBP, BMI, heartRate, glucose
```

The loader is tolerant of common identifier columns such as `id`,
`patient_id`, or `Unnamed: 0`; those columns are dropped automatically.

## Where to get it

The dataset is commonly distributed on Kaggle as the "Framingham Heart Study"
CSV. Download it from the source you are permitted to use, then rename the file
to `framingham.csv`. If your course provides the dataset, use that copy.

Do **not** commit the raw CSV to this repository. `data/*.csv` is ignored by
`.gitignore` to avoid accidentally publishing patient-level data.

## Preprocessing performed by `src/data.py`

- rows with a missing target are dropped
- numeric columns are median-imputed using training statistics only
- categorical columns are most-frequent-imputed and one-hot encoded using
  training categories only
- the split is stratified: 70% train / 10% validation / 20% test by default
- features are standardised with a `StandardScaler` fitted on the training
  split only
- the test split is kept untouched until final evaluation

## If you do not have the dataset yet

The automated tests use a synthetic dataset with the same column names, so you
can run `pytest` without the real CSV. To try the CLI end-to-end, see
`tests/conftest.py` for the synthetic data generator or ask a teammate for the
project copy.
