"""FedDef Streamlit app shell.

Task 1 provides the data/evaluation foundation and this app skeleton.  The
baseline, attack, and defense tabs are placeholders for Tasks 2-4.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.linear_model import LogisticRegression

from src.data import DataBundle, load_data, prepare_data
from src.evaluate import evaluate_model, format_metrics

st.set_page_config(
    page_title="FedDef",
    page_icon="🛡️",
    layout="wide",
)

DEFAULT_DATA_PATH = Path("data/framingham.csv")
TARGET_COLUMN = "TenYearCHD"


def _load_uploaded(
    uploaded_file,
    *,
    seed: int,
    test_size: float,
    val_size: float,
) -> DataBundle:
    frame = pd.read_csv(uploaded_file)
    return prepare_data(
        frame,
        target_col=TARGET_COLUMN,
        test_size=test_size,
        val_size=val_size,
        random_state=seed,
        source=getattr(uploaded_file, "name", "uploaded CSV"),
    )


def _load_default(
    *,
    seed: int,
    test_size: float,
    val_size: float,
) -> DataBundle | None:
    if not DEFAULT_DATA_PATH.exists():
        return None
    return load_data(
        DEFAULT_DATA_PATH,
        target_col=TARGET_COLUMN,
        test_size=test_size,
        val_size=val_size,
        random_state=seed,
    )


st.title("FedDef — Federated Learning Robustness Demo")
st.caption(
    "Task 1 foundation: reproducible data splits and a trustworthy evaluation "
    "harness. FedAvg, poisoning attacks, and the defense are added in Tasks 2-4."
)

with st.sidebar:
    st.header("Data")
    uploaded_file = st.file_uploader("Upload Framingham CSV", type=["csv"])
    seed = int(
        st.number_input(
            "Random seed",
            min_value=0,
            max_value=100_000,
            value=42,
            step=1,
        )
    )
    test_size = float(
        st.slider(
            "Test fraction",
            min_value=0.10,
            max_value=0.40,
            value=0.20,
            step=0.05,
        )
    )
    val_size = float(
        st.slider(
            "Validation fraction",
            min_value=0.05,
            max_value=0.30,
            value=0.10,
            step=0.05,
        )
    )
    use_default = st.checkbox(
        "Use data/framingham.csv if no upload",
        value=True,
    )
    st.divider()
    st.info(
        "The test split is reserved for final evaluation. The defense will use "
        "the validation split only."
    )

if test_size + val_size >= 0.9:
    st.error("Test fraction + validation fraction must be less than 0.9.")
    st.stop()

bundle: DataBundle | None = None

if uploaded_file is not None:
    try:
        bundle = _load_uploaded(
            uploaded_file,
            seed=seed,
            test_size=test_size,
            val_size=val_size,
        )
    except (ValueError, TypeError, pd.errors.ParserError) as exc:
        st.error(f"Could not prepare the uploaded CSV: {exc}")
elif use_default:
    try:
        bundle = _load_default(
            seed=seed,
            test_size=test_size,
            val_size=val_size,
        )
    except (ValueError, TypeError) as exc:
        st.error(f"Could not prepare data/framingham.csv: {exc}")

if bundle is None:
    st.info(
        "Upload a Framingham CSV in the sidebar, or place it at "
        "`data/framingham.csv`. See `data/README.md` for the expected columns."
    )
    st.stop()

metadata = bundle.metadata
st.success(
    f"Loaded **{metadata.get('source') or 'dataset'}** — "
    f"{metadata['n_rows_after_target_drop']:,} rows, "
    f"{bundle.n_features} features."
)

metric_columns = st.columns(3)
for column, split_name in zip(metric_columns, ("train", "val", "test")):
    column.metric(
        f"{split_name.capitalize()} rows",
        f"{metadata['split_sizes'][split_name]:,}",
    )
    column.caption(f"{metadata['positive_rate'][split_name]:.1%} positive")

data_tab, baseline_tab, attack_tab, defense_tab = st.tabs(
    [
        "Data",
        "Baseline (Task 2)",
        "Attack (Task 4)",
        "Defense (Task 3)",
    ]
)

with data_tab:
    st.subheader("Prepared training features")
    st.caption(
        "Features are median/mode imputed, one-hot encoded, and standardised. "
        "Preprocessing is fitted on the training split only."
    )
    st.dataframe(bundle.X_train.head(10), width="stretch")

    with st.expander("Dataset metadata"):
        st.json(metadata)

    st.subheader("Centralized smoke check")
    st.caption(
        "This is a data/evaluation sanity check, not the Task 2 FedAvg "
        "baseline. It trains one logistic regression on the training split and "
        "reports validation metrics."
    )
    if st.button("Train smoke-check model"):
        with st.spinner("Training logistic regression..."):
            model = LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
            )
            model.fit(bundle.X_train, bundle.y_train)
            metrics = evaluate_model(model, bundle.X_val, bundle.y_val)

        st.text(format_metrics(metrics))
        st.dataframe(
            pd.DataFrame(
                metrics["confusion_matrix"],
                index=["True 0", "True 1"],
                columns=["Predicted 0", "Predicted 1"],
            ),
            width="stretch",
        )
        st.caption(
            "Validation split only. The test split is not touched by this "
            "smoke check."
        )

with baseline_tab:
    st.subheader("Clean FedAvg baseline")
    st.info(
        "Task 2 (Dewansh) will add the no-attack FedAvg baseline and compare it "
        "with a centralized logistic regression."
    )
    st.markdown(
        """
        Planned outputs:
        - client partition and seeded local training
        - weighted FedAvg aggregation
        - validation/test metrics and a convergence curve
        - a centralized reference model for sanity checking
        """
    )

with attack_tab:
    st.subheader("Model-poisoning attacks")
    st.info(
        "Task 4 (Pratik) will add configurable attacks and the experiment matrix."
    )
    st.markdown(
        """
        Planned attacks:
        - random noise
        - sign flip
        - scaling
        - label flipping
        - colluding attackers
        """
    )

with defense_tab:
    st.subheader("Anomaly-aware aggregation")
    st.info(
        "Task 3 (Kunal V) will add robust z-score client scoring and the "
        "defense-enabled aggregation."
    )
    st.markdown(
        """
        Planned outputs:
        - per-client anomaly scores on the validation split
        - confidence accumulation with a threshold
        - down-weighting/exclusion before FedAvg
        - clean-case and attack-case comparison
        """
    )
