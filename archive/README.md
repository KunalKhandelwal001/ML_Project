# Archived prototype

`streamlit_prototype.py` is the original single-file Streamlit demo from the
initial commit. It is kept for reference only.

It is **not** the maintained application. Use `app.py` instead.

Known limitations of the prototype:

- It tries to fit a one-row LogisticRegression model, which raises
  `ValueError: This solver needs samples of at least 2 classes`.
- It implements FedAvg under a random-coefficient attack only.
- It does not include the z-score defense.
- It reports accuracy only and has no reproducible data split.
