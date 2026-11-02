# Reproducing the cleaned data

Git supplies the code. Private storage supplies the source data. Cloning the repo alone cannot recover the dataset because `dataset/` is intentionally git-ignored.

## Option A: use the exact prepared release

For teammates who want to train/evaluate rather than rerun cleaning:

1. Obtain the same code revision and a private copy of the complete `dataset/qoherent_v3/` folder from the data owner. Share through the team's Drive folder or another agreed private location.
2. Put the folder under your repo's `dataset/` directory, preserving all files, particularly `metadata/`, `manifest.json` and `checksums.sha256`.
3. From the repo root:

```powershell
python -m pip install -r data/requirements.txt
python data/validate.py --dataset dataset/qoherent_v3 --skip-source-equivalence
```

This checks the release checksums, structure, masks and splits without requiring the original raw folders or old v2 package. It does not rerun cleaning. Use the three top-level JSONL files for training/validation/test, or the `evaluation/` question files for the current evaluator.

## Option B: rebuild from the original inputs

Obtain the four original folders privately and preserve their names and contents. Clone this repo with the local directory name `nanbeige-modal` because the historical source pointers and evaluation path use that name. Keep the same pinned code revision and evaluation question/ID files for comparisons.

```text
workspace/
    GRC_dataset/
    GRC_Agent_data/
    NewData1/
    NewData2/
    nanbeige-modal/       code repository
        data/
        trial_eval/      includes the fixed trial questions and IDs
```

From `workspace/nanbeige-modal`:

```powershell
python -m pip install -r data/requirements.txt
python data/rebuild.py --workspace .. --output dataset/qoherent_v3_reproduced
```

One command runs the complete local pipeline:

1. Clean and select the original Q&A and legacy agent sessions.
2. Reconcile the new single-task/two-turn scenarios and create the protected splits.
3. Export the model-neutral v3 format and evaluation files.
4. Verify the resulting release.

Intermediate stages appear in `workspace/processed/reproduction_v1/` and `reproduction_v2/`. Logs and a receipt appear in `dataset/rebuild_logs/reproduction/`. The wrapper refuses to overwrite existing stages or releases. For another run, select a new `--tag` and `--output`; do not remove or alter the original inputs.

If the reference v3 release is also present, compare the actual train/validation/test data, not path-dependent reports or ZIP timestamps:

```powershell
python data/rebuild.py --workspace .. --tag comparison --output dataset/qoherent_v3_comparison --compare-to dataset/qoherent_v3
```

The three primary JSONL files should be byte-identical with the same source bytes, code and benchmark. Metadata/reports can differ because source paths, intermediate directory names and code snapshots differ. Record the release checksums and code revision with every experiment.

## What must be shared first?

- Use a shared commit containing `data/` and the UTF-8 evaluation-reader fixes. For this release, use branch `data/model-compatible-dataset` until it is merged into `main`; pin the commit with your experiment.
- For Option A, share the complete prepared release privately. Original sources are unnecessary.
- For Option B, share all four raw folders privately. A v2 package is unnecessary because the full rebuild scripts now live under `data/legacy/`.
- Python 3.10+ and `data/requirements.txt` are sufficient for data preparation. GPU access, model weights and API keys are not needed.

The current prepared v3 folder is about 133 MiB; the four raw input folders total about 790 MiB. These are measured sizes, not training-token counts. The repo's data-sharing policy is private storage rather than committing datasets to Git.
