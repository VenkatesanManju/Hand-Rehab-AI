# Agent rules
- Read SCOPE.md and PROGRESS.md at the start of every session; update PROGRESS.md at the end.
- Python 3.11, PyTorch, fixed seed 42 everywhere. Do not mix in TensorFlow.
- Only build items under CORE and DEMO in SCOPE.md. Never start FUTURE WORK.
- Preprocessing must KEEP negations ("no", "not", "never", "without").
- All four models use identical train/val/test splits, embeddings and metrics.
- Label synthetic data as synthetic in code, filenames and docs. Never claim clinical validity.
- Never present the system as diagnostic; alerts say "consult your physiotherapist".
- Small steps: one module + a test per task, then stop and summarise. Save outputs to results/.
- Do not delete or overwrite files in data/raw/.
