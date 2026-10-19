# MASTER PLAN - for the coding agent (read with AGENTS.md and SCOPE.md)

## Goal
An NLP-centred hand-rehabilitation monitoring prototype. A patient writes a daily diary;
NLP extracts symptoms and severity; a sequence model tracks recovery over time and warns on
plateau/relapse; a webcam demo adds objective hand-movement data; a dashboard shows it all.
Primary contribution = diary NLP + comparison of RNN, LSTM, GRU, BiLSTM(+attention).

## Datasets (everything is generated except embeddings)
- Download manually: GloVe 6B (glove.6B.100d.txt) -> data/embeddings/glove.6B.100d.txt
- Diary data: build in Phase 1 (synthetic, labelled, source column marks it synthetic).
- Recovery curves: simulated in Phase 4.
- MediaPipe Hands: pip package, pretrained, no download.

## Phase 1 - Diary dataset (src/nlp/generate_diary_data.py)
- 2,000+ rows to data/raw/diary_synthetic.csv; columns per data/labelling_schema.md.
- Template + slot-filling generator with varied phrasing, negation, mixed symptoms,
  time words, typos, informal text. Add a token-level BIO tag column for symptom words
  (SYMPTOM, SEVERITY_WORD) to support sequence labelling.
- Separate hand-written test set from seed_examples.jsonl (+ any rows the user adds): never train on it.
- Output: label distribution table. Accept: every class (0-3) has >= 10% per symptom.

## Phase 2 - Preprocessing and embeddings (src/nlp/)
- preprocess.py: lowercase, tokenise, lemmatise, KEEP negations. Unit tests for "no pain but stiff".
- embeddings.py: load GloVe 100d, build vocab and embedding matrix, handle OOV.
- Fixed train/val/test split (seed 42, stratified) saved to data/processed/.

## Phase 3 - Diary models and comparison (src/nlp/, src/compare/)
- Multi-head classifier: shared encoder, 4 output heads (pain, stiffness, mobility, grip), 4 classes each.
- Encoders: simple RNN, LSTM, GRU, BiLSTM, BiLSTM+attention. Same data, embeddings, epochs, early stopping.
- Also a TF-IDF + Logistic Regression baseline (classical NLP baseline).
- Metrics per symptom: accuracy, macro precision/recall/F1; confusion matrices; mean +/- std over 3 seeds.
- Evaluate on BOTH synthetic test split and hand-written test set; report the gap honestly.
- Attention explainability: save top-attended words for sample entries (results/attention_examples.md).
- Error analysis: 20 misclassified entries with a short note on why (negation, mixed symptoms...).
- Outputs: results/diary_comparison.csv, confusion matrix PNGs, best model saved to results/best_diary_model.pt.
- Optional if time: sequence-labelling (BiLSTM-CRF or BiLSTM tagger) on BIO tags; BioBERT extra row.

## Phase 4 - Recovery trend (src/trend/)
- simulate.py: daily sequences (60-120 days) of [pain, stiffness, mobility, grip, ROM] with
  gradual recovery, noise, plateaus, relapses; labels: normal / plateau / relapse. Save to data/simulated/.
- Models: RNN, LSTM, GRU. Tasks: next-day forecast (MAE, RMSE) and plateau/relapse
  classification (precision, recall, F1). Window length 14 days.
- Output: results/trend_comparison.csv + forecast plot. State clearly: simulated data.

## Phase 5 - Browser-based hand tracking (app/frontend/vendor/)
- Browser-based hand tracking in JavaScript using MediaPipe Tasks Vision (hand landmarker).
- 21 landmarks, finger joint angles, range of motion, repetition counter (finger curl),
  and form score vs target angle computed directly in JS on the webcam feed.
- Library and model files stored locally in app/frontend/vendor/ so the demo works offline.

## Phase 6 - Flask API and Web Dashboard (app/)
- Flask backend (app/backend/) exposing:
  * POST /api/diary: severities + attention words from the best validation-selected diary model.
  * POST /api/trend: next-day forecast + plateau/relapse alert.
- Frontend in app/frontend/ (index.html, style.css, app.js) with Chart.js (vendored locally):
  * Diary input box with highlighted causal words.
  * Live webcam panel with real-time angles, ROM, and rep counts.
  * Recovery graph showing longitudinal trajectory.
  * Alerts ("consult your physiotherapist" on plateau/relapse).
  * Prominent, visible "monitoring aid, not diagnosis" disclaimer on the page.
  * Daily entries stored in browser localStorage.

## Phase 7 - Report assets (results/, docs/)
- Final tables and figures; docs/REPORT_NOTES.md with method, results, limitations, future work
  (sign-language dexterity score, Whisper voice diary, recommender).

## Rules of engagement
- One phase step at a time; tick PROGRESS.md; stop and summarise after each.
- Never fabricate results: all numbers in results/ must come from executed code.
- Never claim clinical validity; label synthetic data as synthetic everywhere.
