# Scope (read before every task)

## CORE - build and evaluate fully
1. Diary NLP: preprocessing -> embeddings -> BiLSTM + attention. Predict severity (0-3)
   for pain, stiffness, mobility, grip. Attention highlights the causal words.
2. Model comparison: RNN, LSTM, GRU, BiLSTM on the SAME splits and metrics.
   Report accuracy, precision, recall, F1, confusion matrix in a results table.
3. Trend model: LSTM/GRU on simulated daily-score sequences; forecast + plateau/relapse
   detection (MAE, RMSE, precision, recall).

## DEMO - light, no training
4. Browser-based hand tracking in JavaScript using MediaPipe Tasks Vision (hand landmarker).
   Joint angles, range of motion, rep counter and form score computed in JS. Library and
   model files stored locally in app/frontend/vendor/ so the demo works offline.
5. Flask backend (app/backend/) exposing POST /api/diary (severities + attention words from the
   best validation-selected diary model) and POST /api/trend (forecast + plateau/relapse alert).
   Frontend in app/frontend/ (index.html, style.css, app.js) with Chart.js (also vendored locally),
   a diary box with highlighted words, a webcam panel, a recovery graph, alerts, and a visible
   "monitoring aid, not diagnosis" disclaimer. Daily entries stored in localStorage.

## FUTURE WORK - slides only, do NOT build
- Sign-language dexterity score, Whisper voice diary, sentence-embedding recommender,
  fine-tuned BioBERT (optional extra comparison row if time remains).

## Honest limitations (must appear in report and slides)
- Diary and recovery data are synthetic/simulated.
- Not a diagnosis tool; alerts advise consulting a physiotherapist.
