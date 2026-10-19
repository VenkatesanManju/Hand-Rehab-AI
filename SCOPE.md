# Scope (read before every task)

## CORE - build and evaluate fully
1. Diary NLP: preprocessing -> embeddings -> BiLSTM + attention. Predict severity (0-3)
   for pain, stiffness, mobility, grip. Attention highlights the causal words.
2. Model comparison: RNN, LSTM, GRU, BiLSTM on the SAME splits and metrics.
   Report accuracy, precision, recall, F1, confusion matrix in a results table.
3. Trend model: LSTM/GRU on simulated daily-score sequences; forecast + plateau/relapse
   detection (MAE, RMSE, precision, recall).

## DEMO - light, no training
4. MediaPipe Hands: joint angles, range of motion, repetition count, form score on webcam.
5. Streamlit dashboard showing diary severity, ROM, trend graph and alerts.

## FUTURE WORK - slides only, do NOT build
- Sign-language dexterity score, Whisper voice diary, sentence-embedding recommender,
  fine-tuned BioBERT (optional extra comparison row if time remains).

## Honest limitations (must appear in report and slides)
- Diary and recovery data are synthetic/simulated.
- Not a diagnosis tool; alerts advise consulting a physiotherapist.
