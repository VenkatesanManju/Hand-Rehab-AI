# HandRehab-AI: A Multimodal Framework Fusing Attentive Patient Diaries and Markerless Joint Kinematics for Longitudinal Telerehabilitation Monitoring

> **Official Conference Paper Title:** *HandRehab-AI: A Multimodal Framework Fusing Attentive Patient Diaries and Markerless Joint Kinematics for Longitudinal Telerehabilitation Monitoring*  
> **Subtitle:** *Integrating Explainable Natural Language Processing with Real-Time Edge Vision for Post-Surgical Recovery Forecasting and Trajectory Anomaly Detection*

NLP and Multimodal Deep Learning Project: patient diary NLP (BiLSTM + additive attention with causal token occlusion), systematic recurrent comparison (RNN/LSTM/GRU/BiLSTM), longitudinal 14-day recovery trend forecasting (plateau & relapse detection), and offline browser-based MediaPipe hand kinematics.

Full plan: `docs/PROJECT_PLAN.md`. Build scope: `SCOPE.md`. Agent rules: `AGENTS.md`. Detailed report: `docs/REPORT_NOTES.md`.

## Setup
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m nltk.downloader punkt wordnet stopwords
```
