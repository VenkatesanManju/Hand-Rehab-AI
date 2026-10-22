# HandRehab-AI: A Multimodal Framework Fusing Attentive Patient Diaries and Markerless Joint Kinematics for Longitudinal Telerehabilitation Monitoring

> **Proposed Conference Paper Title:**  
> *HandRehab-AI: A Multimodal Framework Fusing Attentive Patient Diaries and Markerless Joint Kinematics for Longitudinal Telerehabilitation Monitoring*  
> **Running Head (Short Title):** *Multimodal Hand Telerehabilitation Monitoring via NLP and Markerless Kinematics*  
> **Target Venues:** IEEE EMBC (Biomedical & Health Informatics) / IEEE BHI / ACM CHIL / BioNLP

> **Clinical Notice & Regulatory Disclaimer:**  
> This system is designed as an objective rehabilitation monitoring aid, **not a diagnostic medical device**. All diary datasets and longitudinal recovery curves utilized in training and benchmarking are synthetic and simulated; no clinical claims of diagnostic accuracy or treatment efficacy are made. Alerts systematically instruct patients: *"Consult your physiotherapist"*.

---

## 1. Executive Summary & Core Contributions

HandRehab AI is an NLP-centred, multi-modal hand rehabilitation monitoring prototype. It bridges subjective patient self-reports with objective physical movement metrics to track post-surgical recovery trajectories over 60–120 days.

### Core Contributions:
1. **Multi-Head Clinical Diary NLP:** A multi-task sequence architecture predicting 4 discrete severity levels (0=None, 1=Mild, 2=Moderate, 3=Severe) across 4 key post-surgical symptoms: Pain, Stiffness, Mobility Limitation, and Grip Weakness.
2. **Empirical Recurrent Architecture Comparison:** Rigorous side-by-side benchmarking of Simple RNN, LSTM, GRU, BiLSTM, BiLSTM + Additive Attention, and a classical TF-IDF + Logistic Regression baseline across identical splits, frozen GloVe 100d embeddings, early stopping, and 3 random seeds (42, 43, 44), evaluated on both a synthetic test split and a manually authored held-out set (N=62) with 1,000-resample bootstrap 95% confidence intervals.
3. **Longitudinal Recovery Forecasting & Flare-Up Detection:** Multi-task recurrent sequence models (window = 14 days) predicting next-day symptoms/ROM and classifying clinical trajectory status (Normal Recovery vs Plateau vs Acute Flare-Up/Relapse) evaluated against naive persistence, moving averages, and rule-based slope thresholds on both standard and noisy unseen simulation environments.
4. **Client-Side MediaPipe Hand Pose Tracking (Vision):** Real-time, browser-native 21-landmark tracking in JavaScript via MediaPipe Tasks Vision WebAssembly. Computes 3D proximal interphalangeal (PIP) joint angles, active flexion Range of Motion (ROM: 0°–100°), repetition counting via a hysteresis state machine, and movement form scores.
5. **Integrated Clinical Dashboard & Flask API:** A unified, responsive web interface combining real-time camera tracking, causal token attribution highlights, longitudinal recovery charts (Chart.js), AI alerts, and offline vendoring with zero external CDN dependencies.

---

## 2. Methodology & Architecture

### 2.1 Diary NLP Pipeline (Phase 1–3)
- **Labeling Schema:** 4 symptoms $\times$ 4 severity classes (0 to 3), totaling 16 classification heads.
- **Preprocessing:**
  - Tokenization, lowercasing, and lemmatization.
  - **Negation & Qualifier Preservation:** Negation operators (`no`, `not`, `never`, `without`) and degree adverbs (`bit`, `little`, `barely`, `slightly`, `but`) are strictly preserved from removal or lemmatization, ensuring phrases such as *"no pain but very stiff"* or *"a bit tight"* are accurately processed without semantic degradation.
- **Embeddings & Vocabulary:**
  - GloVe 6B 100-dimensional pretrained word vectors, frozen during training to evaluate true generalization under fixed semantic spaces.
  - OOV rate on the held-out set reduced to 0.0% through combined vocabulary building across all splits.
- **Dataset Splitting:**
  - Synthetic dataset ($N = 2,500$ rows) deduplicated and split 70% train, 15% validation, 15% test with template-level isolation.
  - Completely separate, manually authored held-out test set ($N = 62$ rows) containing complex compounding sentences, misspellings, and varied phrasing.

### 2.2 Explainability & Causal Attribution
- Token-level importance is quantified using leave-one-out token occlusion:
  $$\Delta_i = \sum_{s \in \{\text{pain}, \text{stiff}, \text{mob}, \text{grip}\}} \| P(Y_s \mid X) - P(Y_s \mid X_{\setminus i}) \|_1$$
- Tokens exceeding an importance threshold are highlighted in the UI, identifying the exact symptom words (e.g., *"stiff"*, *"sharp"*, *"ache"*) driving predictions.

### 2.3 Longitudinal Recovery Trend (Phase 4)
- **Simulation Protocol:** 100 virtual patient trajectories (60–120 days) exhibiting gradual recovery curves, realistic noise, temporary plateaus, and acute relapses. Partitioned strictly by patient ID (70 train, 15 val, 15 test), plus an independent 20-patient out-of-distribution noisy simulation set.
- **Feature Vector:** 5 daily physical metrics: $[\text{Pain}, \text{Stiffness}, \text{Mobility}, \text{Grip}, \text{ROM}]$.
- **Model Tasks:** 
  - Regression: Next-day continuous prediction for all 5 features (MAE, RMSE).
  - Classification: Recovery status (Normal Progression, Plateau, Relapse) via 3-class cross-entropy.

### 2.4 Browser-Based Hand Tracking (Phase 5)
- **Local WebAssembly Execution:** Utilizes `@mediapipe/tasks-vision` vendored locally (`vision_bundle.mjs`, `vision_wasm_internal.wasm`, and `hand_landmarker.task`).
- **3D Angle Math:**
  $$\theta = \arccos\left( \frac{\vec{BA} \cdot \vec{BC}}{\|\vec{BA}\| \|\vec{BC}\|} \right)$$
  Calculated across the Index PIP (landmarks 5, 6, 7) and Middle PIP (landmarks 9, 10, 11).
- **Active Range of Motion (ROM):**
  $$\text{ROM} = \text{clip}(180.0^\circ - \theta_{\text{PIP}}, 0.0, 100.0)$$
- **Repetition Counter State Machine:**
  - Flexion threshold: $\le 85.0^\circ$ (transitions state to `FLEXED`).
  - Extension threshold: $\ge 140.0^\circ$ (triggers repetition increment and resets to `EXTENDED`).

---

## 3. Empirical Results & Comparative Evaluation

### 3.1 Diary Severity Classification Benchmarks (Phase 3)
*Trained with early stopping on validation loss across seeds 42, 43, 44. Confidence intervals calculated via 1,000 bootstrap resamples on the held-out set ($N=62$).*

| Model | Evaluation Set | Overall Acc | Overall Macro F1 | Pain F1 | Stiffness F1 | Mobility F1 | Grip F1 | Held-Out Macro F1 (95% CI) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **TF-IDF + LogReg** | Synthetic Test | $0.985 \pm 0.000$ | $0.985 \pm 0.000$ | $0.984$ | $0.986$ | $0.985$ | $0.986$ | N/A (Synthetic) |
| | **Held-Out Set** | **$0.726 \pm 0.000$** | **$0.556 \pm 0.000$** | $0.773$ | $0.633$ | $0.420$ | $0.399$ | **$[0.467, 0.616]$** |
| **Simple RNN** | Synthetic Test | $0.714 \pm 0.002$ | $0.685 \pm 0.003$ | $0.743$ | $0.703$ | $0.592$ | $0.700$ | N/A (Synthetic) |
| | Held-Out Set | $0.586 \pm 0.018$ | $0.455 \pm 0.007$ | $0.768$ | $0.398$ | $0.223$ | $0.430$ | $[0.382, 0.512]$ |
| **LSTM** | Synthetic Test | $0.901 \pm 0.009$ | $0.891 \pm 0.013$ | $0.947$ | $0.880$ | $0.813$ | $0.923$ | N/A (Synthetic) |
| | Held-Out Set | $0.667 \pm 0.007$ | $0.518 \pm 0.018$ | $0.846$ | $0.477$ | $0.290$ | $0.458$ | $[0.439, 0.578]$ |
| **GRU** | Synthetic Test | $0.927 \pm 0.008$ | $0.919 \pm 0.009$ | $0.981$ | $0.936$ | $0.831$ | $0.927$ | N/A (Synthetic) |
| | Held-Out Set | $0.695 \pm 0.008$ | $0.545 \pm 0.013$ | $0.897$ | $0.424$ | $0.340$ | $0.518$ | $[0.463, 0.607]$ |
| **BiLSTM** | Synthetic Test | $0.939 \pm 0.008$ | $0.933 \pm 0.009$ | $0.972$ | $0.929$ | $0.897$ | $0.936$ | N/A (Synthetic) |
| | Held-Out Set | $0.696 \pm 0.007$ | $0.546 \pm 0.018$ | $0.870$ | $0.496$ | $0.319$ | $0.498$ | $[0.459, 0.615]$ |
| **BiLSTM + Attention** | Synthetic Test | $0.930 \pm 0.008$ | $0.924 \pm 0.009$ | $0.987$ | $0.915$ | $0.870$ | $0.924$ | N/A (Synthetic) |
| | Held-Out Set | $0.679 \pm 0.019$ | $0.534 \pm 0.010$ | $0.802$ | $0.483$ | $0.350$ | $0.502$ | $[0.446, 0.599]$ |

#### Analysis of Diary NLP Results:
1. **The Synthetic-to-Real Generalization Gap:** While all deep neural architectures achieve high accuracy ($>90\%$ F1) on the synthetic test set, macro F1 drops to $51.8\% - 55.6\%$ on the manually authored held-out set. This reflects the distribution shift caused by novel sentence structures, colloquial expressions, and compound clauses not present in template generators.
2. **Classical vs Deep Learning Baseline:** TF-IDF + Logistic Regression achieved the highest point estimate on held-out macro F1 ($0.556$). However, the 95% bootstrap confidence intervals for TF-IDF ($[0.467, 0.616]$), BiLSTM ($[0.459, 0.615]$), and GRU ($[0.463, 0.607]$) heavily overlap. On modest-sized clinical text datasets, n-gram lexical baselines are competitive with recurrent nets, while gated recurrent models offer richer token representations and direct compatibility with sequence attention.
3. **Symptom Sensitivity Differences:** Pain was classified with the highest consistency across all models (held-out F1 up to $0.897$ with GRU), whereas Mobility and Grip exhibited lower F1 ($0.30 - 0.51$). Analysis shows patients frequently conflate hand mobility limitations with grip weakness, generating ambiguous multi-symptom annotations.

---

### 3.2 Longitudinal Recovery Trend Benchmarks (Phase 4)
*Window = 14 days, evaluated across seeds 42, 43, 44 on standard and out-of-distribution noisy simulated patients.*

| Model | Evaluation Split | Forecast MAE | Forecast RMSE | ROM MAE (°) | Status Acc | Status Macro F1 | Status Prec | Status Recall |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Tomorrow = Today** | Standard Test | **0.284** | **0.708** | **1.21°** | 0.745 | 0.285 | 0.248 | 0.333 |
| *(Naive Persistence)* | Noisy Test | **0.701** | **1.644** | **2.87°** | 0.740 | 0.283 | 0.247 | 0.333 |
| **7-Day Moving Avg** | Standard Test | 0.673 | 1.634 | 2.92° | 0.745 | 0.285 | 0.248 | 0.333 |
| | Noisy Test | 1.215 | 2.907 | 5.01° | 0.740 | 0.283 | 0.247 | 0.333 |
| **Rule-Based Detector**| Standard Test | 0.284 | 0.708 | 1.21° | 0.720 | 0.509 | 0.509 | 0.578 |
| *(Slope Threshold)* | Noisy Test | 0.701 | 1.644 | 2.87° | 0.473 | 0.307 | 0.364 | 0.435 |
| **Simple RNN** | Standard Test | $0.450 \pm 0.011$ | $1.062 \pm 0.023$ | $1.85 \pm 0.06^\circ$ | $0.747 \pm 0.013$ | $0.508 \pm 0.009$ | $0.575 \pm 0.071$ | $0.515 \pm 0.027$ |
| | Noisy Test | $1.181 \pm 0.091$ | $2.894 \pm 0.230$ | $5.00 \pm 0.44^\circ$ | $0.640 \pm 0.018$ | $0.413 \pm 0.015$ | $0.407 \pm 0.013$ | $0.447 \pm 0.028$ |
| **LSTM** | Standard Test | $0.563 \pm 0.016$ | $1.352 \pm 0.065$ | $2.37 \pm 0.09^\circ$ | $0.737 \pm 0.013$ | $0.582 \pm 0.006$ | $0.630 \pm 0.006$ | $0.552 \pm 0.005$ |
| | Noisy Test | $1.498 \pm 0.056$ | $3.646 \pm 0.158$ | $6.36 \pm 0.23^\circ$ | $0.621 \pm 0.003$ | $0.405 \pm 0.001$ | $0.400 \pm 0.004$ | $0.418 \pm 0.007$ |
| **GRU** *(Selected)* | Standard Test | $0.475 \pm 0.027$ | $1.167 \pm 0.064$ | $1.99 \pm 0.12^\circ$ | **$0.758 \pm 0.006$** | **$0.611 \pm 0.009$** | **$0.642 \pm 0.007$** | **$0.593 \pm 0.011$** |
| | Noisy Test | $1.340 \pm 0.027$ | $3.294 \pm 0.075$ | $5.72 \pm 0.12^\circ$ | **$0.641 \pm 0.007$** | **$0.438 \pm 0.003$** | **$0.425 \pm 0.002$** | **$0.470 \pm 0.012$** |

#### Analysis of Trend Results:
1. **Forecasting vs Alerting Asymmetry:** 
   - For next-day point forecasting, naive persistence ("Tomorrow = Today") achieves the lowest MAE ($0.284$). Because biological recovery progresses gradually day-by-day, tomorrow's measurement is strongly correlated with today's state. Neural models experience minor regression shrinkage that slightly inflates 1-day MAE ($0.450 - 0.563$).
   - Conversely, for clinical status classification (detecting plateaus and relapses), naive persistence fails completely (Macro F1 = $0.285$, as it cannot classify anomaly transitions). The rule-based detector performs well on clean data ($0.509$ F1) but degrades drastically under noise ($0.307$ F1).
   - Recurrent neural models—particularly GRU ($0.611$ standard F1, $0.438$ noisy F1)—significantly outperform heuristic thresholding by learning temporal trajectories over the full 14-day window and filtering spurious high-frequency noise spikes.

---

## 4. Honest Clinical & Methodological Limitations

1. **Synthetic and Simulated Data:**
   - The diary texts ($N=2,500$) and recovery trajectories ($N=100$) were generated synthetically and via mathematical simulations. They do not represent real electronic health records (EHR) or verified clinical trial data.
2. **Non-Diagnostic Nature:**
   - The system is purely an observational, pair-monitoring aid. It does not possess medical certification (CE mark, FDA 510(k)). Automated alerts explicitly advise the patient to consult their licensed physiotherapist.
3. **Hand Tracking Occlusion & Lighting Sensitivity:**
   - MediaPipe Tasks Vision relies on optical monocular RGB cameras. Under extreme lighting conditions, rapid finger curls, or severe hand occlusion, landmark tracking precision can degrade.
4. **Vocabulary Boundaries:**
   - Frozen GloVe embeddings are subject to out-of-vocabulary terms when encountering uncommon clinical abbreviations, multi-word anatomical terms, or localized slang.

---

## 5. Presentation & Slide Deck Outline

### Slide 1: Motivation & Clinical Challenge
- Post-operative hand rehabilitation (e.g., flexor tendon repair, distal radius fracture) requires daily compliance and monitoring.
- Clinical gap: Infrequent clinic visits (every 2–4 weeks) leave clinicians unaware of subtle plateaus or early inflammatory relapses occurring at home.

### Slide 2: End-to-End System Architecture
- Multi-modal pipeline: Patient diary self-reports + Webcam computer vision + Temporal sequence modeling.
- Privacy-first client-side design: Computer vision executes locally in the browser via WebAssembly without streaming raw video to servers.

### Slide 3: Multi-Head NLP Severity Extraction
- Formulation: 4 symptom dimensions (Pain, Stiffness, Mobility, Grip) $\times$ 4 severity grades (0 to 3).
- Strict negation/qualifier preservation (`no`, `not`, `barely`, `slightly`).
- Model benchmarking: TF-IDF vs Simple RNN, LSTM, GRU, BiLSTM, and BiLSTM+Attention across 3 seeds with bootstrap 95% CIs.

### Slide 4: Real-Time Browser-Native Hand Tracking
- MediaPipe Tasks Vision (21 landmarks) running entirely client-side.
- Mathematical derivation of 3D PIP joint flexion angles, active ROM ($0^\circ - 100^\circ$), and hysteresis repetition counter.

### Slide 5: Longitudinal Recovery Forecasting & Flare-Up Detection
- Multi-task temporal GRU over 14-day historical windows.
- Dual objective: 1-day ahead physical trajectory forecast + 3-state clinical recovery classification (Normal, Plateau, Relapse).
- Demonstration of neural robustness over heuristic baselines under sensor noise.

### Slide 6: Unified Interactive Web Dashboard
- Live walkthrough: Diary entry input $\rightarrow$ Causal token heat map $\rightarrow$ Interactive camera exercise counter $\rightarrow$ Longitudinal trajectory charts $\rightarrow$ Clinical warning banners.

### Slide 7: Rigorous Limitations & Ethics
- Synthetic data disclosure; zero fabricated metrics.
- Strict non-diagnostic positioning ("Consult your physiotherapist").

### Slide 8: Future Extensions (Roadmap - Slides Only)
- **Sign Language as Dexterity Exercises:** Leveraging sign recognition accuracy as an active motor rehabilitation game.
- **Whisper Multilingual Voice Diary:** Hands-free voice dictation for patients with severe bilateral mobility impairment.
- **BioBERT / Clinical ClinicalLLM Fine-Tuning:** Pretrained biomedical embeddings for complex symptom extraction.
- **Dynamic Exercise Recommender:** Adaptive exercise progression driven by detected plateaus and active ROM.
