**AI-Based Hand Rehabilitation Recovery Monitoring System**

*Using NLP, Hand-Pose Tracking and Sequence Models (RNN, LSTM, GRU,
BiLSTM) with Sign Language Support*

**1. Project Overview**

The system is a web application that helps patients recover hand
function. Patients write (or speak) a daily rehab diary, perform
exercises in front of a webcam, and practise sign language as a
dexterity exercise. NLP extracts symptoms from the diary, hand tracking
measures real movement, and sequence models analyse the recovery trend
over time and warn when recovery stalls or relapses. Everything is shown
on a single user dashboard.

**2. Novel Features Added**

- **Webcam Exercise Form Checker:** The user performs an exercise on
  camera. MediaPipe Hands tracks 21 hand landmarks and the system
  calculates joint angles, range of motion, repetition count and a form
  score against the target in the Exercise DB. This adds objective
  measurement to the diary-based (subjective) data.

- **Sign Language as a Rehab Exercise:** Practising signs is itself a
  hand-dexterity exercise. Sign recognition accuracy and speed become a
  dexterity score, linking the Sign Language module with rehabilitation.
  Deaf users can also use signs as input.

- **Plateau and Relapse Early Warning:** The temporal model flags when
  recovery stalls or when pain and stiffness start rising again, and
  advises the user to consult their physiotherapist.

**3. Workflow Diagram**

<img src="media/ae8ee40d41efd5864cf627cdc401d80d23f28ddf.png"
title="Workflow" style="width:5.83333in;height:6.21875in"
alt="Project workflow diagram" />

*Figure 1: Overall project workflow*

**4. How the Project Works**

1.  **Step 1 - Access:** The user logs in to the web application and
    chooses a diary entry, an exercise session, or a sign practice
    session.

2.  **Step 2 - Diary Analysis (NLP):** The user writes or speaks a diary
    entry (voice is converted to text using Whisper). The text is
    cleaned, tokenised and converted to word embeddings. A BiLSTM with
    attention extracts symptoms (pain, stiffness, mobility,
    grip/activity) and predicts a severity level for each, and
    highlights the phrases that caused the score.

3.  **Step 3 - Exercise Form Checking:** The webcam records the
    exercise. MediaPipe Hands extracts landmarks per frame, from which
    joint angles, range of motion, repetitions and a form score are
    calculated.

4.  **Step 4 - Sign Practice:** The user imitates signs from the Sign
    DB. A landmark-based GRU model recognises the sign. Accuracy and
    completion time give a dexterity score.

5.  **Step 5 - Feature Fusion:** Text severity, range of motion and
    dexterity score are combined into one daily feature vector, which is
    stored with the date.

6.  **Step 6 - Temporal Analysis:** The recent daily vectors form a time
    sequence. An LSTM/GRU forecasts the recovery trend and detects
    plateau or relapse patterns, and an alert is raised when needed.

7.  **Step 7 - Recommendation:** Sentence embeddings match the user's
    symptoms to suitable exercises in the Exercise DB and to sign videos
    in the Sign DB.

8.  **Step 8 - Dashboard:** The dashboard shows symptom scores, recovery
    graph, progress comparison, recommended exercises, sign resources
    and alerts.

**5. Datasets That Can Be Used**

*No public dataset of hand-rehab diaries exists, so the diary dataset
must be created by the team (see note below the table).*

| **Module**                  | **Dataset / Source**                                                                                       | **What it provides**                                                     | **Note**                                                                    |
|-----------------------------|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------|-----------------------------------------------------------------------------|
| **Diary NLP**               | Self-built diary dataset (synthetic entries generated with an LLM, then reviewed and labelled by the team) | Entries labelled with pain, stiffness, mobility, grip and severity (0-3) | Aim for 1,500-3,000 entries; have a physiotherapist check a sample          |
| **Text embeddings**         | GloVe / FastText, or BioBERT / ClinicalBERT embeddings                                                     | Pretrained word vectors for medical-style text                           | Use as input to the BiLSTM                                                  |
| **Clinical NLP (optional)** | MIMIC-III clinical notes                                                                                   | Pretraining or NER practice on clinical language                         | Needs credentialed access                                                   |
| **Hand tracking**           | MediaPipe Hands (pretrained), FreiHAND, HaGRID                                                             | Hand landmarks and hand images for validation                            | MediaPipe needs no training                                                 |
| **Rehab movement scoring**  | KIMORE, UI-PRMD                                                                                            | Movement-quality scoring methods                                         | Whole-body rehab, not hand-specific; use for methodology                    |
| **Sign recognition**        | WLASL, MS-ASL (American); INCLUDE by AI4Bharat (Indian)                                                    | Labelled sign videos                                                     | INCLUDE suits Indian users; extract landmarks with MediaPipe                |
| **Voice diary (optional)**  | Whisper (pretrained), Mozilla Common Voice (Tamil / English)                                               | Speech-to-text                                                           | Fine-tune only if needed                                                    |
| **Recovery trend**          | Simulated recovery curves (generated from diary scores and ROM with noise, plateaus and relapses)          | Daily score sequences with plateau / relapse labels                      | Validate on a small pilot; state clearly in the report that it is simulated |

**6. Deep Learning and NLP Ideas**

**6.1 NLP Pipeline**

- Preprocessing: lowercasing, tokenisation, stop-word handling (keep
  negations such as 'no', 'not'), lemmatisation.

- Representation: GloVe/FastText or BioBERT embeddings fed into the
  recurrent models.

- Information extraction: sequence labelling to tag symptom words, and
  classification to give a severity level for each symptom.

- Explainability: attention weights (or SHAP) to highlight the words
  behind each severity score.

**6.2 Where Each Model Is Used**

| **Model**                | **Used for**                                                             | **Why**                                                                                           |
|--------------------------|--------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------|
| **RNN (simple)**         | Baseline for diary classification and trend prediction                   | Simple, but suffers from vanishing gradients on long entries, so it is the comparison baseline    |
| **LSTM**                 | Recovery trend forecasting from daily score sequences                    | Memory cells keep long-term patterns such as slow improvement over weeks                          |
| **GRU**                  | Plateau / relapse detection and sign recognition from landmark sequences | Fewer parameters than LSTM, trains faster, works well with small datasets                         |
| **BiLSTM (+ attention)** | Diary symptom extraction and severity classification                     | Reads text in both directions, so it handles phrases like 'no pain but very stiff in the morning' |

**6.3 Planned Model Comparison**

Train RNN, LSTM, GRU and BiLSTM on the same data and compare them. Use
diary classification for the text task and the simulated recovery
sequences for the trend task. Report the results in a table and select
the best model for each module.

**6.4 Evaluation Metrics**

- Diary NLP: accuracy, precision, recall, F1-score, confusion matrix.

- Exercise form checker: angle error against manual measurement,
  repetition-count accuracy.

- Sign recognition: top-1 accuracy.

- Trend forecasting: MAE and RMSE; plateau / relapse detection:
  precision and recall.

**7. Limitations to State in the Report**

- The diary and recovery-trend data are built or simulated, so results
  should be confirmed with a small pilot or physiotherapist review.

- The system supports recovery monitoring and is not a medical diagnosis
  tool. Alerts advise consulting a physiotherapist.
