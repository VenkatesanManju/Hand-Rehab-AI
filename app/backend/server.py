"""
Flask Backend API for Hand Rehabilitation Monitoring (Phase 6)
==============================================================
Endpoints:
- GET /                    : Serves frontend dashboard
- POST /api/diary          : Predicts severity (0-3) for pain, stiffness, mobility, grip + causal attention words
- POST /api/trend          : Predicts next-day forecast (symptoms + ROM) + plateau/relapse alerts
- GET /api/sample_patient  : Returns simulated patient trajectory for instant demonstration
"""

import os
import sys
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, jsonify, send_from_directory

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)

from src.nlp.preprocess import preprocess_text
from src.nlp.embeddings import GloVeEmbeddingLoader, DEFAULT_GLOVE_PATH
from src.nlp.models import MultiHeadDiaryClassifier
from src.trend.models import MultiTaskTrendModel
from src.trend.dataset import FEATURE_COLS
from app.backend.db import init_db, register_user, authenticate_user, get_user_records, add_user_record

STATIC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../frontend"))
app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEVERITY_LABELS = {0: "None (0)", 1: "Mild (1)", 2: "Moderate (2)", 3: "Severe (3)"}


# Global holders for models and scalers
EMBEDDING_LOADER = None
DIARY_MODEL = None
TREND_MODEL = None
SCALER_MEAN = None
SCALER_STD = None


def init_models():
    global EMBEDDING_LOADER, DIARY_MODEL, TREND_MODEL, SCALER_MEAN, SCALER_STD

    # 1. Load GloVe loader
    vocab_path = os.path.join(PROJECT_ROOT, "data/processed/vocab.json")
    matrix_path = os.path.join(PROJECT_ROOT, "data/processed/glove_100d_matrix.pt")
    EMBEDDING_LOADER = GloVeEmbeddingLoader(embedding_path=DEFAULT_GLOVE_PATH, embedding_dim=100)
    EMBEDDING_LOADER.load_vocab_and_matrix(vocab_path, matrix_path)

    # 2. Load Best Diary Model
    diary_weights_path = os.path.join(PROJECT_ROOT, "results/best_diary_model.pt")
    DIARY_MODEL = MultiHeadDiaryClassifier(
        vocab_size=len(EMBEDDING_LOADER.word2idx),
        embedding_dim=100,
        hidden_dim=128,
        encoder_type="gru",
        num_classes=4,
        pretrained_embeddings=EMBEDDING_LOADER.embedding_matrix,
        freeze_embeddings=True,
    ).to(DEVICE)
    DIARY_MODEL.load_state_dict(torch.load(diary_weights_path, map_location=DEVICE))
    DIARY_MODEL.eval()

    # 3. Load Best Trend Model & Scaler parameters
    trend_weights_path = os.path.join(PROJECT_ROOT, "results/best_trend_model.pt")
    TREND_MODEL = MultiTaskTrendModel(
        input_dim=5,
        hidden_dim=64,
        encoder_type="gru",
        num_layers=1,
    ).to(DEVICE)
    TREND_MODEL.load_state_dict(torch.load(trend_weights_path, map_location=DEVICE))
    TREND_MODEL.eval()

    train_trend_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data/simulated/recovery_curves_train.csv"))
    raw_feats = train_trend_df[FEATURE_COLS].values.astype(np.float32)
    SCALER_MEAN = np.mean(raw_feats, axis=0)
    SCALER_STD = np.std(raw_feats, axis=0) + 1e-6

    # 4. Initialize SQLite User Database
    init_db()


# Initialize models and DB upon module import
init_models()



@app.route("/")
def serve_index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(STATIC_DIR, path)


@app.route("/api/diary", methods=["POST"])
def analyze_diary():
    """
    POST /api/diary
    Body: {"text": "Patient daily diary entry..."}
    Returns: Predicted severities (0-3) for pain, stiffness, mobility, grip,
             plus token-level causal attribution weights.
    """
    data = request.get_json(force=True, silent=True) or {}
    text = str(data.get("text", "")).strip()

    if not text:
        return jsonify({"error": "Empty text provided"}), 400

    tokens = preprocess_text(text, remove_stopwords=True, lemmatize=True, keep_negations=True)
    if not tokens:
        tokens = ["symptom"]

    # Convert to indices
    indices = EMBEDDING_LOADER.tokens_to_indices(tokens, max_len=len(tokens), padding=False)
    input_ids = torch.tensor([indices], dtype=torch.long, device=DEVICE)
    lengths = torch.tensor([len(indices)], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        out = DIARY_MODEL(input_ids, lengths=lengths)

    # Predicted classes
    pred_pain = int(torch.argmax(out["pain"], dim=-1).item())
    pred_stiffness = int(torch.argmax(out["stiffness"], dim=-1).item())
    pred_mobility = int(torch.argmax(out["mobility"], dim=-1).item())
    pred_grip = int(torch.argmax(out["grip"], dim=-1).item())

    # Causal attribution weights via token occlusion (leave-one-out impact)
    base_probs = {
        sym: F.softmax(out[sym][0], dim=-1).cpu().numpy()
        for sym in ["pain", "stiffness", "mobility", "grip"]
    }

    token_importance = []
    with torch.no_grad():
        for i in range(len(indices)):
            if len(indices) <= 1:
                token_importance.append(1.0)
                continue
            occ_indices = [idx for j, idx in enumerate(indices) if j != i]
            occ_ids = torch.tensor([occ_indices], dtype=torch.long, device=DEVICE)
            occ_lengths = torch.tensor([len(occ_indices)], dtype=torch.long, device=DEVICE)
            occ_out = DIARY_MODEL(occ_ids, lengths=occ_lengths)

            diff = 0.0
            for sym in ["pain", "stiffness", "mobility", "grip"]:
                occ_p = F.softmax(occ_out[sym][0], dim=-1).cpu().numpy()
                diff += float(np.sum(np.abs(base_probs[sym] - occ_p)))
            token_importance.append(diff)

    sum_imp = float(np.sum(token_importance))
    if sum_imp > 0:
        attn_weights = [round(float(w / sum_imp), 3) for w in token_importance]
    else:
        attn_weights = [round(1.0 / len(tokens), 3)] * len(tokens)

    highlighted = []
    threshold = (1.2 / len(tokens)) if len(tokens) > 1 else 0.5
    for t, w in zip(tokens, attn_weights):
        highlighted.append({
            "word": t,
            "weight": w,
            "is_causal": bool(w >= threshold),
        })

    return jsonify({
        "pain": pred_pain,
        "pain_label": SEVERITY_LABELS[pred_pain],
        "stiffness": pred_stiffness,
        "stiffness_label": SEVERITY_LABELS[pred_stiffness],
        "mobility": pred_mobility,
        "mobility_label": SEVERITY_LABELS[pred_mobility],
        "grip": pred_grip,
        "grip_label": SEVERITY_LABELS[pred_grip],
        "tokens": tokens,
        "highlighted_words": highlighted,
        "disclaimer": "Monitoring aid, not a diagnostic medical device. Consult your physiotherapist.",
    })


@app.route("/api/trend", methods=["POST"])
def analyze_trend():
    """
    POST /api/trend
    Body: {"history": [{"day": 0, "pain": 2.5, "stiffness": 2.2, "mobility": 2.0, "grip": 2.3, "rom": 35.0}, ...]}
    Requires at least 14 days of history.
    Returns: Next-day forecast + plateau/relapse alerts.
    """
    data = request.get_json(force=True, silent=True) or {}
    history = data.get("history", [])

    if len(history) < 14:
        return jsonify({
            "error": f"Requires at least 14 days of rehabilitation history (received {len(history)})"
        }), 400

    # Extract last 14 days
    last_14 = history[-14:]
    raw_array = np.zeros((14, 5), dtype=np.float32)
    for i, day_dict in enumerate(last_14):
        raw_array[i, 0] = float(day_dict.get("pain", 1.0))
        raw_array[i, 1] = float(day_dict.get("stiffness", 1.0))
        raw_array[i, 2] = float(day_dict.get("mobility", 1.0))
        raw_array[i, 3] = float(day_dict.get("grip", 1.0))
        raw_array[i, 4] = float(day_dict.get("rom", 50.0))

    # Normalize input
    norm_array = (raw_array - SCALER_MEAN) / SCALER_STD
    x_tensor = torch.tensor(np.expand_dims(norm_array, 0), dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        pred_f_norm, pred_s_logits = TREND_MODEL(x_tensor)

        # Denormalize forecast
        mean_t = torch.tensor(SCALER_MEAN, device=DEVICE)
        std_t = torch.tensor(SCALER_STD, device=DEVICE)
        pred_f = (pred_f_norm[0] * std_t + mean_t).cpu().numpy()
        pred_status = int(torch.argmax(pred_s_logits[0], dim=-1).item())

    # Clamp physical predictions
    forecast_dict = {
        "pain": round(float(np.clip(pred_f[0], 0.0, 3.0)), 2),
        "stiffness": round(float(np.clip(pred_f[1], 0.0, 3.0)), 2),
        "mobility": round(float(np.clip(pred_f[2], 0.0, 3.0)), 2),
        "grip": round(float(np.clip(pred_f[3], 0.0, 3.0)), 2),
        "rom": round(float(np.clip(pred_f[4], 0.0, 100.0)), 1),
    }

    status_name = {0: "normal", 1: "plateau", 2: "relapse"}[pred_status]

    if pred_status == 2:
        alert = "Alert: Potential Relapse Detected (Acute worsening in symptoms or drop in active ROM). Please consult your physiotherapist."
        alert_level = "danger"
    elif pred_status == 1:
        alert = "Warning: Recovery Plateau Detected (Rehabilitation progress has stalled over the past 14 days). Please consult your physiotherapist."
        alert_level = "warning"
    else:
        alert = "Progress On Track: Steady functional recovery progression observed."
        alert_level = "success"

    return jsonify({
        "forecast": forecast_dict,
        "status": status_name,
        "status_code": pred_status,
        "alert": alert,
        "alert_level": alert_level,
        "disclaimer": "Monitoring aid, not a diagnostic medical device. Consult your physiotherapist.",
    })


@app.route("/api/sample_patient", methods=["GET"])
def get_sample_patient():
    """
    GET /api/sample_patient?pid=PATIENT_STD_002
    Returns complete trajectory of a sample test patient for easy 1-click frontend loading.
    """
    test_csv = os.path.join(PROJECT_ROOT, "data/simulated/recovery_curves_test.csv")
    df = pd.read_csv(test_csv)
    pid = request.args.get("pid", "PATIENT_STD_002")

    if pid not in df["patient_id"].values:
        pid = df["patient_id"].iloc[0]

    p_df = df[df["patient_id"] == pid].sort_values("day")
    records = p_df[["day", "pain", "stiffness", "mobility", "grip", "rom", "status_label"]].to_dict(orient="records")

    return jsonify({
        "patient_id": pid,
        "total_days": len(records),
        "trajectory": records,
    })


@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    data = request.get_json(force=True, silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()
    full_name = str(data.get("full_name", "")).strip()
    surgery_type = str(data.get("surgery_type", "")).strip()

    res = register_user(username, password, full_name, surgery_type)
    if res.get("success"):
        return jsonify(res), 200
    return jsonify(res), 400


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json(force=True, silent=True) or {}
    username = str(data.get("username", "")).strip()
    password = str(data.get("password", "")).strip()

    res = authenticate_user(username, password)
    if res.get("success"):
        return jsonify(res), 200
    return jsonify(res), 401


@app.route("/api/user/history", methods=["GET"])
def user_history():
    user_id = request.args.get("user_id", type=int)
    if not user_id:
        return jsonify({"error": "Missing user_id parameter"}), 400

    records = get_user_records(user_id)
    return jsonify({
        "user_id": user_id,
        "total_records": len(records),
        "records": records
    }), 200


@app.route("/api/user/record", methods=["POST"])
def user_add_record():
    data = request.get_json(force=True, silent=True) or {}
    user_id = data.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400

    day = int(data.get("day", 1))
    pain = float(data.get("pain", 0.0))
    stiffness = float(data.get("stiffness", 0.0))
    mobility = float(data.get("mobility", 0.0))
    grip = float(data.get("grip", 0.0))
    rom = float(data.get("rom", 50.0))
    reps = int(data.get("reps", 0))
    diary_text = str(data.get("diary_text", "")).strip()
    update_if_same_day = bool(data.get("update_if_same_day", False))

    res = add_user_record(user_id, day, pain, stiffness, mobility, grip, rom, reps, diary_text, update_if_same_day=update_if_same_day)
    return jsonify(res), 200



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"Starting HandRehab Flask Server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)

