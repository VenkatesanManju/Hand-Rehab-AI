"""
Training & Evaluation Pipeline for Longitudinal Trend Recovery (Phase 4)
========================================================================
SIMULATED DATA ONLY - NO CLINICAL VALIDITY - FOR ALGORITHMIC EXPERIMENTATION ONLY.

Compares:
1. Tomorrow = Today (Persistence baseline)
2. 7-Day Moving Average
3. Rule-Based Detector (Slope thresholding, tuned on validation only)
4. Multi-Task Simple RNN (Window 14)
5. Multi-Task LSTM (Window 14)
6. Multi-Task GRU (Window 14)

Evaluated on BOTH:
- Standard Held-Out Test Patients (Unseen patients)
- Noisier Simulation Test Patients (Higher noise setting not seen in training)

Tasks:
- Next-Day Forecast: MAE, RMSE (overall, symptom MAE, ROM MAE in natural units)
- Recovery State Classification: Macro Precision, Macro Recall, Macro F1, Accuracy

Outputs:
- results/trend_comparison.csv
- results/trend_comparison.txt
- results/trend_forecast_plot.png
- results/best_trend_model.pt
"""

import os
import sys
import logging
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support, accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.trend.dataset import TrendWindowDataset, collate_trend_batch, FEATURE_COLS
from src.trend.models import MultiTaskTrendModel, compute_multitask_loss
from src.trend.baselines import TomorrowIsTodayBaseline, MovingAverageBaseline, RuleBasedStatusDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS = [42, 43, 44]


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def evaluate_neural_model(
    model: nn.Module,
    loader: DataLoader,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    device: torch.device,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluates next-day forecast and status classification on a DataLoader.
    Returns metrics dict and prediction arrays.
    """
    model.eval()
    all_preds_forecast = []
    all_targets_forecast = []
    all_preds_status = []
    all_targets_status = []

    mean_t = torch.tensor(scaler_mean, device=device)
    std_t = torch.tensor(scaler_std, device=device)

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            target_f = batch["y_forecast"].to(device)
            target_s = batch["y_status"].to(device)

            pred_f_norm, pred_s_logits = model(x)

            # Denormalize forecast predictions to natural physical units
            pred_f = pred_f_norm * std_t + mean_t

            pred_status = torch.argmax(pred_s_logits, dim=-1)

            all_preds_forecast.append(pred_f.cpu().numpy())
            all_targets_forecast.append(target_f.cpu().numpy())
            all_preds_status.append(pred_status.cpu().numpy())
            all_targets_status.append(target_s.cpu().numpy())

    preds_f = np.concatenate(all_preds_forecast, axis=0)
    targets_f = np.concatenate(all_targets_forecast, axis=0)
    preds_s = np.concatenate(all_preds_status, axis=0)
    targets_s = np.concatenate(all_targets_status, axis=0)

    # 1. Forecast metrics
    mae_overall = float(mean_absolute_error(targets_f, preds_f))
    rmse_overall = float(np.sqrt(mean_squared_error(targets_f, preds_f)))

    # Symptoms (first 4 columns) in [0, 3]
    mae_symptoms = float(mean_absolute_error(targets_f[:, :4], preds_f[:, :4]))
    # ROM (column 4) in degrees [0, 100]
    mae_rom = float(mean_absolute_error(targets_f[:, 4], preds_f[:, 4]))

    # 2. Status classification metrics
    acc = float(accuracy_score(targets_s, preds_s))
    prec, rec, f1, _ = precision_recall_fscore_support(
        targets_s, preds_s, average="macro", zero_division=0
    )

    metrics = {
        "forecast_mae_overall": mae_overall,
        "forecast_rmse_overall": rmse_overall,
        "forecast_mae_symptoms": mae_symptoms,
        "forecast_mae_rom": mae_rom,
        "status_accuracy": acc,
        "status_macro_precision": float(prec),
        "status_macro_recall": float(rec),
        "status_macro_f1": float(f1),
    }

    return metrics, preds_f, targets_f, preds_s, targets_s


def train_neural_trend_model(
    encoder_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    seed: int = 42,
    epochs: int = 20,
    patience: int = 5,
    device: torch.device = DEVICE,
) -> Tuple[nn.Module, float]:
    set_seed(seed)
    model = MultiTaskTrendModel(
        input_dim=5,
        hidden_dim=64,
        encoder_type=encoder_type,
        num_layers=1,
        dropout=0.2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    mean_t = torch.tensor(scaler_mean, device=device)
    std_t = torch.tensor(scaler_std, device=device)

    best_val_score = -float("inf")
    best_weights = None
    patience_cnt = 0

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device)
            target_f = batch["y_forecast"].to(device)
            target_s = batch["y_status"].to(device)

            # Normalize target forecast to match normalized input space
            target_f_norm = (target_f - mean_t) / std_t

            optimizer.zero_grad()
            pred_f_norm, pred_s_logits = model(x)
            loss, mse_val, ce_val = compute_multitask_loss(
                pred_f_norm, target_f_norm, pred_s_logits, target_s, forecast_weight=1.0, cls_weight=1.0
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()

        # Validation evaluation (strictly on validation set)
        val_metrics, _, _, _, _ = evaluate_neural_model(model, val_loader, scaler_mean, scaler_std, device)
        # Composite validation score: validation macro F1 minus normalized forecast MAE
        val_f1 = val_metrics["status_macro_f1"]
        val_mae = val_metrics["forecast_mae_overall"]
        val_score = val_f1 - 0.5 * val_mae

        if val_score > best_val_score:
            best_val_score = val_score
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                break

    if best_weights:
        model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})

    return model, best_val_score


def evaluate_baselines_on_dataset(
    dataset: TrendWindowDataset,
    rule_detector: RuleBasedStatusDetector,
) -> Dict[str, Dict[str, float]]:
    """
    Evaluates Tomorrow=Today, 7-Day MA, and Rule-Based Detector on a TrendWindowDataset.
    """
    # Extract raw arrays
    raw_windows = np.stack([s["raw_window"] for s in dataset.samples])
    targets_f = np.stack([s["target_forecast"] for s in dataset.samples])
    targets_s = np.array([s["target_status"] for s in dataset.samples])

    results = {}

    # 1. Tomorrow = Today (Persistence)
    p_base = TomorrowIsTodayBaseline()
    pred_f_p = p_base.predict_forecast(raw_windows)
    pred_s_p = p_base.predict_status(raw_windows)

    mae_p = float(mean_absolute_error(targets_f, pred_f_p))
    rmse_p = float(np.sqrt(mean_squared_error(targets_f, pred_f_p)))
    mae_p_sym = float(mean_absolute_error(targets_f[:, :4], pred_f_p[:, :4]))
    mae_p_rom = float(mean_absolute_error(targets_f[:, 4], pred_f_p[:, 4]))

    acc_p = float(accuracy_score(targets_s, pred_s_p))
    prec_p, rec_p, f1_p, _ = precision_recall_fscore_support(targets_s, pred_s_p, average="macro", zero_division=0)

    results["Tomorrow = Today"] = {
        "forecast_mae_overall": mae_p,
        "forecast_rmse_overall": rmse_p,
        "forecast_mae_symptoms": mae_p_sym,
        "forecast_mae_rom": mae_p_rom,
        "status_accuracy": acc_p,
        "status_macro_precision": float(prec_p),
        "status_macro_recall": float(rec_p),
        "status_macro_f1": float(f1_p),
    }

    # 2. 7-Day Moving Average
    ma_base = MovingAverageBaseline(ma_days=7)
    pred_f_ma = ma_base.predict_forecast(raw_windows)
    pred_s_ma = ma_base.predict_status(raw_windows)

    mae_ma = float(mean_absolute_error(targets_f, pred_f_ma))
    rmse_ma = float(np.sqrt(mean_squared_error(targets_f, pred_f_ma)))
    mae_ma_sym = float(mean_absolute_error(targets_f[:, :4], pred_f_ma[:, :4]))
    mae_ma_rom = float(mean_absolute_error(targets_f[:, 4], pred_f_ma[:, 4]))

    acc_ma = float(accuracy_score(targets_s, pred_s_ma))
    prec_ma, rec_ma, f1_ma, _ = precision_recall_fscore_support(targets_s, pred_s_ma, average="macro", zero_division=0)

    results["7-Day Moving Average"] = {
        "forecast_mae_overall": mae_ma,
        "forecast_rmse_overall": rmse_ma,
        "forecast_mae_symptoms": mae_ma_sym,
        "forecast_mae_rom": mae_ma_rom,
        "status_accuracy": acc_ma,
        "status_macro_precision": float(prec_ma),
        "status_macro_recall": float(rec_ma),
        "status_macro_f1": float(f1_ma),
    }

    # 3. Rule-Based Status Detector (combines with Persistence forecast)
    pred_s_rule = rule_detector.predict_status(raw_windows)
    acc_rule = float(accuracy_score(targets_s, pred_s_rule))
    prec_r, rec_r, f1_r, _ = precision_recall_fscore_support(targets_s, pred_s_rule, average="macro", zero_division=0)

    results["Rule-Based Detector"] = {
        "forecast_mae_overall": mae_p,
        "forecast_rmse_overall": rmse_p,
        "forecast_mae_symptoms": mae_p_sym,
        "forecast_mae_rom": mae_p_rom,
        "status_accuracy": acc_rule,
        "status_macro_precision": float(prec_r),
        "status_macro_recall": float(rec_r),
        "status_macro_f1": float(f1_r),
    }

    return results


def run_trend_experiment():
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # 1. Load data
    train_df = pd.read_csv("data/simulated/recovery_curves_train.csv")
    val_df = pd.read_csv("data/simulated/recovery_curves_val.csv")
    test_df = pd.read_csv("data/simulated/recovery_curves_test.csv")
    noisy_test_df = pd.read_csv("data/simulated/recovery_curves_noisy_test.csv")

    logger.info(f"Loaded Trajectories: Train={train_df['patient_id'].nunique()} pts ({len(train_df)} days), "
                f"Val={val_df['patient_id'].nunique()} pts ({len(val_df)} days), "
                f"Test={test_df['patient_id'].nunique()} pts ({len(test_df)} days), "
                f"Noisy={noisy_test_df['patient_id'].nunique()} pts ({len(noisy_test_df)} days)")

    # 2. Build datasets (fit scaler on train only)
    train_ds = TrendWindowDataset(train_df, window_size=14, fit_scaler=True)
    scaler_mean = train_ds.scaler_mean
    scaler_std = train_ds.scaler_std

    val_ds = TrendWindowDataset(val_df, window_size=14, scaler_mean=scaler_mean, scaler_std=scaler_std)
    test_ds = TrendWindowDataset(test_df, window_size=14, scaler_mean=scaler_mean, scaler_std=scaler_std)
    noisy_test_ds = TrendWindowDataset(noisy_test_df, window_size=14, scaler_mean=scaler_mean, scaler_std=scaler_std)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_trend_batch)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, collate_fn=collate_trend_batch)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate_trend_batch)
    noisy_test_loader = DataLoader(noisy_test_ds, batch_size=32, shuffle=False, collate_fn=collate_trend_batch)

    # 3. Tune Rule-Based Detector strictly on validation set
    logger.info("Tuning Rule-Based Detector on validation set...")
    val_raw_windows = np.stack([s["raw_window"] for s in val_ds.samples])
    val_targets_s = np.array([s["target_status"] for s in val_ds.samples])
    rule_detector = RuleBasedStatusDetector()
    rule_detector.tune_on_validation(val_raw_windows, val_targets_s)
    logger.info(f"Tuned Rule-Based Detector: relapse_sym={rule_detector.relapse_sym_threshold:.3f}, plateau_sym={rule_detector.plateau_sym_threshold:.3f}")

    # 4. Evaluate Baselines on Standard Test & Noisy Test
    base_std_results = evaluate_baselines_on_dataset(test_ds, rule_detector)
    base_noisy_results = evaluate_baselines_on_dataset(noisy_test_ds, rule_detector)

    all_models = ["Tomorrow = Today", "7-Day Moving Average", "Rule-Based Detector", "RNN", "LSTM", "GRU"]
    comparison_records: List[Dict[str, Any]] = []

    # Store baseline results
    for b_name in ["Tomorrow = Today", "7-Day Moving Average", "Rule-Based Detector"]:
        for eval_set, res_dict in [("Standard Test Split", base_std_results[b_name]), ("Noisy Simulation Test", base_noisy_results[b_name])]:
            comparison_records.append({
                "Model": b_name,
                "Evaluation_Set": eval_set,
                "Forecast_MAE": f"{res_dict['forecast_mae_overall']:.3f}",
                "Forecast_RMSE": f"{res_dict['forecast_rmse_overall']:.3f}",
                "Symptoms_MAE": f"{res_dict['forecast_mae_symptoms']:.3f}",
                "ROM_MAE_deg": f"{res_dict['forecast_mae_rom']:.2f}",
                "Status_Accuracy": f"{res_dict['status_accuracy']:.3f}",
                "Status_Macro_F1": f"{res_dict['status_macro_f1']:.3f}",
                "Status_Precision": f"{res_dict['status_macro_precision']:.3f}",
                "Status_Recall": f"{res_dict['status_macro_recall']:.3f}",
            })

    # 5. Train & Evaluate Neural Models Across Seeds (Selected on Validation Only)
    neural_models = ["rnn", "lstm", "gru"]
    best_overall_val_score = -float("inf")
    best_model_obj = None
    best_model_name = ""

    for enc in neural_models:
        logger.info(f"--- Training & Evaluating {enc.upper()} across seeds {SEEDS} ---")
        seed_res_std: Dict[str, List[float]] = {k: [] for k in ["forecast_mae_overall", "forecast_rmse_overall", "forecast_mae_symptoms", "forecast_mae_rom", "status_accuracy", "status_macro_f1", "status_macro_precision", "status_macro_recall"]}
        seed_res_noisy: Dict[str, List[float]] = {k: [] for k in ["forecast_mae_overall", "forecast_rmse_overall", "forecast_mae_symptoms", "forecast_mae_rom", "status_accuracy", "status_macro_f1", "status_macro_precision", "status_macro_recall"]}

        for seed in SEEDS:
            model, val_score = train_neural_trend_model(
                encoder_type=enc,
                train_loader=train_loader,
                val_loader=val_loader,
                scaler_mean=scaler_mean,
                scaler_std=scaler_std,
                seed=seed,
                epochs=20,
                patience=5,
                device=DEVICE,
            )

            # Evaluate on standard test
            std_metrics, _, _, _, _ = evaluate_neural_model(model, test_loader, scaler_mean, scaler_std, DEVICE)
            for k in seed_res_std:
                seed_res_std[k].append(std_metrics[k])

            # Evaluate on noisy test
            noisy_metrics, _, _, _, _ = evaluate_neural_model(model, noisy_test_loader, scaler_mean, scaler_std, DEVICE)
            for k in seed_res_noisy:
                seed_res_noisy[k].append(noisy_metrics[k])

            # Model selection: strictly based on validation score
            if val_score > best_overall_val_score:
                best_overall_val_score = val_score
                best_model_obj = model
                best_model_name = enc

        display_name = enc.upper()
        # Compile mean ± std for standard and noisy sets
        for eval_set, res_data in [("Standard Test Split", seed_res_std), ("Noisy Simulation Test", seed_res_noisy)]:
            comparison_records.append({
                "Model": display_name,
                "Evaluation_Set": eval_set,
                "Forecast_MAE": f"{np.mean(res_data['forecast_mae_overall']):.3f} ± {np.std(res_data['forecast_mae_overall']):.3f}",
                "Forecast_RMSE": f"{np.mean(res_data['forecast_rmse_overall']):.3f} ± {np.std(res_data['forecast_rmse_overall']):.3f}",
                "Symptoms_MAE": f"{np.mean(res_data['forecast_mae_symptoms']):.3f} ± {np.std(res_data['forecast_mae_symptoms']):.3f}",
                "ROM_MAE_deg": f"{np.mean(res_data['forecast_mae_rom']):.2f} ± {np.std(res_data['forecast_mae_rom']):.2f}",
                "Status_Accuracy": f"{np.mean(res_data['status_accuracy']):.3f} ± {np.std(res_data['status_accuracy']):.3f}",
                "Status_Macro_F1": f"{np.mean(res_data['status_macro_f1']):.3f} ± {np.std(res_data['status_macro_f1']):.3f}",
                "Status_Precision": f"{np.mean(res_data['status_macro_precision']):.3f} ± {np.std(res_data['status_macro_precision']):.3f}",
                "Status_Recall": f"{np.mean(res_data['status_macro_recall']):.3f} ± {np.std(res_data['status_macro_recall']):.3f}",
            })

    # 6. Save Best Model Checkpoint
    best_model_path = os.path.join(results_dir, "best_trend_model.pt")
    torch.save(best_model_obj.state_dict(), best_model_path)
    logger.info(f"Saved best trend model ({best_model_name.upper()}) checkpoint to {best_model_path}")

    # 7. Write results/trend_comparison.csv & results/trend_comparison.txt
    trend_df = pd.DataFrame(comparison_records)
    csv_path = os.path.join(results_dir, "trend_comparison.csv")
    txt_path = os.path.join(results_dir, "trend_comparison.txt")

    trend_df.to_csv(csv_path, index=False)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Longitudinal Recovery Trend Model Comparison (Phase 4)\n")
        f.write("=====================================================\n")
        f.write("SIMULATED DATA ONLY - NO CLINICAL VALIDITY\n")
        f.write(f"Protocol: Window 14 Days, Model Selection on Val Set Only, Seeds {SEEDS}\n\n")
        f.write(trend_df.to_string(index=False))
        f.write("\n\nNote: All numbers computed from executed runs. Never fabricated.\n")

    logger.info(f"Saved comparison to {csv_path} and {txt_path}")

    # 8. Generate Forecast Plot (results/trend_forecast_plot.png)
    logger.info("Generating forecast and status trajectory plot for a sample patient...")
    generate_forecast_plot(
        test_df=test_df,
        model=best_model_obj,
        scaler_mean=scaler_mean,
        scaler_std=scaler_std,
        output_path=os.path.join(results_dir, "trend_forecast_plot.png"),
    )

    logger.info("Phase 4 trend experiment complete!")
    return trend_df


def generate_forecast_plot(
    test_df: pd.DataFrame,
    model: nn.Module,
    scaler_mean: np.ndarray,
    scaler_std: np.ndarray,
    output_path: str = "results/trend_forecast_plot.png",
):
    """
    Plots a sample test patient's full trajectory:
    - Pain severity & ROM true vs predicted trajectory.
    - Ground-truth plateau (yellow) and relapse (red) regions vs model predicted status alerts.
    """
    # Pick a patient that has both plateau and relapse if possible
    grouped = test_df.groupby("patient_id")
    chosen_pid = None
    for pid, group in grouped:
        st = set(group["status"].unique())
        if 1 in st and 2 in st:
            chosen_pid = pid
            break
    if chosen_pid is None:
        chosen_pid = test_df["patient_id"].iloc[0]

    p_df = test_df[test_df["patient_id"] == chosen_pid].sort_values("day").reset_index(drop=True)
    days = p_df["day"].values
    pain_true = p_df["pain"].values
    rom_true = p_df["rom"].values
    status_true = p_df["status"].values

    # Run inference across sliding windows for this patient
    ds = TrendWindowDataset(p_df, window_size=14, scaler_mean=scaler_mean, scaler_std=scaler_std)
    loader = DataLoader(ds, batch_size=len(ds), shuffle=False, collate_fn=collate_trend_batch)

    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(DEVICE)
            pred_f_norm, pred_s_logits = model(x)
            mean_t = torch.tensor(scaler_mean, device=DEVICE)
            std_t = torch.tensor(scaler_std, device=DEVICE)
            pred_f = (pred_f_norm * std_t + mean_t).cpu().numpy()
            pred_s = torch.argmax(pred_s_logits, dim=-1).cpu().numpy()

    eval_days = np.array([s["target_day"] for s in ds.samples])
    pain_pred = pred_f[:, 0]
    rom_pred = pred_f[:, 4]

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1]})

    # Subplot 1: Pain Severity
    ax1.plot(days, pain_true, label="Ground Truth Pain", color="#1f77b4", linewidth=2.0)
    ax1.plot(eval_days, pain_pred, label="Model 1-Day Forecast", color="#ff7f0e", linestyle="--", linewidth=2.0)
    ax1.set_ylabel("Pain Severity (0-3)", fontsize=11, fontweight="bold")
    ax1.set_ylim(-0.1, 3.2)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.set_title(f"Simulated Longitudinal Recovery Trajectory: {chosen_pid} (Simulated Data - No Clinical Validity)", fontsize=13, fontweight="bold")

    # Subplot 2: Range of Motion (ROM)
    ax2.plot(days, rom_true, label="Ground Truth ROM (°)", color="#2ca02c", linewidth=2.0)
    ax2.plot(eval_days, rom_pred, label="Model 1-Day Forecast (°)", color="#9467bd", linestyle="--", linewidth=2.0)
    ax2.set_ylabel("ROM (degrees)", fontsize=11, fontweight="bold")
    ax2.set_ylim(15, 95)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", framealpha=0.9)

    # Shade plateau and relapse regions on ax1 and ax2
    for t_idx in range(len(days)):
        d = days[t_idx]
        st = status_true[t_idx]
        if st == 1:  # Plateau
            ax1.axvspan(d - 0.5, d + 0.5, color="gold", alpha=0.18)
            ax2.axvspan(d - 0.5, d + 0.5, color="gold", alpha=0.18)
        elif st == 2:  # Relapse
            ax1.axvspan(d - 0.5, d + 0.5, color="crimson", alpha=0.22)
            ax2.axvspan(d - 0.5, d + 0.5, color="crimson", alpha=0.22)

    # Subplot 3: Recovery Status vs Alert (0: Normal, 1: Plateau, 2: Relapse)
    ax3.step(days, status_true, label="True Status", color="black", linewidth=2.0, where="mid")
    ax3.step(eval_days, pred_s, label="Detected Status Alert", color="crimson", linestyle=":", linewidth=2.0, where="mid")
    ax3.set_yticks([0, 1, 2])
    ax3.set_yticklabels(["Normal", "Plateau", "Relapse"], fontsize=10)
    ax3.set_xlabel("Rehabilitation Days", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Status", fontsize=11, fontweight="bold")
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="upper right", framealpha=0.9)

    # Shading legend patches
    patch_plat = mpatches.Patch(color="gold", alpha=0.3, label="Plateau Phase (Progress Stalled)")
    patch_relapse = mpatches.Patch(color="crimson", alpha=0.3, label="Relapse Phase (Acute Flare-up)")
    ax1.legend(handles=[
        plt.Line2D([0], [0], color="#1f77b4", lw=2, label="True Pain"),
        plt.Line2D([0], [0], color="#ff7f0e", lw=2, ls="--", label="Model Forecast"),
        patch_plat,
        patch_relapse,
    ], loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    logger.info(f"Saved forecast plot to {output_path}")


if __name__ == "__main__":
    run_trend_experiment()
