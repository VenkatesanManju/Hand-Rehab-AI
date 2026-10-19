"""
Simulated Daily Recovery Trajectories (Phase 4)
==============================================
SIMULATED DATA ONLY - NO CLINICAL VALIDITY - FOR ALGORITHMIC EXPERIMENTATION ONLY.

Generates synthetic longitudinal rehabilitation curves for hand surgery recovery.
Each patient has 60-120 days of continuous multi-variate monitoring:
- Symptoms: [pain, stiffness, mobility, grip] in [0.0, 3.0]
- Functional ROM: [rom] in degrees [0.0, 100.0]
- Recovery States:
  * 0: normal (gradual steady improvement)
  * 1: plateau (progress stalls for 10-25 days)
  * 2: relapse (temporary acute deterioration for 5-14 days)

Splits by patient (NEVER by day) into train, val, and test sets.
Also generates a held-out noisier simulation setting not used in training.
"""

import os
import argparse
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

# Fixed seed for reproducibility per AGENTS.md
DEFAULT_SEED = 42

FEATURE_COLS = ["pain", "stiffness", "mobility", "grip", "rom"]
STATE_MAP = {0: "normal", 1: "plateau", 2: "relapse"}
STATE_REV_MAP = {"normal": 0, "plateau": 1, "relapse": 2}


def simulate_single_patient(
    patient_id: str,
    rng: np.random.RandomState,
    min_days: int = 60,
    max_days: int = 120,
    is_noisy_setting: bool = False,
) -> pd.DataFrame:
    """
    Simulates a single patient's recovery trajectory over 60-120 days.
    """
    num_days = rng.randint(min_days, max_days + 1)

    # Initial baseline values at day 0 (post-surgery / early rehab)
    pain = float(rng.uniform(2.3, 2.9))
    stiffness = float(rng.uniform(2.1, 2.8))
    mobility = float(rng.uniform(2.0, 2.7))
    grip = float(rng.uniform(2.1, 2.8))
    rom = float(rng.uniform(25.0, 42.0))

    # Base recovery slopes (improvement per day under normal recovery)
    slope_pain = float(rng.uniform(0.018, 0.035))
    slope_stiffness = float(rng.uniform(0.016, 0.032))
    slope_mobility = float(rng.uniform(0.015, 0.030))
    slope_grip = float(rng.uniform(0.015, 0.032))
    slope_rom = float(rng.uniform(0.55, 0.95))

    # Patient-specific noise level
    if is_noisy_setting:
        sigma_sym = float(rng.uniform(0.16, 0.26))
        sigma_rom = float(rng.uniform(2.5, 4.5))
        spike_prob = 0.08
    else:
        sigma_sym = float(rng.uniform(0.04, 0.09))
        sigma_rom = float(rng.uniform(0.8, 1.8))
        spike_prob = 0.02

    # Schedule plateau and relapse phases
    # Assign states for each day: 0 = normal, 1 = plateau, 2 = relapse
    states = np.zeros(num_days, dtype=int)

    # 70% chance of 1 plateau, 30% chance of 2 plateaus
    num_plateaus = 1 if rng.rand() < 0.7 else (2 if rng.rand() < 0.4 else 0)
    for _ in range(num_plateaus):
        plat_len = rng.randint(10, 24)
        if num_days - plat_len - 15 > 15:
            start_p = rng.randint(15, num_days - plat_len - 10)
            states[start_p : start_p + plat_len] = 1

    # 55% chance of a relapse phase
    if rng.rand() < 0.55:
        relapse_len = rng.randint(5, 13)
        available_days = [d for d in range(20, num_days - relapse_len - 10) if states[d] == 0]
        if available_days:
            start_r = rng.choice(available_days)
            states[start_r : start_r + relapse_len] = 2

    records: List[Dict[str, Any]] = []

    for day in range(num_days):
        current_state = int(states[day])

        # State-dependent progression
        if current_state == 0:
            # Normal recovery: symptoms decline, ROM increases
            dpain = -slope_pain
            dstiffness = -slope_stiffness
            dmobility = -slope_mobility
            dgrip = -slope_grip
            drom = slope_rom
        elif current_state == 1:
            # Plateau: progress halts (slope ~ 0)
            dpain = rng.normal(0, 0.005)
            dstiffness = rng.normal(0, 0.005)
            dmobility = rng.normal(0, 0.005)
            dgrip = rng.normal(0, 0.005)
            drom = rng.normal(0, 0.05)
        else:
            # Relapse: acute deterioration (symptoms worsen, ROM drops)
            dpain = float(rng.uniform(0.05, 0.12))
            dstiffness = float(rng.uniform(0.05, 0.11))
            dmobility = float(rng.uniform(0.04, 0.10))
            dgrip = float(rng.uniform(0.04, 0.10))
            drom = -float(rng.uniform(0.9, 2.0))

        # Add Gaussian noise
        noise_p = rng.normal(0, sigma_sym)
        noise_s = rng.normal(0, sigma_sym)
        noise_m = rng.normal(0, sigma_sym)
        noise_g = rng.normal(0, sigma_sym)
        noise_r = rng.normal(0, sigma_rom)

        # Sporadic transient spike
        if rng.rand() < spike_prob:
            spike_dir = 1 if rng.rand() < 0.8 else -1
            noise_p += spike_dir * rng.uniform(0.2, 0.4)
            noise_r -= spike_dir * rng.uniform(3.0, 6.0)

        pain = float(np.clip(pain + dpain + noise_p, 0.0, 3.0))
        stiffness = float(np.clip(stiffness + dstiffness + noise_s, 0.0, 3.0))
        mobility = float(np.clip(mobility + dmobility + noise_m, 0.0, 3.0))
        grip = float(np.clip(grip + dgrip + noise_g, 0.0, 3.0))
        rom = float(np.clip(rom + drom + noise_r, 0.0, 100.0))

        records.append({
            "patient_id": patient_id,
            "day": day,
            "pain": round(pain, 3),
            "stiffness": round(stiffness, 3),
            "mobility": round(mobility, 3),
            "grip": round(grip, 3),
            "rom": round(rom, 2),
            "status": current_state,
            "status_label": STATE_MAP[current_state],
            "is_simulated": True,
        })

    return pd.DataFrame(records)


def generate_simulated_dataset(
    num_patients: int = 100,
    seed: int = DEFAULT_SEED,
    is_noisy_setting: bool = False,
    id_prefix: str = "PATIENT",
) -> pd.DataFrame:
    """
    Generates a collection of simulated patient trajectories.
    """
    rng = np.random.RandomState(seed)
    dfs = []
    for i in range(num_patients):
        pid = f"{id_prefix}_{i:03d}"
        df_p = simulate_single_patient(
            patient_id=pid,
            rng=rng,
            min_days=60,
            max_days=120,
            is_noisy_setting=is_noisy_setting,
        )
        dfs.append(df_p)
    return pd.concat(dfs, ignore_index=True)


def split_by_patient(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = DEFAULT_SEED,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits trajectories strictly by patient_id (NEVER by day).
    Guarantees that no patient's data is shared between train, val, or test.
    """
    rng = np.random.RandomState(seed)
    unique_patients = df["patient_id"].unique()
    rng.shuffle(unique_patients)

    n_total = len(unique_patients)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_pids = set(unique_patients[:n_train])
    val_pids = set(unique_patients[n_train : n_train + n_val])
    test_pids = set(unique_patients[n_train + n_val :])

    train_df = df[df["patient_id"].isin(train_pids)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_pids)].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(test_pids)].reset_index(drop=True)

    return train_df, val_df, test_df


def save_simulated_datasets(output_dir: str = "data/simulated", seed: int = DEFAULT_SEED):
    """
    Creates and saves standard train/val/test splits and a held-out noisy test set.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Generating standard simulation dataset (100 patients)...")
    df_standard = generate_simulated_dataset(
        num_patients=100,
        seed=seed,
        is_noisy_setting=False,
        id_prefix="PATIENT_STD",
    )

    train_df, val_df, test_df = split_by_patient(df_standard, train_ratio=0.70, val_ratio=0.15, seed=seed)

    print("Generating unseen noisy simulation test dataset (20 patients)...")
    df_noisy = generate_simulated_dataset(
        num_patients=20,
        seed=seed + 999,
        is_noisy_setting=True,
        id_prefix="PATIENT_NOISY",
    )

    # Save to data/simulated/
    train_path = os.path.join(output_dir, "recovery_curves_train.csv")
    val_path = os.path.join(output_dir, "recovery_curves_val.csv")
    test_path = os.path.join(output_dir, "recovery_curves_test.csv")
    noisy_test_path = os.path.join(output_dir, "recovery_curves_noisy_test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    df_noisy.to_csv(noisy_test_path, index=False)

    print(f"Saved Train: {len(train_df)} days across {train_df['patient_id'].nunique()} patients -> {train_path}")
    print(f"Saved Val:   {len(val_df)} days across {val_df['patient_id'].nunique()} patients -> {val_path}")
    print(f"Saved Test:  {len(test_df)} days across {test_df['patient_id'].nunique()} patients -> {test_path}")
    print(f"Saved Noisy: {len(df_noisy)} days across {df_noisy['patient_id'].nunique()} patients -> {noisy_test_path}")

    # Summary report
    print("\nState distribution in Training Set:")
    print(train_df["status_label"].value_counts(normalize=True).round(3))
    print("\nState distribution in Noisy Test Set:")
    print(df_noisy["status_label"].value_counts(normalize=True).round(3))


if __name__ == "__main__":
    save_simulated_datasets()
