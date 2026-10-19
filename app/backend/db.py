"""
Database Layer for HandRehab AI (Authentication & Patient History Persistence)
Uses SQLite for zero-configuration, robust local persistence.
"""

import os
import sqlite3
import pandas as pd
from typing import Optional, Dict, Any, List
from werkzeug.security import generate_password_hash, check_password_hash

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DB_PATH = os.path.join(PROJECT_ROOT, "data/users.db")


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Initializes SQLite tables and seeds the default demo patient account."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                full_name TEXT NOT NULL,
                surgery_type TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS rehab_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                day INTEGER NOT NULL,
                pain REAL NOT NULL,
                stiffness REAL NOT NULL,
                mobility REAL NOT NULL,
                grip REAL NOT NULL,
                rom REAL NOT NULL,
                reps INTEGER DEFAULT 0,
                diary_text TEXT,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        """)

        conn.commit()

        # Seed demo patient if not exists
        cursor.execute("SELECT id FROM users WHERE username = 'demo_patient'")
        row = cursor.fetchone()
        if not row:
            hashed = generate_password_hash("rehab123")
            cursor.execute("""
                INSERT INTO users (username, password_hash, full_name, surgery_type)
                VALUES (?, ?, ?, ?)
            """, ("demo_patient", hashed, "Alex Mercer", "Flexor Tendon Repair (Zone II)"))
            demo_user_id = cursor.lastrowid
            conn.commit()

            # Seed 60-day baseline trajectory for demo user from simulated test data
            test_csv = os.path.join(PROJECT_ROOT, "data/simulated/recovery_curves_test.csv")
            if os.path.exists(test_csv):
                df = pd.read_csv(test_csv)
                p_df = df[df["patient_id"] == "PATIENT_STD_002"].sort_values("day")
                if len(p_df) == 0:
                    p_df = df.iloc[:60]

                records_to_insert = [
                    (demo_user_id, int(r["day"]), float(r["pain"]), float(r["stiffness"]),
                     float(r["mobility"]), float(r["grip"]), float(r["rom"]), 10,
                     "Daily routine exercise and flexion training completed.")
                    for _, r in p_df.iterrows()
                ]
                cursor.executemany("""
                    INSERT INTO rehab_records (user_id, day, pain, stiffness, mobility, grip, rom, reps, diary_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, records_to_insert)
                conn.commit()


def register_user(username: str, password: str, full_name: str, surgery_type: str) -> Dict[str, Any]:
    """Registers a new patient account."""
    username = username.strip().lower()
    full_name = full_name.strip()
    surgery_type = surgery_type.strip() or "General Hand Rehabilitation"

    if not username or not password or not full_name:
        return {"success": False, "error": "Username, password, and full name are required."}

    if len(password) < 4:
        return {"success": False, "error": "Password must be at least 4 characters long."}

    password_hash = generate_password_hash(password)

    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO users (username, password_hash, full_name, surgery_type)
                VALUES (?, ?, ?, ?)
            """, (username, password_hash, full_name, surgery_type))
            user_id = cursor.lastrowid
            conn.commit()

            # Seed initial Day 0 baseline for the new user so they immediately have a reference point
            cursor.execute("""
                INSERT INTO rehab_records (user_id, day, pain, stiffness, mobility, grip, rom, reps, diary_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (user_id, 0, 2.5, 2.5, 2.2, 2.4, 30.0, 0, "Initial post-evaluation baseline."))
            conn.commit()

            return {
                "success": True,
                "user": {
                    "id": user_id,
                    "username": username,
                    "full_name": full_name,
                    "surgery_type": surgery_type
                }
            }
    except sqlite3.IntegrityError:
        return {"success": False, "error": "Username is already registered. Please choose another username or log in."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def authenticate_user(username: str, password: str) -> Dict[str, Any]:
    """Authenticates a patient against stored hashed credentials."""
    username = username.strip().lower()

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, password_hash, full_name, surgery_type FROM users WHERE username = ?", (username,))
        row = cursor.fetchone()

        if not row:
            return {"success": False, "error": "Invalid username or password."}

        if not check_password_hash(row["password_hash"], password):
            return {"success": False, "error": "Invalid username or password."}

        return {
            "success": True,
            "user": {
                "id": row["id"],
                "username": row["username"],
                "full_name": row["full_name"],
                "surgery_type": row["surgery_type"]
            }
        }


def get_user_records(user_id: int) -> List[Dict[str, Any]]:
    """Retrieves all rehabilitation session records for a user in chronological order."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT day, pain, stiffness, mobility, grip, rom, reps, diary_text, recorded_at
            FROM rehab_records
            WHERE user_id = ?
            ORDER BY day ASC, id ASC
        """, (user_id,))
        rows = cursor.fetchall()
        return [dict(r) for r in rows]


def add_user_record(user_id: int, day: int, pain: float, stiffness: float,
                    mobility: float, grip: float, rom: float, reps: int = 0,
                    diary_text: str = "") -> Dict[str, Any]:
    """Appends a new daily rehabilitation session to the user's history."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO rehab_records (user_id, day, pain, stiffness, mobility, grip, rom, reps, diary_text)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (user_id, day, pain, stiffness, mobility, grip, rom, reps, diary_text))
        record_id = cursor.lastrowid
        conn.commit()

        return {"success": True, "record_id": record_id}
