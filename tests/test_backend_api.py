"""
Unit tests for Flask API backend endpoints (Phase 6)
"""

import pytest
import json
from app.backend.server import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_diary_api_valid(client):
    payload = {"text": "No pain today but my fingers were very stiff in the morning."}
    response = client.post("/api/diary", data=json.dumps(payload), content_type="application/json")
    assert response.status_code == 200
    data = json.loads(response.data)

    assert "pain" in data
    assert "stiffness" in data
    assert "mobility" in data
    assert "grip" in data
    assert "highlighted_words" in data
    assert len(data["highlighted_words"]) > 0
    assert "disclaimer" in data
    assert "consult your physiotherapist" in data["disclaimer"].lower()


def test_diary_api_empty_text(client):
    response = client.post("/api/diary", data=json.dumps({"text": ""}), content_type="application/json")
    assert response.status_code == 400


def test_trend_api_valid(client):
    # 14 days of history
    history = [
        {"day": i, "pain": 2.5 - 0.02 * i, "stiffness": 2.2 - 0.02 * i, "mobility": 2.0 - 0.01 * i, "grip": 2.1 - 0.01 * i, "rom": 35.0 + 1.2 * i}
        for i in range(14)
    ]
    response = client.post("/api/trend", data=json.dumps({"history": history}), content_type="application/json")
    assert response.status_code == 200
    data = json.loads(response.data)

    assert "forecast" in data
    assert "pain" in data["forecast"]
    assert "rom" in data["forecast"]
    assert "status" in data
    assert data["status"] in ["normal", "plateau", "relapse"]
    assert "alert" in data
    assert "disclaimer" in data


def test_trend_api_insufficient_history(client):
    # Only 5 days (requires 14)
    history = [{"day": i, "pain": 2.0, "stiffness": 2.0, "mobility": 2.0, "grip": 2.0, "rom": 40.0} for i in range(5)]
    response = client.post("/api/trend", data=json.dumps({"history": history}), content_type="application/json")
    assert response.status_code == 400


def test_sample_patient_api(client):
    response = client.get("/api/sample_patient?pid=PATIENT_STD_002")
    assert response.status_code == 200
    data = json.loads(response.data)
    assert "patient_id" in data
    assert "trajectory" in data
    assert len(data["trajectory"]) >= 60


def test_static_frontend_routes(client):
    # Verify index page
    res_index = client.get("/")
    assert res_index.status_code == 200
    assert b"HandRehab AI" in res_index.data
    assert b"CLINICAL NOTICE" in res_index.data

    # Verify style and script assets
    res_css = client.get("/style.css")
    assert res_css.status_code == 200

    res_js = client.get("/app.js")
    assert res_js.status_code == 200

    # Verify vendored offline dependencies
    res_chart = client.get("/vendor/chart.umd.min.js")
    assert res_chart.status_code == 200

    res_bundle = client.get("/vendor/vision_bundle.mjs")
    assert res_bundle.status_code == 200

    res_task = client.get("/vendor/hand_landmarker.task")
    assert res_task.status_code == 200

    res_wasm = client.get("/vendor/wasm/vision_wasm_internal.wasm")
    assert res_wasm.status_code == 200


def test_auth_and_user_history(client):
    import time
    unique_user = f"user_{int(time.time() * 1000)}"

    # 1. Register new user
    reg_payload = {
        "username": unique_user,
        "password": "secretpassword",
        "full_name": "Test Patient",
        "surgery_type": "Tendon Repair"
    }
    res_reg = client.post("/api/auth/register", data=json.dumps(reg_payload), content_type="application/json")
    assert res_reg.status_code == 200
    data_reg = json.loads(res_reg.data)
    assert data_reg["success"] is True
    user_id = data_reg["user"]["id"]

    # 2. Login with correct credentials
    login_payload = {"username": unique_user, "password": "secretpassword"}
    res_login = client.post("/api/auth/login", data=json.dumps(login_payload), content_type="application/json")
    assert res_login.status_code == 200
    data_login = json.loads(res_login.data)
    assert data_login["success"] is True
    assert data_login["user"]["full_name"] == "Test Patient"

    # 3. Login with wrong credentials should fail
    bad_login = client.post("/api/auth/login", data=json.dumps({"username": unique_user, "password": "wrong"}), content_type="application/json")
    assert bad_login.status_code == 401

    # 4. Fetch history
    res_hist = client.get(f"/api/user/history?user_id={user_id}")
    assert res_hist.status_code == 200
    data_hist = json.loads(res_hist.data)
    assert data_hist["total_records"] >= 1

    # 5. Add a new record
    rec_payload = {
        "user_id": user_id,
        "day": 1,
        "pain": 1.5,
        "stiffness": 1.2,
        "mobility": 1.0,
        "grip": 1.1,
        "rom": 55.0,
        "reps": 10,
        "diary_text": "Good flexion exercise."
    }
    res_add = client.post("/api/user/record", data=json.dumps(rec_payload), content_type="application/json")
    assert res_add.status_code == 200

    # 6. Verify history updated
    res_hist2 = client.get(f"/api/user/history?user_id={user_id}")
    data_hist2 = json.loads(res_hist2.data)
    assert data_hist2["total_records"] == 2


def test_recovery_report_three_charts_and_patient_friendly_ui(client):
    """
    Verifies that the Recovery Report page contains the patient-friendly header,
    six summary cards, ROM educational definition box, three dedicated charts,
    history duration filters, and clearly labeled AI estimated recovery.
    """
    res = client.get("/")
    assert res.status_code == 200
    html = res.data.decode("utf-8")

    # Header and Navigation
    assert "Step 3: Your Recovery Progress" in html
    assert "3. Your Recovery Progress" in html

    # Recovery Summary Cards (6 cards)
    assert 'id="card-metric-pain"' in html
    assert 'id="card-metric-stiffness"' in html
    assert 'id="card-metric-mobility"' in html
    assert 'id="card-metric-grip"' in html
    assert 'id="card-metric-rom"' in html
    assert 'id="card-metric-reps"' in html
    assert "Your Recovery Summary" in html

    # ROM Educational Explanation
    assert "Understanding Range of Motion (ROM)" in html
    assert "ROM measures how far a joint can move, in degrees." in html
    assert "Index &amp; Middle Finger Knuckles" in html

    # Chart 1: Recovery Progress Over Time (Pain & Stiffness)
    assert 'id="painStiffnessChart"' in html
    assert "Pain and Stiffness Trend" in html
    assert "Daily reported Pain and Stiffness on a 0–3 scale" in html

    # Chart 2: Hand Movement (ROM) Trend
    assert 'id="romChart"' in html
    assert "Hand Movement (ROM) Trend" in html
    assert "Range of Motion (ROM)" in html

    # Chart 3: Mobility and Grip Strength Trend
    assert 'id="mobilityGripChart"' in html
    assert "Mobility and Grip Strength Trend" in html

    # Empty State & 1-Session Notice
    assert "Record your first session to see your recovery graph." in html
    assert "1 session recorded." in html

    # AI Estimated Recovery clearly separated and labeled
    assert "Tomorrow's Estimated Recovery" in html
    assert "requires at least 14 days of recorded sessions" in html
    assert "Multi-Task GRU (AI Estimate)" in html

    # History Table Filters
    assert 'id="filter-all"' in html
    assert 'id="filter-14"' in html
    assert 'id="filter-7"' in html


def test_session_records_zero_one_multiple_flow(client):
    """
    Tests data retrieval and logic for zero, one, and multiple real session records.
    """
    import time
    from app.backend.db import get_db_connection

    unique_user = f"patient_{int(time.time() * 1000)}"
    reg_res = client.post("/api/auth/register", data=json.dumps({
        "username": unique_user,
        "password": "password123",
        "full_name": "Chart Test Patient",
        "surgery_type": "Tendon Repair"
    }), content_type="application/json")
    user_id = json.loads(reg_res.data)["user"]["id"]

    # Test 1: Zero sessions state (clear initial seed record to test 0 sessions)
    with get_db_connection() as conn:
        conn.cursor().execute("DELETE FROM rehab_records WHERE user_id = ?", (user_id,))
        conn.commit()

    res_zero = client.get(f"/api/user/history?user_id={user_id}")
    assert res_zero.status_code == 200
    data_zero = json.loads(res_zero.data)
    assert data_zero["total_records"] == 0
    assert data_zero["records"] == []

    # Test 2: Exactly One session state
    add_res_1 = client.post("/api/user/record", data=json.dumps({
        "user_id": user_id,
        "day": 1,
        "pain": 2.0,
        "stiffness": 2.5,
        "mobility": 1.5,
        "grip": 1.8,
        "rom": 45.0,
        "reps": 10,
        "diary_text": "First recorded day."
    }), content_type="application/json")
    assert add_res_1.status_code == 200

    res_one = client.get(f"/api/user/history?user_id={user_id}")
    data_one = json.loads(res_one.data)
    assert data_one["total_records"] == 1
    rec1 = data_one["records"][0]
    assert rec1["day"] == 1
    assert rec1["rom"] == 45.0
    assert rec1["pain"] == 2.0

    # Test 3: Same-day update accumulates reps and preserves peak ROM
    update_res = client.post("/api/user/record", data=json.dumps({
        "user_id": user_id,
        "day": 1,
        "pain": 1.5,
        "stiffness": 2.0,
        "mobility": 1.0,
        "grip": 1.5,
        "rom": 52.0, # Higher ROM
        "reps": 10,  # 10 more reps -> total 20
        "diary_text": "Second session today.",
        "update_if_same_day": True
    }), content_type="application/json")
    assert update_res.status_code == 200
    assert json.loads(update_res.data)["updated"] is True

    res_updated = client.get(f"/api/user/history?user_id={user_id}")
    data_updated = json.loads(res_updated.data)
    assert data_updated["total_records"] == 1
    assert data_updated["records"][0]["rom"] == 52.0
    assert data_updated["records"][0]["reps"] == 20

    # Test 4: Multiple sessions state
    for day in range(2, 6):
        client.post("/api/user/record", data=json.dumps({
            "user_id": user_id,
            "day": day,
            "pain": max(0.5, 2.0 - 0.2 * day),
            "stiffness": max(0.5, 2.0 - 0.2 * day),
            "mobility": 1.0,
            "grip": 1.0,
            "rom": min(85.0, 52.0 + 3.0 * day),
            "reps": 10,
            "diary_text": f"Day {day} routine."
        }), content_type="application/json")

    res_multi = client.get(f"/api/user/history?user_id={user_id}")
    data_multi = json.loads(res_multi.data)
    assert data_multi["total_records"] == 5
    assert [r["day"] for r in data_multi["records"]] == [1, 2, 3, 4, 5]
    # Verify ROM increases properly
    assert data_multi["records"][-1]["rom"] > data_multi["records"][0]["rom"]



