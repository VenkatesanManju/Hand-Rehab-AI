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


def test_diary_page_session_save_and_elements(client):
    """
    Verifies that Step 1: Diary Rehab page contains the Today's Session Ready box,
    all 6 summary items, save/advance actions, and navigation buttons.
    """
    # 1. Verify HTML elements
    res_index = client.get("/")
    assert res_index.status_code == 200
    html = res_index.data.decode("utf-8")

    assert 'id="diary-session-summary-card"' in html
    assert 'id="diary-summary-pain"' in html
    assert 'id="diary-summary-stiffness"' in html
    assert 'id="diary-summary-mobility"' in html
    assert 'id="diary-summary-grip"' in html
    assert 'id="diary-summary-rom"' in html
    assert 'id="diary-summary-reps"' in html

    assert 'id="btn-diary-save-session"' in html
    assert 'id="label-diary-save-day"' in html
    assert 'id="btn-diary-advance-day"' not in html
    assert 'id="btn-advance-day"' not in html
    assert 'id="btn-goto-exercise"' in html
    assert 'id="btn-diary-goto-report"' in html

    # 2. Verify CSS styles
    res_css = client.get("/style.css")
    assert res_css.status_code == 200
    css = res_css.data.decode("utf-8")
    assert ".diary-session-box" in css
    assert ".diary-nav-actions-row" in css

    # 3. Verify JavaScript logic
    res_js = client.get("/app.js")
    assert res_js.status_code == 200
    js = res_js.data.decode("utf-8")
    assert "btn-diary-save-session" in js
    assert "btn-diary-advance-day" not in js
    assert "label-diary-save-day" in js
    assert "diary-summary-pain" in js

    # 4. Verify user registration seeds Day 0 baseline, and first rehab session is Day 1
    import time
    user_test = f"diary_saver_{int(time.time() * 1000)}"
    reg_res = client.post("/api/auth/register", data=json.dumps({
        "username": user_test,
        "password": "secretpassword",
        "full_name": "Diary Saver",
        "surgery_type": "Carpal Tunnel"
    }), content_type="application/json")
    user_id = json.loads(reg_res.data)["user"]["id"]

    # History starts with 1 record at Day 0
    hist1 = json.loads(client.get(f"/api/user/history?user_id={user_id}").data)
    assert hist1["total_records"] == 1
    assert hist1["records"][0]["day"] == 0

    # User saves from Diary Page -> logs Day 1
    add_day1 = client.post("/api/user/record", data=json.dumps({
        "user_id": user_id,
        "day": 1,
        "pain": 2.0,
        "stiffness": 2.0,
        "mobility": 1.5,
        "grip": 1.5,
        "rom": 40.0,
        "reps": 5,
        "diary_text": "Saved directly from Step 1 Diary page.",
        "update_if_same_day": False
    }), content_type="application/json")
    assert add_day1.status_code == 200

    hist2 = json.loads(client.get(f"/api/user/history?user_id={user_id}").data)
    assert hist2["total_records"] == 2
    assert [r["day"] for r in hist2["records"]] == [0, 1]


def test_registration_and_camera_exercise_guidance(client):
    """
    Verifies that the registration form includes easy-to-understand plain English
    condition descriptions, dynamic hint card, and that the Camera Exercise Tracker
    adapts guided exercise instructions to the patient's selected condition.
    """
    # 1. HTML Verification: Plain-English options & dynamic hint card
    res_index = client.get("/")
    assert res_index.status_code == 200
    html = res_index.data.decode("utf-8")

    # Friendly option labels
    assert "Palm / Finger Tendon Repair" in html
    assert "Back of Hand / Finger Straightening" in html
    assert "Broken Wrist Recovery (Distal Radius Fracture)" in html
    assert "Carpal Tunnel Release" in html
    assert "Broken Hand Bone Recovery" not in html
    assert "General Hand Mobility &amp; Stiffness" in html

    # Registration Hint Card
    assert 'id="reg-condition-info"' in html
    assert 'id="reg-card-title"' in html
    assert 'id="reg-card-desc"' in html
    assert 'id="reg-card-exercise"' in html

    # Camera Exercise Tracker Condition Guide Elements
    assert 'id="guide-condition-badge"' in html
    assert 'id="guide-condition-select"' in html
    assert 'id="guide-exercise-title"' in html
    assert 'id="guide-condition-goal"' in html
    assert 'id="btn-gesture-1"' in html
    assert 'id="btn-gesture-2"' in html
    assert 'id="guide-clinical-advisory"' in html
    assert 'id="guide-step-1-title"' in html
    assert 'id="guide-step-2-title"' in html
    assert 'id="guide-step-3-title"' in html

    # 2. CSS Verification
    res_css = client.get("/style.css")
    assert res_css.status_code == 200
    css = res_css.data.decode("utf-8")
    assert ".reg-condition-card" in css
    assert ".guide-condition-pill" in css
    assert ".guide-condition-select" in css
    assert ".guide-goal-text" in css
    assert ".gesture-switcher-row" in css
    assert ".btn-gesture" in css
    assert ".guide-advisory-text" in css

    # 3. JavaScript Verification
    res_js = client.get("/app.js")
    assert res_js.status_code == 200
    js = res_js.data.decode("utf-8")
    assert "CONDITION_GUIDES" in js
    assert "updateRegistrationHint" in js
    assert "updateExerciseGuide" in js
    assert "getConditionKey" in js
    assert "getActiveGesture" in js


def test_candidate_gestures_and_clinical_advisory(client):
    """
    Verifies that all 5 rehabilitation conditions provide the required condition-specific
    candidate gestures rather than generic exercises, and that the mandatory clinical
    safety advisory is displayed.
    """
    res_index = client.get("/")
    assert res_index.status_code == 200
    html = res_index.data.decode("utf-8")

    res_js = client.get("/app.js")
    assert res_js.status_code == 200
    js = res_js.data.decode("utf-8")

    # Required 5 rehabilitation conditions and their pairs of candidate gestures
    expected_gestures = {
        "Palm / Finger Tendon Repair (Flexor)": ["Hook Fist", "Individual Finger Bend"],
        "Back of Hand Tendon Repair (Extensor)": ["Straight Finger Extension", "Tabletop Hand Position"],
        "Broken Wrist Recovery (Distal Radius)": ["Wrist Flexion", "Wrist Extension"],
        "Carpal Tunnel Release (Post-Op)": ["Thumb-to-Index Fingertip Touch", "Thumb-to-Little Finger Touch"],
        "General Hand Mobility & Stiffness": ["Finger Spreading and Closing", "Thumb Opposition"]
    }

    for condition, gestures in expected_gestures.items():
        for gesture in gestures:
            assert gesture in js, f"Expected gesture '{gesture}' for condition '{condition}' in app.js"

    # Verbatim clinical warning requirement
    advisory_text = (
        "Important: These are candidate gestures for your project. "
        "For tendon repairs and postoperative conditions, show or recommend "
        "a gesture only when it is permitted by the patient's surgeon or hand therapist."
    )
    assert advisory_text in html
    assert advisory_text in js






