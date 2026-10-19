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

