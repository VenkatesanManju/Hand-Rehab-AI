"""
Vendoring script for offline MediaPipe Tasks Vision and Chart.js (Phase 5)
Downloads required library, WASM binaries, and model weights to app/frontend/vendor/.
"""

import os
import urllib.request

VENDOR_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "vendor"))
WASM_DIR = os.path.join(VENDOR_DIR, "wasm")

ASSETS = {
    os.path.join(VENDOR_DIR, "chart.umd.min.js"): "https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js",
    os.path.join(VENDOR_DIR, "vision_bundle.mjs"): "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs",
    os.path.join(WASM_DIR, "vision_wasm_internal.js"): "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm/vision_wasm_internal.js",
    os.path.join(WASM_DIR, "vision_wasm_internal.wasm"): "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm/vision_wasm_internal.wasm",
    os.path.join(VENDOR_DIR, "hand_landmarker.task"): "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
}

def download_assets():
    os.makedirs(WASM_DIR, exist_ok=True)
    print(f"Downloading vendor assets into {VENDOR_DIR}...")

    for local_path, url in ASSETS.items():
        rel_name = os.path.relpath(local_path, VENDOR_DIR)
        if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
            print(f" - {rel_name} already exists ({os.path.getsize(local_path)} bytes), skipping.")
            continue

        print(f" - Downloading {rel_name} from {url}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            with open(local_path, "wb") as f:
                f.write(data)
        print(f"   Done ({len(data)} bytes).")

    print("All vendor assets successfully downloaded and ready for offline use.")

if __name__ == "__main__":
    download_assets()
