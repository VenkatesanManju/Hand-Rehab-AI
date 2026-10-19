"""
HandRehab AI — Main Application Entry Point
Run simply with:
    python app.py
"""

import os
import sys
import importlib.util

# Set project root in path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Dynamically import backend server module avoiding app.py vs app/ directory clash
server_path = os.path.join(PROJECT_ROOT, "app", "backend", "server.py")
spec = importlib.util.spec_from_file_location("server", server_path)
server_mod = importlib.util.module_from_spec(spec)
sys.modules["server"] = server_mod
spec.loader.exec_module(server_mod)

flask_app = server_mod.app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 70)
    print("  HANDREHAB AI - RECOVERY MONITORING DASHBOARD")
    print("=" * 70)
    print(f"  * Local Web Server: http://127.0.0.1:{port}")
    print("  * Open your web browser (Chrome, Edge, or Firefox) to view.")
    print("  * Press CTRL+C in this terminal window to stop the server.")
    print("=" * 70)
    flask_app.run(host="0.0.0.0", port=port, debug=False)
