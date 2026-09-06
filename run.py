"""
IRISCOPE entry point.

Run from the project root:
    python run.py

Starts the Flask development server. Serves the web frontend at "/" and
the analysis API at "/api/analyze" and "/api/match".

Bound to 0.0.0.0 by default so the app can also be reached from a phone
on the same network (useful for testing the camera flow on a real device
that isn't the laptop running the server). Flask's debug mode is OFF by
default -- with debug=True, Werkzeug's interactive debugger allows
arbitrary code execution from anyone who can reach the port, which is a
real risk on a network-bound address. Set IRISCOPE_DEBUG=1 to turn it on
for local-only development.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

from app import app  # noqa: E402  (path must be set up before this import)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("IRISCOPE_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
