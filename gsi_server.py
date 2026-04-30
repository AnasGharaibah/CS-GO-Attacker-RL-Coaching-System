# This file has moved to  recording/gsi_server.py
# It is the full-featured GSI server with REST API endpoints.
# Use it if you want to read live game state from other apps.
# For data recording only, use recording/gsi_recorder.py instead.
#
# Run from the project root with:
#
#     python recording/gsi_server.py

import subprocess, sys
subprocess.run([sys.executable, "recording/gsi_server.py"] + sys.argv[1:])
