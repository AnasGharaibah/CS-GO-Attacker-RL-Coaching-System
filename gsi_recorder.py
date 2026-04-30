# This file has moved to  recording/gsi_recorder.py
# Run from the project root with:
#
#     python recording/gsi_recorder.py
#
# Or use the launcher:
#
#     python run.py

import subprocess, sys
subprocess.run([sys.executable, "recording/gsi_recorder.py"] + sys.argv[1:])
