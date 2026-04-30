# This file has moved to  recording/capture.py
# Run from the project root with:
#
#     python recording/capture.py --model models/csgo_yolo.pt
#
# Or use the launcher:
#
#     python run.py --model models/csgo_yolo.pt

import subprocess, sys
subprocess.run([sys.executable, "recording/capture.py"] + sys.argv[1:])
