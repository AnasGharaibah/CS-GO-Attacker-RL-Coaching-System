import sys
import platform

print(f"Python  {sys.version}")
print(f"OS      {platform.system()} {platform.release()}")
print()

required = [
    ("torch",              "torch"),
    ("numpy",              "numpy"),
    ("gymnasium",          "gymnasium"),
    ("stable_baselines3",  "stable-baselines3"),
    ("sb3_contrib",        "sb3-contrib"),
    ("ultralytics",        "ultralytics"),
    ("mss",                "mss"),
    ("cv2",                "opencv-python-headless"),
    ("requests",           "requests"),
    ("fastapi",            "fastapi"),
    ("uvicorn",            "uvicorn"),
    ("matplotlib",         "matplotlib"),
    ("tensorboard",        "tensorboard"),
    ("h5py",               "h5py"),
]

ok  = []
bad = []

for module, pkg in required:
    try:
        m = __import__(module)
        ver = getattr(m, "__version__", "?")
        ok.append((pkg, ver))
    except ImportError:
        bad.append(pkg)

print("Installed:")
for pkg, ver in ok:
    print(f"  [OK]  {pkg:<35} {ver}")

if bad:
    print("\nMissing:")
    for pkg in bad:
        print(f"  [!!]  {pkg}")
    print(f"\nRun:  pip install {' '.join(bad)}")
else:
    print("\nAll packages installed.")

print()

from pathlib import Path
model = Path("cs2_yolo_model/weights/best.pt")
if model.exists():
    print(f"[OK]  YOLO model found: {model}")
else:
    print(f"[!!]  YOLO model not found at {model}")
    print("      Download cs2_yolo_model/ folder and place it here.")
