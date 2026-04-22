import os
import random
import shutil
from pathlib import Path

# ===== CONFIG =====
random.seed(42)

SOURCE_DIR = Path(r"C:/Users/fastf/Downloads/rclone-v1.73.2-windows-amd64/rclone-v1.73.2-windows-amd64/genimage/SD1_4/sd1_4/data/train")   # paths
TEST_DIR = Path(r"C:/Users/fastf/Downloads/rclone-v1.73.2-windows-amd64/rclone-v1.73.2-windows-amd64/genimage/SD1_4/sd1_4/data/test")

SPLIT_RATIO = 0.20  # 20% rule of thumb 20%

CLASSES = ["real", "fake"]

# ===== CREATE TEST DIRS =====
for cls in CLASSES:
    (TEST_DIR / cls).mkdir(parents=True, exist_ok=True)

# ===== SPLIT FUNCTION =====
def split_class(cls):
    src_class_path = SOURCE_DIR / cls
    dst_class_path = TEST_DIR / cls

    files = list(src_class_path.glob("*"))
    total = len(files)
    n_test = int(total * SPLIT_RATIO)

    print(f"\nClass: {cls}")
    print(f"Total: {total}")
    print(f"Moving to test: {n_test}")

    # Shuffle for randomness
    random.shuffle(files)

    test_files = files[:n_test]

    for f in test_files:
        shutil.move(str(f), str(dst_class_path / f.name))

# ===== RUN =====
for cls in CLASSES:
    split_class(cls)

print("\nDone. Test set created successfully.")