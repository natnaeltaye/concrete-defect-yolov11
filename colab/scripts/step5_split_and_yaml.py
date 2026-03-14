import os, shutil, random, math
from pathlib import Path

# ==== CONFIG ====
BASE = Path("/content/drive/MyDrive/YOLOv11_Concrete-defect-dataset-08202025")
DATASET = BASE / "dataset"         # where your mixed images+txt live
TRAIN   = BASE / "train"           # will be created/overwritten
VAL     = BASE / "val"             # will be created/overwritten
TEST    = BASE / "test"            # new test folder
VAL_RATIO = 0.17                   # 17% for validation
TEST_RATIO = 0.10                  # 10% for test
RANDOM_SEED = 42                   # reproducible split
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}  # case-insensitive
OVERWRITE = True                   # delete existing train/val/test before writing

# ==== CHECK RATIOS ====
if VAL_RATIO + TEST_RATIO >= 1.0:
    raise ValueError("VAL_RATIO + TEST_RATIO must be less than 1.0")

# ==== PREP ====
if not DATASET.exists():
    raise FileNotFoundError(f"Dataset folder not found: {DATASET}")

def clean_dir(d: Path):
    if d.exists():
        if OVERWRITE:
            shutil.rmtree(d)
        else:
            raise RuntimeError(f"{d} exists. Set OVERWRITE=True or rename it first.")
    d.mkdir(parents=True, exist_ok=True)

clean_dir(TRAIN)
clean_dir(VAL)
clean_dir(TEST)

# ==== COLLECT ELIGIBLE SAMPLES (image + matching .txt) ====
samples = []
for p in DATASET.rglob("*"):
    if p.is_file() and p.suffix.lower() in IMG_EXTS:
        txt = p.with_suffix(".txt")
        if txt.exists():
            samples.append((p, txt))

if not samples:
    raise RuntimeError(f"No (image, txt) pairs found under {DATASET}")

# ==== SHUFFLE & SPLIT ====
random.seed(RANDOM_SEED)
random.shuffle(samples)

n_total = len(samples)
n_val = math.floor(n_total * VAL_RATIO)
n_test = math.floor(n_total * TEST_RATIO)

val_samples = samples[:n_val]
test_samples = samples[n_val:n_val + n_test]
train_samples = samples[n_val + n_test:]

# ==== COPY ====
def copy_pair(img: Path, txt: Path, dst_dir: Path):
    # Flattens into train/, val/, and test/
    shutil.copy2(img, dst_dir / img.name)
    shutil.copy2(txt, dst_dir / txt.name)

for img, txt in train_samples:
    copy_pair(img, txt, TRAIN)

for img, txt in val_samples:
    copy_pair(img, txt, VAL)

for img, txt in test_samples:
    copy_pair(img, txt, TEST)

# ==== REPORT ====
def count_images(d: Path):
    return sum(1 for f in d.glob("*") if f.suffix.lower() in IMG_EXTS)

def count_txts(d: Path):
    return sum(1 for f in d.glob("*.txt"))

print(f"Total pairs found: {n_total}")
print(f"Train pairs: {len(train_samples)} | images={count_images(TRAIN)} | labels={count_txts(TRAIN)}")
print(f"Val   pairs: {len(val_samples)}   | images={count_images(VAL)}   | labels={count_txts(VAL)}")
print(f"Test  pairs: {len(test_samples)}  | images={count_images(TEST)}  | labels={count_txts(TEST)}")

# ==== WRITE A YOLO DATA YAML ====
yaml_text = f"""# Auto-generated for Colab
path: {BASE}
train: {TRAIN}
val: {VAL}
test: {TEST}

names:
  0: Crack
  1: ACrack
  2: Efflorescence
  3: WConccor
  4: Spalling
  5: Wetspot
  6: Rust
  7: ExposedRebars
"""

yaml_path = BASE / "data_colab.yaml"
yaml_path.write_text(yaml_text)
print(f"Wrote {yaml_path}")
