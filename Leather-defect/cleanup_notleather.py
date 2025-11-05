import os
from PIL import Image

folder = "Assets/Leather Defect Classification/NotLeather"
valid_exts = {".jpg", ".jpeg", ".png"}

for fname in os.listdir(folder):
    path = os.path.join(folder, fname)
    ext = os.path.splitext(fname)[1].lower()
    if ext not in valid_exts:
        os.remove(path)
        continue
    try:
        img = Image.open(path)
        img.verify()
    except Exception:
        print(f"❌ Removing corrupted file: {fname}")
        os.remove(path)

print("✅ Cleanup complete.")
