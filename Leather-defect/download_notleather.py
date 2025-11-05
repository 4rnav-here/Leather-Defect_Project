from bing_image_downloader import downloader
import os
import shutil

# Folder to store images
OUTPUT_DIR = "Assets/Leather Defect Classification/NotLeather"
TEMP_DIR = OUTPUT_DIR + "_tmp"

# Categories of "non-leather" objects we want
queries = [
    "person face portrait",
    "wooden table texture",
    "metal surface close up",
    "fabric cloth texture",
    "paper texture",
    "plastic material close up",
    "concrete wall texture",
    "stone floor texture",
    "car interior dashboard",
    "landscape scenery",
    "glass bottle close up",
    "food plate macro",
    "animal fur texture"
]

# Step 1: Download into temporary folder
for q in queries:
    print(f"🔹 Downloading: {q}")
    downloader.download(
        query=q,
        limit=100,                     # you can increase to 150–200 per query
        output_dir=TEMP_DIR,
        adult_filter_off=True,
        force_replace=False,
        timeout=60
    )

# Step 2: Flatten into single "NotLeather" folder
os.makedirs(OUTPUT_DIR, exist_ok=True)
count = 0
for root, dirs, files in os.walk(TEMP_DIR):
    for file in files:
        src_path = os.path.join(root, file)
        dst_path = os.path.join(OUTPUT_DIR, f"notleather_{count}.jpg")
        shutil.move(src_path, dst_path)
        count += 1

# Step 3: Clean up
shutil.rmtree(TEMP_DIR, ignore_errors=True)

print(f"\n✅ Download complete! {count} images saved to {OUTPUT_DIR}")
