# parse_kaggle_xml_and_crop.py

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

# ✅ Define paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KAGGLE_IMG_DIR = PROJECT_ROOT / "data" / "external" / "khmer_kaggle" / "images"
KAGGLE_XML_DIR = PROJECT_ROOT / "data" / "external" / "khmer_kaggle" / "labels"
CROP_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "general_khmer_docs" / "line_crops"
TROCR_JSON_OUT = PROJECT_ROOT / "data" / "annotated" / "trocr_transcriptions" / "general_khmer_lines" / "train" / "batches" / "train_data-kaggle.json"

CROP_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
trocr_data = []

# ✅ Iterate over all XML files
for xml_file in sorted(KAGGLE_XML_DIR.glob("*.xml")):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Get image name from <image> tag
    image_filename = root.find("image").text
    image_path = KAGGLE_IMG_DIR / image_filename

    if not image_path.exists():
        print(f"⚠️ Image file missing: {image_path}")
        continue

    img = Image.open(image_path).convert("RGB")
    img_w, img_h = img.size

    # Each <word> contains one line/word and its bounding box
    for i, word_tag in enumerate(root.findall("word")):
        label_text = word_tag.find("text").text.strip()
        bbox_tag = word_tag.find("bbox")
        xmin = int(float(bbox_tag.find("x1").text))
        ymin = int(float(bbox_tag.find("y1").text))
        xmax = int(float(bbox_tag.find("x2").text))
        ymax = int(float(bbox_tag.find("y2").text))

        # ✅ Clamp to image bounds
        xmin = max(0, min(xmin, img_w - 1))
        xmax = max(0, min(xmax, img_w))
        ymin = max(0, min(ymin, img_h - 1))
        ymax = max(0, min(ymax, img_h))

        if xmax <= xmin or ymax <= ymin:
            print(f"⚠️ Invalid crop in {xml_file.name} — Skipping line {i+1}")
            continue

        # Crop and save line image
        cropped = img.crop((xmin, ymin, xmax, ymax))
        crop_filename = f"{image_filename.replace('.png','')}_line_{i+1:03}.png"
        crop_path = CROP_OUTPUT_DIR / crop_filename
        cropped.save(crop_path)

        # Save TrOCR-format entry
        trocr_data.append({
            "image": crop_path.relative_to(PROJECT_ROOT).as_posix(),
            "text": label_text
        })

        print(f"✅ Cropped: {crop_filename} — '{label_text}'")

# ✅ Save JSON for this batch
with open(TROCR_JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(trocr_data, f, ensure_ascii=False, indent=2)

print(f"\n✅ Done! Saved {len(trocr_data)} entries to {TROCR_JSON_OUT.relative_to(PROJECT_ROOT)}")
