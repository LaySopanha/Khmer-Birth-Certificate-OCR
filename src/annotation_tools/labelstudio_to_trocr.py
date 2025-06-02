import json
from pathlib import Path

# ✅ Set your batch number here
BATCH_ID = "001"

# ✅ Define project root (2 levels up from this script)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 📂 Input: Label Studio export file
input_file = PROJECT_ROOT / "data" / "annotated" / "trocr_transcriptions" / "general_khmer_lines" / "label_studio_annotated" / f"a-image-{BATCH_ID}.json"

# 📂 Output: Save into batches folder
output_dir = PROJECT_ROOT / "data" / "annotated" / "trocr_transcriptions" / "general_khmer_lines" / "train" / "batches"
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / f"train_data-{BATCH_ID}.json"

# ✅ Read Label Studio export
with open(input_file, "r", encoding="utf-8") as f:
    labelstudio_data = json.load(f)

trocr_data = []

for item in labelstudio_data:
    try:
        # 🧠 Extract filename (e.g. '8295530e-line_001.png')
        full_path = item["data"]["image"].replace("/data/local-files/?d=", "")
        filename = Path(full_path).name

        # 🔧 Strip Label Studio hash prefix (e.g. '8295530e-')
        if "-" in filename:
            clean_filename = "-".join(filename.split("-")[1:])
        else:
            clean_filename = filename

        # ✅ Correct line_crops path
        image_path = Path("data") / "raw" / "general_khmer_docs" / "line_crops" / clean_filename

        # 📝 Transcription
        transcription = item["annotations"][0]["result"][0]["value"]["text"][0]

        trocr_data.append({
            "image": image_path.as_posix(),
            "text": transcription
        })

    except Exception as e:
        print(f"⚠️ Skipped one item due to error: {e}")

# ✅ Save batch file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(trocr_data, f, ensure_ascii=False, indent=2)

print(f"✅ Saved batch {BATCH_ID} with {len(trocr_data)} items → {output_file.relative_to(PROJECT_ROOT)}")
