import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
batch_dir = PROJECT_ROOT / "data" / "annotated" / "trocr_transcriptions" / "general_khmer_lines" / "train" / "batches"
output_file = PROJECT_ROOT / "data" / "annotated" / "trocr_transcriptions" / "general_khmer_lines" / "train" / "train_data.json"

merged_data = []

for batch_file in sorted(batch_dir.glob("train_data-*.json")):
    print(f"📦 Adding {batch_file.name}")
    with open(batch_file, encoding="utf-8") as f:
        merged_data.extend(json.load(f))

# Deduplicate (optional)
seen = set()
unique_data = []
for item in merged_data:
    key = (item["image"], item["text"])
    if key not in seen:
        seen.add(key)
        unique_data.append(item)

# Save merged file
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(unique_data, f, ensure_ascii=False, indent=2)

print(f"✅ Merged {len(unique_data)} items into {output_file}")
