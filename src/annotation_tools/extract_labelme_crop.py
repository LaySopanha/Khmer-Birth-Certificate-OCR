import json
import cv2
from pathlib import Path

# ✅ Define project root (2 levels up from this script)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ✅ Input paths
json_file = PROJECT_ROOT / "data" / "raw" / "general_khmer_docs" / "cropped" / "image-003.json"
img_file = PROJECT_ROOT / "data" / "raw" / "general_khmer_docs" / "khmer_image" / "image-003.png"
output_dir = PROJECT_ROOT / "data" / "raw" / "general_khmer_docs" / "line_crops"
output_dir.mkdir(parents=True, exist_ok=True)

# ✅ Load the image
img = cv2.imread(str(img_file))
if img is None:
    raise FileNotFoundError(f"Could not read image: {img_file}")

# ✅ Load the annotations
with open(json_file, encoding="utf-8") as f:
    data = json.load(f)

for i, shape in enumerate(data["shapes"]):
    label = shape.get("label", f"line_{i+1:03}")
    points = shape["points"]
    x_coords = [int(p[0]) for p in points]
    y_coords = [int(p[1]) for p in points]
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)

    cropped = img[y_min:y_max, x_min:x_max]
    crop_path = output_dir / f"line_{i+1:03}.png"
    cv2.imwrite(str(crop_path), cropped)

    print(f"✅ Saved: {crop_path.relative_to(PROJECT_ROOT)}")
