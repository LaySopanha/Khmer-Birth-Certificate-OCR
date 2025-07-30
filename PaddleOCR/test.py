from paddleocr import LayoutDetection
image_path = './image/birth_certificate.jpg'
model = LayoutDetection(model_name="PP-DocLayout_plus-L")
output = model.predict(image_path, batch_size=1, layout_nms=True)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
    
import cv2
from paddleocr import PaddleOCR, LayoutDetection
import json

# --- Your original image path ---
img = cv2.imread(image_path)
if img is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

# --- STEP 1: Layout Detection (Your current code) ---
print("Step 1: Running Layout Detection...")
# Using a faster, lighter model for this example, but 'PP-DocLayout_plus-L' is also great.
layout_model = LayoutDetection(model_name='PP-DocLayout_plus-L')
layouts = layout_model.predict(img)

# --- STEP 2: Initialize OCR for Khmer Language ---
# Make sure you have the Khmer model downloaded. If not, it will download automatically.
# lang='km' is the language code for Khmer.
print("Step 2: Initializing OCR model for Khmer...")
ocr_engine = PaddleOCR(use_angle_cls=True, lang='km')

# --- STEP 3: Iterate through layouts and perform OCR on text regions ---
print("Step 3: Performing OCR on detected text regions...")
all_results = []
for i, layout in enumerate(layouts):
    # We only care about regions classified as 'text' or 'title' for OCR
    if layout.label in ['text', 'title', 'list']:
        # Get the coordinates of the layout box
        x1, y1, x2, y2 = layout.bbox
        
        # Crop the image to this specific region
        # Note: Slicing is [y1:y2, x1:x2]
        region_image = img[int(y1):int(y2), int(x1):int(x2)]
        
        # Run OCR on the cropped region
        ocr_result = ocr_engine.ocr(region_image, cls=True)
        
        # The result from ocr() is nested, so flatten it for easier processing
        if ocr_result and ocr_result[0]:
            print(f"  - Found text in region {i} ({layout.label}):")
            for line in ocr_result[0]:
                text = line[1][0]
                confidence = line[1][1]
                print(f"    '{text}' (Confidence: {confidence:.2f})")
                all_results.append({'layout': layout.label, 'text': text, 'confidence': confidence})

# --- Save the final combined results to a file ---
with open('./output/final_extracted_text.json', 'w', encoding='utf-8') as f:
    json.dump(all_results, f, ensure_ascii=False, indent=4)

print("\nProcessing complete. Final results saved to ./output/final_extracted_text.json")