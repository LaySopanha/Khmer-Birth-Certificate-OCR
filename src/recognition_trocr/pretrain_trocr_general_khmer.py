import json
import torch
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import wandb
from jiwer import cer 

#Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#W&B setup
wandb.init(project="khmer_ocr", name="trocr_stage1_pretrain", config={
    "epochs": 5,
    "batch_size": 2,
    "learning_rate": 5e-5
})
config = wandb.config

#Load model and processor
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

#Load train data
data_file = PROJECT_ROOT / "data" / "annotated" / "trocr_transcriptions" / "general_khmer_lines" / "train" / "train_data.json"
with open(data_file, "r", encoding="utf-8") as f:
    dataset_json = json.load(f)

#Dataset class
class KhmerOCRDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = PROJECT_ROOT / item["image"]
        text = item["text"]

        image = Image.open(image_path).convert("RGB")
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        input_ids = processor.tokenizer(text, return_tensors="pt").input_ids.squeeze(0)

        return {"pixel_values": pixel_values, "labels": input_ids}

#Collate function with padding
def custom_collate_fn(batch):
    pixel_values = torch.stack([item["pixel_values"] for item in batch])
    labels = [item["labels"] for item in batch]
    labels_padded = processor.tokenizer.pad(
        {"input_ids": labels}, padding=True, return_tensors="pt"
    )["input_ids"]
    return {"pixel_values": pixel_values, "labels": labels_padded}

# ✅ Evaluation function
@torch.no_grad()
def evaluate(model, processor, dataset, num_samples=3):
    model.eval()
    predictions = []
    cer_scores = []

    for i in range(min(num_samples, len(dataset))):
        item = dataset[i]
        image = item["pixel_values"].unsqueeze(0).to(device)
        label_text = processor.tokenizer.decode(item["labels"], skip_special_tokens=True)

        output_ids = model.generate(image)
        pred_text = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        cer_score = cer(label_text, pred_text)
        cer_scores.append(cer_score)

        predictions.append({
            "gt": label_text,
            "pred": pred_text
        })

    avg_cer = sum(cer_scores) / len(cer_scores)
    return avg_cer, predictions

# ✅ DataLoader
dataset = KhmerOCRDataset(dataset_json)
dataloader = DataLoader(dataset, batch_size=config.batch_size, collate_fn=custom_collate_fn)

# ✅ Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

# ✅ Training loop
for epoch in range(config.epochs):
    model.train()
    total_loss = 0

    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    # ✅ Evaluate and log
    avg_cer, predictions = evaluate(model, processor, dataset)
    print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}, CER = {avg_cer:.4f}")

    wandb.log({
        "loss": avg_loss,
        "cer": avg_cer,
        "epoch": epoch + 1,
        "examples": wandb.Table(columns=["GT", "Prediction"],
                                data=[[p["gt"], p["pred"]] for p in predictions])
    })

# ✅ Save model
save_path = PROJECT_ROOT / "models" / "trocr_khmer_recognizer" / "pre_trained"
save_path.mkdir(parents=True, exist_ok=True)
model.save_pretrained(save_path)
processor.save_pretrained(save_path)

print(f"✅ Model saved to: {save_path}")
