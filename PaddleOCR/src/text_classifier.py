# text_classifier.py
import yaml
import paddle
import cv2
import numpy as np
from ppcls.arch import build_model
# --- THIS IS THE CORRECTED IMPORT FOR YOUR VERSION ---
from ppcls.data import create_operators 

class TextClassifier:
    def __init__(self, config_path, model_weights_path):
        print("Initializing Text Classifier...")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        self.model = build_model(config)
        
        model_dict = paddle.load(model_weights_path)
        self.model.set_state_dict(model_dict)
        self.model.eval()

        # --- THIS USES THE CORRECT FUNCTION ---
        self.preprocess_ops = create_operators(config['Infer']['transforms'])
        
        self.class_labels = self._load_class_labels(config['Infer']['PostProcess']['class_id_map_file'])
        print("Text Classifier initialized successfully.")

    def _load_class_labels(self, file_path):
        labels_map = {}
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        class_id = int(parts[0])
                        class_name = " ".join(parts[1:])
                        labels_map[class_id] = class_name
        except Exception as e:
            print(f"Warning: Could not load class labels from {file_path}. Error: {e}")
        return labels_map

    def _preprocess(self, image_np):
        # The create_operators pipeline expects the raw image
        data = image_np
        for op in self.preprocess_ops:
            data = op(data)
        return data

    def predict(self, image_np):
        """
        Predicts the class for a given image.
        Args:
            image_np (np.ndarray): Image in BGR format from cv2.
        Returns:
            str: The predicted class label.
        """
        # Ensure image is in RGB format for consistency with training
        image_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

        preprocessed_image = self._preprocess(image_rgb)
        
        input_tensor = paddle.to_tensor(preprocessed_image).unsqueeze(0)

        with paddle.no_grad():
            output = self.model(input_tensor)
        
        predicted_class_id = paddle.argmax(output, axis=1).numpy()[0]
        
        return self.class_labels.get(predicted_class_id, "unknown")