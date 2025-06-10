import os
import json
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.nn as nn
import timm
from torch.amp import autocast

# Load class mapping
with open('data\\class_mapping.json', 'r') as f:
    class_mapping = json.load(f)
class_names = list(class_mapping.keys())
num_classes = len(class_names)

# Define your model class as in training
class VerticalImageClassifier(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.3):
        super().__init__()
        self.base_model = timm.create_model('efficientnet_b3', pretrained=False)
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.base_model.classifier.in_features, num_classes)
        )
    def forward(self, x):
        return self.base_model(x)

# Device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Preprocessing transforms (must match training transform!)
transform = transforms.Compose([
    transforms.Resize((400, 200)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image_path, model, class_names, device=device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(image)
            probs = F.softmax(outputs, dim=1)
            _, pred_idx = torch.max(probs, 1)
            predicted_label = class_names[pred_idx.item()]
    return predicted_label

def main():
    # Initialize and load model weights
    model = VerticalImageClassifier(num_classes=num_classes)
    checkpoint = torch.load('saved/metal_plate_classifier_best.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    test_folder = 'testing_data'
    correct = 0
    total = 0

    for filename in os.listdir(test_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(test_folder, filename)
            true_label = filename.split('_')[0]  # e.g., A0253 from A0253_0012.jpg
            predicted_label = predict_image(image_path, model, class_names, device)
            is_correct = (predicted_label == true_label)
            print(f"{filename}: {predicted_label} - {is_correct}")
            correct += int(is_correct)
            total += 1

    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"Accuracy on test set: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
