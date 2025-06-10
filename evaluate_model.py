import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import timm
from torch.amp import autocast
import json

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

def load_model(checkpoint_path, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VerticalImageClassifier(num_classes=num_classes).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Handle both possible checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    return model, device

def predict_image(model, image_path, transform, device):
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
    
    return predicted_class, confidence

def main():
    checkpoint_path = 'saved\\metal_plate_classifier_best.pth'
    test_dir = 'data\\testing_data'
    results_file = 'test_results.json'
    
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Model checkpoint not found at {checkpoint_path}")
    
    with open('data\\class_mapping.json', 'r') as f:
        class_mapping = json.load(f)
    
    num_classes = len(class_mapping)
    
    model, device = load_model(checkpoint_path, num_classes)
    print(f"Model loaded successfully. Using device: {device}")
    
    transform = transforms.Compose([
        transforms.Resize((400, 200), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    results = []
    total_images = 0
    correct_predictions = 0
    
    if not os.path.exists(test_dir):
        raise RuntimeError(f"Test directory '{test_dir}' does not exist")
    
    print(f"\nProcessing images from {test_dir}...")
    
    for image_name in os.listdir(test_dir):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        image_path = os.path.join(test_dir, image_name)
        predicted_class, confidence = predict_image(model, image_path, transform, device)
        
        # Maping the prediction to the json respomnmse/...............

        predicted_class_name = list(class_mapping.keys())[predicted_class]
        
        result = {
            'image_path': image_path,
            'predicted_class': predicted_class_name,
            'confidence': confidence
        }
        results.append(result)
        total_images += 1
        
        print(f"\nImage: {image_name}")
        print(f"Predicted class: {predicted_class_name}")
        print(f"Confidence: {confidence:.2%}")
        print("-" * 50)
    
    with open(results_file, 'w') as f:
        json.dump({
            'total_images': total_images,
            'detailed_results': results
        }, f, indent=4)
    
    print(f"\nProcessed {total_images} images")
    print(f"Detailed results saved to {results_file}")

if __name__ == "__main__":
    main()
