import os
import json
from torchvision.datasets import ImageFolder

def create_class_mapping():
    train_dataset = ImageFolder(root='data\\training_data')
    
    class_mapping = {class_name: idx for idx, class_name in enumerate(train_dataset.classes)}
    
    with open('data\\class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=4)
    
    print(f"Created class mapping with {len(class_mapping)} classes:")
    for class_name, idx in class_mapping.items():
        print(f"{class_name}: {idx}")

if __name__ == "__main__":
    create_class_mapping() 