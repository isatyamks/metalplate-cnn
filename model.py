import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import InterpolationMode
import os
import json
from datetime import datetime

class VerticalImageClassifier(nn.Module):
    def __init__(self, num_classes=361):
        super().__init__()
        self.base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.base_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

def save_checkpoint(model, optimizer, scheduler, epoch, best_val_acc, checkpoint_dir, is_best=False):
    """Save model checkpoint with training metadata"""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
        'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best model if this is the best so far
    if is_best:
        best_model_path = os.path.join(checkpoint_dir, 'metal_plate_classifier_best.pth')
        torch.save(checkpoint, best_model_path)
        print(f"New best model saved with validation accuracy: {best_val_acc:.2f}%")

def main():
    # Create checkpoints directory
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Save training configuration
    config = {
        'batch_size': 32,
        'num_epochs': 20,
        'learning_rate': 1e-4,
        'model_architecture': 'ResNet50',
        'num_classes': 361,
        'optimizer': 'Adam',
        'scheduler': 'CosineAnnealingLR'
    }
    
    with open(os.path.join(checkpoint_dir, 'training_config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    train_transform = transforms.Compose([
        transforms.Resize((420, 210), interpolation=InterpolationMode.BILINEAR),
        transforms.RandomResizedCrop((400, 200), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((400, 200), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    train_data_path = os.path.abspath(os.path.join('data', 'training_data'))
    val_data_path = os.path.abspath(os.path.join('data', 'validation_data'))

    if not os.path.exists(train_data_path):
        raise FileNotFoundError(f"Training data path does not exist: {train_data_path}")
    if not os.path.exists(val_data_path):
        raise FileNotFoundError(f"Validation data path does not exist: {val_data_path}")

    train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_data_path, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = VerticalImageClassifier(num_classes=361).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)

    scaler = torch.amp.GradScaler(device="cuda")

    num_epochs = 20
    best_val_acc = 0
    training_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

        train_loss = running_loss / len(train_dataset)
        train_acc = 100 * correct / len(train_dataset)

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                with torch.amp.autocast(device_type='cuda'):
                    val_outputs = model(val_inputs)
                val_preds = val_outputs.argmax(dim=1)
                val_correct += (val_preds == val_labels).sum().item()
                val_total += val_labels.size(0)

        val_acc = 100 * val_correct / val_total

        # Save training history
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        training_history.append(epoch_stats)

        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Acc: {val_acc:.2f}%")

        scheduler.step()

        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
        
        save_checkpoint(model, optimizer, scheduler, epoch + 1, best_val_acc, checkpoint_dir, is_best)

    # Save final training history
    with open(os.path.join(checkpoint_dir, 'training_history.json'), 'w') as f:
        json.dump(training_history, f, indent=4)

    print(f"Training completed. Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Checkpoints saved in: {checkpoint_dir}")

if __name__ == "__main__":
    main()
