import os
import sys
import json
import numpy as np
import torch
import multiprocessing
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.amp import GradScaler, autocast

# Ensure script runs in the background without interruption
sys.stdout = open('training_output.log', 'w')
sys.stderr = sys.stdout

# -----------------------
# Dataset Definition
# -----------------------
class CarDataset(Dataset):
    def __init__(self, file_path, root_dir, transform=None):
        self.image_paths = []
        self.labels = []
        self.root_dir = root_dir
        self.transform = transform

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('/')
                    if len(parts) < 2:
                        raise ValueError(f"Invalid path format: {line}")
                    label = int(parts[1])
                    self.labels.append(label)
                    self.image_paths.append(os.path.join(root_dir, line))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Error loading image at {img_path}: {e}")
        if self.transform:
            image = self.transform(image)
        return image, label

# -----------------------
# Data Transformations
# -----------------------
# Use slightly milder augmentation for early training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    # You can add RandomErasing later if needed
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# -----------------------
# Setup Directories & Device
# -----------------------
root_dir = "/data/NNDL/data/part/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = min(multiprocessing.cpu_count(), 16)

# Training parameters
batch_size = 32
num_epochs = 100
early_stopping_patience = 15

# -----------------------
# Prepare Global label_to_idx
# -----------------------
all_labels = set()
for i in range(1, 9):
    train_file_path = f"/data/NNDL/data/train_test_split/part/train_part_{i}.txt"
    test_file_path = f"/data/NNDL/data/train_test_split/part/test_part_{i}.txt"
    train_val_dataset = CarDataset(train_file_path, root_dir, transform=train_transform)
    test_dataset = CarDataset(test_file_path, root_dir, transform=val_test_transform)
    all_labels.update(train_val_dataset.labels + test_dataset.labels)

label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
with open("label_to_idx.json", "w") as f:
    json.dump(label_to_idx, f)

# -----------------------
# (Optional) Mixup Augmentation Functions
# -----------------------
def mixup_data(x, y, alpha=0.2):  # Use a lower alpha initially
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# -----------------------
# Label Smoothing Loss (optionally reduce smoothing factor)
# -----------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.05):  # Lower smoothing factor
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        n_classes = pred.size(1)
        log_preds = torch.log_softmax(pred, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_preds, dim=1))

# -----------------------
# Training Loop for Each Dataset Part
# -----------------------
for i in range(1, 9):
    print(f"Starting training for dataset part {i}...")
    train_file_path = f"/data/NNDL/data/train_test_split/part/train_part_{i}.txt"
    test_file_path = f"/data/NNDL/data/train_test_split/part/test_part_{i}.txt"
    model_save_path = f"part_{i}_predictor_resnet50_model.pth"
    training_history_path = f"part_{i}_predictor_resnet50_training.json"

    # Create datasets
    train_val_dataset = CarDataset(train_file_path, root_dir, transform=train_transform)
    test_dataset = CarDataset(test_file_path, root_dir, transform=val_test_transform)

    # Aggregate all labels from the datasets
    all_labels = set(train_val_dataset.labels + test_dataset.labels)

    # Create a mapping for labels to 0-based indices
    label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
    
    # Map labels using label_to_idx
    train_val_dataset.labels = [label_to_idx[label] for label in train_val_dataset.labels]
    test_dataset.labels = [label_to_idx[label] for label in test_dataset.labels]
    num_classes = len(label_to_idx)

    # Split train and validation sets
    train_size = int(0.8 * len(train_val_dataset))
    val_size = len(train_val_dataset) - train_size
    train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # -----------------------
    # Load Pretrained ResNet50 and Modify the Final Layer
    # -----------------------
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Freeze the entire network initially
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    # Use a smaller dropout rate (0.2 instead of 0.5)
    model.fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, num_classes)
    )
    model = model.to(device)

    # -----------------------
    # Define Loss Function, Optimizer, and Schedulers
    # -----------------------
    criterion = LabelSmoothingCrossEntropy(smoothing=0.05)
    # Use AdamW optimizer with differential learning rates:
    optimizer = optim.AdamW([
        {'params': model.fc.parameters(), 'lr': 1e-3},
        {'params': [p for name, p in model.named_parameters() if "fc" not in name], 'lr': 1e-4}
    ], weight_decay=1e-4)

    # Implement a learning rate warmup for the first 2 epochs
    def lr_lambda(epoch):
        if epoch < 2:
            return (epoch + 1) / 2  # linear warmup from 0 to 1 over 2 epochs
        else:
            return 1
    warmup_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    # After warmup, use cosine annealing scheduler over the remaining epochs:
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - 2)

    scaler = GradScaler()

    best_val_accuracy = 0.0
    early_stopping_counter = 0
    train_accuracies = []
    val_accuracies = []

    # -----------------------
    # Epoch Loop
    # -----------------------
    for epoch in range(num_epochs):
        print(f"Dataset {i} | Epoch {epoch + 1}/{num_epochs}")

        # Unfreeze layer4 and fc earlier (e.g., after 2 epochs)
        if epoch == 2:
            for name, param in model.named_parameters():
                if "layer4" in name or "fc" in name:
                    param.requires_grad = True
            print("Unfroze layer4 and fc parameters for fine-tuning.")

        model.train()
        running_corrects = 0
        total_samples = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            # Optionally, you can disable mixup in the first few epochs if needed
            # For now, we use mixup (with a lower alpha)
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2)
            with autocast(device_type=device.type):
                outputs = model(images)
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # For accuracy, use outputs with original labels (note: mixup makes accuracy estimates rough)
            _, preds = torch.max(outputs, 1)
            running_corrects += (preds == labels).sum().item()
            total_samples += labels.size(0)
        train_accuracy = running_corrects / total_samples
        train_accuracies.append(train_accuracy)
        print(f"Training Accuracy: {train_accuracy:.4f}")

        # -----------------------
        # Validation Phase
        # -----------------------
        model.eval()
        running_corrects = 0
        total_samples = 0
        total_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)
                _, preds = torch.max(outputs, 1)
                running_corrects += (preds == labels).sum().item()
                total_samples += labels.size(0)
        val_accuracy = running_corrects / total_samples
        avg_val_loss = total_loss / len(val_dataset)
        val_accuracies.append(val_accuracy)
        print(f"Validation Accuracy: {val_accuracy:.4f}, Loss: {avg_val_loss:.4f}")

        # Update LR schedulers
        if epoch < 2:
            warmup_scheduler.step()
        else:
            cosine_scheduler.step()

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with validation accuracy: {best_val_accuracy:.4f}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"No improvement for {early_stopping_counter} epoch(s).")

        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    # Save training history for this dataset part
    training_history = {
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_val_accuracy': best_val_accuracy
    }
    with open(training_history_path, "w") as f:
        json.dump(training_history, f)

    print(f"Training complete for Dataset part {i}.\n")
