import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import multiprocessing


# Define the dataset class
class CarDataset(Dataset):
    def __init__(self, file_path, root_dir, transform=None):
        """
        Args:
            file_path (str): Path to the text file containing image paths.
            root_dir (str): Root directory containing the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_paths = []
        self.labels = []
        self.root_dir = root_dir
        self.transform = transform

        # Load image paths and labels
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Extract label (model_id is the second element in the path)
                    parts = line.split('/')
                    if len(parts) < 2:
                        raise ValueError(f"Invalid path format: {line}")
                    label = int(parts[1])  # Correctly extract model_id
                    self.labels.append(label)
                    self.image_paths.append(os.path.join(root_dir, line))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Error loading image at {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        return image, label


# Directories and file paths
train_file_path = "/data/NNDL/data/train_test_split/part/train_part_1.txt"
test_file_path = "/data/NNDL/data/train_test_split/part/test_part_1.txt"
root_dir = "/data/NNDL/data/part/"

# Data transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create the dataset
train_val_dataset = CarDataset(train_file_path, root_dir, transform=train_transform)
test_dataset = CarDataset(test_file_path, root_dir, transform=val_test_transform)

# Aggregate all labels from the datasets
all_labels = set(train_val_dataset.labels + test_dataset.labels)

# Create a mapping for labels to 0-based indices
label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}

# Update the labels in all datasets
train_val_dataset.labels = [label_to_idx[label] for label in train_val_dataset.labels]
test_dataset.labels = [label_to_idx[label] for label in test_dataset.labels]

# Number of classes
num_classes = len(label_to_idx)

# Split train/validation set (80% train, 20% validation)
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

# Data loaders
batch_size = 32
num_workers = min(multiprocessing.cpu_count(), 16)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Load ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

# Training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
early_stopping_patience = 10
best_val_accuracy = 0.0
early_stopping_counter = 0
num_epochs = 100

train_accuracies = []
val_accuracies = []

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Training phase
    model.train()
    running_corrects = 0
    total_samples = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_corrects += (preds == labels).sum().item()
        total_samples += labels.size(0)

    train_accuracy = running_corrects / total_samples
    train_accuracies.append(train_accuracy)
    print(f"Training Accuracy: {train_accuracy:.4f}")

    # Validation phase
    model.eval()
    running_corrects = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            running_corrects += (preds == labels).sum().item()
            total_samples += labels.size(0)

    val_accuracy = running_corrects / total_samples
    val_accuracies.append(val_accuracy)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Learning rate adjustment
    scheduler.step(val_accuracy)

    # Early stopping
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        early_stopping_counter = 0
        torch.save(model.state_dict(), 'part_1_predictor_resnet50_model.pth')
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

# Save training history
with open('part_1_predictor_resnet50_training.json', 'w') as f:
    json.dump({'train_accuracies': train_accuracies, 'val_accuracies': val_accuracies}, f)

print("Training complete.")
