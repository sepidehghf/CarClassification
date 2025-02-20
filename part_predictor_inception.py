import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
import multiprocessing
import sys

# Ensure script runs in the background without interruption
sys.stdout = open('training_output_inception.log', 'w')
sys.stderr = sys.stdout

# Define the dataset class
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

# Directories and device setup
root_dir = "/data/NNDL/data/part/"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = min(multiprocessing.cpu_count(), 16)

# Data transformations for Inception v3 (input size 299x299)
train_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_test_transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Training parameters
batch_size = 32
num_epochs = 100
early_stopping_patience = 15

# Prepare global label_to_idx
all_labels = set()
for i in range(1, 9):
    train_file_path = f"/data/NNDL/data/train_test_split/part/train_part_{i}.txt"
    test_file_path = f"/data/NNDL/data/train_test_split/part/test_part_{i}.txt"
    train_val_dataset = CarDataset(train_file_path, root_dir, transform=train_transform)
    test_dataset = CarDataset(test_file_path, root_dir, transform=val_test_transform)
    all_labels.update(train_val_dataset.labels + test_dataset.labels)

label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}

# Save label_to_idx for later use
with open("label_to_idx.json", "w") as f:
    json.dump(label_to_idx, f)

# Loop through each dataset part
for i in range(1, 9):
    print(f"Starting training for dataset part {i}...")
    train_file_path = f"/data/NNDL/data/train_test_split/part/train_part_{i}.txt"
    test_file_path = f"/data/NNDL/data/train_test_split/part/test_part_{i}.txt"
    model_save_path = f"part_{i}_predictor_inception_model.pth"
    training_history_path = f"part_{i}_predictor_inception_training.json"

    # Create datasets
    train_dataset = CarDataset(train_file_path, root_dir, transform=train_transform)
    test_dataset = CarDataset(test_file_path, root_dir, transform=val_test_transform)
    
    # Map original labels to consistent indices
    train_dataset.labels = [label_to_idx[label] for label in train_dataset.labels]
    test_dataset.labels = [label_to_idx[label] for label in test_dataset.labels]
    num_classes = len(label_to_idx)

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load pre-trained Inception v3 and modify the final layers
    model = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, aux_logits=True)
    # Modify the main classifier
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    # Modify the auxiliary classifier
    if model.AuxLogits is not None:
        num_ftrs_aux = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs_aux, num_classes)
    
    model = model.to(device)

    # Define loss function, optimizer, and learning rate scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

    # Initialize training tracking variables
    best_test_accuracy = 0.0
    early_stopping_counter = 0
    train_accuracies = []
    test_accuracies = []

    # Training loop for current dataset part
    for epoch in range(num_epochs):
        print(f"Dataset {i} | Epoch {epoch + 1}/{num_epochs}")

        # --- Training Phase ---
        model.train()
        running_corrects = 0
        total_samples = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            # In training mode, Inception v3 returns (output, aux_output)
            if isinstance(model, models.Inception3) and model.training:
                outputs, aux_outputs = model(images)
                loss1 = criterion(outputs, labels)
                loss2 = criterion(aux_outputs, labels)
                loss = loss1 + 0.4 * loss2
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # Use only the main output for accuracy calculation
            _, preds = torch.max(outputs, 1)
            running_corrects += (preds == labels).sum().item()
            total_samples += labels.size(0)
        train_accuracy = running_corrects / total_samples
        train_accuracies.append(train_accuracy)
        print(f"Training Accuracy: {train_accuracy:.4f}")

        # --- Test Phase ---
        model.eval()
        running_corrects = 0
        total_samples = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                # In eval mode, only the main output is returned
                _, preds = torch.max(outputs, 1)
                running_corrects += (preds == labels).sum().item()
                total_samples += labels.size(0)
        test_accuracy = running_corrects / total_samples
        test_accuracies.append(test_accuracy)
        print(f"Test Accuracy: {test_accuracy:.4f}")

        # Adjust learning rate based on test accuracy
        scheduler.step(test_accuracy)

        # Save best model checkpoint based on test accuracy
        if test_accuracy > best_test_accuracy:
            best_test_accuracy = test_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with test accuracy: {best_test_accuracy:.4f}")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            print(f"No improvement for {early_stopping_counter} epoch(s).")

        # Early stopping check
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    # Save training history for this dataset part
    training_history = {
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'best_test_accuracy': best_test_accuracy
    }
    with open(training_history_path, "w") as f:
        json.dump(training_history, f)

    print(f"Training complete for Dataset part {i}.\n")
