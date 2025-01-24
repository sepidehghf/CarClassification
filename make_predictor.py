import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json

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
                    # Extract label (first number in the path)
                    label = int(line.split('/')[0])
                    self.labels.append(label)
                    self.image_paths.append(os.path.join(root_dir, line))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
    
# Directories and file paths
train_file_path = "/data/NNDL/data/train_test_split/classification/train.txt"
test_file_path = "/data/NNDL/data/train_test_split/classification/test.txt"
root_dir = "/data/NNDL/data/image/"

# Data augmentation and normalization for training and validation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to a fixed size
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(15),  # Random rotation within 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random changes in brightness, contrast, etc.
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop and resize
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
    transforms.RandomGrayscale(p=0.1),  # Randomly convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet stats
])

# Create the dataset
train_val_dataset = CarDataset(train_file_path, root_dir, transform=transform)
test_dataset = CarDataset(test_file_path, root_dir, transform=transform)

# Aggregate all labels from the datasets
all_labels = set(train_val_dataset.labels + test_dataset.labels)

# Check labels in all datasets
print(f"Min label: {min(all_labels)}, Max label: {max(all_labels)}")
print(f"Expected label range: [0, {len(all_labels) - 1}]")


# Split train/validation set (80% train, 20% validation)
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

# Create a mapping for labels to 0-based indices
label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
train_val_dataset.labels = [label_to_idx[label] for label in train_val_dataset.labels]
test_dataset.labels = [label_to_idx[label] for label in test_dataset.labels]

# Update the number of classes
num_classes = len(label_to_idx)


# Data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Load a pre-defined model structure (ResNet without pretrained weights)
model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)  # Adjust for the number of classes

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler with ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

# Training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Early stopping parameters
early_stopping_patience = 10
best_val_accuracy = 0.0
early_stopping_counter = 0

num_epochs = 100
train_accuracies = []  # Store training accuracies for visualization
val_accuracies = []  # Store validation accuracies for visualization

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Training phase
    model.train()
    running_corrects = 0
    total_samples = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_corrects += (preds == labels).sum().item()
        total_samples += labels.size(0)

    train_accuracy = running_corrects / total_samples
    train_accuracies.append(train_accuracy)  # Save training accuracy
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
    val_accuracies.append(val_accuracy)  # Save validation accuracy
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Learning rate adjustment
    scheduler.step(val_accuracy)

    # Early stopping
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        early_stopping_counter = 0
        torch.save(model.state_dict(), 'make_predictor_resnet50_model.pth')  # Save the best model
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

print("Training complete.")

# Save accuracies for visualization later
with open('make_predictor_resnet50_training.json', 'w') as f:
    json.dump({'train_accuracies': train_accuracies, 'val_accuracies': val_accuracies}, f)



# Data augmentation and normalization for training and validation
transform = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize images to a fixed size
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(15),  # Random rotation within 15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random changes in brightness, contrast, etc.
    transforms.RandomResizedCrop(299, scale=(0.8, 1.0)),  # Random crop and resize
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
    transforms.RandomGrayscale(p=0.1),  # Randomly convert to grayscale
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet stats
])

# Create the dataset
train_val_dataset = CarDataset(train_file_path, root_dir, transform=transform)
test_dataset = CarDataset(test_file_path, root_dir, transform=transform)

# Aggregate all labels from the datasets
all_labels = set(train_val_dataset.labels + test_dataset.labels)

# Check labels in all datasets
print(f"Min label: {min(all_labels)}, Max label: {max(all_labels)}")
print(f"Expected label range: [0, {len(all_labels) - 1}]")


# Split train/validation set (80% train, 20% validation)
train_size = int(0.8 * len(train_val_dataset))
val_size = len(train_val_dataset) - train_size
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

# Create a mapping for labels to 0-based indices
label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
train_val_dataset.labels = [label_to_idx[label] for label in train_val_dataset.labels]
test_dataset.labels = [label_to_idx[label] for label in test_dataset.labels]

# Update the number of classes
num_classes = len(label_to_idx)


# Data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=16)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=16)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=16)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

model = models.inception_v3(pretrained=True, aux_logits=True)  # Use Inception-v3
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)  # Adjust the final layer for your number of classes

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler with ReduceLROnPlateau
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=5, verbose=True)

# Training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Early stopping parameters
early_stopping_patience = 10
best_val_accuracy = 0.0
early_stopping_counter = 0

num_epochs = 100
train_accuracies = []  # Store training accuracies for visualization
val_accuracies = []  # Store validation accuracies for visualization

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Training phase
    model.train()
    running_corrects = 0
    total_samples = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        _, preds = torch.max(outputs.logits, 1)
        loss = criterion(outputs.logits, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_corrects += (preds == labels).sum().item()
        total_samples += labels.size(0)

    train_accuracy = running_corrects / total_samples
    train_accuracies.append(train_accuracy)  # Save training accuracy
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
    val_accuracies.append(val_accuracy)  # Save validation accuracy
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Learning rate adjustment
    scheduler.step(val_accuracy)

    # Early stopping
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        early_stopping_counter = 0
        torch.save(model.state_dict(), 'make_predictor_inception_model.pth')  # Save the best model
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break
        
print("Training complete.")

with open('make_predictor_inception_training.json', 'w') as f:
    json.dump({'train_accuracies': train_accuracies, 'val_accuracies': val_accuracies}, f)
