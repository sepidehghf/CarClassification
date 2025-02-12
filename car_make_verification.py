import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm  # For progress bar
from torch.utils.data import DataLoader, random_split


# Path to the image dataset directory
root_dir = "/data/NNDL/data/image"

# Create a dictionary to group images by car make
car_make_dict = {}

for dirpath, _, filenames in os.walk(root_dir):
    for file in filenames:
        if file.endswith(".jpg"):
            # Extract car make from the path (3rd-level directory)
            parts = dirpath.split(os.sep)
            car_make = parts[-3]  # The 3rd-level directory is the car make
            
            if car_make not in car_make_dict:
                car_make_dict[car_make] = []
            car_make_dict[car_make].append(os.path.join(dirpath, file))

# Function to create positive and negative pairs
def create_pairs(car_make_dict, num_pairs=10000):
    pairs = []
    makes = list(car_make_dict.keys())
    
    for _ in range(num_pairs):
        # Positive pair (same car make)
        make = random.choice(makes)
        if len(car_make_dict[make]) >= 2:
            img1, img2 = random.sample(car_make_dict[make], 2)
            pairs.append((img1, img2, 1))  # label 1 for similar
        
        # Negative pair (different car makes)
        make1, make2 = random.sample(makes, 2)
        img1 = random.choice(car_make_dict[make1])
        img2 = random.choice(car_make_dict[make2])
        pairs.append((img1, img2, 0))  # label 0 for dissimilar

    return pairs

# Generate 10,000 pairs (adjust num_pairs as needed)
pairs = create_pairs(car_make_dict, num_pairs=10000)

# Convert to a DataFrame for easier processing
pairs_df = pd.DataFrame(pairs, columns=["img1", "img2", "label"])

# Save the pairs to a CSV file
pairs_df.to_csv("car_pairs.csv", index=False)
print("Dataset pairs created and saved as car_pairs.csv!")


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

# Custom dataset class for loading and preprocessing image pairs
class CarPairDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.pairs_df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.pairs_df)

    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        img1_path = row["img1"]
        img2_path = row["img2"]
        label = torch.tensor(row["label"], dtype=torch.float32)

        img1 = self.load_image(img1_path)
        img2 = self.load_image(img2_path)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return (img1, img2), label

    def load_image(self, image_path):
        """Load an image from a given path and convert it to PIL format."""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: Could not read {image_path}")
            img = np.zeros((224, 224, 3), dtype=np.uint8)  # Return a black image if loading fails
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        return Image.fromarray(img)

# Load the dataset
csv_file = "car_pairs.csv"  # Update this to your actual CSV file path
car_dataset = CarPairDataset(csv_file=csv_file, transform=transform)



# Total dataset size
total_size = len(car_dataset)

# Split sizes: 80% for training, 20% for validation
train_size = int(0.8 * total_size)
val_size = total_size - train_size

# Randomly split the dataset
train_dataset, val_dataset = random_split(car_dataset, [train_size, val_size])

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False)

print(f"Train set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")




# Use ResNet50 pretrained on ImageNet as the feature extractor
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Load pretrained ResNet50 and remove the last fully connected layer
        resnet50 = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])  # Remove the FC layer

        # Add a fully connected layer for feature comparison (optional)
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )

    def forward_once(self, x):
        """Pass the input through the feature extractor."""
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc(x)
        return x

    def forward(self, input1, input2):
        """Compute embeddings for both inputs and return the distance."""
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

# Define the contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        """Compute contrastive loss."""
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                          label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss

# Create the model, loss function, and optimizer
model = SiameseNetwork().cuda()  # Move model to GPU if available
criterion = ContrastiveLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)  # Reduce LR if no improvement

# Training loop with validation using the split dataset
num_epochs = 30
best_val_loss = float('inf')
early_stop_counter = 0
patience = 5  # Early stopping patience

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    
    # Training phase
    for (img1, img2), labels in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        img1, img2, labels = img1.cuda(), img2.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        output1, output2 = model(img1, img2)
        loss = criterion(output1, output2, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (val_img1, val_img2), val_labels in val_dataloader:
            val_img1, val_img2, val_labels = val_img1.cuda(), val_img2.cuda(), val_labels.cuda()
            val_output1, val_output2 = model(val_img1, val_img2)
            val_loss += criterion(val_output1, val_output2, val_labels).item()
    
    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Validation Loss: {avg_val_loss:.4f}")

    # Early stopping and model saving
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), "best_siamese_model.pth")
        print("Best model saved!")
    else:
        early_stop_counter += 1
        print(f"Early stopping counter: {early_stop_counter}/{patience}")

    # Reduce learning rate if validation loss doesn't improve
    scheduler.step(avg_val_loss)
    
    if early_stop_counter >= patience:
        print("Early stopping triggered. Training stopped.")
        break


