# Cell 1: Imports and utility libraries
import os
import random
import time
import pickle
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from skimage.measure import label
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models

# Cell 2: Parse file paths and labels
def parse_file(file_path, root_dir, root_dir_view):
    image_paths = []
    labels = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                label_file_path = os.path.join(root_dir_view, os.path.splitext(line)[0] + ".txt")
                if os.path.exists(label_file_path):
                    with open(label_file_path, 'r') as label_file:
                        first_line = label_file.readline().strip()
                        if first_line in {'1', '2'}:
                            # Here the label is taken as the integer from the first folder in the path.
                            label_val = int(line.split('/')[0])
                            labels.append(label_val)
                            image_paths.append(os.path.join(root_dir, line))
    return image_paths, labels

# Define directories and file paths (change these as needed)
train_file_path = "/data/NNDL/data/train_test_split/classification/train.txt"
test_file_path  = "/data/NNDL/data/train_test_split/classification/test.txt"
root_dir      = "/data/NNDL/data/image/"
root_dir_view = "/data/NNDL/data/label/"

# Parse train and test data
train_image_paths, train_labels = parse_file(train_file_path, root_dir, root_dir_view)
test_image_paths, test_labels   = parse_file(test_file_path, root_dir, root_dir_view)

# Aggregate all labels and build a mapping to 0-based indices
all_labels = sorted(set(train_labels + test_labels))
print(f"Min label: {min(all_labels)}, Max label: {max(all_labels)}")
print(f"Expected label range: [0, {len(all_labels) - 1}]")

label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
# Remap labels
train_labels = [label_to_idx[label] for label in train_labels]
test_labels  = [label_to_idx[label] for label in test_labels]

num_classes = len(label_to_idx)
print("Number of classes:", num_classes)


# Cell 3: Custom Dataset
class CarDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image and convert to RGB
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Cell 4: Data transforms and splitting dataset
# Define transforms (feel free to adjust augmentation parameters)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomAffine(degrees=15, shear=0.2, scale=(0.8, 1.2)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Split the training data into training and validation sets (80/20 split)
train_paths, val_paths, train_labels_split, val_labels_split = train_test_split(
    train_image_paths, train_labels, test_size=0.2, random_state=42
)


# Cell 5: Create DataLoaders
batch_size = 128

train_dataset = CarDataset(train_paths, train_labels_split, transform=train_transforms)
val_dataset   = CarDataset(val_paths, val_labels_split, transform=val_transforms)
test_dataset  = CarDataset(test_image_paths, test_labels, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers= 16, pin_memory=True, prefetch_factor=2, persistent_workers=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers= 16, pin_memory=True, prefetch_factor=2, persistent_workers=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers= 16, pin_memory=True, prefetch_factor=2, persistent_workers=True)

print(f"Training dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


# Cell 7: Define the model (ResNet50)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)  # modify the final layer
model = model.to(device)

# Print a summary of the model
print(model)


 #Ensure num_classes is defined (should be the same as used during training)
# For example:
# num_classes = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Recreate the model architecture. In this example we use ResNet50.
global_model = models.resnet50(pretrained=False)
num_ftrs = global_model.fc.in_features
global_model.fc = nn.Linear(num_ftrs, num_classes)

# Load the saved state dictionary
state_dict = torch.load('make_predictor_resnet50_attention_model.pth', map_location=device)
global_model.load_state_dict(state_dict)
global_model.to(device)
global_model.eval()  # Set the model to evaluation mode

print("Model loaded successfully!")

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Global ImageNet normalization parameters (used for unnormalizing images)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])

# --- Preprocessing ---
def preprocess_input(images):
    """
    Converts images (if needed) to a torch.FloatTensor.
    Assumes images are already scaled in [0, 1] (or normalized).
    If images come in as a numpy array, converts them.
    """
    if isinstance(images, np.ndarray):
        images = torch.tensor(images, dtype=torch.float32)
    return images

# --- Heatmap Extraction ---
def get_heatmap(image, model, last_conv_layer_name, pred_index=None):
    """
    Extracts a heatmap from a single image using the activations
    from the specified convolutional layer.
    
    Args:
        image: A torch tensor of shape (1, C, H, W).
        model: The PyTorch model.
        last_conv_layer_name: String name of the target layer (e.g. 'layer4' for ResNet50).
        pred_index: (Optional) index of the class to be used for a weighted heatmap.
    
    Returns:
        A torch tensor of shape (1, H, W) with values normalized to [0, 1].
    """
    activation = None
    def hook_fn(module, input, output):
        nonlocal activation
        activation = output

    # Retrieve the layer by name from the model's modules
    layer = dict(model.named_modules()).get(last_conv_layer_name, None)
    if layer is None:
        print(f"Layer {last_conv_layer_name} not found in model.")
        return None

    # Register the forward hook
    hook = layer.register_forward_hook(hook_fn)
    model.eval()
    device = next(model.parameters()).device  # get model device
    with torch.no_grad():
        _ = model(image.to(device))
    hook.remove()

    if activation is None:
        return None

    # activation has shape (1, C, H, W). Compute the heatmap by taking
    # the maximum absolute activation over the channel dimension.
    heatmap = torch.max(torch.abs(activation), dim=1)[0]  # shape: (1, H, W)
    heatmap = torch.clamp(heatmap, min=0)
    max_val = heatmap.max().item()
    heatmap = heatmap / (max_val + 1e-7)
    return heatmap

# --- Attention Mask Inference ---
def attention_guided_mask_inference_batch(generator, model, last_conv_layer_name='layer4', threshold=0.7):
    """
    Given a DataLoader iterator, computes a binary attention mask for each image
    in the batch based on the heatmap from the specified layer.
    
    Returns:
        masks_tensor: Tensor of shape (N, 1, H_mask, W_mask).
        images: The batch of images (as a torch tensor).
    """
    masks = []
    images, labels = next(generator)  # images shape: (N, C, H, W)
    images = preprocess_input(images)
    for img in images:
        # Add batch dimension: (1, C, H, W)
        img = img.unsqueeze(0)
        heatmap = get_heatmap(img, model, last_conv_layer_name)
        if heatmap is None:
            continue
        mask = (heatmap > threshold).float()  # binary mask
        masks.append(mask)
    if len(masks) == 0:
        masks_tensor = torch.tensor([])
    else:
        masks_tensor = torch.stack(masks, dim=0)  # shape: (N, 1, H, W)
    return masks_tensor, images

# --- Local Region Cropping ---
def crop_local_region_batch(images, masks, input_shape=(224, 224)):
    """
    For each image in the batch, uses the corresponding mask to detect contours
    and crop the most salient region. If no contours are found, the full image is resized.
    
    Args:
        images: Tensor of shape (N, C, H, W).
        masks: Tensor of shape (N, 1, H_mask, W_mask).
        input_shape: Target size as (height, width).
        
    Returns:
        A tensor of cropped images of shape (N, C, input_shape[0], input_shape[1]).
    """
    cropped_images = []
    # Convert images to numpy for use with OpenCV (shape: N x C x H x W)
    images_np = images.cpu().numpy()
    for i in range(images.shape[0]):
        if i >= masks.shape[0]:
            continue
        # Convert mask to numpy and squeeze to (H_mask, W_mask)
        mask_np = masks[i].cpu().numpy().astype(np.uint8)
        mask_np = np.squeeze(mask_np)
        # Resize mask to target dimensions (note: cv2.resize expects (width, height))
        mask_resized = cv2.resize(mask_np, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_NEAREST)
        # Convert corresponding image to numpy and change from (C, H, W) to (H, W, C)
        image_np = images_np[i]
        image_np = np.transpose(image_np, (1, 2, 0))
        # NOTE: image_np remains in normalized space. Cropping is done on the raw data.
        # Prepare the mask for contour detection (scale to 0-255)
        mask_for_contours = (mask_resized * 255).astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask_for_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            # If no contour found, simply resize the full image.
            cropped = cv2.resize(image_np, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cropped = image_np[y:y+h, x:x+w]
            cropped = cv2.resize(cropped, (input_shape[1], input_shape[0]), interpolation=cv2.INTER_LINEAR)
        cropped_images.append(cropped)
    cropped_images = np.array(cropped_images)
    # Convert back to tensor and permute to (N, C, H, W)
    cropped_tensor = torch.tensor(cropped_images, dtype=torch.float32)
    cropped_tensor = cropped_tensor.permute(0, 3, 1, 2)
    return cropped_tensor

# --- Saving Cropped Images ---
def save_cropped_images_to_directory(generator, model, output_dir, subset, classes, last_conv_layer_name='layer4'):
    """
    Iterates through the generator (DataLoader) and saves both the original and
    cropped images to disk, organizing them by class.
    Assumes that `classes` is a mapping from label indices to class names.
    """
    original_images_dir = os.path.join(output_dir, f'{subset}_original')
    cropped_images_dir = os.path.join(output_dir, f'{subset}_cropped')
    os.makedirs(original_images_dir, exist_ok=True)
    os.makedirs(cropped_images_dir, exist_ok=True)
    total_samples = 0
    data_iter = iter(generator)
    total_samples_in_dataset = len(generator.dataset)
    
    while total_samples < total_samples_in_dataset:
        try:
            images, labels = next(data_iter)
        except StopIteration:
            break
        images = preprocess_input(images)
        # Compute masks for the current batch
        masks = []
        for img in images:
            img_unsq = img.unsqueeze(0)
            heatmap = get_heatmap(img_unsq, model, last_conv_layer_name)
            if heatmap is None:
                continue
            mask = (heatmap > 0.7).float()
            masks.append(mask)
        if len(masks) == 0:
            continue
        masks_tensor = torch.stack(masks, dim=0)
        cropped_images = crop_local_region_batch(images, masks_tensor)
        
        # Unnormalize images for saving
        for i, (original_img, cropped_img, label) in enumerate(zip(images, cropped_images, labels)):
            class_name = classes[int(label)]
            orig_class_dir = os.path.join(original_images_dir, class_name)
            crop_class_dir = os.path.join(cropped_images_dir, class_name)
            os.makedirs(orig_class_dir, exist_ok=True)
            os.makedirs(crop_class_dir, exist_ok=True)
            
            orig_img_path = os.path.join(orig_class_dir, f"{total_samples + i}.png")
            crop_img_path = os.path.join(crop_class_dir, f"{total_samples + i}.png")
            
            # Unnormalize original image
            original_img_np = original_img.cpu().numpy().transpose(1, 2, 0)
            original_img_np = original_img_np * IMAGENET_STD + IMAGENET_MEAN
            original_img_np = np.clip(original_img_np, 0, 1)
            original_img_np = (original_img_np * 255).astype(np.uint8)
            
            # Unnormalize cropped image
            cropped_img_np = cropped_img.cpu().numpy().transpose(1, 2, 0)
            cropped_img_np = cropped_img_np * IMAGENET_STD + IMAGENET_MEAN
            cropped_img_np = np.clip(cropped_img_np, 0, 1)
            cropped_img_np = (cropped_img_np * 255).astype(np.uint8)
            
            cv2.imwrite(orig_img_path, original_img_np.squeeze())
            cv2.imwrite(crop_img_path, cropped_img_np.squeeze())
        
        total_samples += images.size(0)

# --- Plotting Samples with Masks and Crops ---
def plot_samples_with_masks_and_crops(generator, model, classes, num_samples=5, last_conv_layer_name='layer4'):
    """
    Plots samples with their attention masks, overlays, and cropped regions.
    Assumes images are normalized with ImageNet stats.
    """
    # Use global ImageNet normalization statistics
    mean = IMAGENET_MEAN
    std  = IMAGENET_STD
    
    data_iter = iter(generator)
    batch_images, batch_labels = next(data_iter)
    num_samples = min(num_samples, batch_images.size(0))
    batch_images = preprocess_input(batch_images)
    
    # Compute masks for the batch
    masks = []
    for img in batch_images:
        img_unsq = img.unsqueeze(0)
        heatmap = get_heatmap(img_unsq, model, last_conv_layer_name)
        if heatmap is None:
            continue
        mask = (heatmap > 0.7).float()
        masks.append(mask)
    if len(masks) == 0:
        print("No masks computed.")
        return
    masks_tensor = torch.stack(masks, dim=0)
    
    # Crop local regions based on the masks
    cropped_images = crop_local_region_batch(batch_images, masks_tensor)
    
    # Get model predictions for the batch
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model(batch_images.to(device))
        predictions = outputs.cpu().numpy()
    predicted_labels = np.argmax(predictions, axis=1)
    
    plt.figure(figsize=(20, num_samples * 5))
    for i in range(num_samples):
        # Process the original image
        original_img = batch_images[i].cpu().numpy().squeeze()  # shape: (C, H, W)
        original_img = np.transpose(original_img, (1, 2, 0))       # shape: (H, W, C)
        # Unnormalize: x_unnorm = x * std + mean
        original_img = original_img * std + mean
        original_img = np.clip(original_img, 0, 1)
        original_img_disp = (original_img * 255).astype(np.uint8)
        
        # Process the cropped image (derived from the original normalized image)
        cropped_img = cropped_images[i].cpu().numpy().squeeze()  # shape: (C, H, W)
        cropped_img = np.transpose(cropped_img, (1, 2, 0))         # shape: (H, W, C)
        cropped_img = cropped_img * std + mean
        cropped_img = np.clip(cropped_img, 0, 1)
        cropped_img_disp = (cropped_img * 255).astype(np.uint8)
        
        # Get the mask and prepare an overlay
        mask = masks_tensor[i].cpu().numpy().squeeze()  # shape: (H_mask, W_mask)
        heatmap = cv2.resize(mask, (original_img_disp.shape[1], original_img_disp.shape[0]), interpolation=cv2.INTER_NEAREST)
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        alpha = 0.3
        superimposed_img = cv2.addWeighted(original_img_disp, 1 - alpha, heatmap_colored, alpha, 0)
        
        # Get class names
        true_label = classes[int(batch_labels[i])]
        predicted_label = classes[int(predicted_labels[i])]
        
        plt.subplot(num_samples, 4, i * 4 + 1)
        plt.imshow(original_img_disp)
        plt.title(f"Original: {true_label}")
        plt.axis('off')
        
        plt.subplot(num_samples, 4, i * 4 + 2)
        plt.imshow(heatmap_colored)
        plt.title(f"Mask: {true_label}")
        plt.axis('off')
        
        plt.subplot(num_samples, 4, i * 4 + 3)
        plt.imshow(superimposed_img)
        plt.title("Mask Overlayed")
        plt.axis('off')
        
        plt.subplot(num_samples, 4, i * 4 + 4)
        plt.imshow(cropped_img_disp)
        plt.title(f"True: {true_label}\nPred: {predicted_label}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


# Ensure classes mapping is defined
classes = {idx: str(label) for label, idx in label_to_idx.items()}

# -------------------------------
# 4. Save Cropped Images to Disk
# -------------------------------

output_dir = './cropped_data/'

# Save images from each subset
save_cropped_images_to_directory(train_loader, global_model, output_dir, 'train', classes, last_conv_layer_name='layer4')
save_cropped_images_to_directory(val_loader, global_model, output_dir, 'validation', classes, last_conv_layer_name='layer4')
save_cropped_images_to_directory(test_loader, global_model, output_dir, 'test', classes, last_conv_layer_name='layer4')

print(f"Original and cropped images have been organized and saved to {output_dir}")

# (Optional) To visualize some examples, uncomment the following:
# plot_samples_with_masks_and_crops(train_loader, global_model, classes, num_samples=32, last_conv_layer_name='layer4')
