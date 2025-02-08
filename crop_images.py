import os
import random
import time
import pickle
import zipfile
import logging
import requests

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageFont
from tqdm import tqdm
from skimage.measure import label
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, metrics
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import (
    AvgPool2D,
    GlobalAveragePooling2D,
    MaxPool2D,
    Input,
    Conv2D,
    BatchNormalization,
    Dense,
    ReLU,
    concatenate,
    Lambda,
)
from tensorflow.keras.models import load_model, clone_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K

import visualkeras



# Directories and file paths
train_file_path = "/data/NNDL/data/train_test_split/classification/train.txt"
test_file_path = "/data/NNDL/data/train_test_split/classification/test.txt"
root_dir = "/data/NNDL/data/image/"

# Helper function to parse file paths and labels
def parse_file(file_path, root_dir):
    image_paths, labels = [], []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                label = int(line.split('/')[0])
                labels.append(label)
                image_paths.append(os.path.join(root_dir, line))
    return image_paths, labels

# Parse train and test data
train_image_paths, train_labels = parse_file(train_file_path, root_dir)
test_image_paths, test_labels = parse_file(test_file_path, root_dir)

# Aggregate all labels from train and test datasets
all_labels = set(train_labels + test_labels)
print(f"Min label: {min(all_labels)}, Max label: {max(all_labels)}")
print(f"Expected label range: [0, {len(all_labels) - 1}]")

# Create a mapping for labels to 0-based indices
label_to_idx = {label: idx for idx, label in enumerate(sorted(all_labels))}
train_labels = [label_to_idx[label] for label in train_labels]
test_labels = [label_to_idx[label] for label in test_labels]

# Update the number of classes
num_classes = len(label_to_idx)

# Prepare ImageDataGenerator for data augmentation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values to [0, 1]
    rotation_range=15,  # Random rotation within 15 degrees
    width_shift_range=0.1,  # Random horizontal shift
    height_shift_range=0.1,  # Random vertical shift
    shear_range=0.2,  # Random shear
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random horizontal flip
    fill_mode='nearest'  # Fill empty pixels
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

# Helper function to create datasets from ImageDataGenerator
def create_dataset(datagen, image_paths, labels, batch_size):
    # Ensure labels are strings
    labels = list(map(str, labels))
    dataframe = pd.DataFrame({"filename": image_paths, "class": labels})
    dataset = datagen.flow_from_dataframe(
        dataframe=dataframe,
        x_col="filename",
        y_col="class",
        target_size=(224, 224),
        class_mode="sparse",  # Sparse expects string labels internally
        batch_size=batch_size,
        shuffle=True
    )
    return dataset

train_paths, val_paths, train_labels_split, val_labels_split = train_test_split(
    train_image_paths, train_labels, test_size=0.2, random_state=42
)

# Batch size
batch_size = 128

# Create datasets
train_dataset = create_dataset(train_datagen, train_paths, train_labels_split, batch_size)
val_dataset = create_dataset(val_test_datagen, val_paths, val_labels_split, batch_size)
test_dataset = create_dataset(val_test_datagen, test_image_paths, test_labels, batch_size)

print(f"Training dataset size: {len(train_paths)}")
print(f"Validation dataset size: {len(val_paths)}")
print(f"Test dataset size: {len(test_image_paths)}")

# Define classes dynamically from the labels
classes = train_dataset.class_indices
classes = {v: k for k, v in classes.items()}  # Reverse to map indices to class names
plt.rcParams.update({'font.size': 10})


from tensorflow.keras.applications import ResNet50

def resnet50_pt(input_shape, n_classes, filters=32):
    
    inputs = Input(shape=input_shape)
    
    
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    
    x = base_model(inputs, training=True)
    
    x = GlobalAveragePooling2D()(x)
    
    # x = Dense(256, activation='relu')(x)  # Added dense layer with 256 units
    
    output = Dense(n_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=output)
    return model

input_shape = (224, 224, 3)  # Grayscale image input

resnet50_pt = resnet50_pt(input_shape, num_classes)

resnet50_pt.compile(optimizer=optimizers.Adam(learning_rate=0.0001),
                     loss=losses.SparseCategoricalCrossentropy(),
                     metrics=['accuracy'])

resnet50_pt.summary()

global_model = load_model('make_predictor_resnet50_attention_model.keras')

def preprocess_input(images):
    # Convert numpy array to tensor
    images = tf.convert_to_tensor(images, dtype=tf.float32)
    return images

def get_heatmap(images, model, last_conv_layer_name, pred_index=None):
    resnet_model = model.get_layer('resnet50')
    last_conv_layer = resnet_model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.Model(inputs=resnet_model.input, outputs=last_conv_layer.output)
    conv_output = last_conv_layer_model(images, training=False)
    heatmap = tf.reduce_max(tf.abs(conv_output), axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-7)
    return heatmap

def attention_guided_mask_inference_batch(generator, model, last_conv_layer_name='conv5_block3_out'):
    masks = []
    images, _ = next(generator)
    images = preprocess_input(images)  # Ensure images are normalized
    for img in images:
        img = tf.expand_dims(img, axis=0)
        #img = tf.image.grayscale_to_rgb(img)  # Convert grayscale to RGB for DenseNet
        heatmap = get_heatmap(img, model, last_conv_layer_name)
        if heatmap is None:
            continue
        threshold = 0.7
        mask = heatmap > threshold
        masks.append(tf.cast(mask, tf.float32))
    masks = tf.convert_to_tensor(masks, dtype=tf.float32)
    masks = tf.expand_dims(masks, axis=-1)
    return masks, images

def crop_local_region_batch(images, masks, input_shape=(224, 224)):
    cropped_images = []
    for i in range(images.shape[0]):
        if i >= len(masks):
            continue
        mask = masks[i].numpy().astype(np.uint8)
        if mask.size == 0:
            continue
        mask = np.squeeze(mask)
        mask = cv2.resize(mask, (input_shape[1], input_shape[0]))
        image = images[i]
        if isinstance(image, tf.Tensor):
            image = image.numpy()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            cropped_image = cv2.resize(image, (input_shape[1], input_shape[0]))
            cropped_images.append(cropped_image)
            continue
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = image[y:y+h, x:x+w]
        cropped_image = cv2.resize(cropped_image, (input_shape[1], input_shape[0]))
        cropped_images.append(cropped_image)
    cropped_images = np.array(cropped_images)
    return tf.convert_to_tensor(cropped_images, dtype=tf.float32)

def save_cropped_images_to_directory(generator, model, output_dir, subset, last_conv_layer_name='conv5_block3_out'):
    original_images_dir = os.path.join(output_dir, f'{subset}_original')
    cropped_images_dir = os.path.join(output_dir, f'{subset}_cropped')
    #classes = classes
    os.makedirs(original_images_dir, exist_ok=True)
    os.makedirs(cropped_images_dir, exist_ok=True)
    generator.reset()
    total_samples = 0
    while total_samples < generator.samples:
        images, labels = next(generator)
        images = preprocess_input(images)  # Ensure images are normalized
        masks, _ = attention_guided_mask_inference_batch(generator, model, last_conv_layer_name)
        cropped_images = crop_local_region_batch(images, masks)
        for i, (original_img, cropped_img, label) in enumerate(zip(images, cropped_images, labels)):
            class_name = classes[int(label)]
            orig_class_dir = os.path.join(original_images_dir, class_name)
            crop_class_dir = os.path.join(cropped_images_dir, class_name)
            os.makedirs(orig_class_dir, exist_ok=True)
            os.makedirs(crop_class_dir, exist_ok=True)
            orig_img_path = os.path.join(orig_class_dir, f"{total_samples + i}.png")
            crop_img_path = os.path.join(crop_class_dir, f"{total_samples + i}.png")
            original_img = (original_img.numpy() * 255).astype(np.uint8)
            cropped_img = (cropped_img.numpy() * 255).astype(np.uint8)
            cv2.imwrite(orig_img_path, original_img.squeeze())
            cv2.imwrite(crop_img_path, cropped_img.squeeze())
        total_samples += len(images)

def plot_samples_with_masks_and_crops(generator, model, num_samples=5, last_conv_layer_name='conv5_block3_out'):
    #classes = classes
    batch_images, batch_labels = next(generator)
    num_samples = min(num_samples, len(batch_images))
    batch_images = preprocess_input(batch_images)  # Ensure images are normalized
    masks, _ = attention_guided_mask_inference_batch(generator, model, last_conv_layer_name)
    cropped_images = crop_local_region_batch(batch_images, masks)
    predictions = model.predict(batch_images)
    predicted_labels = np.argmax(predictions, axis=1)
    plt.figure(figsize=(20, num_samples * 5))
    for i in range(num_samples):
        original_img = batch_images[i].numpy().squeeze()
        mask = masks[i].numpy().squeeze()
        cropped_img = cropped_images[i].numpy().squeeze()
        true_label = classes[int(batch_labels[i])]
        predicted_label = classes[int(predicted_labels[i])]
        
        heatmap = cv2.resize(mask, (original_img.shape[1], original_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        alpha = 0.3
        if original_img.ndim == 2:  # Grayscale
            original_img = cv2.cvtColor((original_img * 255).astype('uint8'), cv2.COLOR_GRAY2RGB)
        else:  # RGB
             original_img = (original_img * 255).astype('uint8')
             
        superimposed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_colored, alpha, 0)
        
        plt.subplot(num_samples, 4, i * 4 + 1)
        plt.imshow(original_img)
        plt.title(f"Original: {true_label}")
        plt.axis('off')
        
        plt.subplot(num_samples, 4, i * 4 + 2)
        plt.imshow(heatmap_colored)
        plt.title(f"Mask: {true_label}")
        plt.axis('off')
        
        plt.subplot(num_samples, 4, i * 4 + 3)
        plt.imshow(superimposed_img)
        plt.title('Mask Overlayed')
        plt.axis('off')
        
        plt.subplot(num_samples, 4, i * 4 + 4)
        plt.imshow(cropped_img)
        plt.title(f"True Label: {true_label}\nPredicted Label: {predicted_label}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

output_dir = './cropped_data/'

# Now you can use this function with the data generators
save_cropped_images_to_directory(train_dataset, global_model, output_dir, 'train')
save_cropped_images_to_directory(val_dataset, global_model, output_dir, 'validation')
save_cropped_images_to_directory(test_dataset, global_model, output_dir, 'test')

print(f"Original and cropped images have been organized and saved to {output_dir}")