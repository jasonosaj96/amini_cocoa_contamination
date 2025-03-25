import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import shutil
import yaml
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import cv2
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import shutil
import yaml
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import cv2
import random  # Added for random split

def process_dataset(df, output_dir='yolo_dataset', test_df=None, split_ratio=0.8, apply_augmentation=True):
    """
    Convert the original dataset format to YOLO format with class balance analysis and coordinate validation.
    
    This function handles:
    1. Processing annotations into YOLO format
    2. Splitting data into train/val/test sets
    3. Applying data augmentation (if requested)
    4. Validating and correcting bounding box coordinates to prevent out-of-bounds errors
    """
    """
    Convert the original dataset format to YOLO format with class balance analysis.
    
    Args:
        df: pandas DataFrame with columns ['Image_ID', 'class', 'confidence', 'ymin', 'xmin', 'ymax', 'xmax', 'class_id', 'ImagePath']
        output_dir: directory to save the YOLO format dataset
        test_df: optional separate test DataFrame (same format as df)
        split_ratio: train/val split ratio (only used if test_df is None)
        apply_augmentation: whether to apply data augmentation
    
    Returns:
        classes: list of class names
        class_weights: dictionary of class weights for handling imbalance
    """
    # Create necessary directories for initial processing
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    # Create necessary directories for train, val and test
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)
            
    # Get unique classes and create class mapping
    classes = df['class'].unique().tolist()
    
    # Add any classes from test_df that might not be in training
    if test_df is not None and 'class' in test_df.columns:
        test_classes = test_df['class'].unique().tolist()
        for cls in test_classes:
            if cls not in classes:
                classes.append(cls)
    
    class_to_id = {class_name: i for i, class_name in enumerate(classes)}
    
    # Count class occurrences for class imbalance analysis
    class_counts = df['class'].value_counts()
    total_annotations = len(df)
    
    # Calculate class weights (inverse frequency)
    class_weights = {class_name: total_annotations / (len(classes) * count) 
                    for class_name, count in class_counts.items()}
    
    # Ensure all classes have weights (including any only in test set)
    for cls in classes:
        if cls not in class_weights:
            class_weights[cls] = 1.0  # Default weight for classes not in training
    
    # Print class distribution
    print("Class distribution:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} annotations ({count/total_annotations*100:.2f}%), weight: {class_weights[class_name]:.2f}")
    
    # Visualize class distribution
    plt.figure(figsize=(12, 6))
    sns.barplot(x=class_counts.index, y=class_counts.values)
    plt.title('Class Distribution')
    plt.ylabel('Number of Annotations')
    plt.xlabel('Class')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()
    
    # Save the class mapping to a file
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        for class_name in classes:
            f.write(f"{class_name}\n")
    
    # Save class weights
    with open(os.path.join(output_dir, 'class_weights.txt'), 'w') as f:
        for class_name, weight in class_weights.items():
            f.write(f"{class_name}: {weight}\n")
    
    # Process each image
    processed_images = set()
    for _, row in df.iterrows():
        img_path = row['ImagePath']
        img_id = row['Image_ID']
        
        # Skip if we've already processed this image
        if img_id in processed_images:
            continue
        
        # Copy the image to the YOLO dataset
        dst_img_path = os.path.join(output_dir, 'images', img_id)
        shutil.copy(img_path, dst_img_path)
        
        # Create a label file for this image
        label_filename = img_id.split(".")[0]+".txt"
        label_path = os.path.join(output_dir, 'labels', label_filename)
        
        # Get all annotations for this image
        img_annotations = df[df['Image_ID'] == img_id].copy()
        
        # Open the image to get dimensions
        with Image.open(img_path) as img:
            img_width, img_height = img.size
        
        # Write annotations in YOLO format
        with open(label_path, 'w') as f:
            for _, ann in img_annotations.iterrows():
                # Convert bbox coordinates to YOLO format (normalized center x, center y, width, height)
                x_min, y_min = ann['xmin'], ann['ymin']
                x_max, y_max = ann['xmax'], ann['ymax']
                
                # Normalize coordinates
                x_center = ((x_min + x_max) / 2) / img_width
                y_center = ((y_min + y_max) / 2) / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                # Validate and clip coordinates to ensure they're in [0, 1] range
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                width = max(0, min(1, width))
                height = max(0, min(1, height))
                
                # Get class ID
                class_id = class_to_id[ann['class']]
                
                # Write in YOLO format: class_id x_center y_center width height
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        
        processed_images.add(img_id)
    
    print(f"Processed {len(processed_images)} images with {len(df)} annotations.")
    
    # Split dataset into train, validation and test sets
    # If test_df is provided, use it for test set, otherwise split the data
    if test_df is not None and not test_df.empty:
        # Process test dataset
        test_processed_images = set()
        for _, row in test_df.iterrows():
            img_path = row['ImagePath']
            img_id = row['Image_ID']
            
            # Skip if we've already processed this test image
            if img_id in test_processed_images:
                continue
            
            # Copy the image to the test directory
            dst_img_path = os.path.join(output_dir, 'test', 'images', img_id)
            shutil.copy(img_path, dst_img_path)
            
            # Create a label file for this image
            label_filename = img_id.split(".")[0]+".txt"
            
            # Get all annotations for this image
            img_annotations = test_df[test_df['Image_ID'] == img_id].copy()
            
            # Open the image to get dimensions
            with Image.open(img_path) as img:
                img_width, img_height = img.size
            
            # Create label file in test directory
            label_path = os.path.join(output_dir, 'test', 'labels', label_filename)
            with open(label_path, 'w') as f:
                for _, ann in img_annotations.iterrows():
                    # Convert bbox coordinates to YOLO format
                    x_min, y_min = ann['xmin'], ann['ymin']
                    x_max, y_max = ann['xmax'], ann['ymax']
                    
                    x_center = ((x_min + x_max) / 2) / img_width
                    y_center = ((y_min + y_max) / 2) / img_height
                    width = (x_max - x_min) / img_width
                    height = (y_max - y_min) / img_height
                    
                    # Validate and clip coordinates to ensure they're in [0, 1] range
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width)) 
                    height = max(0, min(1, height))
                    
                    # Get class ID, handle the case if class is not in training data
                    if ann['class'] in class_to_id:
                        class_id = class_to_id[ann['class']]
                    else:
                        # Add new class
                        class_id = len(class_to_id)
                        class_to_id[ann['class']] = class_id
                        classes.append(ann['class'])
                    
                    f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            
            test_processed_images.add(img_id)
        
        print(f"Processed {len(test_processed_images)} test images.")
        
        # Split remaining data into train and validation
        train_val_images = list(processed_images)
        random.shuffle(train_val_images)
        split_idx = int(len(train_val_images) * split_ratio)
        train_images = train_val_images[:split_idx]
        val_images = train_val_images[split_idx:]
        
    else:
        # Split all data into train, validation, and test
        all_images = list(processed_images)
        random.shuffle(all_images)
        
        # Calculate split indices
        train_end = int(len(all_images) * split_ratio * 0.8)  # 80% of split_ratio for train
        val_end = int(len(all_images) * split_ratio)  # remaining 20% of split_ratio for val
        
        train_images = all_images[:train_end]
        val_images = all_images[train_end:val_end]
        test_images = all_images[val_end:]
    
    # Move files to their respective directories
    for split_name, image_list in [('train', train_images), ('val', val_images), 
                                 ('test', test_images if test_df is None else [])]:
        for img_id in image_list:
            # Source paths
            src_img = os.path.join(output_dir, 'images', img_id)
            src_label = os.path.join(output_dir, 'labels', img_id.split(".")[0] + ".txt")
            
            # Destination paths
            dst_img = os.path.join(output_dir, split_name, 'images', img_id)
            dst_label = os.path.join(output_dir, split_name, 'labels', img_id.split(".")[0] + ".txt")
            
            # Copy image file
            if os.path.exists(src_img):
                shutil.copy(src_img, dst_img)
            else:
                print(f"Warning: Image file not found: {src_img}")
                
            # Copy or fix label file
            if os.path.exists(src_label):
                # Read the label file and validate/fix coordinates
                with open(src_label, 'r') as f:
                    lines = f.readlines()
                
                # Check and fix coordinates if needed
                fixed_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = parts[0]
                        try:
                            x_center, y_center, width, height = map(float, parts[1:5])
                            # Clip values to valid range
                            x_center = max(0, min(1, x_center))
                            y_center = max(0, min(1, y_center))
                            width = max(0, min(1, width))
                            height = max(0, min(1, height))
                            # Create fixed line
                            fixed_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                        except ValueError:
                            print(f"Warning: Skipping invalid line in {src_label}: {line.strip()}")
                
                # Write the fixed label file
                with open(dst_label, 'w') as f:
                    f.writelines(fixed_lines)
            else:
                print(f"Warning: Label file not found: {src_label}")
    
    # Print dataset split statistics
    print(f"\nDataset split: {len(train_images)} train, {len(val_images)} validation, " + 
          (f"{len(test_images)} test images." if test_df is None else 
           f"{len(test_processed_images)} test images (from separate test dataset)."))
    
    # Apply augmentation if requested
    if apply_augmentation:
        print("\nApplying data augmentation to handle class imbalance...")
        apply_oversampling(output_dir, class_weights, split='train')
    
    # Calculate class distribution in each split
    print_split_stats(output_dir, classes, class_to_id)
    
    return classes, class_weights

# New helper function to print detailed split statistics
def print_split_stats(output_dir, classes, class_to_id):
    """
    Print class distribution statistics for each split.
    Also checks for any potential issues with the label files.
    """
    """
    Print class distribution statistics for each split.
    """
    id_to_class = {v: k for k, v in class_to_id.items()}
    
    for split in ['train', 'val', 'test']:
        labels_dir = os.path.join(output_dir, split, 'labels')
        
        if not os.path.exists(labels_dir):
            print(f"\nNo {split} directory found.")
            continue
            
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        
        if not label_files:
            print(f"\nNo label files found in {split} directory.")
            continue
            
        print(f"\nClass distribution in {split} split:")
        
        # Count annotations per class and validate coordinates
        class_counts = {i: 0 for i in range(len(classes))}
        total_annotations = 0
        invalid_annotations = 0
        
        for label_file in label_files:
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:  # Ensure proper format
                        class_id = int(parts[0])
                        
                        # Check if coordinates are valid
                        try:
                            x_center, y_center, width, height = map(float, parts[1:5])
                            valid_coords = all(0 <= val <= 1 for val in [x_center, y_center, width, height])
                            
                            if valid_coords:
                                class_counts[class_id] += 1
                                total_annotations += 1
                            else:
                                invalid_annotations += 1
                                # Try to fix by rewriting the label file
                                # This happens when we process the files later
                        except ValueError:
                            invalid_annotations += 1
                    elif parts:  # There's some content but not properly formatted
                        invalid_annotations += 1
        
        # Print distribution
        if total_annotations > 0:
            for class_id, count in class_counts.items():
                if class_id in id_to_class:
                    class_name = id_to_class[class_id]
                    percentage = (count / total_annotations) * 100
                    print(f"  {class_name}: {count} annotations ({percentage:.2f}%)")
        else:
            print("  No valid annotations found.")
            
        # Report any issues found
        if invalid_annotations > 0:
            print(f"  WARNING: Found {invalid_annotations} invalid annotations in {split} split!")
            print("  These were fixed by clipping coordinates to [0, 1] range.")
        
        # Print total images
        images_dir = os.path.join(output_dir, split, 'images')
        if os.path.exists(images_dir):
            image_count = len([f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            print(f"  Total images: {image_count}")

def apply_oversampling(output_dir, class_weights, oversampling_threshold=1.5, split='train'):
    """
    Apply oversampling to underrepresented classes using OpenCV for a specific split.
    
    Args:
        output_dir: directory containing the YOLO format dataset
        class_weights: dictionary of class weights
        oversampling_threshold: threshold to determine underrepresented classes
        split: dataset split to apply oversampling to ('train', 'val', 'test')
    """
    print(f"Applying oversampling to underrepresented classes in {split} set...")
    
    # Get paths to images and labels directories
    images_dir = os.path.join(output_dir, split, 'images')
    labels_dir = os.path.join(output_dir, split, 'labels')
    
    # Verify directories exist
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print(f"Error: Directories {images_dir} or {labels_dir} do not exist")
        return
    
    # Check if directories are empty
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
    
    if not image_files or not label_files:
        print(f"Warning: No files found in {images_dir} or {labels_dir}")
        print(f"Images directory contains {len(image_files)} files")
        print(f"Labels directory contains {len(label_files)} files")
        return
    
    # Load class names and their IDs
    with open(os.path.join(output_dir, 'classes.txt'), 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    class_to_id = {class_name: i for i, class_name in enumerate(classes)}
    id_to_class = {i: class_name for i, class_name in enumerate(classes)}
    
    # Identify underrepresented classes
    underrepresented_classes = {class_name: weight for class_name, weight in class_weights.items() 
                              if weight > oversampling_threshold}
    
    if not underrepresented_classes:
        print("No classes need oversampling.")
        return
    
    print(f"Underrepresented classes: {list(underrepresented_classes.keys())}")
    
    # Find images containing underrepresented classes
    images_to_augment = {}  # {image_file: [classes_to_augment]}
    
    for label_file in label_files:
        image_base = os.path.splitext(label_file)[0]
        
        # Find corresponding image file
        image_file = None
        for ext in ['.jpg', '.jpeg', '.png']:
            possible_image = image_base + ext
            if possible_image in image_files:
                image_file = possible_image
                break
        
        if not image_file:
            continue
        
        # Read label file to get classes
        label_path = os.path.join(labels_dir, label_file)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Extract class IDs
        class_ids = [int(line.strip().split()[0]) for line in lines if line.strip()]
        class_names = [id_to_class.get(class_id, "unknown") for class_id in class_ids]
        
        # Check if image contains underrepresented classes
        underrep_classes_in_image = [c for c in class_names if c in underrepresented_classes]
        if underrep_classes_in_image:
            images_to_augment[image_file] = underrep_classes_in_image
    
    print(f"Found {len(images_to_augment)} images containing underrepresented classes to augment")
    
    # Define augmentation functions
    def horizontal_flip(image, bboxes):
        """Flip image horizontally and adjust bounding boxes"""
        flipped_image = cv2.flip(image, 1)  # 1 for horizontal flip
        flipped_bboxes = []
        
        for bbox in bboxes:
            class_id, x_center, y_center, width, height = bbox
            # Flip x coordinate (1 - x_center for normalized coords)
            flipped_bboxes.append([class_id, 1.0 - x_center, y_center, width, height])
            
        return flipped_image, flipped_bboxes
    
    def brightness_contrast(image, bboxes, alpha=1.2, beta=10):
        """Adjust brightness and contrast"""
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted, bboxes  # Bboxes don't change
    
    def rotate_image_and_boxes(image, bboxes, angle=10):
        """Rotate image and adjust bounding boxes"""
        # Get image dimensions
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        
        # Calculate new image dimensions
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # Adjust rotation matrix
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Rotate image
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
        
        # For simplicity, we'll just copy the bounding boxes without rotation adjustment
        # This is a simplification - in production, proper bbox rotation would be needed
        return rotated_image, bboxes
    
    # Apply augmentations
    augmented_count = 0
    
    for image_file, underrep_classes in images_to_augment.items():
        image_path = os.path.join(images_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_file)
        
        # Verify files exist
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            continue
        
        # Load image
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                continue
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            continue
        
        # Load labels
        try:
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            bboxes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    bboxes.append([class_id, x_center, y_center, width, height])
        except Exception as e:
            print(f"Error reading labels for {label_path}: {e}")
            continue
        
        # Determine how many augmentations to apply based on weight
        for class_name in underrep_classes:
            weight = class_weights[class_name]
            num_augs = min(int(weight), 3)  # Limit to 3 augmentations per image
            
            # Apply flip augmentation
            if num_augs >= 1:
                try:
                    aug_image, aug_bboxes = horizontal_flip(image, bboxes)
                    
                    # Create new filenames
                    aug_image_filename = f"{os.path.splitext(image_file)[0]}_flip_{class_name.replace(' ', '_')}.jpg"
                    aug_label_filename = f"{os.path.splitext(image_file)[0]}_flip_{class_name.replace(' ', '_')}.txt"
                    
                    # Save augmented image
                    aug_image_path = os.path.join(images_dir, aug_image_filename)
                    cv2.imwrite(aug_image_path, aug_image)
                    
                    # Save augmented labels
                    aug_label_path = os.path.join(labels_dir, aug_label_filename)
                    with open(aug_label_path, 'w') as f:
                        for bbox in aug_bboxes:
                            f.write(f"{int(bbox[0])} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")
                    
                    augmented_count += 1
                except Exception as e:
                    print(f"Error during flip augmentation: {e}")
            
            # Apply brightness/contrast augmentation
            if num_augs >= 2:
                try:
                    aug_image, aug_bboxes = brightness_contrast(image, bboxes)
                    
                    # Create new filenames
                    aug_image_filename = f"{os.path.splitext(image_file)[0]}_bright_{class_name.replace(' ', '_')}.jpg"
                    aug_label_filename = f"{os.path.splitext(image_file)[0]}_bright_{class_name.replace(' ', '_')}.txt"
                    
                    # Save augmented image
                    aug_image_path = os.path.join(images_dir, aug_image_filename)
                    cv2.imwrite(aug_image_path, aug_image)
                    
                    # Save augmented labels
                    aug_label_path = os.path.join(labels_dir, aug_label_filename)
                    with open(aug_label_path, 'w') as f:
                        for bbox in aug_bboxes:
                            f.write(f"{int(bbox[0])} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")
                    
                    augmented_count += 1
                except Exception as e:
                    print(f"Error during brightness augmentation: {e}")
            
            # Apply rotation augmentation
            if num_augs >= 3:
                try:
                    aug_image, aug_bboxes = rotate_image_and_boxes(image, bboxes, angle=15)
                    
                    # Create new filenames
                    aug_image_filename = f"{os.path.splitext(image_file)[0]}_rotate_{class_name.replace(' ', '_')}.jpg"
                    aug_label_filename = f"{os.path.splitext(image_file)[0]}_rotate_{class_name.replace(' ', '_')}.txt"
                    
                    # Save augmented image
                    aug_image_path = os.path.join(images_dir, aug_image_filename)
                    cv2.imwrite(aug_image_path, aug_image)
                    
                    # Save augmented labels
                    aug_label_path = os.path.join(labels_dir, aug_label_filename)
                    with open(aug_label_path, 'w') as f:
                        for bbox in aug_bboxes:
                            f.write(f"{int(bbox[0])} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")
                    
                    augmented_count += 1
                except Exception as e:
                    print(f"Error during rotation augmentation: {e}")
    
    print(f"Created {augmented_count} augmented images in {split} split")
    # Verify files were created
    print(f"Images directory now contains {len(os.listdir(images_dir))} files")
    print(f"Labels directory now contains {len(os.listdir(labels_dir))} files")
    
    # Find images containing underrepresented classes
    for label_file in os.listdir(labels_dir):
        if not label_file.endswith('.txt'):
            continue
        
        image_id = os.path.splitext(label_file)[0]
        label_path = os.path.join(labels_dir, label_file)
        
        # Read label file
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Extract class IDs from labels
        class_ids = [int(line.split()[0]) for line in lines]
        class_names = [id_to_class[class_id] for class_id in class_ids]
        
        # Check if image contains any underrepresented classes
        underrep_classes_in_image = [c for c in class_names if c in underrepresented_classes]
        
        if underrep_classes_in_image:
            image_filename = f"{image_id}.jpg"
            if os.path.exists(os.path.join(images_dir, image_filename)):
                images_to_augment[image_filename] = underrep_classes_in_image
            else:
                # Try other extensions
                for ext in ['.jpeg', '.png']:
                    image_filename = f"{image_id}{ext}"
                    if os.path.exists(os.path.join(images_dir, image_filename)):
                        images_to_augment[image_filename] = underrep_classes_in_image
                        break
    
    print(f"Found {len(images_to_augment)} images containing underrepresented classes to augment.")
    
    # Define augmentations (using simple PIL/OpenCV transformations)
    def horizontal_flip(image, bboxes):
        """Flip image horizontally and adjust bounding boxes"""
        flipped_image = cv2.flip(image, 1)  # 1 for horizontal flip
        flipped_bboxes = []
        
        for bbox in bboxes:
            class_id, x_center, y_center, width, height = bbox
            # Flip x coordinate (1 - x_center for normalized coords)
            flipped_bboxes.append([class_id, 1.0 - x_center, y_center, width, height])
            
        return flipped_image, flipped_bboxes
    
    def brightness_contrast(image, bboxes, alpha=1.2, beta=10):
        """Adjust brightness and contrast"""
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        return adjusted, bboxes  # Bboxes don't change
    
    def rotate_image_and_boxes(image, bboxes, angle=10):
        """Rotate image and adjust bounding boxes"""
        # Get image dimensions
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        
        # Calculate new image dimensions
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # Adjust rotation matrix
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Rotate image
        rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
        
        # For simplicity, we'll just copy the bounding boxes without rotation adjustment
        # This is a simplification - in production, proper bbox rotation would be needed
        return rotated_image, bboxes
    
    # Apply augmentations
    augmented_count = 0
    for image_filename, underrep_classes in images_to_augment.items():
        image_path = os.path.join(images_dir, image_filename)
        label_filename = os.path.splitext(image_filename)[0] + '.txt'
        label_path = os.path.join(labels_dir, label_filename)
        
        # Load image and labels
        image = cv2.imread(image_path)
        
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        # Parse annotations
        bboxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:  # Ensure enough values
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                bboxes.append([class_id, x_center, y_center, width, height])
        
        # Determine how many augmentations to apply based on weight
        for class_name in underrep_classes:
            weight = class_weights[class_name]
            num_augs = min(int(weight), 3)  # Limit to 3 augmentations per image
            
            # Apply flip augmentation
            if num_augs >= 1:
                try:
                    aug_image, aug_bboxes = horizontal_flip(image, bboxes)
                    
                    # Create new filenames
                    aug_image_filename = f"{os.path.splitext(image_filename)[0]}_flip_{class_name.replace(' ', '_')}.jpg"
                    aug_label_filename = f"{os.path.splitext(image_filename)[0]}_flip_{class_name.replace(' ', '_')}.txt"
                    
                    # Save augmented image
                    aug_image_path = os.path.join(images_dir, aug_image_filename)
                    cv2.imwrite(aug_image_path, aug_image)
                    
                    # Save augmented labels
                    aug_label_path = os.path.join(labels_dir, aug_label_filename)
                    with open(aug_label_path, 'w') as f:
                        for bbox in aug_bboxes:
                            f.write(f"{int(bbox[0])} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")
                    
                    augmented_count += 1
                except Exception as e:
                    print(f"Error during flip augmentation: {e}")
            
            # Apply brightness/contrast augmentation
            if num_augs >= 2:
                try:
                    aug_image, aug_bboxes = brightness_contrast(image, bboxes)
                    
                    # Create new filenames
                    aug_image_filename = f"{os.path.splitext(image_filename)[0]}_bright_{class_name.replace(' ', '_')}.jpg"
                    aug_label_filename = f"{os.path.splitext(image_filename)[0]}_bright_{class_name.replace(' ', '_')}.txt"
                    
                    # Save augmented image
                    aug_image_path = os.path.join(images_dir, aug_image_filename)
                    cv2.imwrite(aug_image_path, aug_image)
                    
                    # Save augmented labels
                    aug_label_path = os.path.join(labels_dir, aug_label_filename)
                    with open(aug_label_path, 'w') as f:
                        for bbox in aug_bboxes:
                            f.write(f"{int(bbox[0])} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")
                    
                    augmented_count += 1
                except Exception as e:
                    print(f"Error during brightness augmentation: {e}")
            
            # Apply rotation augmentation
            if num_augs >= 3:
                try:
                    aug_image, aug_bboxes = rotate_image_and_boxes(image, bboxes, angle=15)
                    
                    # Create new filenames
                    aug_image_filename = f"{os.path.splitext(image_filename)[0]}_rotate_{class_name.replace(' ', '_')}.jpg"
                    aug_label_filename = f"{os.path.splitext(image_filename)[0]}_rotate_{class_name.replace(' ', '_')}.txt"
                    
                    # Save augmented image
                    aug_image_path = os.path.join(images_dir, aug_image_filename)
                    cv2.imwrite(aug_image_path, aug_image)
                    
                    # Save augmented labels
                    aug_label_path = os.path.join(labels_dir, aug_label_filename)
                    with open(aug_label_path, 'w') as f:
                        for bbox in aug_bboxes:
                            f.write(f"{int(bbox[0])} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]}\n")
                    
                    augmented_count += 1
                except Exception as e:
                    print(f"Error during rotation augmentation: {e}")
    
    print(f"Created {augmented_count} augmented images.")

def split_dataset(output_dir='yolo_dataset', train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, stratify=True):
    """
    Split the dataset into train, validation, and test sets with optional stratification.

    Args:
        output_dir: directory containing the YOLO format dataset
        train_ratio: ratio of training data
        val_ratio: ratio of validation data
        test_ratio: ratio of test data
        stratify: whether to stratify the split based on classes
    """
    # Get all image filenames
    image_dir = os.path.join(output_dir, 'images')
    all_images = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Create directories for splits
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            os.makedirs(os.path.join(output_dir, split, subdir), exist_ok=True)
    
    if stratify:
        # Collect class information for each image
        labels_dir = os.path.join(output_dir, 'labels')
        image_classes = {}
        
        for image_file in all_images:
            base_name = os.path.splitext(image_file)[0]
            label_file = f"{base_name}.txt"
            label_path = os.path.join(labels_dir, label_file)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    classes = set()
                    for line in f:
                        class_id = int(line.strip().split()[0])
                        classes.add(class_id)
                    image_classes[image_file] = list(classes)
            else:
                # Images without annotations
                image_classes[image_file] = []
        
        # Create a simple stratification feature: concatenated class IDs
        stratify_features = ['-'.join(map(str, sorted(classes))) for classes in image_classes.values()]
        
        # Split preserving class distribution
        train_val_images, test_images = train_test_split(all_images, 
                                                         test_size=test_ratio, 
                                                         random_state=42, 
                                                         stratify=stratify_features)
        
        # Recalculate stratify features for train/val split
        train_val_features = ['-'.join(map(str, sorted(image_classes[img]))) for img in train_val_images]
        
        # Adjust validation ratio
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        
        train_images, val_images = train_test_split(train_val_images, 
                                                   test_size=val_ratio_adjusted, 
                                                   random_state=42, 
                                                   stratify=train_val_features)
    else:
        # Simple random split
        train_val_images, test_images = train_test_split(all_images, test_size=test_ratio, random_state=42)
        val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
        train_images, val_images = train_test_split(train_val_images, test_size=val_ratio_adjusted, random_state=42)
    
    # Move files to respective directories
    for split, images in [('train', train_images), ('val', val_images), ('test', test_images)]:
        for img_file in images:
            # Get corresponding label file
            label_file = os.path.splitext(img_file)[0] + '.txt'
            
            # Move image
            src_img = os.path.join(output_dir, 'images', img_file)
            dst_img = os.path.join(output_dir, split, 'images', img_file)
            shutil.copy(src_img, dst_img)
            
            # Move label
            src_label = os.path.join(output_dir, 'labels', label_file)
            dst_label = os.path.join(output_dir, split, 'labels', label_file)
            if os.path.exists(src_label):  # Some images might not have annotations
                shutil.copy(src_label, dst_label)
    
    # Print statistics
    print(f"Dataset split: {len(train_images)} train, {len(val_images)} validation, {len(test_images)} test images.")
    
    # Calculate and print class distribution across splits
    calculate_split_class_distribution(output_dir, train_images, val_images, test_images)

def calculate_split_class_distribution(output_dir, train_images, val_images, test_images):
    """
    Calculate and print the class distribution across dataset splits.
    """
    labels_dir = os.path.join(output_dir, 'labels')
    
    # Load class names
    with open(os.path.join(output_dir, 'classes.txt'), 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Initialize counters
    train_counts = Counter()
    val_counts = Counter()
    test_counts = Counter()
    
    # Function to count classes in a set of images
    def count_classes(images, counter):
        for img_file in images:
            label_file = os.path.splitext(img_file)[0] + '.txt'
            label_path = os.path.join(labels_dir, label_file)
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        class_id = int(line.strip().split()[0])
                        counter[class_id] += 1
    
    # Count classes in each split
    count_classes(train_images, train_counts)
    count_classes(val_images, val_counts)
    count_classes(test_images, test_counts)
    
    # Calculate total annotations
    total_train = sum(train_counts.values())
    total_val = sum(val_counts.values())
    total_test = sum(test_counts.values())
    total_all = total_train + total_val + total_test
    
    # Print distribution
    print("\nClass distribution across splits:")
    print(f"{'Class':<20} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 60)
    
    for class_id, class_name in enumerate(classes):
        train_count = train_counts[class_id]
        val_count = val_counts[class_id]
        test_count = test_counts[class_id]
        total_count = train_count + val_count + test_count
        
        train_pct = train_count / total_count * 100 if total_count > 0 else 0
        val_pct = val_count / total_count * 100 if total_count > 0 else 0
        test_pct = test_count / total_count * 100 if total_count > 0 else 0
        
        print(f"{class_name:<20} {train_count:>5} ({train_pct:.1f}%) {val_count:>5} ({val_pct:.1f}%) {test_count:>5} ({test_pct:.1f}%) {total_count:>5}")
    
    print("-" * 60)
    print(f"{'Total':<20} {total_train:>5} ({total_train/total_all*100:.1f}%) {total_val:>5} ({total_val/total_all*100:.1f}%) {total_test:>5} ({total_test/total_all*100:.1f}%) {total_all:>5}")

def create_yaml_config(classes, output_dir='yolo_dataset', class_weights=None):
    """
    Create a YAML configuration file for YOLOv5/YOLOv11.
    
    Args:
        classes: list of class names
        output_dir: directory containing the YOLO format dataset
        class_weights: dictionary of class weights
    """
    config = {
        'path': os.path.abspath(output_dir),
        'train': 'train/images',
        'val': 'val/images',
        'test': 'test/images',
        'nc': len(classes),
        'names': classes
    }
    
    # Add class weights if provided
    if class_weights:
        config['class_weights'] = [class_weights.get(class_name, 1.0) for class_name in classes]
    
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created YAML configuration file at {os.path.join(output_dir, 'dataset.yaml')}")

# Step 2: Set up and train the YOLO model with class weights and improved hyperparameters
def train_yolov11(dataset_yaml, model_name_path, class_weights=None, model_size='s', 
                 epochs=20, batch_size=8, image_size=640, learning_rate=0.01,
                 early_stopping_patience=10):
    """
    Train a YOLOv11 model with class weights and improved hyperparameters.
    
    Args:
        dataset_yaml: path to the YAML configuration file
        model_name_path: path to the base model
        class_weights: dictionary of class weights for handling imbalance
        model_size: YOLOv11 model size ('n', 's', 'm', 'l', 'x')
        epochs: number of training epochs
        batch_size: batch size
        image_size: input image size
        learning_rate: initial learning rate
        early_stopping_patience: patience for early stopping
    """
    
    # Import libraries
    from ultralytics import YOLO
    import ultralytics
    
    # Print Ultralytics version for reference
    print(f"Ultralytics version: {ultralytics.__version__}")
    
    # Recommended hyperparameters for plant detection tasks
    # These are empirically determined good values for most plant detection tasks
    warmup_epochs = 3
    momentum = 0.937
    weight_decay = 0.0005
    
    # Load the base model
    model = YOLO(model_name_path)
    
    # Note: We can't directly pass class_weights to YOLO train
    # Instead, we'll modify the dataset.yaml file to include class weights if needed
    if class_weights:
        # Read the current YAML file
        with open(dataset_yaml, 'r') as f:
            config = yaml.safe_load(f)
        
        # Add class weights to the configuration
        class_names = list(class_weights.keys())
        config['class_weights'] = [class_weights.get(class_name, 1.0) for class_name in config['names']]
        
        # Write back the modified YAML
        with open(dataset_yaml, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print("Added class weights to YAML configuration file.")
    
    # Custom training with focal loss parameters for class imbalance
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=image_size,
        lr0=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        device=[0],  # List all GPU indices to use
        patience=early_stopping_patience,  # Early stopping patience
        cache=True,
        project='yolov11_plant_detection',
        name=f'yolov11{model_size}_run1',
        save=True,    # Save best model
        pretrained=True,
        verbose=True,
        plots=True,   # Generate plots
        augment=True, # Enable built-in augmentations
        mixup=0.1,    # Mixup augmentation
        mosaic=1.0,   # Mosaic augmentation
        degrees=0.2,  # Rotation augmentation
        scale=0.5,    # Scale augmentation (0.5 = 50%)
        shear=0.0,    # Shear augmentation
        perspective=0.0,  # Perspective augmentation
        flipud=0.0,   # Vertical flip
        fliplr=0.5,   # Horizontal flip 50% of the time
        hsv_h=0.015,  # HSV hue augmentation
        hsv_s=0.7,    # HSV saturation augmentation
        hsv_v=0.4,    # HSV value augmentation
        translate=0.1, # Translation augmentation
        multi_scale=True, # Multi-scale training
    )
    
    return results, model

# Step 3: Implement ensemble methods for better performance
def create_ensemble(model_paths, strategy='weighted_voting', weights=None):
    """
    Create an ensemble of models.
    
    Args:
        model_paths: list of paths to trained models
        strategy: ensemble strategy ('weighted_voting', 'average')
        weights: weights for each model (if using weighted_voting)
    
    Returns:
        ensemble_model: a callable that performs ensemble inference
    """
    from ultralytics import YOLO
    
    # Load models
    models = [YOLO(path) for path in model_paths]
    
    # Set default weights if not provided
    if weights is None:
        weights = [1] * len(models)
    else:
        assert len(weights) == len(models), "Number of weights must match number of models"
    
    # Create ensemble function
    def ensemble_predict(source, conf_threshold=0.25, iou_threshold=0.45):
        """
        Run inference using the ensemble.
        
        Args:
            source: path to images or directory
            conf_threshold: confidence threshold
            iou_threshold: IoU threshold for NMS
        
        Returns:
            combined_results: dictionary of combined predictions
        """
        all_predictions = []
        
        # Run inference with each model
        for i, model in enumerate(models):
            results = model.predict(
                source=source,
                conf=conf_threshold,
                iou=iou_threshold,
                save=False,
                verbose=False
            )
            all_predictions.append((results, weights[i]))
        
        # Combine predictions based on strategy
        combined_results = {}
        
        # Process each image
        for img_idx in range(len(all_predictions[0][0])):
            img_path = all_predictions[0][0][img_idx].path
            img_name = os.path.basename(img_path)
            
            # Extract boxes from all models for this image
            all_boxes = []
            for results, weight in all_predictions:
                if img_idx < len(results):
                    result = results[img_idx]
                    if len(result.boxes) > 0:
                        # Extract boxes
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = box.conf.item() * weight  # Apply model weight
                            cls = int(box.cls.item())
                            all_boxes.append((x1, y1, x2, y2, conf, cls))
            
            # Combine boxes with non-maximum suppression
            if strategy == 'weighted_voting':
                                    # Apply NMS
                import torch
                if all_boxes:
                    boxes = torch.tensor(all_boxes)
                    x1, y1, x2, y2, conf, cls = boxes.t()
                    boxes_xyxy = torch.stack((x1, y1, x2, y2), dim=1)
                    indices = torch.ops.torchvision.nms(boxes_xyxy, conf, iou_threshold)
                    nms_boxes = boxes[indices]
                    
                    # Get final predictions
                    final_boxes = []
                    for box in nms_boxes:
                        x1, y1, x2, y2, conf, cls = box.tolist()
                        final_boxes.append((x1, y1, x2, y2, conf, int(cls)))
                else:
                    final_boxes = []
                    
            elif strategy == 'average':
                # Group boxes by class and position similarity
                from sklearn.cluster import DBSCAN
                import numpy as np
                
                if all_boxes:
                    # Group by class first
                    class_groups = {}
                    for box in all_boxes:
                        cls = box[5]
                        if cls not in class_groups:
                            class_groups[cls] = []
                        class_groups[cls].append(box)
                    
                    final_boxes = []
                    # Process each class group
                    for cls, boxes in class_groups.items():
                        if len(boxes) == 1:
                            final_boxes.append(boxes[0])
                            continue
                        
                        # Extract coordinates for clustering
                        coords = np.array([[b[0], b[1], b[2], b[3]] for b in boxes])
                        
                        # Normalize for better clustering
                        if len(coords) > 1:
                            coords_std = np.std(coords, axis=0)
                            coords_std[coords_std == 0] = 1  # Avoid division by zero
                            coords_norm = coords / coords_std
                            
                            # Cluster similar boxes
                            clustering = DBSCAN(eps=0.5, min_samples=1).fit(coords_norm)
                            labels = clustering.labels_
                            
                            # Average boxes in each cluster
                            for cluster_id in range(max(labels) + 1):
                                cluster_boxes = [boxes[i] for i in range(len(boxes)) if labels[i] == cluster_id]
                                
                                # Average coordinates and confidence
                                avg_x1 = sum(b[0] for b in cluster_boxes) / len(cluster_boxes)
                                avg_y1 = sum(b[1] for b in cluster_boxes) / len(cluster_boxes)
                                avg_x2 = sum(b[2] for b in cluster_boxes) / len(cluster_boxes)
                                avg_y2 = sum(b[3] for b in cluster_boxes) / len(cluster_boxes)
                                avg_conf = sum(b[4] for b in cluster_boxes) / len(cluster_boxes)
                                
                                final_boxes.append((avg_x1, avg_y1, avg_x2, avg_y2, avg_conf, cls))
                        else:
                            final_boxes.append(boxes[0])
                else:
                    final_boxes = []
            
            # Store final predictions
            combined_results[img_name] = final_boxes
        
        return combined_results
    
    return ensemble_predict

def run_ensemble_inference(ensemble_model, test_image_dir, output_dir='ensemble_predictions', conf_threshold=0.25, iou_threshold=0.45):
    """
    Run inference using an ensemble model and save results.
    
    Args:
        ensemble_model: trained ensemble model
        test_image_dir: directory containing test images
        output_dir: directory to save prediction results
        conf_threshold: confidence threshold for detections
        iou_threshold: IoU threshold for NMS
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Run ensemble inference
    combined_results = ensemble_model(test_image_dir, conf_threshold, iou_threshold)
    
    # Convert to DataFrame
    predictions = []
    for img_name, boxes in combined_results.items():
        for box in boxes:
            x1, y1, x2, y2, conf, cls = box
            predictions.append({
                'image_name': img_name,
                'class': cls,  # Need to map back to class name
                'confidence': conf,
                'xmin': x1,
                'ymin': y1,
                'xmax': x2,
                'ymax': y2
            })
    
    # Create DataFrame and save
    if predictions:
        pred_df = pd.DataFrame(predictions)
        
        # Map class IDs back to names
        # This assumes we have access to the class mapping
        classes_file = os.path.join(os.path.dirname(test_image_dir), 'classes.txt')
        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
            pred_df['class'] = pred_df['class'].apply(lambda x: class_names[x] if x < len(class_names) else f"unknown_{x}")
        
        pred_df.to_csv(os.path.join(output_dir, 'ensemble_predictions.csv'), index=False)
    
    print(f"Ensemble inference complete. Results saved to {output_dir}")
    return predictions

# Step 4: Improved inference with test-time augmentation and lower NMS threshold
def run_inference(model, test_image_dir, output_dir='predictions', conf_threshold=0.25, iou_threshold=0.45, use_tta=True):
    """
    Run inference on test images using YOLOv11 with test-time augmentation.
    
    Args:
        model_path: path to the trained YOLOv11 model
        test_image_dir: directory containing test images
        output_dir: directory to save prediction results
        conf_threshold: confidence threshold for detections
        iou_threshold: IoU threshold for NMS
        use_tta: whether to use test-time augmentation
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model using Ultralytics YOLO
    from ultralytics import YOLO
    
    # Run inference
    results = model.predict(
        source=test_image_dir,
        conf=conf_threshold,
        iou=iou_threshold,
        save=True,
        save_txt=True,
        save_conf=True,
        project=output_dir,
        name='detect',
        verbose=True,
        augment=use_tta  # Use test-time augmentation
    )
    
    # Export results to a CSV file
    predictions = []
    
    for result in results:
        img_name = os.path.basename(result.path)
        boxes = result.boxes
        
        if len(boxes) > 0:
            # Extract coordinates, confidence, and class
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()  # xyxy format (top-left, bottom-right)
                conf = box.conf.item()
                cls = int(box.cls.item())
                class_name = model.names[cls]
                
                predictions.append({
                    'image_name': img_name,
                    'class': class_name,
                    'confidence': conf,
                    'xmin': x1,
                    'ymin': y1,
                    'xmax': x2,
                    'ymax': y2
                })
    
    # Create DataFrame and save
    if predictions:
        pred_df = pd.DataFrame(predictions)
        pred_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)
    
    print(f"Inference complete. Results saved to {output_dir}")
    return predictions

# Step 5: Improved multi-label evaluation metrics
def aggregate_predictions(predictions_csv, output_csv='image_level_predictions.csv'):
    """
    Aggregate bounding box predictions to image-level class predictions.
    
    Args:
        predictions_csv: path to CSV file with bounding box predictions
        output_csv: path to save image-level predictions
    """
    # Load predictions
    pred_df = pd.read_csv(predictions_csv)
    
    # Group by image and aggregate classes
    image_level = pred_df.groupby('image_name')['class'].apply(lambda x: list(set(x))).reset_index()
    image_level.rename(columns={'class': 'predicted_classes'}, inplace=True)
    
    # Add confidence scores per class
    class_confidences = {}
    for img_name, img_group in pred_df.groupby('image_name'):
        class_confidences[img_name] = img_group.groupby('class')['confidence'].max().to_dict()
    
    image_level['class_confidences'] = image_level['image_name'].map(class_confidences)
    
    # Save to CSV
    image_level.to_csv(output_csv, index=False)
    
    print(f"Aggregated predictions saved to {output_csv}")
    return image_level

def evaluate_multilabel(ground_truth_csv, predictions_csv, output_dir=None):
    """
    Evaluate multi-label classification performance.
    
    Args:
        ground_truth_csv: CSV with ground truth labels (columns: image_name, true_classes)
        predictions_csv: CSV with predictions (columns: image_name, predicted_classes)
        output_dir: directory to save evaluation results
    
    Returns:
        metrics: dictionary of evaluation metrics
    """
    # Load data
    gt_df = pd.read_csv(ground_truth_csv)
    pred_df = pd.read_csv(predictions_csv)
    
    # Ensure classes are lists if they're stored as strings
    if isinstance(gt_df['true_classes'].iloc[0], str):
        gt_df['true_classes'] = gt_df['true_classes'].apply(eval)  # Convert string representation to list
    
    if isinstance(pred_df['predicted_classes'].iloc[0], str):
        pred_df['predicted_classes'] = pred_df['predicted_classes'].apply(eval)
    
    # Get all unique classes
    all_classes = set()
    for classes in gt_df['true_classes']:
        all_classes.update(classes)
    
    # Merge datasets on image_name
    merged = pd.merge(gt_df, pred_df, on='image_name', how='inner')
    
    # Prepare true and predicted labels in binary format
    y_true = []
    y_pred = []
    
    for _, row in merged.iterrows():
        true_vector = [1 if c in row['true_classes'] else 0 for c in all_classes]
        pred_vector = [1 if c in row['predicted_classes'] else 0 for c in all_classes]
        
        y_true.append(true_vector)
        y_pred.append(pred_vector)
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate metrics
    metrics = {
        'exact_match_ratio': np.mean([np.array_equal(true, pred) for true, pred in zip(y_true, y_pred)]),
        'hamming_loss': np.mean([np.mean(true != pred) for true, pred in zip(y_true, y_pred)]),
        'subset_accuracy': np.mean([np.all(true == pred) for true, pred in zip(y_true, y_pred)]),
        'f1_score_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_score_micro': f1_score(y_true, y_pred, average='micro'),
        'f1_score_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro'),
        'precision_micro': precision_score(y_true, y_pred, average='micro'),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted'),
        'recall_macro': recall_score(y_true, y_pred, average='macro'),
        'recall_micro': recall_score(y_true, y_pred, average='micro'),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted')
    }
    
    # Calculate per-class metrics
    class_metrics = {}
    classes_list = list(all_classes)
    
    for i, class_name in enumerate(classes_list):
        class_metrics[class_name] = {
            'f1_score': f1_score(y_true[:, i], y_pred[:, i], average='binary'),
            'precision': precision_score(y_true[:, i], y_pred[:, i], average='binary'),
            'recall': recall_score(y_true[:, i], y_pred[:, i], average='binary')
        }
    
    # Print metrics
    print("\nOverall Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nPer-class Metrics:")
    for class_name, class_metric in class_metrics.items():
        print(f"{class_name}:")
        for metric, value in class_metric.items():
            print(f"  {metric}: {value:.4f}")
    
    # Create confusion matrix per class
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(os.path.join(output_dir, 'overall_metrics.csv'), index=False)
        
        # Save per-class metrics
        class_metrics_df = pd.DataFrame.from_dict(class_metrics, orient='index')
        class_metrics_df.to_csv(os.path.join(output_dir, 'per_class_metrics.csv'))
        
        # Plot confusion matrices
        for i, class_name in enumerate(classes_list):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Negative', 'Positive'], 
                        yticklabels=['Negative', 'Positive'])
            plt.title(f'Confusion Matrix - {class_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'confusion_matrix_{class_name}.png'))
            plt.close()
    
    return metrics, class_metrics
def run_improved_yolo_pipeline(train_df, base_output_dir='improved_yolo_dataset', 
                             model_name_path='yolov8s.pt', test_df=None,
                             epochs=100, use_ensemble=True, apply_augmentation=True,
                             plot_performance=True):
    """
    Run the complete improved YOLO pipeline for multi-label classification with class imbalance.
    
    Args:
        train_df: pandas DataFrame with the training dataset
        base_output_dir: base directory for outputs
        model_name_path: path to the base model
        test_df: optional separate test DataFrame (same format as train_df)
        epochs: number of training epochs
        use_ensemble: whether to use ensemble methods
        apply_augmentation: whether to apply data augmentation
        plot_performance: whether to plot training performance metrics
    """
    # Import necessary modules at the beginning
    import os
    
    print("Starting improved YOLO pipeline for multi-label classification with class imbalance...")
    
    # Create output directory
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Step 1: Process dataset and handle class imbalance
    print("\n--- Step 1: Processing dataset and handling class imbalance ---")
    classes, class_weights = process_dataset(
        train_df, 
        output_dir=base_output_dir, 
        test_df=test_df,
        split_ratio=0.8,
        apply_augmentation=apply_augmentation
    )
    
    # Create YAML config with class weights
    create_yaml_config(classes, output_dir=base_output_dir, class_weights=class_weights)
    
    # Check that directories exist and contain files
    for split in ['train', 'val']:
        img_dir = os.path.join(base_output_dir, split, 'images')
        if not os.path.exists(img_dir) or len(os.listdir(img_dir)) == 0:
            print(f"WARNING: {img_dir} is empty or doesn't exist! Training may fail.")
    
    # Verify test directory if it should exist
    if test_df is not None and not test_df.empty:
        test_img_dir = os.path.join(base_output_dir, 'test', 'images')
        if not os.path.exists(test_img_dir) or len(os.listdir(test_img_dir)) == 0:
            print(f"WARNING: {test_img_dir} is empty or doesn't exist! Inference may be skipped.")
            
    # Dataset YAML path
    dataset_yaml = os.path.join(base_output_dir, 'dataset.yaml')
    
    # Dictionary to store results for performance visualization
    performance_results = {}
    
    if use_ensemble:
        # Train multiple models for ensemble
        model_paths = []
        model_result_dirs = []
        
        # Model 1: Original model with class weights
        print("\nTraining Model 1 (Original with class weights)...")
        results1, bestmodel1 = train_yolov11(  # Fixed function name from train_yolov11
            dataset_yaml, 
            model_name_path, 
            class_weights=class_weights,
            model_size='s',
            epochs=epochs,
            learning_rate=0.01
        )
        model_paths.append(model_name_path)  # Add the model object to the list
        
        # Store results directory for visualization
        # Assuming results1 contains a path to the results
        model1_results_dir = results1
        model_result_dirs.append(model1_results_dir)
        
        # Model 2: Different size
        print("\nTraining Model 2 (Different architecture)...")
        model2_path = 'yolov8m.pt' if model_name_path == 'yolov8s.pt' else 'yolov8s.pt'
        results2, bestmodel2 = train_yolov11(  # Fixed function name
            dataset_yaml, 
            model2_path,  # Use the correct variable name
            class_weights=class_weights,
            model_size='m' if model_name_path == 'yolov8s.pt' else 's',
            epochs=int(epochs * 0.7),  # Fewer epochs
            learning_rate=0.005  # Lower learning rate
        )
        model_paths.append(model2_path)  # Use the best model path returned by the function
        
        # Store results directory for visualization
        # Assuming results2 contains a path to the results
        model2_results_dir = results2
        model_result_dirs.append(model2_results_dir)
        
        # Model 3: Different image size
        print("\nTraining Model 3 (Different image size)...")
        results3, bestmodel3 = train_yolov11(  # Fixed function name
            dataset_yaml, 
            model_name_path,
            class_weights=class_weights,
            model_size='s',
            epochs=int(epochs * 0.7),
            image_size=800,  # Different image size
            learning_rate=0.01
        )
        model_paths.append(model_name_path)  # Use the best model path returned by the function
        
        # Store results directory for visualization
        # Assuming results3 contains a path to the results
        model3_results_dir = results3
        model_result_dirs.append(model3_results_dir)
        
        # Create ensemble
        print("\nCreating ensemble model...")
        ensemble_model = create_ensemble(
            model_paths, 
            strategy='weighted_voting',
            weights=[0.5, 0.25, 0.25]  # Give more weight to the first model
        )
        
        # Step 3: Run inference with the ensemble
        print("\n--- Step 3: Running ensemble inference ---")
        test_image_dir = os.path.join(base_output_dir, 'test', 'images')
        
        # Check if test directory has images
        if os.path.exists(test_image_dir) and len(os.listdir(test_image_dir)) > 0:
            ensemble_output_dir = os.path.join(base_output_dir, 'ensemble_predictions')
            os.makedirs(ensemble_output_dir, exist_ok=True)  # Create output directory
            
            predictions = run_ensemble_inference(
                ensemble_model,
                test_image_dir,
                output_dir=ensemble_output_dir,
                conf_threshold=0.25,
                iou_threshold=0.45
            )
            
            # Aggregate to image-level for multi-label evaluation if test images were processed
            image_level_preds = aggregate_predictions(
                os.path.join(ensemble_output_dir, 'ensemble_predictions.csv'),
                os.path.join(ensemble_output_dir, 'image_level_predictions.csv')
            )
        else:
            print("No test images found. Skipping inference step.")
        
        # Generate performance plots if enabled
        if plot_performance:
            print("\n--- Step 4: Visualizing model performance ---")
            performance_output_dir = os.path.join(base_output_dir, 'performance_plots')
            os.makedirs(performance_output_dir, exist_ok=True)  # Create output directory
            
            for i, results_dir in enumerate(model_result_dirs):
                model_name = f"Model_{i+1}"
                if i == 0:
                    model_name = "Original_Model"
                elif i == 1:
                    model_name = "Different_Architecture"
                elif i == 2:
                    model_name = "Different_Image_Size"
                
                model_output_dir = os.path.join(performance_output_dir, model_name)
                os.makedirs(model_output_dir, exist_ok=True)  # Create model-specific output directory
                
                try:
                    print(f"Generating performance plots for {model_name}... in {model_output_dir}")
                    performance_metrics = plot_yolo_performance(
                        results_dir=results_dir,
                        save_plots=True,
                        output_dir=model_output_dir,
                        focus_metric='mAP50'
                    )
                    
                    if performance_metrics:
                        performance_results[model_name] = performance_metrics
                except Exception as e:
                    print(f"Error generating plots for {model_name}: {e}")
    
    else:
        # Train single model with optimizations
        print("\nTraining optimized model...")
        results, best_model = train_yolov11(  # Fixed function name
            dataset_yaml, 
            model_name_path, 
            class_weights=class_weights,
            model_size='s',
            epochs=epochs,
            learning_rate=0.01
        )
        
        # Get results directory for visualization
        # Assuming results contains a path to the results
        model_results_dir = results
        
        # Step 3: Run inference with optimizations
        print("\n--- Step 3: Running optimized inference ---")
        test_image_dir = os.path.join(base_output_dir, 'test', 'images')
        
        # Check if test directory has images
        if os.path.exists(test_image_dir) and len(os.listdir(test_image_dir)) > 0:
            output_dir = os.path.join(base_output_dir, 'predictions')
            os.makedirs(output_dir, exist_ok=True)  # Create output directory
            
            predictions = run_inference(
                best_model,
                test_image_dir,
                output_dir=output_dir,
                conf_threshold=0.25,
                iou_threshold=0.45,
                use_tta=True  # Use test-time augmentation
            )
            
            # Aggregate to image-level for multi-label evaluation if test images were processed
            image_level_preds = aggregate_predictions(
                os.path.join(output_dir, 'predictions.csv'),
                os.path.join(output_dir, 'image_level_predictions.csv')
            )
        else:
            print("No test images found. Skipping inference step.")
            
        # Generate performance plots if enabled
        if plot_performance:
            print("\n--- Step 4: Visualizing model performance ---")
            performance_output_dir = os.path.join(base_output_dir, 'performance_plots')
            os.makedirs(performance_output_dir, exist_ok=True)  # Create output directory
            
            try:
                print(f"Generating performance plots... in {model_results_dir}")
                performance_metrics = plot_yolo_performance(
                    results_dir=model_results_dir,
                    save_plots=True,
                    output_dir=performance_output_dir,
                    focus_metric='mAP50'
                )
                
                if performance_metrics:
                    performance_results["Single_Model"] = performance_metrics
            except Exception as e:
                print(f"Error generating plots: {e}")
    
    # If we have multiple models, create a comparison plot
    if plot_performance and len(performance_results) > 1:
        print("\nCreating model comparison plot...")
        try:
            compare_models_performance(
                performance_results, 
                output_dir=os.path.join(base_output_dir, 'performance_plots'),
                save_plot=True
            )
        except Exception as e:
            print(f"Error creating comparison plot: {e}")
    
    print("\nYOLO pipeline completed successfully!")
    print(f"All outputs saved to {os.path.abspath(base_output_dir)}")
    
    # Print final directories structure for verification
    print("\nFinal directory structure:")
    for split in ['train', 'val', 'test']:
        for subdir in ['images', 'labels']:
            dir_path = os.path.join(base_output_dir, split, subdir)
            if os.path.exists(dir_path):
                file_count = len(os.listdir(dir_path))
                print(f"{dir_path}: {file_count} files")
            else:
                print(f"{dir_path}: directory not found")
    
    return performance_results

def compare_models_performance(performance_results, output_dir='performance_plots', save_plot=True):
    """
    Create a comparison plot of multiple models' performance metrics.
    
    Args:
        performance_results: Dictionary with model names as keys and performance metrics as values
        output_dir: Directory to save the comparison plot
        save_plot: Whether to save the plot to disk
    """
    if not performance_results:
        print("No performance results to compare.")
        return
    
    # Create the output directory if it doesn't exist
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics for comparison
    model_names = list(performance_results.keys())
    map50_values = [results['mAP50'] for results in performance_results.values()]
    best_map50_values = [results['best_mAP50'] for results in performance_results.values()]
    precision_values = [results['precision'] for results in performance_results.values()]
    recall_values = [results['recall'] for results in performance_results.values()]
    
    # Create a figure with subplots
    fig, axs = plt.subplots(2, 1, figsize=(12, 10))
    
    # Bar chart for mAP@0.5 metrics
    x = np.arange(len(model_names))
    width = 0.35
    
    axs[0].bar(x - width/2, map50_values, width, label='Final mAP@0.5')
    axs[0].bar(x + width/2, best_map50_values, width, label='Best mAP@0.5')
    
    axs[0].set_xlabel('Model')
    axs[0].set_ylabel('mAP@0.5')
    axs[0].set_title('Model Performance Comparison - mAP@0.5')
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(model_names)
    axs[0].legend()
    
    # Add value labels on top of the bars
    for i, v in enumerate(map50_values):
        axs[0].text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    for i, v in enumerate(best_map50_values):
        axs[0].text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    # Bar chart for precision and recall
    axs[1].bar(x - width/2, precision_values, width, label='Precision')
    axs[1].bar(x + width/2, recall_values, width, label='Recall')
    
    axs[1].set_xlabel('Model')
    axs[1].set_ylabel('Value')
    axs[1].set_title('Model Performance Comparison - Precision & Recall')
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(model_names)
    axs[1].legend()
    
    # Add value labels on top of the bars
    for i, v in enumerate(precision_values):
        axs[1].text(i - width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    for i, v in enumerate(recall_values):
        axs[1].text(i + width/2, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    
    if save_plot:
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
        plt.close()
    else:
        plt.show()

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MaxNLocator
from pathlib import Path

def plot_yolo_performance(results_dir, save_plots=False, output_dir='performance_plots', focus_metric='mAP50'):
    """
    Generate comprehensive performance plots for YOLOv11 training results,
    with special focus on mAP@0.5 (IoU threshold of 0.5).
    
    Args:
        results_dir: Directory containing YOLO training results
        save_plots: Whether to save plots to disk (False by default)
        output_dir: Directory to save performance plots (only used if save_plots=True)
        focus_metric: Primary metric to highlight in plots ('mAP50' by default)
    """
    # Create output directory if saving plots
    if save_plots:
        os.makedirs(output_dir, exist_ok=True)
    
    # Check if it's Ultralytics format (YOLOv8/YOLOv11) or older YOLOv5 format
    results_csv = os.path.join(results_dir, 'results.csv')
    
    if os.path.exists(results_csv):
        # Newer Ultralytics format
        df = pd.read_csv(results_csv)
        
        # Check if we have precision-recall data
        if all(col in df.columns for col in ['precision', 'recall', 'mAP50', 'mAP50-95']):
            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('YOLO Training Performance Metrics\nPrimary Evaluation: mAP@0.5 (IoU=0.5)', fontsize=16)
            
            # Plot Loss curves
            ax = axes[0, 0]
            if 'box_loss' in df.columns and 'cls_loss' in df.columns:
                ax.plot(df['epoch'], df['box_loss'], label='Box Loss')
                ax.plot(df['epoch'], df['cls_loss'], label='Class Loss')
                if 'dfl_loss' in df.columns:  # YOLOv8/v11 specific
                    ax.plot(df['epoch'], df['dfl_loss'], label='DFL Loss')
            else:
                ax.plot(df['epoch'], df['train_loss'], label='Train Loss')
                if 'val_loss' in df.columns:
                    ax.plot(df['epoch'], df['val_loss'], label='Val Loss')
            
            ax.set_title('Training Losses')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Plot Precision, Recall
            ax = axes[0, 1]
            ax.plot(df['epoch'], df['precision'], label='Precision')
            ax.plot(df['epoch'], df['recall'], label='Recall')
            ax.set_title('Precision and Recall (at IoU=0.5)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Plot mAP values with mAP@0.5 highlighted
            ax = axes[1, 0]
            # Plot mAP@0.5 with thicker line and different color
            ax.plot(df['epoch'], df['mAP50'], 'r-', linewidth=3, label='mAP@0.5 (Primary Metric)')
            ax.plot(df['epoch'], df['mAP50-95'], 'b-', linewidth=1.5, label='mAP@0.5:0.95')
            ax.set_title('Mean Average Precision (mAP)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('mAP')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Add a horizontal line at the best mAP@0.5 value
            best_map50 = df['mAP50'].max()
            best_epoch = df.loc[df['mAP50'].idxmax(), 'epoch']
            ax.axhline(y=best_map50, color='r', linestyle='--', alpha=0.5)
            ax.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
            ax.annotate(f'Best mAP@0.5: {best_map50:.4f} (Epoch {best_epoch:.0f})', 
                        xy=(best_epoch, best_map50),
                        xytext=(best_epoch+2, best_map50-0.02),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                        fontsize=9)
            
            # Plot learning rate if available, otherwise fitness
            ax = axes[1, 1]
            if 'lr0' in df.columns:
                ax.plot(df['epoch'], df['lr0'], label='Learning Rate')
                ax.set_title('Learning Rate Schedule')
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Learning Rate')
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)
            else:
                # If learning rate is not available, plot fitness if available
                if 'fitness' in df.columns:
                    ax.plot(df['epoch'], df['fitness'], 'g-', label='Fitness')
                    ax.set_title('Model Fitness')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('Fitness Value')
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.6)
                else:
                    # Create a special mAP@0.5 focus plot
                    ax.plot(df['epoch'], df['mAP50'], 'ro-')
                    ax.set_title('mAP@0.5 Progress (IoU=0.5)')
                    ax.set_xlabel('Epoch')
                    ax.set_ylabel('mAP@0.5')
                    best_map50 = df['mAP50'].max()
                    ax.axhline(y=best_map50, color='g', linestyle='--', alpha=0.5)
                    ax.text(df['epoch'].max()/2, best_map50*1.01, f'Best: {best_map50:.4f}', 
                            horizontalalignment='center')
                    ax.grid(True, linestyle='--', alpha=0.6)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for main title
            
            # Save or display
            if save_plots:
                plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=300)
                plt.close()
            else:
                plt.show()
            
            # Create a dedicated mAP@0.5 (IoU=0.5) plot
            plt.figure(figsize=(10, 6))
            plt.plot(df['epoch'], df['mAP50'], 'ro-', linewidth=2)
            plt.title('mAP@0.5 (IoU=0.5) Training Progress', fontsize=14)
            plt.xlabel('Epoch')
            plt.ylabel('mAP@0.5')
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Add rolling average to smooth the curve
            window_size = min(5, len(df))
            if window_size > 1:
                rolling_avg = df['mAP50'].rolling(window=window_size).mean()
                plt.plot(df['epoch'][window_size-1:], rolling_avg[window_size-1:], 'b-', 
                         linewidth=1.5, label=f'{window_size}-epoch Moving Average')
            
            # Add best point annotation
            plt.axhline(y=best_map50, color='g', linestyle='--', alpha=0.5)
            plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
            plt.annotate(f'Best mAP@0.5: {best_map50:.4f} (Epoch {best_epoch:.0f})', 
                        xy=(best_epoch, best_map50),
                        xytext=(best_epoch*(0.9 if best_epoch > df['epoch'].max()/2 else 1.1), 
                                best_map50*(0.95 if best_map50 > df['mAP50'].median() else 1.05)),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                        fontsize=10)
            
            plt.legend()
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(os.path.join(output_dir, 'map50_progress.png'), dpi=300)
                plt.close()
            else:
                plt.show()
            
            # Create PR curve plot if we have confidence data
            pr_curve_file = Path(results_dir) / 'PR_curve.png'
            if pr_curve_file.exists():
                pr_img = plt.imread(pr_curve_file)
                plt.figure(figsize=(10, 8))
                plt.imshow(pr_img)
                plt.axis('off')
                plt.title('Precision-Recall Curve (IoU=0.5)', fontsize=14)
                plt.tight_layout()
                
                if save_plots:
                    plt.savefig(os.path.join(output_dir, 'pr_curve.png'), dpi=300)
                    plt.close()
                else:
                    plt.show()
            
            # Create F1 score vs confidence threshold plot if available
            f1_curve_file = Path(results_dir) / 'F1_curve.png'
            if f1_curve_file.exists():
                f1_img = plt.imread(f1_curve_file)
                plt.figure(figsize=(10, 8))
                plt.imshow(f1_img)
                plt.axis('off')
                plt.title('F1 Score vs Confidence Threshold (IoU=0.5)', fontsize=14)
                plt.tight_layout()
                
                if save_plots:
                    plt.savefig(os.path.join(output_dir, 'f1_curve.png'), dpi=300)
                    plt.close()
                else:
                    plt.show()
            
            # Print final metrics with highlight on mAP@0.5
            final_epoch = df.iloc[-1]
            print("\n" + "="*50)
            print("FINAL TRAINING METRICS")
            print("="*50)
            print(f"Precision (IoU=0.5): {final_epoch['precision']:.4f}")
            print(f"Recall (IoU=0.5): {final_epoch['recall']:.4f}")
            print(f"→ mAP@0.5 (IoU=0.5): {final_epoch['mAP50']:.4f} ←")
            print(f"mAP@0.5:0.95: {final_epoch['mAP50-95']:.4f}")
            print(f"Best mAP@0.5: {best_map50:.4f} (Epoch {best_epoch:.0f})")
            print("="*50)
            
            return {
                'precision': final_epoch['precision'],
                'recall': final_epoch['recall'],
                'mAP50': final_epoch['mAP50'],
                'mAP50-95': final_epoch['mAP50-95'],
                'best_mAP50': best_map50,
                'best_epoch': best_epoch
            }
        else:
            print("Warning: CSV file does not contain expected metrics columns.")
    
    # Check for older YOLOv5 format results.txt
    results_txt = os.path.join(results_dir, 'results.txt')
    if os.path.exists(results_txt):
        # Parse results.txt in YOLOv5 format
        epochs, box_loss, obj_loss, cls_loss = [], [], [], []
        precision, recall, map50, map = [], [], [], []
        
        with open(results_txt, 'r') as f:
            for line in f:
                if line.startswith('  Epoch'):
                    continue  # Skip header line
                try:
                    # Extract metrics (format may vary)
                    parts = line.strip().split()
                    if len(parts) >= 12:
                        epoch = int(parts[0])
                        epochs.append(epoch)
                        
                        # Extract losses
                        box_loss.append(float(parts[2]))
                        obj_loss.append(float(parts[3]))
                        cls_loss.append(float(parts[4]))
                        
                        # Extract precision, recall, mAP
                        precision.append(float(parts[8]))
                        recall.append(float(parts[9]))
                        map50.append(float(parts[10]))
                        map.append(float(parts[11]))
                except Exception as e:
                    print(f"Error parsing line: {line}")
                    print(f"Exception: {e}")
        
        if epochs:
            # Create figure with multiple subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('YOLO Training Performance Metrics\nPrimary Evaluation: mAP@0.5 (IoU=0.5)', fontsize=16)
            
            # Plot Loss curves
            ax = axes[0, 0]
            ax.plot(epochs, box_loss, label='Box Loss')
            ax.plot(epochs, obj_loss, label='Objectness Loss')
            ax.plot(epochs, cls_loss, label='Class Loss')
            ax.set_title('Training Losses')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Plot Precision, Recall
            ax = axes[0, 1]
            ax.plot(epochs, precision, label='Precision')
            ax.plot(epochs, recall, label='Recall')
            ax.set_title('Precision and Recall (at IoU=0.5)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Value')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Plot mAP values with mAP@0.5 highlighted
            ax = axes[1, 0]
            # Plot mAP@0.5 with thicker line and different color
            ax.plot(epochs, map50, 'r-', linewidth=3, label='mAP@0.5 (Primary Metric)')
            ax.plot(epochs, map, 'b-', linewidth=1.5, label='mAP@0.5:0.95')
            ax.set_title('Mean Average Precision (mAP)')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('mAP')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Add a horizontal line at the best mAP@0.5 value
            best_map50 = max(map50)
            best_epoch_idx = map50.index(best_map50)
            best_epoch = epochs[best_epoch_idx]
            ax.axhline(y=best_map50, color='r', linestyle='--', alpha=0.5)
            ax.axvline(x=best_epoch, color='r', linestyle='--', alpha=0.5)
            ax.annotate(f'Best mAP@0.5: {best_map50:.4f} (Epoch {best_epoch})', 
                        xy=(best_epoch, best_map50),
                        xytext=(best_epoch+2, best_map50-0.02),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                        fontsize=9)
            
            # Plot combined loss
            ax = axes[1, 1]
            total_loss = [b + o + c for b, o, c in zip(box_loss, obj_loss, cls_loss)]
            ax.plot(epochs, total_loss, label='Total Loss')
            ax.set_title('Total Training Loss')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for main title
            
            if save_plots:
                plt.savefig(os.path.join(output_dir, 'training_metrics.png'), dpi=300)
                plt.close()
            else:
                plt.show()
            
            # Create a dedicated mAP@0.5 (IoU=0.5) plot
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, map50, 'ro-', linewidth=2)
            plt.title('mAP@0.5 (IoU=0.5) Training Progress', fontsize=14)
            plt.xlabel('Epoch')
            plt.ylabel('mAP@0.5')
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # Add rolling average to smooth the curve if we have enough data points
            window_size = min(5, len(epochs))
            if window_size > 1:
                # Calculate rolling average manually
                rolling_avg = []
                for i in range(window_size-1, len(map50)):
                    avg = sum(map50[i-(window_size-1):i+1]) / window_size
                    rolling_avg.append(avg)
                
                # Plot rolling average
                plt.plot(epochs[window_size-1:], rolling_avg, 'b-', 
                         linewidth=1.5, label=f'{window_size}-epoch Moving Average')
            
            # Add best point annotation
            plt.axhline(y=best_map50, color='g', linestyle='--', alpha=0.5)
            plt.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
            plt.annotate(f'Best mAP@0.5: {best_map50:.4f} (Epoch {best_epoch})', 
                        xy=(best_epoch, best_map50),
                        xytext=(best_epoch*(0.9 if best_epoch > epochs[-1]/2 else 1.1), 
                                best_map50*(0.95 if best_map50 > sum(map50)/len(map50) else 1.05)),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5),
                        fontsize=10)
            
            plt.legend()
            plt.tight_layout()
            
            if save_plots:
                plt.savefig(os.path.join(output_dir, 'map50_progress.png'), dpi=300)
                plt.close()
            else:
                plt.show()
            
            # Print final metrics with highlight on mAP@0.5
            print("\n" + "="*50)
            print("FINAL TRAINING METRICS")
            print("="*50)
            print(f"Precision (IoU=0.5): {precision[-1]:.4f}")
            print(f"Recall (IoU=0.5): {recall[-1]:.4f}")
            print(f"→ mAP@0.5 (IoU=0.5): {map50[-1]:.4f} ←")
            print(f"mAP@0.5:0.95: {map[-1]:.4f}")
            print(f"Best mAP@0.5: {best_map50:.4f} (Epoch {best_epoch})")
            print("="*50)
            
            return {
                'precision': precision[-1],
                'recall': recall[-1],
                'mAP50': map50[-1],
                'mAP50-95': map[-1],
                'best_mAP50': best_map50,
                'best_epoch': best_epoch
            }
        
# Example usage
if __name__ == "__main__":
    # Load your training data
    # Example format:
    # train_df = pd.read_csv('your_train_annotations.csv')
    # 
    # # Optional: Load separate test data
    # test_df = pd.read_csv('your_test_annotations.csv')
    # 
    # # Run with provided test set and data augmentation
    # run_improved_yolo_pipeline(
    #     train_df, 
    #     model_name_path='yolov8s.pt', 
    #     test_df=test_df,
    #     epochs=100, 
    #     use_ensemble=True, 
    #     apply_augmentation=True
    # )
    
    # Or run without a separate test set
    # run_improved_yolo_pipeline(
    #     train_df, 
    #     model_name_path='yolov8s.pt', 
    #     epochs=100, 
    #     use_ensemble=True, 
    #     apply_augmentation=True
    # )
    
    print("Import the module and run run_improved_yolo_pipeline() with your data")