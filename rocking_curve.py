import os
import fabio
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.util import view_as_blocks



def load_edf_images(directory, pat, scan=00, nb=31):
    # Load images, normalize them, and log-transform the data
    images = [np.log(fabio.open(pat.format(i=i, scan=scan)).data) for i in range(nb)]
    return np.array(images)

def process_images(images):
    processed_images = []
    for image in images:
        # Compute mean intensity
        mean_intensity = np.mean(image)
        
        # Subtract mean intensity to calculate background
        background = image - mean_intensity
        
        # Ensure non-negative values
        background[background < 0] = 0
        
        # Normalize the background
        normalized = background / np.max(background)
        
        # Append the processed image
        processed_images.append(normalized)
    
    return np.array(processed_images)
    

def load_images(directory):
    """
    Load images from a directory containing .edf or .h5 files.
    
    Parameters:
        directory (str): Directory containing .edf and/or .h5 files.
    
    Returns:
        list: A list of images (numpy arrays).
    """
    images = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filename.endswith(".edf"):
            edf_image = np.log(fabio.open(file_path).data)
            images.append(edf_image)
        elif filename.endswith(".h5"):
            with fabio.open_series(first_filename=file_path) as series:
                for frame in series.frames():
                    filtered_data = frame.data
                    images.append(np.log(filtered_data))  # Apply log to the data
    return images


def crop_image_to_fit(image, patch_size):
    """
    Crop the image to ensure its dimensions are divisible by the patch size.
    
    Parameters:
        image (np.array): Input image.
        patch_size (tuple): Size of the patches (height, width).
    
    Returns:
        np.array: Cropped image.
    """
    new_height = (image.shape[0] // patch_size[0]) * patch_size[0]
    new_width = (image.shape[1] // patch_size[1]) * patch_size[1]
    return image[:new_height, :new_width]


def extract_patches(image, patch_size):
    """
    Extract patches of a given size from an image.
    
    Parameters:
        image (np.array): Input image.
        patch_size (tuple): Size of each patch (height, width).
    
    Returns:
        tuple: Patches as an array and the cropped image shape.
    """
    cropped_image = crop_image_to_fit(image, patch_size)
    patches = view_as_blocks(cropped_image, block_shape=patch_size)
    patches = patches.reshape(-1, patch_size[0], patch_size[1])
    return patches, cropped_image.shape


def classify_patches(patches, model, mean, std, device):
    """
    Classify patches using a trained model.
    
    Parameters:
        patches (np.array): Patches to classify.
        model (torch.nn.Module): Trained PyTorch model.
        mean (float): Mean for normalization.
        std (float): Standard deviation for normalization.
        device (torch.device): Device to run the model on.
    
    Returns:
        list: Labels predicted for each patch.
    """
    model.eval()
    classified_labels = []
    with torch.no_grad():
        for patch in patches:
            # Convert patch to float32 and normalize using mean and std
            patch = (patch.astype(np.float32) - mean) / std
            
            # Convert patch to tensor and add channel and batch dimensions
            patch_tensor = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            
            # Pass through the model
            output = model(patch_tensor)
            _, predicted = torch.max(output, 1)
            classified_labels.append(predicted.item())
    return classified_labels


def reconstruct_image_with_background(patches, labels, original_shape, patch_size, background_intensity):
    """
    Reconstruct an image with patches replaced by background intensity based on labels.
    
    Parameters:
        patches (np.array): Patches to use for reconstruction.
        labels (list): Labels for each patch.
        original_shape (tuple): Shape of the original image.
        patch_size (tuple): Size of the patches.
        background_intensity (float): Intensity value for background patches.
    
    Returns:
        np.array: Reconstructed image.
    """
    reconstructed_image = np.zeros(original_shape, dtype=np.float32)
    patch_idx = 0
    for i in range(0, original_shape[0], patch_size[0]):
        for j in range(0, original_shape[1], patch_size[1]):
            if labels[patch_idx] == 0:  # Use weak beam patches
                reconstructed_image[i:i + patch_size[0], j:j + patch_size[1]] = patches[patch_idx]
            else:  # Replace with background intensity patch
                reconstructed_image[i:i + patch_size[0], j:j + patch_size[1]] = background_intensity
            patch_idx += 1
    return reconstructed_image


def normalize_image(image):
    """
    Normalize an image to the range [0, 255].
    
    Parameters:
        image (np.array): Image to normalize.
    
    Returns:
        np.array: Normalized image.
    """
    image -= np.min(image)  # Shift to make the minimum 0
    image /= np.max(image)  # Scale to make the maximum 1
    image *= 255  # Scale to the range [0, 255]
    return image.astype(np.uint8)


