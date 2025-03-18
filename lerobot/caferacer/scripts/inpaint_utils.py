import modal
from diffusers.utils import load_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL import ImageDraw
import cv2
from tqdm.notebook import tqdm
from typing import List, Optional, Dict, Union
import os
import subprocess
import torch
import inspect

from lerobot.caferacer.scripts.image_utils import tensor_to_pil, display_images

def create_inpaint_mask(
    im: Union[Image.Image, np.ndarray],
    direction: str,
) -> tuple[Image.Image, Image.Image]:
    """
    Create a mask for inpainting a corner region of the image
    
    Args:
        im: Input image (PIL Image or numpy array)
        direction: Corner to mask - 'top-left', 'top-right', 'bottom-left', 'bottom-right'
    
    Returns:
        tuple: (mask, overlaid_image) where overlaid_image shows the masked region in white
    """
    # Convert numpy array to PIL if needed
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    
    width, height = im.size
    corner_width = width // 4
    corner_height = height // 4
    
    # Create mask (black background with white corner)
    mask = Image.new('RGB', (width, height), 'black')
    draw = ImageDraw.Draw(mask)
    
    # Define corner coordinates based on direction
    if direction == 'top-left':
        coords = [0, 0, corner_width, corner_height]
    elif direction == 'top-right':
        coords = [width - corner_width, 0, width, corner_height]
    elif direction == 'bottom-left':
        coords = [0, height - corner_height, corner_width, height]
    elif direction == 'bottom-right':
        coords = [width - corner_width, height - corner_height, width, height]
    else:
        raise ValueError("Direction must be one of: 'top-left', 'top-right', 'bottom-left', 'bottom-right'")
    
    # Draw white rectangle for the corner in mask
    draw.rectangle(coords, fill='white')
    
    # Create overlaid image by copying original and drawing white rectangle
    overlaid_im = im.copy()
    draw = ImageDraw.Draw(overlaid_im)
    draw.rectangle(coords, fill='white')
    
    return mask, overlaid_im

def create_outpaint_mask(
    im: Image.Image,
    direction: str,
    plot: bool = True
) -> tuple[Image.Image, Image.Image, np.ndarray]:
    """
    Create mask for outpainting by extending image in specified direction
    
    Args:
        im: Input PIL Image (keeps original dimensions)
        direction: One of ['left-top', 'left', 'top', 'right-top', 
                         'right', 'right-bottom', 'bottom', 'left-bottom']
        plot: Whether to display visualization
    
    Returns:
        tuple: (modified_image, mask, overlay)
        - modified_image: Original image cropped and extended with white space
        - mask: White in new area, black in original area
        - overlay: Visualization with translucent overlay
    """
    # Get original dimensions
    width, height = im.size
    
    # Calculate crop/extend sizes (1/10 of image)
    shift_x = width // 10
    shift_y = height // 10
    
    # Create new white image with original dimensions
    new_im = Image.new('RGB', (width, height), 'white')
    
    # Create base mask (black)
    mask = Image.new('RGB', (width, height), 'black')
    
    # Process based on direction
    if 'left' in direction:
        paste_x = shift_x
    elif 'right' in direction:
        paste_x = -shift_x
    else:
        paste_x = 0
        
    if 'top' in direction:
        paste_y = shift_y
    elif 'bottom' in direction:
        paste_y = -shift_y
    else:
        paste_y = 0
    
    # Crop original image
    crop_box = (
        max(-paste_x, 0),
        max(-paste_y, 0),
        width - max(paste_x, 0),
        height - max(paste_y, 0)
    )
    cropped_im = im.crop(crop_box)
    
    # Paste cropped image into new white image
    paste_box = (
        max(paste_x, 0),
        max(paste_y, 0),
        width - max(-paste_x, 0),
        height - max(-paste_y, 0)
    )
    new_im.paste(cropped_im, paste_box)
    
    # Create mask (white in new area)
    white_box = []
    if 'left' in direction:
        white_box.append((0, 0, shift_x, height))
    if 'right' in direction:
        white_box.append((width-shift_x, 0, width, height))
    if 'top' in direction:
        white_box.append((0, 0, width, shift_y))
    if 'bottom' in direction:
        white_box.append((0, height-shift_y, width, shift_y))
    
    # Fill mask with white in new areas
    mask_draw = ImageDraw.Draw(mask)
    for box in white_box:
        mask_draw.rectangle(box, fill='white')
    
    return new_im, mask

def apply_filled_region(
    target_im: Image.Image,
    filled_im: Image.Image,
    direction: str,
    blur_width: int = 8,    # Increased from 4 to 8
    sigma: float = 3.0      # Increased from 1.0 to 2.0
) -> Image.Image:
    """
    Paste filled region with stronger Gaussian blur at boundary
    
    Args:
        target_im: Image to modify
        filled_im: Image with filled region
        direction: 'left', 'right', 'top', or 'bottom'
        blur_width: Width of blur region on each side of boundary
        sigma: Strength of the Gaussian blur
    """
    width, height = target_im.size
    shift_x = width // 10
    shift_y = height // 10
    
    # Do the basic paste first
    result_im = target_im.copy()
    
    if direction == 'left':
        filled_region = filled_im.crop((0, 0, shift_x, height))
        result_im.paste(filled_region, (0, 0))
        
        # Apply stronger Gaussian blur at boundary
        arr = np.array(result_im)
        boundary_region = arr[:, shift_x-blur_width:shift_x+blur_width]
        arr[:, shift_x-blur_width:shift_x+blur_width] = cv2.GaussianBlur(
            boundary_region, 
            (blur_width*2-1, blur_width*2-1),  # Kernel size must be odd
            sigma
        )
        result_im = Image.fromarray(arr)
        
    elif direction == 'right':
        filled_region = filled_im.crop((width-shift_x, 0, width, height))
        result_im.paste(filled_region, (width-shift_x, 0))
        
        arr = np.array(result_im)
        boundary_region = arr[:, (width-shift_x-blur_width):(width-shift_x+blur_width)]
        arr[:, (width-shift_x-blur_width):(width-shift_x+blur_width)] = cv2.GaussianBlur(
            boundary_region,
            (blur_width*2-1, blur_width*2-1),
            sigma
        )
        result_im = Image.fromarray(arr)
        
    elif direction == 'top':
        filled_region = filled_im.crop((0, 0, width, shift_y))
        result_im.paste(filled_region, (0, 0))
        
        arr = np.array(result_im)
        boundary_region = arr[shift_y-blur_width:shift_y+blur_width, :]
        arr[shift_y-blur_width:shift_y+blur_width, :] = cv2.GaussianBlur(
            boundary_region,
            (blur_width*2-1, blur_width*2-1),
            sigma
        )
        result_im = Image.fromarray(arr)
        
    elif direction == 'bottom':
        filled_region = filled_im.crop((0, height-shift_y, width, height))
        result_im.paste(filled_region, (0, height-shift_y))
        
        arr = np.array(result_im)
        boundary_region = arr[(height-shift_y-blur_width):(height-shift_y+blur_width), :]
        arr[(height-shift_y-blur_width):(height-shift_y+blur_width), :] = cv2.GaussianBlur(
            boundary_region,
            (blur_width*2-1, blur_width*2-1),
            sigma
        )
        result_im = Image.fromarray(arr)
    
    return result_im

def paste_wrapped_image(
    target_im: Image.Image,
    wrapped_im: Image.Image,
    mask: Image.Image,
    direction: str
) -> tuple[Image.Image, Image.Image]:
    """
    Paste wrapped trapezoid image into target image's masked region
    
    Args:
        target_im: Target image with masked region
        wrapped_im: Perspective-transformed image (with black background)
        mask: Original mask (white region indicates where to paste)
        direction: One of 'top-left', 'top-right', 'bottom-left', 'bottom-right'
    
    Returns:
        tuple: (modified_target, new_mask)
    """
    # Convert to PIL if needed
    if isinstance(target_im, np.ndarray):
        target_im = Image.fromarray(target_im)
    if isinstance(wrapped_im, np.ndarray):
        wrapped_im = Image.fromarray(wrapped_im)
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
    
    # Extract non-black regions from wrapped image
    wrapped_arr = np.array(wrapped_im)
    non_black = np.any(wrapped_arr > 10, axis=2)  # Threshold to handle compression artifacts
    coords = np.where(non_black)
    min_y, max_y = coords[0].min(), coords[0].max()
    min_x, max_x = coords[1].min(), coords[1].max()
    
    # Create RGBA version of wrapped image
    wrapped_rgba = Image.new('RGBA', wrapped_im.size, (0,0,0,0))
    wrapped_rgba.paste(wrapped_im)
    wrapped_data = np.array(wrapped_rgba)
    wrapped_data[~non_black] = [0,0,0,0]  # Make black pixels transparent
    trapezoid = Image.fromarray(wrapped_data)
    
    # Find mask coordinates (white region)
    mask_arr = np.array(mask)
    white_pixels = np.all(mask_arr == 255, axis=2)
    mask_coords = np.where(white_pixels)
    mask_min_y, mask_max_y = mask_coords[0].min(), mask_coords[0].max()
    mask_min_x, mask_max_x = mask_coords[1].min(), mask_coords[1].max()
    
    # Create result image with white masked region
    result_im = target_im.copy()
    draw = ImageDraw.Draw(result_im)
    draw.rectangle([mask_min_x, mask_min_y, mask_max_x, mask_max_y], fill='white')
    
    # Calculate paste coordinates based on direction
    if 'top' in direction:
        paste_y = mask_min_y  # Align to top
        if 'left' in direction:
            paste_x = mask_min_x  # Align to left
        else:  # top-right
            paste_x = mask_max_x - (max_x - min_x)  # Align to right
    else:  # bottom
        paste_y = mask_max_y - (max_y - min_y)  # Align to bottom
        if 'left' in direction:
            paste_x = mask_min_x  # Align to left
        else:  # bottom-right
            paste_x = mask_max_x - (max_x - min_x)  # Align to right
    
    # Paste trapezoid image
    result_im.paste(trapezoid, (paste_x - min_x, paste_y - min_y), trapezoid)
    
     # Create new mask for the remaining white space
    new_mask = Image.new('RGB', target_im.size, 'black')
    draw = ImageDraw.Draw(new_mask)
    
    # First fill the entire original masked region
    draw.rectangle([mask_min_x, mask_min_y, mask_max_x, mask_max_y], fill='white')
    
    # Create mask array
    mask_arr = np.array(new_mask)
    
    # Get the region where the trapezoid was pasted
    paste_height = max_y - min_y + 1
    paste_width = max_x - min_x + 1
    
    # First make the entire pasted region black in the mask
    paste_region = np.zeros_like(mask_arr[:,:,0], dtype=bool)
    paste_region[paste_y:paste_y+paste_height, paste_x:paste_x+paste_width] = True
    mask_arr[paste_region] = [0,0,0]
    
    # Then make the black pixels from wrapped image white in the mask
    black_pixels = ~np.any(wrapped_arr > 10, axis=2)  # Find black pixels
    black_region = black_pixels[min_y:max_y+1, min_x:max_x+1]  # Get region within bounds
    
    # Apply black pixels as white to the mask
    paste_mask = np.zeros_like(mask_arr[:,:,0], dtype=bool)
    paste_mask[paste_y:paste_y+paste_height, paste_x:paste_x+paste_width] = black_region
    mask_arr[paste_mask] = [255,255,255]
    
    new_mask = Image.fromarray(mask_arr)
    
    return result_im, new_mask

def extract_masked_region(
    im: Union[Image.Image, np.ndarray],
    mask: Image.Image
) -> Image.Image:
    """
    Extract/crop the region from image where mask is white
    
    Args:
        im: Input image (PIL Image or numpy array)
        mask: Mask image (white pixels indicate region to extract)
    
    Returns:
        Image.Image: Cropped region from original image
    """
    # Convert numpy array to PIL if needed
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    
    # Convert mask to numpy array
    mask_arr = np.array(mask)
    
    # Find white pixels (assuming RGB mask where white is [255,255,255])
    white_pixels = np.all(mask_arr == 255, axis=2)
    white_coords = np.where(white_pixels)
    
    # Get bounding box of white region
    min_y, max_y = white_coords[0].min(), white_coords[0].max()
    min_x, max_x = white_coords[1].min(), white_coords[1].max()
    
    # Crop original image to this region
    cropped_region = im.crop((min_x, min_y, max_x + 1, max_y + 1))
    
    return cropped_region

def paste_masked_region(
    target_im: Image.Image,
    original_im: Image.Image, 
    mask: Image.Image
) -> Image.Image:
    """
    Paste content from original image into target image where mask is white
    
    Args:
        target_im: Target image to paste into
        original_im: Source image to copy content from
        mask: Mask where white pixels indicate where to paste
    
    Returns:
        Image.Image: Modified target image
    """
    # Convert to PIL if needed
    if isinstance(target_im, np.ndarray):
        target_im = Image.fromarray(target_im)
    if isinstance(original_im, np.ndarray):
        original_im = Image.fromarray(original_im)
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask)
        
    # Create result image
    result_im = target_im.copy()
    
    # Convert mask to RGBA for alpha compositing
    mask_rgba = mask.convert('RGBA')
    mask_data = np.array(mask_rgba)
    
    # Make black regions transparent in mask
    mask_data[np.all(mask_data[:,:,:3] == 0, axis=2)] = [0,0,0,0]
    mask_data[np.all(mask_data[:,:,:3] == 255, axis=2)] = [255,255,255,255]
    alpha_mask = Image.fromarray(mask_data)
    
    # Create RGBA version of original image
    original_rgba = Image.new('RGBA', original_im.size)
    original_rgba.paste(original_im)
    
    # Paste original content using mask as alpha
    result_im.paste(original_rgba, (0,0), alpha_mask)
    
    return result_im

