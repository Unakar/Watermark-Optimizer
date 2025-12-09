"""Watermark generation and application utilities."""

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import Tuple, Optional


def generate_letter_mask(
    letter: str = "a",
    matrix_shape: Tuple[int, int] = (1024, 3072),
    letter_size: int = 200,
    position: Tuple[int, int] = (50, 50),
    font_size: int = 180,
) -> np.ndarray:
    """Generate a binary mask with a letter pattern.
    
    Args:
        letter: The letter to draw (default: "a")
        matrix_shape: Shape of the weight matrix (rows, cols)
        letter_size: Size of the letter region in pixels
        position: Top-left position of the letter (row, col)
        font_size: Font size for rendering the letter
        
    Returns:
        Binary mask of shape matrix_shape, True where letter is drawn
    """
    rows, cols = matrix_shape
    
    # Create a temporary image to render the letter
    img = Image.new('L', (letter_size, letter_size), color=0)
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except (IOError, OSError):
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except (IOError, OSError):
            # Use default font
            font = ImageFont.load_default()
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), letter, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Center the letter in the region
    x = (letter_size - text_width) // 2 - bbox[0]
    y = (letter_size - text_height) // 2 - bbox[1]
    
    # Draw the letter
    draw.text((x, y), letter, fill=255, font=font)
    
    # Convert to numpy array
    letter_array = np.array(img) > 128  # Binary threshold
    
    # Create the full mask
    mask = np.zeros((rows, cols), dtype=bool)
    
    # Place the letter in the mask
    row_start, col_start = position
    row_end = min(row_start + letter_size, rows)
    col_end = min(col_start + letter_size, cols)
    
    # Crop letter array if needed
    letter_rows = row_end - row_start
    letter_cols = col_end - col_start
    
    mask[row_start:row_end, col_start:col_end] = letter_array[:letter_rows, :letter_cols]
    
    return mask


def apply_watermark(
    weight: torch.Tensor,
    mask: np.ndarray,
    value: float = 0.0,
) -> torch.Tensor:
    """Apply watermark to weight matrix by setting masked positions to a fixed value.
    
    Args:
        weight: Weight tensor of shape (out_features, in_features)
        mask: Binary mask of same shape as weight
        value: Value to set for masked positions (default: 0)
        
    Returns:
        Modified weight tensor (in-place modification)
    """
    mask_tensor = torch.from_numpy(mask).to(weight.device)
    weight[mask_tensor] = value
    return weight


def compute_watermark_visibility(
    weight: torch.Tensor,
    mask: np.ndarray,
    original_value: float = 0.0,
) -> float:
    """Compute how visible the watermark still is.
    
    This measures how close the masked region weights are to the original watermark value.
    A high value means the watermark is still visible; low value means it's been "erased".
    
    Args:
        weight: Current weight tensor
        mask: Binary mask for the watermark
        original_value: The original value set for the watermark
        
    Returns:
        Visibility score in [0, 1], where 1 means fully visible
    """
    mask_tensor = torch.from_numpy(mask).to(weight.device)
    
    # Get weights in the masked region
    masked_weights = weight[mask_tensor]
    
    # Compute how close they are to the original value
    # Using inverse of normalized L2 distance
    distance = torch.sqrt(torch.mean((masked_weights - original_value) ** 2))
    
    # Get the typical weight magnitude for normalization
    non_masked = weight[~mask_tensor]
    typical_magnitude = torch.std(non_masked)
    
    # Visibility: 1 when distance is 0, approaching 0 as distance increases
    visibility = torch.exp(-distance / (typical_magnitude + 1e-8))
    
    return visibility.item()


def create_watermark_setup(
    weight_shape: Tuple[int, int] = (1024, 3072),
    letter: str = "a",
    watermark_value: Optional[float] = None,
    watermark_ratio: float = 0.5,  # 水印边长占矩阵高度的比例
) -> Tuple[np.ndarray, float, dict]:
    """Create a standard watermark setup for experiments.
    
    Args:
        weight_shape: Shape of the weight matrix
        letter: Letter to use as watermark
        watermark_value: Value to set for watermark (default: 0)
        watermark_ratio: 水印正方形边长 = rows * watermark_ratio
        
    Returns:
        Tuple of (mask, watermark_value, info_dict)
        info_dict contains: letter_size, position, area_ratio
    """
    rows, cols = weight_shape
    
    # 水印是正方形，边长 = rows * watermark_ratio
    letter_size = int(rows * watermark_ratio)
    letter_size = max(letter_size, 100)  # 至少 100 像素
    font_size = int(letter_size * 0.85)
    
    # Position: 左上角，留一些边距
    margin_row = (rows - letter_size) // 4
    margin_col = (cols - letter_size) // 8
    position = (margin_row, margin_col)
    
    mask = generate_letter_mask(
        letter=letter,
        matrix_shape=weight_shape,
        letter_size=letter_size,
        position=position,
        font_size=font_size,
    )
    
    # Default watermark value is 0
    if watermark_value is None:
        watermark_value = 0.0
    
    # 计算信息
    total_pixels = rows * cols
    watermark_region_pixels = letter_size * letter_size
    letter_pixels = mask.sum()
    
    info = {
        'letter_size': letter_size,
        'position': position,
        'region_end': (position[0] + letter_size, position[1] + letter_size),
        'region_ratio': watermark_region_pixels / total_pixels,  # 水印区域占比
        'letter_pixel_ratio': letter_pixels / total_pixels,  # 字母像素占比
        'letter_pixels': int(letter_pixels),
    }
    
    return mask, watermark_value, info


def get_watermark_region(
    weight: np.ndarray,
    info: dict,
) -> np.ndarray:
    """Extract the watermark region from weight matrix.
    
    Args:
        weight: Full weight matrix
        info: Info dict from create_watermark_setup
        
    Returns:
        Cropped weight matrix containing only the watermark region (square)
    """
    r_start, c_start = info['position']
    r_end, c_end = info['region_end']
    return weight[r_start:r_end, c_start:c_end]

