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
) -> Tuple[np.ndarray, float]:
    """Create a standard watermark setup for experiments.
    
    Args:
        weight_shape: Shape of the weight matrix
        letter: Letter to use as watermark
        
    Returns:
        Tuple of (mask, watermark_value)
    """
    rows, cols = weight_shape
    
    # Scale letter size based on matrix dimensions
    letter_size = min(rows // 4, cols // 15, 200)
    font_size = int(letter_size * 0.9)
    
    # Position in top-left area
    position = (rows // 10, cols // 20)
    
    mask = generate_letter_mask(
        letter=letter,
        matrix_shape=weight_shape,
        letter_size=letter_size,
        position=position,
        font_size=font_size,
    )
    
    return mask, 0.0  # Value = 0 for the watermark

