"""Utility functions for evaluation models."""

from typing import List

from PIL import Image


def get_combined_image(images: List[Image.Image], orientation: str = "horizontal") -> Image.Image:
    """
    Combine multiple PIL images into a single image.
    
    Args:
        images: List of PIL Image objects to combine
        orientation: 'horizontal' (side-by-side) or 'vertical' (stacked)
    
    Returns:
        Combined PIL Image
    """
    if not images:
        raise ValueError("images list cannot be empty")
    
    if len(images) == 1:
        return images[0]
    
    # Ensure all images are RGB
    images = [img.convert("RGB") for img in images]
    
    if orientation == "horizontal":
        # Calculate total width and max height
        total_width = sum(img.width for img in images)
        max_height = max(img.height for img in images)
        
        # Create new image
        combined = Image.new("RGB", (total_width, max_height))
        
        # Paste images side by side
        x_offset = 0
        for img in images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width
            
    elif orientation == "vertical":
        # Calculate max width and total height
        max_width = max(img.width for img in images)
        total_height = sum(img.height for img in images)
        
        # Create new image
        combined = Image.new("RGB", (max_width, total_height))
        
        # Paste images vertically
        y_offset = 0
        for img in images:
            combined.paste(img, (0, y_offset))
            y_offset += img.height
    else:
        raise ValueError(f"orientation must be 'horizontal' or 'vertical', got {orientation}")
    
    return combined
