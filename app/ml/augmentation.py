import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import io
import math

class ImageAugmenter:
    def __init__(self):
        pass
        
    def apply_augmentation(self, image, augmentation_type, params=None):
        """Apply specific augmentation to an image while preserving properties"""
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        
        original_mode = image.mode
        params = params or {}
        
        if augmentation_type == "rotation":
            angle = params.get("angle", 10)
            return TF.rotate(image, angle=angle)
            
        elif augmentation_type == "noise":
            noise_factor = params.get("factor", 0.05)
            img_tensor = TF.to_tensor(image)
            noise = torch.randn_like(img_tensor) * noise_factor
            noisy_tensor = torch.clamp(img_tensor + noise, 0., 1.)
            return TF.to_pil_image(noisy_tensor, mode=original_mode)
            
        elif augmentation_type == "brightness":
            factor = params.get("factor", 1.2)
            img_tensor = TF.to_tensor(image)
            brightened = img_tensor * factor
            brightened = torch.clamp(brightened, 0., 1.)
            return TF.to_pil_image(brightened, mode=original_mode)
            
        elif augmentation_type == "affine":
            # Decompose affine parameters for better control
            angle = params.get("angle", 0)  # Rotation angle
            scale = params.get("scale", 1.0)  # Scaling factor
            shear_x = params.get("shear_x", 0)  # Horizontal shear
            shear_y = params.get("shear_y", 0)  # Vertical shear
            translate_x = params.get("translate_x", 0) * image.size[0]  # Horizontal translation
            translate_y = params.get("translate_y", 0) * image.size[1]  # Vertical translation
            
            # Create affine transformation matrix
            return TF.affine(
                image,
                angle=angle,
                translate=(translate_x, translate_y),
                scale=scale,
                shear=(shear_x, shear_y),
                fill=self._get_fill_value(image)
            )
        
        return image

    def _get_fill_value(self, image):
        """Determine appropriate fill value based on image mode"""
        if image.mode == 'L':
            return 0  # Black for grayscale
        elif image.mode in ['RGB', 'RGBA']:
            return (0, 0, 0)  # Black for RGB/RGBA
        return 0  # Default

    def get_available_augmentations(self):
        return [
            "rotation",
            "noise",
            "brightness",
            "affine"
        ] 