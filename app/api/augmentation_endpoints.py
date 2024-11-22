from fastapi import APIRouter, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from ..ml.augmentation import ImageAugmenter
import base64
import io
from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

router = APIRouter()
augmenter = ImageAugmenter()

@router.get("/augmentation-ui", response_class=HTMLResponse)
async def get_augmentation_ui():
    return """
    <html>
        <head>
            <title>MNIST Image Augmentation</title>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
            <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
            <style>
                :root {
                    --primary-color: #2563eb;
                    --secondary-color: #1e40af;
                    --background-color: #f8fafc;
                    --card-background: #ffffff;
                    --text-color: #1e293b;
                    --border-color: #e2e8f0;
                }
                
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: 'Inter', sans-serif;
                    background-color: var(--background-color);
                    color: var(--text-color);
                    line-height: 1.6;
                    padding: 2rem;
                }
                
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                
                .header {
                    text-align: center;
                    margin-bottom: 2rem;
                }
                
                .header h1 {
                    font-size: 2.5rem;
                    color: var(--primary-color);
                    margin-bottom: 1rem;
                }
                
                .header p {
                    color: #64748b;
                    font-size: 1.1rem;
                }
                
                .main-content {
                    display: grid;
                    grid-template-columns: 300px 1fr;
                    gap: 2rem;
                    background: var(--card-background);
                    border-radius: 1rem;
                    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
                    padding: 2rem;
                }
                
                .controls {
                    background: var(--background-color);
                    padding: 1.5rem;
                    border-radius: 0.75rem;
                }
                
                .form-group {
                    margin-bottom: 1.5rem;
                }
                
                label {
                    display: block;
                    font-weight: 500;
                    margin-bottom: 0.5rem;
                    color: var(--text-color);
                }
                
                input[type="file"] {
                    width: 100%;
                    padding: 0.5rem;
                    border: 2px dashed var(--border-color);
                    border-radius: 0.5rem;
                    cursor: pointer;
                }
                
                select, input[type="number"] {
                    width: 100%;
                    padding: 0.75rem;
                    border: 1px solid var(--border-color);
                    border-radius: 0.5rem;
                    font-size: 1rem;
                    transition: border-color 0.2s;
                }
                
                select:focus, input[type="number"]:focus {
                    outline: none;
                    border-color: var(--primary-color);
                    box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1);
                }
                
                button {
                    width: 100%;
                    padding: 0.75rem;
                    background-color: var(--primary-color);
                    color: white;
                    border: none;
                    border-radius: 0.5rem;
                    font-size: 1rem;
                    font-weight: 500;
                    cursor: pointer;
                    transition: background-color 0.2s;
                }
                
                button:hover {
                    background-color: var(--secondary-color);
                }
                
                .preview-container {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 2rem;
                    align-items: start;
                }
                
                .preview-box {
                    background: var(--background-color);
                    padding: 1.5rem;
                    border-radius: 0.75rem;
                    text-align: center;
                }
                
                .preview-box h3 {
                    margin-bottom: 1rem;
                    color: var(--text-color);
                    font-size: 1.1rem;
                }
                
                .image-container {
                    position: relative;
                    background: white;
                    padding: 1rem;
                    border-radius: 0.5rem;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
                }
                
                .image-container img {
                    max-width: 100%;
                    height: auto;
                    border-radius: 0.25rem;
                }
                
                .loading {
                    display: none;
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    color: var(--primary-color);
                }
                
                .parameter-info {
                    margin-top: 1rem;
                    font-size: 0.9rem;
                    color: #64748b;
                }
                
                .tooltip {
                    display: inline-block;
                    margin-left: 0.5rem;
                    color: #64748b;
                    cursor: help;
                }
                
                @media (max-width: 768px) {
                    .main-content {
                        grid-template-columns: 1fr;
                    }
                    
                    .preview-container {
                        grid-template-columns: 1fr;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>MNIST Image Augmentation</h1>
                    <p>Upload an image and apply various augmentation techniques</p>
                </div>
                
                <div class="main-content">
                    <div class="controls">
                        <form id="augmentationForm">
                            <div class="form-group">
                                <label>
                                    Upload Image 
                                    <span class="tooltip" title="Supported formats: PNG, JPG, JPEG">
                                        <i class="fas fa-info-circle"></i>
                                    </span>
                                </label>
                                <input type="file" name="file" accept="image/*" required id="imageInput">
                            </div>
                            
                            <div class="form-group">
                                <label>Augmentation Type</label>
                                <select name="augmentation_type" id="augmentationType">
                                    <option value="rotation">Rotation</option>
                                    <option value="noise">Noise</option>
                                    <option value="brightness">Brightness</option>
                                    <option value="affine">Affine</option>
                                </select>
                            </div>
                            
                            <div class="form-group">
                                <label>
                                    Parameter Value
                                    <span class="tooltip" id="parameterTooltip">
                                        <i class="fas fa-info-circle"></i>
                                    </span>
                                </label>
                                <input type="number" name="param_value" step="0.1" value="1.0" id="paramValue">
                                <div class="parameter-info" id="parameterInfo"></div>
                            </div>
                            
                            <button type="submit">
                                <i class="fas fa-magic"></i> Apply Augmentation
                            </button>
                        </form>
                    </div>
                    
                    <div class="preview-container">
                        <div class="preview-box">
                            <h3>Original Image</h3>
                            <div class="image-container" id="originalPreview">
                                <div class="loading">
                                    <i class="fas fa-spinner fa-spin fa-2x"></i>
                                </div>
                            </div>
                        </div>
                        
                        <div class="preview-box">
                            <h3>Augmented Image</h3>
                            <div class="image-container" id="augmentedPreview">
                                <div class="loading">
                                    <i class="fas fa-spinner fa-spin fa-2x"></i>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="augmentation-info">
                    <h3>Augmentation Description</h3>
                    <div id="augmentationDescription" class="description-box"></div>
                </div>
            </div>
            
            <style>
                .augmentation-info {
                    margin-top: 2rem;
                    padding: 1.5rem;
                    background: var(--card-background);
                    border-radius: 1rem;
                    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
                }

                .description-box {
                    margin-top: 1rem;
                    padding: 1rem;
                    background: var(--background-color);
                    border-radius: 0.5rem;
                    color: var(--text-color);
                    line-height: 1.6;
                }

                .description-box h4 {
                    color: var(--primary-color);
                    margin-bottom: 0.5rem;
                }

                .description-box p {
                    margin-bottom: 1rem;
                }

                .description-box ul {
                    padding-left: 1.5rem;
                }

                .description-box li {
                    margin-bottom: 0.5rem;
                }
            </style>
            
            <script>
                let originalImageData = null;
                
                const augmentationDescriptions = {
                    rotation: {
                        title: "Rotation Augmentation",
                        description: "Rotates the image by a specified angle while preserving the image content and dimensions.",
                        details: [
                            "Helps model learn rotational invariance",
                            "Useful for recognizing digits at different angles",
                            "Preserves image quality and aspect ratio",
                            "Recommended range: -30° to +30°"
                        ]
                    },
                    noise: {
                        title: "Noise Augmentation",
                        description: "Adds random Gaussian noise to the image, simulating real-world image capture conditions.",
                        details: [
                            "Improves model robustness to image noise",
                            "Helps prevent overfitting",
                            "Simulates real-world image imperfections",
                            "Recommended range: 0.01 to 0.1"
                        ]
                    },
                    brightness: {
                        title: "Brightness Augmentation",
                        description: "Adjusts the brightness of the image, helping model adapt to different lighting conditions.",
                        details: [
                            "Simulates different lighting conditions",
                            "Helps model become invariant to brightness changes",
                            "Preserves image content and structure",
                            "Recommended range: 0.5 to 1.5"
                        ]
                    },
                    affine: {
                        title: "Affine Transformation",
                        description: "Applies a combination of rotation, scaling, and translation to the image.",
                        details: [
                            "Combines multiple geometric transformations",
                            "Helps model learn spatial invariance",
                            "Preserves parallel lines and ratios",
                            "Recommended range: 0.5 to 1.5"
                        ]
                    }
                };

                function updateAugmentationDescription(type) {
                    const desc = augmentationDescriptions[type];
                    const descBox = document.getElementById('augmentationDescription');
                    descBox.innerHTML = `
                        <h4>${desc.title}</h4>
                        <p>${desc.description}</p>
                        <ul>
                            ${desc.details.map(detail => `<li>${detail}</li>`).join('')}
                        </ul>
                    `;
                }

                document.getElementById('imageInput').addEventListener('change', function(e) {
                    const file = e.target.files[0];
                    if (file) {
                        const reader = new FileReader();
                        reader.onload = function(e) {
                            originalImageData = e.target.result;
                            const img = document.createElement('img');
                            img.src = originalImageData;
                            const container = document.getElementById('originalPreview');
                            container.innerHTML = '';
                            container.appendChild(img);
                        }
                        reader.readAsDataURL(file);
                    }
                });

                document.getElementById('augmentationType').addEventListener('change', function() {
                    const type = this.value;
                    updateAugmentationDescription(type);
                    if (originalImageData) {
                        applyAugmentation();
                    }
                });

                document.getElementById('paramValue').addEventListener('input', function() {
                    if (originalImageData) {
                        applyAugmentation();
                    }
                });

                async function applyAugmentation() {
                    const formData = new FormData(document.getElementById('augmentationForm'));
                    const augmentedPreview = document.getElementById('augmentedPreview');
                    augmentedPreview.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin fa-2x"></i></div>';
                    
                    try {
                        const response = await fetch('/augmentation/augment', {
                            method: 'POST',
                            body: formData
                        });
                        const data = await response.json();
                        augmentedPreview.innerHTML = `<img src="${data.augmented_image}" alt="Augmented image">`;
                    } catch (error) {
                        console.error('Error:', error);
                        augmentedPreview.innerHTML = '<p class="error">Error applying augmentation</p>';
                    }
                }

                // Initialize description
                updateAugmentationDescription('rotation');
            </script>
        </body>
    </html>
    """

@router.post("/augment")
async def augment_image(
    file: UploadFile = File(...),
    augmentation_type: str = Form(...),
    param_value: float = Form(1.0)
):
    # Read image while preserving original format
    image_data = await file.read()
    original_image = Image.open(io.BytesIO(image_data))
    
    # Store original image mode
    original_mode = original_image.mode
    
    # Apply augmentation while preserving image properties
    if augmentation_type == "rotation":
        # Direct rotation without converting to tensor
        augmented_image = TF.rotate(original_image, angle=param_value)
    
    elif augmentation_type == "brightness":
        # Adjust brightness while preserving color
        augmented_image = TF.adjust_brightness(original_image, brightness_factor=param_value)
    
    elif augmentation_type == "noise":
        # Convert to tensor, add noise, then back to PIL
        img_tensor = TF.to_tensor(original_image)
        noise = torch.randn_like(img_tensor) * param_value
        noisy_tensor = torch.clamp(img_tensor + noise, 0., 1.)
        augmented_image = TF.to_pil_image(noisy_tensor, mode=original_mode)
    
    elif augmentation_type == "affine":
        # Convert param_value to meaningful affine parameters
        angle = 45.0 * param_value  # Max ±45 degrees rotation
        scale = 1.0 + (0.5 * (param_value - 1.0))  # Scale range: 0.5 to 1.5
        shear_x = 15.0 * (param_value - 1.0)  # Max ±15 degrees shear
        shear_y = 15.0 * (param_value - 1.0)
        translate_x = 0.2 * (param_value - 1.0)  # Max ±20% translation
        translate_y = 0.2 * (param_value - 1.0)
        
        augmented_image = augmenter.apply_augmentation(
            original_image,
            "affine",
            {
                "angle": angle,
                "scale": scale,
                "shear_x": shear_x,
                "shear_y": shear_y,
                "translate_x": translate_x,
                "translate_y": translate_y
            }
        )
    else:
        augmented_image = original_image
    
    # Ensure output image has same mode as input
    augmented_image = augmented_image.convert(original_mode)
    
    # Convert both original and augmented images to base64
    def image_to_base64(img):
        buffered = io.BytesIO()
        img.save(buffered, format=file.content_type.split('/')[-1].upper())
        return base64.b64encode(buffered.getvalue()).decode()
    
    original_b64 = image_to_base64(original_image)
    augmented_b64 = image_to_base64(augmented_image)
    
    return JSONResponse({
        "original_image": f"data:{file.content_type};base64,{original_b64}",
        "augmented_image": f"data:{file.content_type};base64,{augmented_b64}",
        "augmentation_type": augmentation_type,
        "parameters": {
            "type": augmentation_type,
            "value": param_value,
            "original_mode": original_mode
        }
    }) 