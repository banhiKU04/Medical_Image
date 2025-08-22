import gradio as gr
from PIL import Image
import numpy as np
import cv2

def enhance(image):
    """
    Enhances medical images using lightweight sharpening and contrast adjustment.
    Works fully on CPU and gives visible improvement.
    """
    # Convert PIL to NumPy array
    img = np.array(image)

    # If color image, convert to grayscale (most medical scans are grayscale)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)

    # Optional: increase contrast slightly
    enhanced = cv2.convertScaleAbs(sharpened, alpha=1.2, beta=10)  # alpha=contrast, beta=brightness

    # Convert back to PIL Image
    enhanced_pil = Image.fromarray(enhanced)

    # Return side-by-side comparison
    return [image, enhanced_pil]

# Gradio UI
demo = gr.Interface(
    fn=enhance,
    inputs=gr.Image(type="pil", label="Upload Medical Image"),
    outputs=[gr.Image(label="Original"), gr.Image(label="Enhanced")],
    title="Medical Image Enhancer",
    description="Upload a medical scan (MRI/X-ray). Compare Original vs Enhanced."
)

if __name__ == "__main__":
    demo.launch()
