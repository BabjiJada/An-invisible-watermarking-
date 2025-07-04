import torch
from PIL import Image
import numpy as np
import cv2
from torchvision import transforms
from models.build_model import build_embedder, build_extractor

# Set up device for model execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained model checkpoint
checkpoint = torch.load("models/watermark_model.pth", map_location=device)

# Build model architecture
embed_net = build_embedder(alpha=0.002, num_blocks=5).to(device)
extract_net = build_extractor(num_blocks=5).to(device)

# Load model weights
embed_net.load_state_dict(checkpoint["embedder"])
extract_net.load_state_dict(checkpoint["extractor"])

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def embed_watermark_with_model(cover_path, watermark_path, output_path):
    """
    Simulates embedding the watermark using the deep learning model.
    """
    cover = Image.open(cover_path).convert("RGB")
    watermark = Image.open(watermark_path).convert("L")

    cover_tensor = transform(cover).unsqueeze(0).to(device)
    watermark_tensor = transform(watermark).unsqueeze(0).to(device)

    # Apply model processing
    watermarked_tensor = embed_net(cover_tensor, watermark_tensor)

    # Convert tensor output back to image
    watermarked_img = transforms.ToPILImage()(watermarked_tensor.squeeze(0).cpu())

    # Save output (simulated)
    watermarked_img.save(output_path)

def extract_watermark_with_model(embedded_path):
    """
    Simulates extracting the watermark using the deep learning model.
    """
    embedded_img = Image.open(embedded_path).convert("RGB")
    embedded_tensor = transform(embedded_img).unsqueeze(0).to(device)

    # Apply model processing
    extracted_tensor = extract_net(embedded_tensor)

    # Convert extracted tensor back to NumPy array
    extracted_img = transforms.ToPILImage()(extracted_tensor.squeeze(0).cpu())
    extracted_np = np.array(extracted_img)

    # Return processed watermark
    return extracted_np
