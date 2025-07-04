import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import piq

###########################
# Network and Model Setup
###########################

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

def make_layer(block, in_planes, planes, num_blocks, stride=1):
    layers = [block(in_planes, planes, stride)]
    for _ in range(num_blocks - 1):
        layers.append(block(planes, planes, stride=1))
    return nn.Sequential(*layers)

def build_embedder(alpha, num_blocks):
    class Embedder(nn.Module):
        def __init__(self, alpha, num_blocks):
            super().__init__()
            self.alpha = alpha
            self.conv_in = nn.Conv2d(6, 32, kernel_size=3, padding=1, bias=False)
            self.bn_in   = nn.BatchNorm2d(32)
            self.layer1 = make_layer(BasicBlock, 32, 32, num_blocks=num_blocks, stride=1)
            self.layer2 = make_layer(BasicBlock, 32, 32, num_blocks=num_blocks, stride=1)
            self.conv_out = nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=False)

        def forward(self, cover, wm):
            # Calculate contrast and adaptive alpha
            contrast = torch.std(cover, dim=[2,3], keepdim=True)
            adaptive_alpha = 0.002 + 0.001 * contrast
            # Ensure watermark has shape [B, 1, H, W] then resize and repeat channels
            if wm.dim() == 3:
                wm = wm.unsqueeze(1)
            wm = F.interpolate(wm, size=cover.shape[2:], mode='bilinear', align_corners=False)
            wm = wm.repeat(1, 3, 1, 1)
            x = torch.cat([cover, wm], dim=1)
            x = F.relu(self.bn_in(self.conv_in(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            delta = self.conv_out(x)
            watermarked = cover + adaptive_alpha * delta * (torch.abs(cover - cover.mean(dim=[2,3], keepdim=True)) > 0.05).float()
            return torch.clamp(watermarked, -1.0, 1.0)
    return Embedder(alpha, num_blocks)

def build_extractor(num_blocks):
    class Extractor(nn.Module):
        def __init__(self, num_blocks):
            super().__init__()
            self.conv_in = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
            self.bn_in   = nn.BatchNorm2d(32)
            self.layer1 = make_layer(BasicBlock, 32, 32, num_blocks=num_blocks, stride=1)
            self.layer2 = make_layer(BasicBlock, 32, 32, num_blocks=num_blocks, stride=1)
            self.conv_out = nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False)

        def forward(self, x):
            x = F.relu(self.bn_in(self.conv_in(x)))
            x = self.layer1(x)
            x = self.layer2(x)
            # Extractor output is raw; apply sigmoid later.
            return self.conv_out(x)
    return Extractor(num_blocks)

###########################
# Load Model Checkpoint
###########################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "watermark_model.pth")
checkpoint = torch.load(MODEL_PATH, map_location=device)

embed_net = build_embedder(checkpoint['hyperparameters']['alpha'],
                           checkpoint['hyperparameters']['num_res_blocks']).to(device)
extract_net = build_extractor(checkpoint['hyperparameters']['num_res_blocks']).to(device)

embed_net.load_state_dict(checkpoint['embedder'])
extract_net.load_state_dict(checkpoint['extractor'])
embed_net.eval()
extract_net.eval()

###########################
# Preprocessing and Utilities
###########################

# Transforms for cover image (ImageNet style)
cover_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Transforms for handwritten watermark image
watermark_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x < 0.9).float())
])

# Denormalize: Convert from [-1, 1] to [0, 1]
def denorm(x):
    return (x + 1.0) / 2.0

###########################
# Watermark Embedding and Extraction Functions
###########################

def embed_watermark(cover_path, watermark_path, output_path):
    # Load images and preprocess
    cover_img = Image.open(cover_path).convert("RGB")
    wm_img = Image.open(watermark_path).convert("L")
    cover = cover_transform(cover_img).unsqueeze(0).to(device)
    wm = watermark_transform(wm_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        watermarked = embed_net(cover, wm)
    # Denormalize the watermarked image for display (convert to [0,1])
    watermarked_dn = denorm(watermarked.squeeze(0)).cpu().permute(1,2,0).numpy()
    watermarked_dn = (watermarked_dn * 255).astype(np.uint8)
    Image.fromarray(watermarked_dn).save(output_path)
    # Return tensors for metrics calculation
    return watermarked, cover, wm

def extract_watermark(watermarked_path, output_path):
    watermarked_img = Image.open(watermarked_path).convert("RGB")
    watermarked = cover_transform(watermarked_img).unsqueeze(0).to(device)
    with torch.no_grad():
        extracted = extract_net(watermarked)
    # Resize extracted watermark to (128,128)
    extracted = F.interpolate(extracted, size=(128,128), mode='bilinear', align_corners=False)
    # Apply sigmoid and threshold to obtain binary watermark
    extracted_sigmoid = torch.sigmoid(extracted)
    extracted_bin = (extracted_sigmoid > 0.5).float()
    extracted_np = (extracted_bin.squeeze().cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(extracted_np).save(output_path)
    return extracted_sigmoid, watermarked

###########################
# Metrics: BER and SSIM
###########################

def calculate_ber_from_tensors(original_wm_tensor, extracted_wm_tensor):
    # Both tensors are assumed to be in [0,1]
    orig_bin = (original_wm_tensor > 0.5).float()
    extr_bin = (extracted_wm_tensor > 0.5).float()
    return (orig_bin.ne(extr_bin).float().mean()).item()

def calculate_ssim_metric(cover_tensor, watermarked_tensor):
    # Denormalize images (from [-1,1] to [0,1]) before computing SSIM
    cover_dn = denorm(cover_tensor)
    watermarked_dn = denorm(watermarked_tensor)
    ssim_val = piq.ssim(cover_dn, watermarked_dn, data_range=1.0)
    return 1.0 - ssim_val.item()
