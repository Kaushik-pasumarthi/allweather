import os
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from train_data_functions import AllWeatherDataset
from utils import to_psnr, adjust_learning_rate
from perceptual import LossNetwork
from torchvision.models import vgg16

from transweather_masked import MaskedResidualTransWeather, MaskNet

# -----------------------------
# Argument parsing
# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument('-exp_name', required=True, type=str)
parser.add_argument('-learning_rate', default=2e-4, type=float)
parser.add_argument('-crop_size', default=[192, 192], nargs='+', type=int)
# 🚨 CRITICAL CHANGE: train_batch_size MUST be 1 for dynamic cropping to work!
parser.add_argument('-train_batch_size', default=1, type=int)
parser.add_argument('-val_batch_size', default=1, type=int)
parser.add_argument('-num_epochs', default=10, type=int)
parser.add_argument('-lambda_loss', default=0.04, type=float)
parser.add_argument('-mask_weight', default=1.0, type=float)  # Increased for BCE loss
parser.add_argument('-seed', default=19, type=int)
parser.add_argument('-vis_interval', default=500, type=int)

args = parser.parse_args()

# -----------------------------
# Reproducibility
# -----------------------------
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Directories
# -----------------------------
os.makedirs(args.exp_name, exist_ok=True)
os.makedirs(f"{args.exp_name}/mask_vis", exist_ok=True)

# -----------------------------
# Dataset
# -----------------------------
train_dataset = AllWeatherDataset(
    root='allweather',
    file_list='allweather/train.txt', # Or 'allweather/input/train.txt' depending on where it is
    crop_size=args.crop_size,
    train=True
)

val_dataset = AllWeatherDataset(
    root='allweather',
    file_list='allweather/val.txt',
    crop_size=args.crop_size,
    train=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.train_batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# -----------------------------
# Model
# -----------------------------
mask_net = MaskNet()
model = MaskedResidualTransWeather(mask_net)
model = model.to(device)

# -----------------------------
# Optimizer
# -----------------------------
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

# --- ADD THIS RESUME LOGIC HERE ---
checkpoint_path = f"{args.exp_name}/latest.pth"
if os.path.exists(checkpoint_path):
    print(f"✅ Found checkpoint at {checkpoint_path}. Resuming training!")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
else:
    print("🚀 No checkpoint found. Starting from scratch!")


def calculate_masked_psnr(pred, gt, mask, data_range=1.0):
    """
    Calculates PSNR strictly inside the masked object region.
    """
    # Ensure the mask is a strict binary stencil
    binary_mask = (mask > 0.5).float()
    psnrs = []

    for i in range(pred.shape[0]):
        p, g, m = pred[i], gt[i], binary_mask[i]

        # Calculate total active pixels (multiplied by 3 for RGB channels)
        active_pixels = torch.sum(m) * p.shape[0]

        # If there's no object in this specific image crop, skip PSNR calculation
        if active_pixels < 1:
            continue

        # Compute MSE only where the mask is 1
        mse = torch.sum(((p - g) * m) ** 2) / active_pixels

        if mse == 0:
            psnrs.append(100.0)  # Exact match / Infinite PSNR
        else:
            psnr = 10 * torch.log10((data_range ** 2) / mse)
            psnrs.append(psnr.item())

    return psnrs
# -----------------------------
# Perceptual loss (VGG16)
# -----------------------------
vgg = vgg16(pretrained=True).features[:16].to(device)
for p in vgg.parameters():
    p.requires_grad = False
loss_network = LossNetwork(vgg)
loss_network.eval()

# -----------------------------
# Training loop
# -----------------------------
for epoch in range(args.num_epochs):
    model.train()
    psnr_list = []
    start_time = time.time()

    adjust_learning_rate(optimizer, epoch)

    # 🚨 CRITICAL CHANGE: Unpack 'gt_mask' from the dataloader
    for i, (inp, gt, gt_mask, name) in enumerate(train_loader):
        inp = inp.to(device)
        gt = gt.to(device)
        gt_mask = gt_mask.to(device)  # Send ground truth mask to GPU

        optimizer.zero_grad()

        # ---- Forward ----
        # Renamed output mask to pred_mask to avoid confusion
        out, pred_mask = model(inp)

        # ---- Losses ----
        recon_loss = F.smooth_l1_loss(out, gt)
        perceptual_loss = loss_network(out, gt)

        # 🚨 CRITICAL CHANGE: Train MaskNet using Binary Cross Entropy
        # This forces the predicted mask to match your OpenCV generated masks!
        mask_bce_loss = F.binary_cross_entropy(pred_mask, gt_mask)

        # Total Loss calculation
        loss = recon_loss + args.lambda_loss * perceptual_loss + (args.mask_weight * mask_bce_loss)

        loss.backward()
        optimizer.step()

        # ---- Metrics ----
        # ---- Metrics ----
        # Calculate PSNR comparing output to GT, strictly inside the ground truth mask
        batch_psnrs = calculate_masked_psnr(out, gt, gt_mask)
        if batch_psnrs:  # Only extend if the list isn't empty (e.g., if a crop had no mask)
            psnr_list.extend(batch_psnrs)

        # ---- Mask visualization ----
        if i % args.vis_interval == 0:
            save_image(pred_mask[0], f"{args.exp_name}/mask_vis/e{epoch}_i{i}_pred_mask.png")
            save_image(gt_mask[0], f"{args.exp_name}/mask_vis/e{epoch}_i{i}_gt_mask.png")  # Save GT to compare
            save_image(out[0], f"{args.exp_name}/mask_vis/e{epoch}_i{i}_out.png")
            save_image(inp[0], f"{args.exp_name}/mask_vis/e{epoch}_i{i}_inp.png")

            print(
                f"[Epoch {epoch} | Iter {i}] "
                f"Loss: {loss.item():.4f} | "
                f"Recon: {recon_loss.item():.4f} | "
                f"Mask BCE: {mask_bce_loss.item():.4f}"
            )

    # ---- Epoch summary ----
    avg_psnr = sum(psnr_list) / len(psnr_list)
    epoch_time = time.time() - start_time

    print(
        f"\nEpoch [{epoch + 1}/{args.num_epochs}] "
        f"PSNR: {avg_psnr:.2f} dB | "
        f"Time: {epoch_time:.1f}s\n"
    )

    # ---- Save checkpoint ----
    torch.save(model.state_dict(), f"{args.exp_name}/latest.pth")

print("✅ Training finished")