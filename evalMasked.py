import time
import torch
import lpips
import numpy as np
from torch.utils.data import DataLoader

from utils import validation
from train_data_functions import AllWeatherDataset
from transweather_masked import MaskedResidualTransWeather, MaskNet

# ------------------ Device ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

class MaskedWrapper(torch.nn.Module):
    def __init__(self, masked_model):
        super().__init__()
        self.masked_model = masked_model

    def forward(self, x):
        out, _ = self.masked_model(x)
        return out


# ------------------ Load model ------------------
mask_net = MaskNet().to(device)
model = MaskedResidualTransWeather(mask_net).to(device)

state = torch.load("masked_baseline/latest.pth", map_location=device)
model.load_state_dict(state, strict=False)
model.eval()

# ------------------ Validation dataset ------------------
val_dataset = AllWeatherDataset(
    root="dataset",
    file_list="dataset/val.txt",
    crop_size=[192, 192],
    train=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0
)

# ------------------ LPIPS ------------------
lpips_fn = lpips.LPIPS(net="alex").to(device)
lpips_fn.eval()

# ------------------ Metrics ------------------
lpips_scores = []
inference_times = []
mask_means = []

torch.set_grad_enabled(False)

# ------------------ Run evaluation ------------------
for idx, batch in enumerate(val_loader):
    inp, gt, _ = batch
    inp = inp.to(device)
    gt = gt.to(device)

    # -------- Timing --------
    torch.cuda.synchronize()
    start = time.time()

    pred, mask = model(inp)

    torch.cuda.synchronize()
    end = time.time()

    inference_times.append(end - start)
    mask_means.append(mask.mean().item())

    # -------- LPIPS --------
    # LPIPS expects [-1, 1]
    lp = lpips_fn(
        2 * pred - 1,
        2 * gt - 1
    )
    lpips_scores.append(lp.item())

    if idx % 50 == 0:
        print(f"Evaluated {idx}/{len(val_loader)} images")

# ------------------ PSNR & SSIM ------------------
wrapped_model = MaskedWrapper(model)

psnr, ssim = validation(
    wrapped_model,
    val_loader,
    device,
    exp_name="masked_transweather"
)


# ------------------ Efficiency ------------------
avg_time = np.mean(inference_times)
fps = 1.0 / avg_time

# ------------------ Results ------------------
print("\n====== Masked TransWeather Results ======")
print(f"PSNR  : {psnr:.2f} dB")
print(f"SSIM  : {ssim:.4f}")
print(f"LPIPS : {np.mean(lpips_scores):.4f}")
print("----------------------------------------")
print(f"Avg inference time : {avg_time * 1000:.2f} ms/image")
print(f"FPS                : {fps:.2f}")
print(f"Mean mask value    : {np.mean(mask_means):.4f}")
print("========================================\n")
