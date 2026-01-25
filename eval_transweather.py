import time
import torch
import lpips
import numpy as np
from torch.utils.data import DataLoader
from transweather_model import Transweather
from utils import validation
from train_data_functions import AllWeatherDataset

# ------------------ Device ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Load model ------------------
model = Transweather().to(device)
state = torch.load("pretrain_tw/best", map_location=device)
model.load_state_dict(state)
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
lpips_fn = lpips.LPIPS(net='alex').to(device)
lpips_fn.eval()

# ------------------ Metrics accumulators ------------------
lpips_scores = []
inference_times = []

# ------------------ Disable gradients ------------------
torch.set_grad_enabled(False)

# ------------------ Run evaluation ------------------
for idx, batch in enumerate(val_loader):
    inp, gt, _ = batch
    inp = inp.to(device)
    gt = gt.to(device)

    # -------- Timing --------
    torch.cuda.synchronize()
    start = time.time()

    pred = model(inp)

    torch.cuda.synchronize()
    end = time.time()

    inference_times.append(end - start)

    # -------- LPIPS --------
    # LPIPS expects inputs in [-1, 1]
    lp = lpips_fn(
        2 * pred - 1,
        2 * gt - 1
    )
    lpips_scores.append(lp.item())

    if idx % 50 == 0:
        print(f"Evaluated {idx}/{len(val_loader)} images")

# ------------------ PSNR & SSIM ------------------
psnr, ssim = validation(
    model,
    val_loader,
    device,
    exp_name="baseline_transweather"
)

# ------------------ Efficiency ------------------
avg_time = np.mean(inference_times)
fps = 1.0 / avg_time

# ------------------ Results ------------------
print("\n====== Baseline TransWeather Results ======")
print(f"PSNR  : {psnr:.2f} dB")
print(f"SSIM  : {ssim:.4f}")
print(f"LPIPS : {np.mean(lpips_scores):.4f}")
print("-------------------------------------------")
print(f"Avg inference time : {avg_time * 1000:.2f} ms/image")
print(f"FPS                : {fps:.2f}")
print("===========================================\n")
