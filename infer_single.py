import torch
import torchvision.transforms as T
from PIL import Image
import os

# CHANGE THIS IMPORT depending on model
from transweather_masked import MaskedResidualTransWeather, MaskNet
# from transweather_model import Transweather

# ---------------- CONFIG ----------------
IMAGE_PATH = "test.png"          # downloaded image
CKPT_PATH  = "masked_baseline/latest.pth"
SAVE_DIR   = "outputs"
IMG_SIZE   = 192
# ----------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------- Load image ----------
img = Image.open(IMAGE_PATH).convert("RGB")

transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])

x = transform(img).unsqueeze(0).to(device)  # [1,3,H,W]

# --------- Load model ----------
mask_net = MaskNet().to(device)
model = MaskedResidualTransWeather(mask_net).to(device)

state = torch.load(CKPT_PATH, map_location=device)
model.load_state_dict(state)
model.eval()

# --------- Inference ----------
with torch.no_grad():
    out, mask = model(x)

# --------- Save results ----------
to_pil = T.ToPILImage()

to_pil(x[0].cpu()).save(f"{SAVE_DIR}/input.png")
to_pil(out[0].clamp(0,1).cpu()).save(f"{SAVE_DIR}/dehazed.png")
to_pil(mask[0].repeat(3,1,1).cpu()).save(f"{SAVE_DIR}/mask.png")

print("Saved:")
print(" - outputs/input.png")
print(" - outputs/dehazed.png")
print(" - outputs/mask.png")
