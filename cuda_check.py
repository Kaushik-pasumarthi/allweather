import torch
import torch.nn as nn

# IMPORT YOUR MASKED MODEL
# adjust the import path if needed
from transweather_masked import MaskedResidualTransWeather
from transweather_masked import Transweather


# -----------------------------
# Simple lightweight mask CNN
# -----------------------------
class DummyMaskNet(nn.Module):
    """
    A tiny CNN that outputs a soft mask in [0,1].
    This is ONLY for sanity checking.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # -----------------------------
    # Instantiate model
    # -----------------------------
    mask_net = DummyMaskNet().to(device)
    model = MaskedResidualTransWeather(mask_net).to(device)

    model.eval()

    # -----------------------------
    # Dummy input
    # -----------------------------
    x = torch.randn(1, 3, 192, 192).to(device)

    # -----------------------------
    # Forward pass
    # -----------------------------
    with torch.no_grad():
        out, mask = model(x)

    # -----------------------------
    # Shape checks
    # -----------------------------
    print("\n--- SANITY CHECK RESULTS ---")
    print("Input shape  :", x.shape)
    print("Output shape :", out.shape)
    print("Mask shape   :", mask.shape)

    # -----------------------------
    # Value range checks
    # -----------------------------
    print("\n--- VALUE CHECKS ---")
    print("Mask min/max :", mask.min().item(), mask.max().item())
    print("Output min/max:", out.min().item(), out.max().item())

    # -----------------------------
    # Assertions (fail fast)
    # -----------------------------
    assert out.shape == x.shape, "❌ Output shape mismatch"
    assert mask.shape[1] == 1, "❌ Mask must be single-channel"
    assert 0.0 <= mask.min() and mask.max() <= 1.0, "❌ Mask not in [0,1]"

    print("\n✅ SANITY CHECK PASSED")
    print("Masked TransWeather forward path is correct.")


if __name__ == "__main__":
    main()
