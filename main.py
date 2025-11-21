import rasterio
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# -------------------------
# 1) 데이터 로더
# -------------------------
class TIFDataset(Dataset):
    def __init__(self, src_path, ref_path):
        self.src_path = src_path
        self.ref_path = ref_path

        with rasterio.open(src_path) as src:
            self.src_img = src.read().astype(np.float32)  # (C,H,W)
            self.profile = src.profile

        with rasterio.open(ref_path) as ref:
            self.ref_img = ref.read().astype(np.float32)  # (C,H,W)

        # normalize 0~1
        self.src_img /= np.max(self.src_img)
        self.ref_img /= np.max(self.ref_img)

    def __len__(self):
        return 1  # 하나의 이미지 pair

    def __getitem__(self, idx):
        # (C,H,W) -> (H,W,C) -> tensor
        src = torch.from_numpy(np.transpose(self.src_img, (1, 2, 0)))
        ref = torch.from_numpy(np.transpose(self.ref_img, (1, 2, 0)))
        return src, ref


# -------------------------
# 2) Lightweight UNet
# -------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1),
                                  nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1),
                                  nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.Conv2d(128, 128, 3, padding=1),
                                        nn.ReLU())

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1),
                                  nn.ReLU())
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1),
                                  nn.ReLU())

        self.out_conv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        # x: (B,C,H,W)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.out_conv(d1)
        return out


# -------------------------
# 3) Histogram Wasserstein Loss
# -------------------------
def histogram_loss(output, reference, num_bins=256):
    # output, reference: (B,H,W,C)
    B, C, H, W = output.shape
    loss = 0.0
    for c in range(C):
        out_flat = output[:, c, :, :].reshape(-1)
        ref_flat = reference[:, c, :, :].reshape(-1)

        # 0~1 normalize
        out_flat = out_flat / (torch.max(out_flat) + 1e-6)
        ref_flat = ref_flat / (torch.max(ref_flat) + 1e-6)

        # histogram 계산
        hist_out = torch.histc(out_flat, bins=num_bins, min=0.0, max=1.0)
        hist_ref = torch.histc(ref_flat, bins=num_bins, min=0.0, max=1.0)

        hist_out /= torch.sum(hist_out)
        hist_ref /= torch.sum(hist_ref)

        # Wasserstein distance (1D EMD)
        cdf_out = torch.cumsum(hist_out, dim=0)
        cdf_ref = torch.cumsum(hist_ref, dim=0)
        loss += torch.mean(torch.abs(cdf_out - cdf_ref))
    return loss / C


# -------------------------
# 4) Training loop
# -------------------------
def train_harmonizer(src_path, ref_path, save_path, device='cuda'):
    dataset = TIFDataset(src_path, ref_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 200  # 조절 가능
    for epoch in range(num_epochs):
        for src, ref in dataloader:
            src = src.permute(0, 3, 1, 2).to(device)  # (B,C,H,W)
            ref = ref.permute(0, 3, 1, 2).to(device)

            optimizer.zero_grad()
            out = model(src)

            loss_hist = histogram_loss(out, ref)

            loss_l1 = nn.L1Loss()(out, src)  # 구조 보존
            loss = loss_hist + 0.1 * loss_l1
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}")

    # -------------------------
    # 5) Inference & Save
    # -------------------------
    model.eval()
    with torch.no_grad(), rasterio.open(src_path) as src_r:
        src_img = src_r.read().astype(np.float32) / np.max(src_r.read())
        input_tensor = torch.from_numpy(src_img).unsqueeze(0).to(device)  # (1,C,H,W)
        output_tensor = model(input_tensor).cpu().squeeze(0).numpy()
        # (C,H,W)
        output_tensor = np.clip(output_tensor, 0, 1).astype(np.float32)

        profile = src_r.profile
        profile.update(dtype=rasterio.float32)
        with rasterio.open(save_path, 'w', **profile) as dst:
            dst.write(output_tensor)
    print(f"Harmonized result saved to {save_path}")


# -------------------------
# 6) Usage
# -------------------------
src_tif = r"F:\02_result\TR\tgt\GK2_GOCI2_L1B_20250403_013905_FD_S007_G133_MG.tif"
ref_tif = r"F:\02_result\TR\src\GK2_GOCI2_L1B_20250403_013544_FD_S003_G119_MG.tif"
save_tif = r"F:\02_result\TR\rst\result_masked_harmonized_133_nomask_200_w119.tif"

train_harmonizer(src_tif, ref_tif, save_tif, device='cpu')
