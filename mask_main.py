import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import rasterio
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# -------------------------
# 1) 데이터 로더 (src/ref 마스크 모두 읽기)
# -------------------------
class TIFDataset(Dataset):
    def __init__(self, src_path, ref_path, mask_src_path=None, mask_ref_path=None):
        self.src_path = src_path
        self.ref_path = ref_path
        self.mask_src_path = mask_src_path
        self.mask_ref_path = mask_ref_path

        with rasterio.open(src_path) as src:
            self.src_img = src.read().astype(np.float32)
            self.profile = src.profile

        with rasterio.open(ref_path) as ref:
            self.ref_img = ref.read().astype(np.float32)

        # normalize 0~1
        self.src_img /= np.max(self.src_img)
        self.ref_img /= np.max(self.ref_img)

        # 마스크 읽기
        if mask_src_path:
            with rasterio.open(mask_src_path) as m1:
                self.mask_src = m1.read(1).astype(np.int32)
        else:
            self.mask_src = None

        if mask_ref_path:
            with rasterio.open(mask_ref_path) as m2:
                self.mask_ref = m2.read(1).astype(np.int32)
        else:
            self.mask_ref = None

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        src = torch.from_numpy(np.transpose(self.src_img, (1, 2, 0)))
        ref = torch.from_numpy(np.transpose(self.ref_img, (1, 2, 0)))

        mask_src = None
        mask_ref = None
        if self.mask_src is not None:
            mask_src = torch.from_numpy(self.mask_src)
        if self.mask_ref is not None:
            mask_ref = torch.from_numpy(self.mask_ref)

        return src, ref, mask_src, mask_ref


# -------------------------
# 2) Lightweight UNet
# -------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.out_conv = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        out = self.out_conv(d1)
        return out


# -------------------------
# 3) 마스크 기반 Histogram Loss (src/ref 공통 클래스만)
# -------------------------
def histogram_loss_masked(output, reference, mask_src, mask_ref):
    # output: (H, W, C) -> (1, C, H, W)
    if output.ndim == 3:  # H,W,C
        output = output.permute(2,0,1).unsqueeze(0)  # -> (1,C,H,W)
    elif output.ndim == 2:  # H,W
        output = output.unsqueeze(0).unsqueeze(0)  # -> (1,1,H,W)

    # reference도 동일
    if reference.ndim == 3:
        reference = reference.permute(2,0,1).unsqueeze(0)
    elif reference.ndim == 2:
        reference = reference.unsqueeze(0).unsqueeze(0)

    # mask는 (H,W) -> (1,H,W)
    if mask_src.ndim == 2:
        mask_src = mask_src.unsqueeze(0)
    if mask_ref.ndim == 2:
        mask_ref = mask_ref.unsqueeze(0)

    # 공통 클래스
    common_classes = torch.from_numpy(
        np.intersect1d(mask_src.cpu().numpy(), mask_ref.cpu().numpy())
    ).to(output.device)

    loss = 0.0
    for cls in common_classes:
        cls = int(cls)
        mask_common = (mask_src == cls) & (mask_ref == cls)
        if mask_common.sum() == 0:
            continue

        for c in range(output.shape[1]):
            out_c = output[0, c][mask_common[0]]
            ref_c = reference[0, c][mask_common[0]]

            if out_c.numel() == 0 or ref_c.numel() == 0:
                continue

            hist_out = torch.histc(out_c, bins=256, min=0, max=1)
            hist_ref = torch.histc(ref_c, bins=256, min=0, max=1)

            hist_out /= hist_out.sum()
            hist_ref /= hist_ref.sum()

            loss += torch.mean((hist_out - hist_ref) ** 2)

    return loss

# -------------------------
# 4) Training loop
# -------------------------
def train_harmonizer(src_path, ref_path, save_path, mask_src_path=None, mask_ref_path=None, device='cpu'):
    dataset = TIFDataset(src_path, ref_path, mask_src_path, mask_ref_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = UNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 50

    for epoch in range(num_epochs):
        epoch_start = time.time()
        for src, ref, mask_src, mask_ref in dataloader:
            src = src.permute(0, 3, 1, 2).to(device)
            ref = ref.permute(0, 3, 1, 2).to(device)
            mask_src = mask_src.to(device)
            mask_ref = mask_ref.to(device)

            optimizer.zero_grad()
            out = model(src)

            loss_hist = histogram_loss_masked(out, ref, mask_src, mask_ref)
            loss_l1 = nn.L1Loss()(out, src)
            loss = loss_hist + 0.1 * loss_l1
            loss.backward()
            optimizer.step()

        epoch_end = time.time()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {loss.item():.6f}")
            print(f"[Epoch {epoch + 1}] 실행 시간: {epoch_end - epoch_start:.2f}초")

    # -------------------------
    # Inference & Save
    # -------------------------
    model.eval()
    with torch.no_grad(), rasterio.open(src_path) as src_r:
        src_img = src_r.read().astype(np.float32) / np.max(src_r.read())
        input_tensor = torch.from_numpy(src_img).unsqueeze(0).to(device)
        output_tensor = model(input_tensor).cpu().squeeze(0).numpy()
        output_tensor = np.clip(output_tensor, 0, 1).astype(np.float32)

        profile = src_r.profile
        profile.update(dtype=rasterio.float32)
        with rasterio.open(save_path, 'w', **profile) as dst:
            dst.write(output_tensor)
    print(f"Harmonized result saved to {save_path}")

def show_image(path, title="Image"):
    with rasterio.open(path) as src:
        img = src.read(1)  # 1 band mask라고 가정
    plt.figure(figsize=(8,8))
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
# -------------------------
# 5) 실행 예시
# -------------------------
# ⏱ 시작 시간 기록
start_time = time.time()
src_tif = r"F:\02_result\TR\tgt\GK2_GOCI2_L1B_20250403_013905_FD_S007_G133_MG.tif"
ref_tif = r"F:\02_result\TR\src\GK2_GOCI2_L1B_20250403_013544_FD_S003_G119_MG.tif"
mask_src = r"F:\02_result\TR\GK2_GOCI2_L1B_20250403_013905_FD_S007_G133_MG_mask.tif"
mask_ref = r"F:\02_result\TR\GK2_GOCI2_L1B_20250403_013544_FD_S003_G119_MG_mask.tif"
save_tif = r"F:\02_result\TR\rst\result_masked_harmonized_133_200_w119.tif"
# show_image(mask_src, "mask_src")
# show_image(mask_ref, "mask_ref")

train_harmonizer(src_tif, ref_tif, save_tif, mask_src_path=mask_src, mask_ref_path=mask_ref, device='cpu')
# ⏱ 끝 시간 기록 및 출력
end_time = time.time()
elapsed = end_time - start_time
print(f"전체 실행 시간: {elapsed:.2f}초")
