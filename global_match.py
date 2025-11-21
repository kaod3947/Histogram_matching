import os
import rasterio
import numpy as np
from skimage.exposure import match_histograms
from tqdm import tqdm

# --------------------------------------------------------------------
# 1) 폴더 내 모든 .tif 파일 검색
# --------------------------------------------------------------------
def list_tif_files(folder):
    tif_files = []
    for f in os.listdir(folder):
        if f.lower().endswith(".tif"):
            tif_files.append(os.path.join(folder, f))
    return tif_files


# --------------------------------------------------------------------
# 2) 폴더 내 모든 tif → 글로벌 평균 히스토그램 계산
# --------------------------------------------------------------------
def compute_global_histogram(image_paths):
    hist_sum = None
    count = 0

    for p in tqdm(image_paths, desc="Building global histogram"):
        with rasterio.open(p) as src:
            img = src.read().astype(np.float32)

        img = img / np.max(img)

        # accumulate
        if hist_sum is None:
            hist_sum = img.copy()
        else:
            # (C,H,W) 맞춰서 resize가 필요할 수 있음 → 여기선 동일 크기 가정
            hist_sum += img

        count += 1

    global_hist = hist_sum / count
    return global_hist


# --------------------------------------------------------------------
# 3) 모든 이미지 → global_hist 기준으로 히스토그램 매칭
# --------------------------------------------------------------------
def match_all_to_global(image_paths, global_hist, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for p in tqdm(image_paths, desc="Matching images"):
        with rasterio.open(p) as src:
            img = src.read().astype(np.float32)
            profile = src.profile

        img_norm = img / np.max(img)

        # match_histograms expects (H,W,C)
        matched = match_histograms(
            img_norm.transpose(1, 2, 0),
            global_hist.transpose(1, 2, 0),
            channel_axis=2
        )

        matched = matched.transpose(2, 0, 1).astype(np.float32)

        profile.update(dtype=rasterio.float32)

        save_path = os.path.join(
            save_dir,
            os.path.basename(p).replace(".tif", "_global_matched.tif")
        )

        with rasterio.open(save_path, "w", **profile) as dst:
            dst.write(matched)

        print(f"Saved: {save_path}")


# --------------------------------------------------------------------
# 4) 실행 함수
# --------------------------------------------------------------------
def process_folder(input_folder, output_folder):
    image_paths = list_tif_files(input_folder)

    if len(image_paths) == 0:
        print("No TIF files found in the folder.")
        return

    print(f"Found {len(image_paths)} TIF files.")

    # 1) 글로벌 히스토그램 계산
    global_hist = compute_global_histogram(image_paths)

    # 2) 전체 영상 매칭
    match_all_to_global(image_paths, global_hist, output_folder)


# --------------------------------------------------------------------
# 5) 실행
# --------------------------------------------------------------------
input_folder = r"F:\01_data\coord\250403"     # 입력 폴더
output_folder = r"F:\02_result\TR\global_matched"  # 저장 폴더

process_folder(input_folder, output_folder)
