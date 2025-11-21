# cloud_land_sea_unsupervised.py
import numpy as np
import rasterio
from rasterio import Affine
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from scipy.ndimage import binary_opening, binary_closing, generate_binary_structure
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def read_image(path):
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        data = src.read()  # shape: (bands, H, W)
        nodata = src.nodata
    return data, profile, nodata

def compute_features(bands, nodata):
    # bands: (B, H, W)
    B, H, W = bands.shape
    mask = np.ones((H, W), dtype=bool)
    if nodata is not None:
        for b in range(B):
            mask &= (bands[b] != nodata)
            # ✅ 3밴드 모두 0인 blank 픽셀 제거
            if B >= 3:
                blank = np.logical_and.reduce([
                    bands[0] == 0,
                    bands[1] == 0,
                    bands[2] == 0
                ])
                mask &= ~blank
    # convert to float
    arr = bands.astype('float32')
    # Basic band names by heuristic:
    # If 4+ bands: assume order R,G,B,NIR (common). If 3 bands: R,G,B.
    # We'll try to choose indices:
    if B >= 4:
        r, g, b, nir = arr[0], arr[1], arr[2], arr[3]
    elif B == 3:
        r, g, b = arr[0], arr[1], arr[2]
        nir = None
    else:
        # fallback: replicate single band
        r = g = b = arr[0]
        nir = None

    # brightness: mean of visible bands
    if B >= 3:
        brightness = (r + g + b) / 3.0
    else:
        brightness = r

    # NDWI: (G - NIR) / (G + NIR) if NIR available (McFeeters)
    # fallback: (G - B) / (G + B) as rough proxy
    eps = 1e-6
    r, g, b = arr[0], arr[1], arr[2]

    brightness = (r + g + b) / 3.0

    blue_ratio = b / (r + g + b + eps)
    green_ratio = g / (r + g + b + eps)
    red_ratio = r / (r + g + b + eps)

    # 픽셀 내부 밝기 변화량 → 구름/물/육지 텍스처 구분에 도움
    std3 = np.std(arr[:3], axis=0)

    if B >= 4:
        nir = arr[3]
        ndwi = (g - nir) / (g + nir + eps)
    else:
        ndwi = (g - b) / (g + b + eps)

    # flatten feature set
    feature_list = [
        brightness.reshape(-1),
        blue_ratio.reshape(-1),
        green_ratio.reshape(-1),
        red_ratio.reshape(-1),
        std3.reshape(-1),
        ndwi.reshape(-1),
    ]
    X = np.vstack(feature_list).T
    valid_mask_flat = mask.reshape(-1)
    return X, valid_mask_flat, (H, W)

# def compute_features(bands, nodata):
#     # bands: (B, H, W)
#     B, H, W = bands.shape
#     mask = np.ones((H, W), dtype=bool)
#     if nodata is not None:
#         for b in range(B):
#             mask &= (bands[b] != nodata)
#             # ✅ 3밴드 모두 0인 blank 픽셀 제거
#             if B >= 3:
#                 blank = np.logical_and.reduce([
#                     bands[0] == 0,
#                     bands[1] == 0,
#                     bands[2] == 0
#                 ])
#                 mask &= ~blank
#     # convert to float
#     arr = bands.astype('float32')
#     # Basic band names by heuristic:
#     # If 4+ bands: assume order R,G,B,NIR (common). If 3 bands: R,G,B.
#     # We'll try to choose indices:
#     if B >= 4:
#         r, g, b, nir = arr[0], arr[1], arr[2], arr[3]
#     elif B == 3:
#         r, g, b = arr[0], arr[1], arr[2]
#         nir = None
#     else:
#         # fallback: replicate single band
#         r = g = b = arr[0]
#         nir = None
#
#     # brightness: mean of visible bands
#     if B >= 3:
#         brightness = (r + g + b) / 3.0
#     else:
#         brightness = r
#
#     # NDWI: (G - NIR) / (G + NIR) if NIR available (McFeeters)
#     # fallback: (G - B) / (G + B) as rough proxy
#     eps = 1e-6
#     if nir is not None:
#         ndwi = (g - nir) / (g + nir + eps)
#     else:
#         ndwi = (g - b) / (g + b + eps)
#
#     # Normalize band values by dynamic range to reduce scale issues
#     # Detect dtype dynamic range
#     # We'll scale each band by its max (for that image)
#     stacked = []
#     for i in range(B):
#         band = arr[i]
#         # mask nodata for max calc
#         valid = band[mask]
#         if valid.size == 0:
#             band_norm = band
#         else:
#             mx = np.percentile(valid, 99.5)  # robust max
#             if mx <= 0:
#                 band_norm = band
#             else:
#                 band_norm = band / (mx + eps)
#         stacked.append(band_norm.reshape(-1))
#     brightness_vec = brightness.reshape(-1) / (np.percentile(brightness[mask], 99.5) + eps)
#     ndwi_vec = ndwi.reshape(-1)
#     # Build feature matrix: normalized bands, brightness, ndwi
#     feature_list = stacked + [brightness_vec, ndwi_vec]
#     X = np.vstack(feature_list).T  # shape (H*W, features)
#     valid_mask_flat = mask.reshape(-1)
#     return X, valid_mask_flat, (H, W)

def cluster_and_map(X, valid_mask_flat, H, W, n_clusters=3, random_state=0):
    # Use only valid pixels for clustering
    X_valid = X[valid_mask_flat]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_valid)

    # kmean 분류
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=8)
    labels_valid = kmeans.fit_predict(Xs)
    # We'll determine which label is cloud/sea/land using centroids in feature space.
    centroids = kmeans.cluster_centers_  # in scaled space
    # To interpret, inverse transform centroids to original feature scale:
    centroids_orig = scaler.inverse_transform(centroids)

    # gmm 분류
    # gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=0)
    # labels_valid = gmm.fit_predict(Xs)
    # centroids_orig = scaler.inverse_transform(gmm.means_)

    # Recall feature order: band0...bandN-1, brightness, ndwi
    ndwi_idx = X.shape[1] - 1
    brightness_idx = X.shape[1] - 2

    centroid_ndwi = centroids_orig[:, ndwi_idx]
    centroid_brightness = centroids_orig[:, brightness_idx]

    # Heuristics:
    # - cloud cluster: highest brightness
    # - sea cluster: highest ndwi
    # - land: the remaining one
    cloud_label = int(np.argmax(centroid_brightness))
    sea_label = int(np.argmax(centroid_ndwi))
    # if same, pick next best by excluding cloud
    if sea_label == cloud_label:
        sorted_ndwi = np.argsort(centroid_ndwi)[::-1]
        for lab in sorted_ndwi:
            if lab != cloud_label:
                sea_label = int(lab)
                break
    # remaining label -> land
    all_labels = set(range(n_clusters))
    land_label = int((all_labels - {cloud_label, sea_label}).pop())

    # build full label array
    labels_full = np.full(H * W, fill_value=-1, dtype=np.int8)
    labels_full[valid_mask_flat] = labels_valid
    labels_full = labels_full.reshape(H, W)

    # Map to classes: 1=cloud, 2=land, 3=sea (you can change)
    mapped = np.zeros_like(labels_full, dtype=np.uint8)
    mapped[labels_full == cloud_label] = 1   # cloud
    mapped[labels_full == land_label] = 2    # land
    mapped[labels_full == sea_label] = 3     # sea

    return mapped, {'cloud_label': cloud_label, 'land_label': land_label, 'sea_label': sea_label,
                    'centroid_ndwi': centroid_ndwi.tolist(), 'centroid_brightness': centroid_brightness.tolist()}

def clean_mask(mask, iterations_open=1, iterations_close=1):
    # small morphological cleaning to remove salt&pepper
    struct = generate_binary_structure(2, 2)
    cleaned = np.zeros_like(mask)
    for class_val in [1,2,3]:
        m = (mask == class_val)
        m = binary_opening(m, structure=struct, iterations=iterations_open)
        m = binary_closing(m, structure=struct, iterations=iterations_close)
        cleaned[m] = class_val
    return cleaned

def save_mask_as_tiff(mask, profile, outpath, nodata_val=0):
    prof = profile.copy()
    prof.update(dtype=rasterio.uint8, count=1, compress='lzw', nodata=nodata_val)
    with rasterio.open(outpath, 'w', **prof) as dst:
        dst.write(mask.astype(rasterio.uint8), 1)

def plot_preview(rgb_bands, mask):
    # rgb_bands: (3,H,W) values normalized 0-1 or 0-255
    img = np.moveaxis(rgb_bands, 0, -1)
    if img.dtype != np.uint8:
        # scale for display
        img = img.astype('float32')
        mx = np.percentile(img, 99.5)
        img = np.clip(img / (mx + 1e-6), 0, 1)
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title("RGB")
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.title("Class mask (1=cloud,2=land,3=sea)")
    plt.imshow(mask, vmin=0, vmax=3)
    plt.axis('off')
    plt.show()

def main(input_tif, output_mask_tif, preview=True):
    bands, profile, nodata = read_image(input_tif)
    X, valid_mask_flat, (H, W) = compute_features(bands, nodata)

    # 3밴드 모두 0인 값 최종 강제 제외
    if bands.shape[0] >= 3:
        blank = (bands[0] == 0) & (bands[1] == 0) & (bands[2] == 0)
        valid_mask_flat = valid_mask_flat & (~blank.reshape(-1))
    mapped, meta = cluster_and_map(X, valid_mask_flat, H, W)
    cleaned = clean_mask(mapped, iterations_open=1, iterations_close=1)
    save_mask_as_tiff(cleaned, profile, output_mask_tif, nodata_val=0)
    print("Saved classified mask to:", output_mask_tif)
    print("Mapping meta:", meta)
    if preview:
        # build an RGB for preview: try to use first 3 bands, scale
        if bands.shape[0] >= 3:
            rgb = bands[:3].astype('float32')
            # simple scaling by 99.5th percentile per band
            for i in range(3):
                p = np.percentile(rgb[i], 99.5)
                if p > 0:
                    rgb[i] = rgb[i] / p
            rgb = np.clip(rgb, 0, 1)
            plot_preview(rgb, cleaned)
        else:
            print("Preview skipped: not enough bands for RGB preview.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unsupervised cloud/land/sea classification")
    parser.add_argument("input_tif", help="Input multi-band GeoTIFF")
    parser.add_argument("output_mask_tif", help="Output classified mask GeoTIFF")
    parser.add_argument("--no-preview", action="store_true", help="Don't show preview plot")
    args = parser.parse_args()
    main(args.input_tif, args.output_mask_tif, preview=not args.no_preview)
