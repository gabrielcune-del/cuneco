import sys
from pathlib import Path
import numpy as np
from PIL import Image
from collections import deque
from scipy.ndimage import binary_dilation

IMG_DIR = Path(__file__).resolve().parent.parent / "public" / "img" / "products"

def is_bgish(px, mn_thresh=195, diff_thresh=18):
    r, g, b = int(px[0]), int(px[1]), int(px[2])
    mx, mn = max(r, g, b), min(r, g, b)
    return mn > mn_thresh and (mx - mn) < diff_thresh

def flood_mask(arr, seed_thresh=(195, 18), grow_thresh=(150, 25)):
    h, w = arr.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    visited = np.zeros((h, w), dtype=bool)
    q = deque()
    border_pts = []
    for x in range(w):
        border_pts.append((0, x))
        border_pts.append((h - 1, x))
    for y in range(h):
        border_pts.append((y, 0))
        border_pts.append((y, w - 1))
    for (y, x) in border_pts:
        if not visited[y, x] and is_bgish(arr[y, x], *seed_thresh):
            visited[y, x] = True
            q.append((y, x))
    while q:
        y, x = q.popleft()
        mask[y, x] = True
        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                visited[ny, nx] = True
                if is_bgish(arr[ny, nx], *grow_thresh):
                    q.append((ny, nx))
    return mask

def process(path: Path, out_dir: Path = None, seed_thresh=(195, 18), grow_thresh=(195, 18), protect_box=None):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    mask = flood_mask(arr, seed_thresh=seed_thresh, grow_thresh=grow_thresh)

    if protect_box is not None:
        x0, y0, x1, y1 = protect_box
        mask[y0:y1, x0:x1] = False

    dilated = binary_dilation(mask, iterations=3)
    edge_zone = dilated & ~mask

    alpha = np.full(arr.shape[:2], 255, dtype=np.float32)
    alpha[mask] = 0

    ez_pixels = arr[edge_zone].astype(np.float32)
    brightness = ez_pixels.mean(axis=1) / 255.0
    whiteness = np.clip((brightness - 0.75) / 0.25, 0, 1)
    alpha[edge_zone] = 255 * (1 - whiteness)

    rgba = np.dstack([arr, alpha.astype(np.uint8)])
    out_img = Image.fromarray(rgba, "RGBA")

    target_dir = out_dir or path.parent
    out_path = target_dir / (path.stem + ".png")
    out_img.save(out_path, "PNG")
    print(f"done: {path.name} -> {out_path.name}")
    return out_path

if __name__ == "__main__":
    targets = sys.argv[1:]
    if targets:
        files = [IMG_DIR / t for t in targets]
    else:
        files = sorted(IMG_DIR.glob("brrr-*"))
    for f in files:
        process(f)
