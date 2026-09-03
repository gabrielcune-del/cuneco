import sys
from pathlib import Path
import numpy as np
from PIL import Image
from collections import deque

CREAM = (0xF5, 0xEF, 0xE3)
IMG_DIR = Path(__file__).resolve().parent.parent / "public" / "img" / "products"

def is_bgish(px):
    r, g, b = int(px[0]), int(px[1]), int(px[2])
    mx, mn = max(r, g, b), min(r, g, b)
    return mn > 195 and (mx - mn) < 18

def flood_mask(arr):
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
        if not visited[y, x] and is_bgish(arr[y, x]):
            visited[y, x] = True
            q.append((y, x))
    while q:
        y, x = q.popleft()
        mask[y, x] = True
        for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                visited[ny, nx] = True
                if is_bgish(arr[ny, nx]):
                    q.append((ny, nx))
    return mask

def process(path: Path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img)
    mask = flood_mask(arr)

    # feather: for pixels adjacent to masked region that are near-white, blend toward cream
    from scipy.ndimage import binary_dilation
    dilated = binary_dilation(mask, iterations=3)
    edge_zone = dilated & ~mask

    out = arr.copy().astype(np.float32)
    cream_arr = np.array(CREAM, dtype=np.float32)

    out[mask] = cream_arr

    # feather edge zone based on whiteness
    ez_pixels = arr[edge_zone].astype(np.float32)
    brightness = ez_pixels.mean(axis=1) / 255.0
    whiteness = np.clip((brightness - 0.75) / 0.25, 0, 1)
    blended = ez_pixels * (1 - whiteness[:, None]) + cream_arr[None, :] * whiteness[:, None]
    out[edge_zone] = blended

    out_img = Image.fromarray(out.astype(np.uint8), "RGB")
    out_path = path.with_suffix(".jpg") if path.suffix.lower() != ".jpg" else path
    out_img.save(out_path, "JPEG", quality=92)
    if out_path != path:
        path.unlink()
    print(f"done: {path.name} -> {out_path.name}")

if __name__ == "__main__":
    targets = sys.argv[1:]
    if targets:
        files = [IMG_DIR / t for t in targets]
    else:
        files = sorted(IMG_DIR.glob("brrr-*"))
    for f in files:
        process(f)
