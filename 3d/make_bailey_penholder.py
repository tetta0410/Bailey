"""ベイリーの写真入りペン立て — Bambu Lab A1 mini + AMS lite で1回印刷。

正面写真から、黒い毛・鼻・口・目（と舌のピンク）をそのまま写し取り、
白い本体の正面に面一ではめ込む。白い毛は本体の白がそのまま見える。中の仕切りはなし。

写真は2種類:
  tongue … photos/bailey_front2.jpg（舌ぺろ。白・黒・ピンクの3色）
  puppy  … photos/bailey_puppy.jpg（子犬のころ。白・黒の2色。背景・リードは除く）

印刷の向きを2通り用意する（<name> は bailey_penholder / bailey_penholder_puppy）:
  <name>_facedown.3mf … 【おすすめ】顔を下にして倒した向き。絵は最初の 0.8mm だけなので
                         色替えは数回。背面が天井になるので、中にサポートが必要。
  <name>.3mf          … 立てた向き。サポート不要だが、絵のある高さで毎層色替え。

使い方: python3 3d/make_bailey_penholder.py [tongue|puppy]   （省略時は両方）
"""
import os
import sys

import numpy as np
import shapely
from PIL import Image, ImageOps
from scipy import ndimage as ndi
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from skimage import measure, morphology

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ittetsu"))
from make_bailey import extrude  # noqa: E402
from make_pencase import write_3mf  # noqa: E402

OUT = os.path.join(HERE, "out")
PHOTO = os.path.join(HERE, "photos", "bailey_front2.jpg")
PUPPY = os.path.join(HERE, "photos", "bailey_puppy.jpg")

# ---- 本体 (mm)。仕切りなしの四角いペン立て
W, D, H = 90.0, 75.0, 100.0
WALL, FRONT, FLOOR = 2.4, 2.8, 2.4   # 正面ははめ込みの分だけ厚め
CORNER = 3.0                         # 角の丸み（小さめにして正面の平らな幅を広く取る）
INLAY = 0.8                          # 絵のはめ込み深さ

# ---- 絵（舌ぺろの写真）
ART_W = 82.0          # 絵の幅（正面の平らな部分は W - 2*CORNER = 84mm）
CROP_BOTTOM = 1310    # 写真のこの行より下（胸・足・名札）は使わない
BLACK_T = 70          # これより暗い画素を黒い毛とみなす
# 目のまわりだけ別の基準で明るい部分（瞳の縁・光）を白に戻す: (中心x, 中心y, しきい値)
EYES = [(300, 505, 85), (690, 480, 55)]


# ---- 絵（子犬の写真）。日向と日陰で明るさが違うので、まわりの白い毛の明るさに対する比で黒を決める
PUPPY_RATIO = 0.70
PUPPY_FIT = (80.0, 90.0)   # 絵が収まる幅・高さ (mm)
# 犬のまわり（少し広め）と、除くリードの範囲（写真の画素座標）
PUPPY_OUTLINE = [(60, 330), (150, 170), (320, 120), (560, 190), (800, 200), (950, 160), (1090, 220),
                 (1150, 400), (1200, 560), (1210, 760), (1190, 1000), (1100, 1150), (960, 1230),
                 (750, 1270), (450, 1270), (330, 1120), (230, 960), (160, 760), (150, 560), (80, 420)]
PUPPY_LEASH = [(1040, 727), (1107, 715), (1300, 768), (1300, 852), (1133, 812), (1053, 800)]
PUPPY_BOTTOM = 1260        # これより下（足）は使わない


def trace_tongue():
    """舌ぺろの写真 → (黒の2値画像, ピンクの2値画像)。"""
    im = ImageOps.exif_transpose(Image.open(PHOTO)).convert("RGB")
    rgb = np.asarray(im).astype(float)
    gray = ndi.gaussian_filter(rgb.mean(axis=2), 1.2)
    hsv = np.asarray(im.convert("HSV")).astype(float)
    hue, sat, val = (ndi.gaussian_filter(hsv[..., i], 1.5) for i in range(3))

    # 舌のピンク
    pink = ((hue > 225) | (hue < 12)) & (sat > 70) & (val > 110)
    pink = morphology.remove_small_objects(pink, max_size=2000)
    pink = ndi.binary_fill_holes(morphology.closing(pink, morphology.disk(6)))

    # 黒い毛（頭につながる部分だけ残し、胸・足・名札は外す）
    black = (gray < BLACK_T) & ~morphology.dilation(pink, morphology.disk(2))
    lab, n = ndi.label(black)
    keep = np.zeros(n + 1, bool)
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl[0].start < 1150 and (lab[sl] == i).sum() > 300:
            keep[i] = True
    black = keep[lab]

    # 目: 瞳の下の明るい縁と光を白に戻す
    g1 = ndi.gaussian_filter(rgb.mean(axis=2), 1.0)
    for cx, cy, thr in EYES:
        y0, y1, x0, x1 = cy - 90, cy + 90, cx - 110, cx + 110
        white = morphology.remove_small_objects(g1[y0:y1, x0:x1] > thr, max_size=25)
        white = morphology.dilation(white, morphology.disk(2))
        yy, xx = np.mgrid[y0:y1, x0:x1]
        inside = ((xx - cx) / 105) ** 2 + ((yy - cy) / 80) ** 2 < 1
        black[y0:y1, x0:x1][white & inside] = False

    black, pink = black[:CROP_BOTTOM], pink[:CROP_BOTTOM]
    # 印刷できない細かい毛やヒゲを整理（1画素 ≒ 0.07mm）
    black = morphology.closing(black, morphology.disk(3))
    black = morphology.opening(black, morphology.disk(4))
    black = morphology.remove_small_objects(black, max_size=120)
    black = morphology.remove_small_holes(black, max_size=60)
    return black, pink


def mask_to_shape(mask, mm_per_px):
    """2値画像 → shapely 図形 (mm, 画像の上が +y)。入れ子は深さの偶奇で塗り/穴を決める。"""
    pad = np.pad(mask, 2).astype(float)
    rings = []
    for c in measure.find_contours(pad, 0.5):
        if len(c) >= 4:
            p = Polygon(np.column_stack([c[:, 1] - 2, -(c[:, 0] - 2)])).buffer(0)
            if p.area > 4:
                rings.append(p)
    tree = STRtree(rings)
    depth = [sum(1 for j in tree.query(p.representative_point(), predicate="within")
                 if j != i and rings[j].area > p.area) for i, p in enumerate(rings)]
    res = shapely.Polygon()
    for d in range(max(depth) + 1):
        layer = shapely.union_all([rings[i] for i in range(len(rings)) if depth[i] == d])
        res = res.union(layer) if d % 2 == 0 else res.difference(layer)
    res = affinity.scale(res, mm_per_px, mm_per_px, origin=(0, 0))
    return res.simplify(0.03).buffer(0)


def trace_puppy():
    """子犬の写真 → (黒の2値画像, None)。草・タイル・リードは除く。"""
    from PIL import ImageDraw
    im = ImageOps.exif_transpose(Image.open(PUPPY)).convert("RGB")
    rgb = np.asarray(im).astype(float)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gray = ndi.gaussian_filter(rgb.mean(axis=2), 1.5)
    # まわりの白い毛の明るさ（近くの明るい側 95% 点）
    small = gray[::8, ::8]
    loc = ndi.gaussian_filter(ndi.percentile_filter(small, 95, size=25), 3)
    loc = ndi.zoom(loc, (gray.shape[0] / loc.shape[0], gray.shape[1] / loc.shape[1]), order=1)
    loc = loc[:gray.shape[0], :gray.shape[1]]
    grass = ndi.binary_dilation((g > r + 12) & (g > b + 5), iterations=6)
    mask = Image.new("L", im.size, 0)
    d = ImageDraw.Draw(mask)
    d.polygon(PUPPY_OUTLINE, fill=255)
    d.polygon(PUPPY_LEASH, fill=0)
    dog = np.asarray(mask) > 0
    black = (gray < PUPPY_RATIO * loc) & dog & ~grass
    black[PUPPY_BOTTOM:] = False
    black = morphology.closing(black, morphology.disk(3))
    black = morphology.opening(black, morphology.disk(4))
    black = morphology.remove_small_objects(black, max_size=200)
    black = morphology.remove_small_holes(black, max_size=80)
    return black, None


def placed_art(photo="tongue"):
    """絵を正面に配置（u=横・中央が 0, v=高さ）。

    tongue: 写真の上端を本体の縁にそろえる（耳の先が縁から出ているように見せる）。
    puppy:  絵全体を PUPPY_FIT に収めて正面の中央に置く。
    """
    if photo == "tongue":
        black, pink = trace_tongue()
        mm = ART_W / black.shape[1]
        b, p = mask_to_shape(black, mm), mask_to_shape(pink, mm)
        t = lambda g: affinity.translate(g, -ART_W / 2, H)
        return t(b), t(p)
    black, _ = trace_puppy()
    ys, xs = np.nonzero(black)
    black = black[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    hpx, wpx = black.shape
    mm = min(PUPPY_FIT[0] / wpx, PUPPY_FIT[1] / hpx)
    b = mask_to_shape(black, mm)          # 左上が原点, 下向きが -y
    return affinity.translate(b, -wpx * mm / 2, H / 2 + hpx * mm / 2), shapely.Polygon()


def rbox(x0, y0, x1, y1, r):
    return shapely.box(x0 + r, y0 + r, x1 - r, y1 - r).buffer(r, quad_segs=16)


def make_body():
    outer = extrude(rbox(0, 0, W, D, CORNER), H)
    cavity = extrude(rbox(WALL, FRONT, W - WALL, D - WALL, 1.5), H, FLOOR)
    return outer.difference(cavity)


def on_front(g, depth, extra=0.0):
    """正面 (y=0) に、平面図形 g を奥行き depth で配置したメッシュ。"""
    m = extrude(g, depth + extra, -extra)
    v = m.vertices.copy()
    m.vertices = np.column_stack([v[:, 0] + W / 2, v[:, 2], v[:, 1]])
    m.fix_normals()
    return m


def face_down(m):
    """正面 (y=0) が下になるよう X 軸まわりに 90° 回す（鏡像にはならない）。"""
    m = m.copy()
    v = m.vertices.copy()
    m.vertices = np.column_stack([v[:, 0], -v[:, 2], v[:, 1]])
    return m


def main(photo="tongue"):
    black, pink = placed_art(photo)
    name = "bailey_penholder" if photo == "tongue" else "bailey_penholder_puppy"
    body = make_body().difference(on_front(shapely.union_all([black, pink]), INLAY, 0.05))
    black_m = on_front(black, INLAY)
    parts = [("body_white", body, 1), ("face_black", black_m, 2)]
    pink_m = None
    if not pink.is_empty:
        pink_m = on_front(pink, INLAY)
        parts.append(("tongue_pink", pink_m, 3))
    os.makedirs(OUT, exist_ok=True)
    write_3mf(os.path.join(OUT, f"{name}.3mf"), [(name, (90, 90), parts)])
    write_3mf(os.path.join(OUT, f"{name}_facedown.3mf"), [(
        f"{name}_facedown", (90, 90), [(n, face_down(m), e) for n, m, e in parts])])
    print(f"== {name}")
    for pname, m, _ in parts:
        print(f"{pname:12s} {m.extents.round(1)} mm watertight={m.is_watertight} vol={m.volume/1000:.2f}cm3")
    zb = black_m.bounds[:, 2]
    zp = pink_m.bounds[:, 2] if pink_m is not None else np.array([0.0, 0.0])
    xb = black_m.bounds[:, 0]
    print(f"art x {xb[0]:.1f}-{xb[1]:.1f} (flat front {CORNER}-{W - CORNER}), "
          f"black z {zb[0]:.1f}-{zb[1]:.1f}, tongue z {zp[0]:.1f}-{zp[1]:.1f}")
    for lh in (0.2, 0.24):
        n2, n3 = (zb[1] - zb[0]) / lh, (zp[1] - zp[0]) / lh
        print(f"standing, layer {lh}: ~{n2:.0f} layers with black, ~{n3:.0f} with pink -> about {n2 + n3:.0f} swaps")
    fd = face_down(body)
    print(f"face-down: footprint {fd.extents[0]:.0f} x {fd.extents[1]:.0f} mm, height {fd.extents[2]:.0f} mm, "
          f"art in the first {INLAY} mm = {INLAY / 0.2:.0f} layers at 0.2 mm -> about {2 * INLAY / 0.2:.0f} swaps; "
          f"inner ceiling span {W - 2 * WALL:.1f} x {H - FLOOR:.1f} mm needs support")
    return body, black_m, pink_m


if __name__ == "__main__":
    for photo in (sys.argv[1:] or ["tongue", "puppy"]):
        main(photo)
