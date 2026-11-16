"""ベイリーの写真入りペン立て — Bambu Lab A1 mini + AMS lite で1回印刷。

正面写真 (photos/bailey_front2.jpg) から、黒い毛・鼻・口・目と、ピンクの舌を
そのまま写し取り、白い本体の正面に面一ではめ込む。白い毛は本体の白がそのまま見える。
中の仕切りはなし。

立てて印刷するので、絵のある高さでは毎層色替えになる。色は 白(本体)・黒・ピンク の3色で、
ピンクは舌の高さだけ。

出力 (3d/out/):
  bailey_penholder.3mf  … body_white / face_black / tongue_pink の3パーツ（1つのオブジェクト）
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

# ---- 本体 (mm)。仕切りなしの四角いペン立て
W, D, H = 90.0, 75.0, 100.0
WALL, FRONT, FLOOR = 2.4, 2.8, 2.4   # 正面ははめ込みの分だけ厚め
CORNER = 3.0                         # 角の丸み（小さめにして正面の平らな幅を広く取る）
INLAY = 0.8                          # 絵のはめ込み深さ

# ---- 絵
ART_W = 82.0          # 絵の幅（正面の平らな部分は W - 2*CORNER = 84mm）
CROP_BOTTOM = 1310    # 写真のこの行より下（胸・足・名札）は使わない
BLACK_T = 70          # これより暗い画素を黒い毛とみなす
# 目のまわりだけ別の基準で明るい部分（瞳の縁・光）を白に戻す: (中心x, 中心y, しきい値)
EYES = [(300, 505, 85), (690, 480, 55)]


def trace():
    """写真 → (黒の2値画像, ピンクの2値画像)。1画素 ≒ ART_W / 写真幅 mm。"""
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


def placed_art():
    """絵を正面に配置（u=横・中央が 0, v=高さ）。写真の上端を本体の縁にそろえる。"""
    black, pink = trace()
    mm = ART_W / black.shape[1]
    b, p = mask_to_shape(black, mm), mask_to_shape(pink, mm)
    t = lambda g: affinity.translate(g, -ART_W / 2, H)
    return t(b), t(p)


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


def main():
    black, pink = placed_art()
    body = make_body().difference(on_front(shapely.union_all([black, pink]), INLAY, 0.05))
    black_m, pink_m = on_front(black, INLAY), on_front(pink, INLAY)
    os.makedirs(OUT, exist_ok=True)
    write_3mf(os.path.join(OUT, "bailey_penholder.3mf"), [("bailey_penholder", (90, 90), [
        ("body_white", body, 1), ("face_black", black_m, 2), ("tongue_pink", pink_m, 3)])])
    for name, m in [("body", body), ("black", black_m), ("pink", pink_m)]:
        print(f"{name:6s} {m.extents.round(1)} mm watertight={m.is_watertight} vol={m.volume/1000:.2f}cm3")
    zb, zp = black_m.bounds[:, 2], pink_m.bounds[:, 2]
    xb = black_m.bounds[:, 0]
    print(f"art x {xb[0]:.1f}-{xb[1]:.1f} (flat front {CORNER}-{W - CORNER}), "
          f"black z {zb[0]:.1f}-{zb[1]:.1f}, tongue z {zp[0]:.1f}-{zp[1]:.1f}")
    for lh in (0.2, 0.24):
        n2, n3 = (zb[1] - zb[0]) / lh, (zp[1] - zp[0]) / lh
        print(f"layer {lh}: ~{n2:.0f} layers with black, ~{n3:.0f} with pink -> about {n2 + n3:.0f} swaps")
    return body, black_m, pink_m


if __name__ == "__main__":
    main()
