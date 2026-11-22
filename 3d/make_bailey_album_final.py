"""ベイリーの4面アルバム・ペン立て（完成版）— 黒い箱 × 白い板 × 黒いベイリー。

写真: 正面=舌ぺろ, 左=子犬, 右=肉球, 背面=鼻アップ。
絵は輪郭を強めにならしてポスター風にし、まわりに細い額縁の線を入れる。舌は白を黒で縁取る。
目は写真の位置・大きさに合わせて、白い縁＋黒目＋光の点で描き直す（黒い毛に埋もれないように）。
肉球は写真から肉球の形とホクロを取り、太い線＋ホクロの点＋ふわふわの毛先でスケッチ風に描く。
箱の底には白で「Bailey / 2026.1.15」。

2つの作り方:
  plates   … 白黒の板4枚を箱の4面のアリ溝に差し込む（色替えはごく少ない）
             bailey_final_box.3mf    箱（黒）＋底の文字（白）
             bailey_final_plates.3mf 板4枚（白地＋黒の絵）を1枚のビルドプレートに
  onepiece … 板と同じ見た目を、箱と一体で立てたまま印刷（組み立てなし・色替え多い）
             bailey_final_onepiece.3mf

使い方: python3 3d/make_bailey_album_final.py
"""
import functools
import os
import random
import sys

import numpy as np
import shapely
import trimesh
from shapely import affinity
from shapely.geometry import Point, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ittetsu"))
import make_bailey_album as ma  # noqa: E402
import make_bailey_album_plates as mp  # noqa: E402
import make_bailey_penholder as mb  # noqa: E402
from make_bailey import extrude, text_poly  # noqa: E402
from make_pencase import write_3mf  # noqa: E402

OUT = mb.OUT
PHOTOS = {"front": "tongue2", "left": "puppy2", "right": "paw", "back": "nose"}

# 目（作業画像=幅1512 の画素で 中心x, 中心y, 黒目の半径）
EYES = {
    "tongue2": [(478, 514, 48), (857, 475, 45)],
    "puppy2": [(345, 423, 35), (641, 457, 47)],
    "nose": [(1146, 764, 60)],
}
EYE_RING = 0.8          # 黒目のまわりの白い縁の太さ (mm)
EYE_GLINT = 0.6         # 光の点の最小半径 (mm)

mb.trace_set = functools.lru_cache()(mb.trace_set)    # 同じ写真を何度もなぞらない
SMOOTH = 0.6            # 輪郭のならし（大きいほどすっきり太い形）
FRAME_INSET = 2.5       # 額縁の線の、板の縁からの距離
FRAME_W = 0.9           # 額縁の線の太さ
ART_MARGIN = 4.0        # 額縁の内側の余白
TONGUE_LINE = 1.1       # 舌の縁取り

BOTTOM_TEXT = ["Bailey", "2026.1.15"]
BOTTOM_SIZES = [11.0, 5.0]
BOTTOM_DEPTH = 0.6

LAYER = 0.2


def panel_size(face):
    """板の見える部分（幅, 高さ）。"""
    return mp.PW[face] - 2 * mp.INSET, mp.PH - 1.0


def placed_eyes(photo):
    """EYES の画素座標を、mb.placed_art と同じ置き方の mm 座標 (x, y, r) にする。"""
    black, pink = mb.trace_set(photo)
    ys, xs = np.nonzero(black | pink)
    y0, x0 = ys.min(), xs.min()
    hpx, wpx = ys.max() + 1 - y0, xs.max() + 1 - x0
    mm = min(mb.SET_FIT[0] / wpx, mb.SET_FIT[1] / hpx)
    return [((x - x0) * mm - wpx * mm / 2, -(y - y0) * mm + mb.H / 2 + hpx * mm / 2, r * mm)
            for x, y, r in EYES.get(photo, [])]


def fit_to(geoms, points, fit):
    """図形をまとめて fit に収まる大きさにして中央 (0, H/2) に置く。points (x, y, r) も同じように動かす。"""
    x0, y0, x1, y1 = shapely.union_all([g for g in geoms if not g.is_empty]).bounds
    k = min(fit[0] / (x1 - x0), fit[1] / (y1 - y0))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    f = lambda g: affinity.translate(affinity.scale(g, k, k, origin=(cx, cy)), -cx, ma.H / 2 - cy)
    return [f(g) for g in geoms], [((x - cx) * k, (y - cy) * k + ma.H / 2, r * k) for x, y, r in points]


def draw_eyes(b, eyes):
    """白い縁 → 黒目 → 光の点（左上）。"""
    for x, y, r in eyes:
        b = b.difference(Point(x, y).buffer(r + EYE_RING, quad_segs=24))
        gr = max(0.28 * r, EYE_GLINT)
        pupil = Point(x, y).buffer(r, quad_segs=24).difference(
            Point(x - 0.35 * r, y + 0.35 * r).buffer(gr, quad_segs=16))
        b = b.union(pupil)
    return b.buffer(0)


# ---------------------------------------------------------------- 肉球
PAW_PHOTO = os.path.join(HERE, "photos", "bailey_paw.jpg")
# 肉球ごとの探す範囲（写真の画素: 中心x, 中心y, 半径x, 半径y）。写真は指が右向きなので、あとで90°回して指を上にする
PAW_PADS = [(1200, 630, 110, 130), (1440, 840, 150, 140), (1340, 1130, 140, 90), (1000, 1170, 130, 90),
            (870, 840, 220, 270)]
PAW_OUTLINE = [(330, 640), (460, 480), (660, 430), (920, 420), (1120, 460), (1280, 540), (1400, 660),
               (1560, 780), (1680, 860), (1720, 960), (1680, 1080), (1580, 1160), (1440, 1240), (1280, 1300),
               (1120, 1280), (900, 1240), (720, 1140), (500, 1040), (360, 940), (320, 800)]
PAD_LINE = 28            # 肉球の線の太さ（画素, ≒1.3mm）
PAD_RATIO = 1.48         # 赤/緑 の比がこれ以上を肉球とみなす
FUR_LINE = 16            # 毛のりんかく線（画素）
FUR_GAP = 90             # 肉球のまわりの毛の厚み（画素）


@functools.lru_cache()
def paw_shape():
    """肉球の写真 → スケッチ風の黒い図形（画素単位, 指が上, y は上向き）。"""
    from PIL import Image, ImageDraw, ImageOps
    from scipy import ndimage as ndi
    from skimage import morphology
    im = ImageOps.exif_transpose(Image.open(PAW_PHOTO)).convert("RGB")
    a = np.asarray(im).astype(float)
    r, g = (ndi.gaussian_filter(a[..., i], 4) for i in range(2))
    gray = ndi.gaussian_filter(a.mean(2), 2)
    ratio = r / (g + 1)                                     # 肉球は赤みが強い（影でも比は残る）
    yy, xx = np.mgrid[0:gray.shape[0], 0:gray.shape[1]]
    pads = np.zeros(gray.shape, bool)
    for cx, cy, rx, ry in PAW_PADS:
        roi = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 < 1
        m = morphology.opening(roi & (ratio > PAD_RATIO), morphology.disk(10))
        lab, n = ndi.label(m)
        if n:
            m = lab == (np.argmax(ndi.sum(m, lab, range(1, n + 1))) + 1)
        pads |= ndi.binary_fill_holes(morphology.closing(m, morphology.disk(15)))
    mk = Image.new("L", im.size, 0)
    ImageDraw.Draw(mk).polygon(PAW_OUTLINE, fill=255)
    near = np.asarray(mk) > 0
    moles = (gray < 42) & near & morphology.dilation(pads, morphology.disk(40))
    moles = morphology.remove_small_objects(morphology.opening(moles, morphology.disk(3)), max_size=300)
    # 指を上に（写真を反時計回りに 90°）
    pads, moles = np.rot90(pads, 1), np.rot90(moles, 1)
    P = mb.mask_to_shape(pads, 1.0)
    M = mb.mask_to_shape(moles, 1.0)
    P = P.buffer(6).buffer(-12).buffer(6)                   # 線をなめらかに
    pad_lines = P.difference(P.buffer(-PAD_LINE))
    # 毛: 肉球それぞれのまわりに毛をまとわせた形（指ごとにふくらむ）を、ふわっとした輪郭にする
    body = shapely.union_all([q.buffer(FUR_GAP, quad_segs=24) for q in getattr(P, "geoms", [P])])
    body = body.buffer(60).buffer(-60)                       # 指のあいだを少しだけつなぐ
    body = max(getattr(body, "geoms", [body]), key=lambda q: q.area)
    ring = body.exterior
    rnd = random.Random(15)
    wisps = []
    n = int(ring.length / 26)
    for i in range(n):
        d = i * ring.length / n + rnd.uniform(-6, 6)
        p0, p1 = ring.interpolate(d), ring.interpolate(d + 2)
        tx, ty = p1.x - p0.x, p1.y - p0.y
        L = (tx * tx + ty * ty) ** 0.5 or 1
        tx, ty = tx / L, ty / L
        nx_, ny_ = ty, -tx                                   # 外向きの法線
        if body.contains(Point(p0.x + nx_ * 5, p0.y + ny_ * 5)):
            nx_, ny_ = -nx_, -ny_
        # 少し曲がった毛束: 根元から外へ、毛先は横に流れる
        ln = rnd.uniform(18, 42)
        lean = rnd.uniform(-0.6, 0.6)
        mid = (p0.x + nx_ * ln * 0.55 + tx * lean * ln * 0.2, p0.y + ny_ * ln * 0.55 + ty * lean * ln * 0.2)
        tip = (p0.x + nx_ * ln + tx * lean * ln * 0.6, p0.y + ny_ * ln + ty * lean * ln * 0.6)
        wisps.append(shapely.LineString([(p0.x, p0.y), mid, tip]).buffer(4.5, cap_style="round"))
    fur = shapely.union_all([ring.buffer(FUR_LINE / 2), *wisps])
    hairs = []
    return shapely.union_all([pad_lines, M.buffer(9), fur, *hairs]).buffer(0)


def art(face):
    """面の絵（黒い部分）: 写真の黒い毛＋舌の縁取り＋目＋額縁の線。板の見える部分の中央 (0, 高さ/2) 基準。"""
    w, h = panel_size(face)
    fit = (w - 2 * (FRAME_INSET + FRAME_W + ART_MARGIN), h - 2 * (FRAME_INSET + FRAME_W + ART_MARGIN))
    photo = PHOTOS[face]
    if photo == "paw":
        (b,), _ = fit_to([paw_shape()], [], fit)
        p = shapely.Polygon()
    else:
        art0 = mb.placed_art(photo)
        (b, p), eyes = fit_to([art0[0], art0[1]], placed_eyes(photo), fit)
        b = ma.clean(b, smooth=SMOOTH, min_area=3.0, min_hole=0.3)
        b = draw_eyes(b, eyes)
        if not p.is_empty:
            p = ma.clean(p, smooth=SMOOTH)
            b = shapely.union_all([b.difference(p), p.difference(p.buffer(-TONGUE_LINE))]).buffer(0)
    b = affinity.translate(b, 0, h / 2 - ma.H / 2)
    r = shapely.box(-w / 2 + FRAME_INSET, FRAME_INSET, w / 2 - FRAME_INSET, h - FRAME_INSET)
    frame = r.difference(r.buffer(-FRAME_W, join_style="mitre"))
    g = shapely.union_all([b, frame]).buffer(0)
    # ほとんど接している点や細すぎるすき間をなくして、立体にできる形に整える
    g = g.buffer(0.02, join_style="round").buffer(-0.02, join_style="round").simplify(0.01).buffer(0)
    return shapely.union_all([q for q in getattr(g, "geoms", [g]) if q.area > 0.5])


def bottom_text():
    """底の文字（外から＝下から見て読める向き）。箱の座標 (x, y) で返す。"""
    lines, y = [], 0.0
    for t, s in zip(BOTTOM_TEXT, BOTTOM_SIZES):
        g = text_poly(t, s)
        lines.append(affinity.translate(g, 0, -y))
        y += s * 0.9 + 3.0
    g = shapely.union_all(lines)
    x0, y0, x1, y1 = g.bounds
    g = affinity.translate(g, -(x0 + x1) / 2, -(y0 + y1) / 2)
    # 箱を手前に倒して底をのぞくと、正面側の縁が上・右が +x になる。その向きで読めるよう上下を反転
    return affinity.translate(affinity.scale(g, 1, -1, origin=(0, 0)), mp.W / 2, mp.D / 2)


def with_bottom_text(box):
    t = bottom_text()
    box = box.difference(extrude(t, BOTTOM_DEPTH + 0.05, -0.05))
    return box, extrude(t, BOTTOM_DEPTH, 0)


# ---------------------------------------------------------------- 板差し込み版
def build_plates():
    mp.art_for = lambda face: art(face)     # 板の絵をこの完成版の絵に差し替える
    box, text = with_bottom_text(mp.make_box())
    write_3mf(os.path.join(OUT, "bailey_final_box.3mf"),
              [("bailey_final_box", (90, 90), [("box_black", box, 1), ("bottom_text_white", text, 2)])])
    plates = {f: mp.plate(f) for f in mp.PW}
    centers = {"front": (49, 46), "back": (127, 46), "left": (41.5, 135), "right": (104.5, 135)}
    write_3mf(os.path.join(OUT, "bailey_final_plates.3mf"),
              [(f"plate_{f}", centers[f], [("plate_white", b, 1), ("art_black", a, 2)])
               for f, (b, a) in plates.items()])
    return box, text, plates


# ---------------------------------------------------------------- 一体版
def panel_on(face, g, depth, extra=0.0):
    """板と同じ位置（溝の底 GROOVE_Z から上）に、面の外から見た向きで図形を置く。"""
    if face == "front":
        g2 = affinity.translate(g, 0, mp.GROOVE_Z)
        return mb.on_front(g2, depth, extra)
    g2 = affinity.translate(g, 0, mp.GROOVE_Z)
    return ma.on_face(g2, face, depth, extra)


def build_onepiece():
    body = mb.make_body()                     # 仕切り付き 90x75x100（溝なし）
    whites = []
    for face in mp.PW:
        w, h = panel_size(face)
        panel = shapely.box(-w / 2, 0, w / 2, h - 1.0)          # 縁から少し下までの白い板の部分
        white = panel.difference(art(face)).buffer(0)          # 黒い絵の部分は本体の黒がそのまま見える
        body = body.difference(panel_on(face, white, mb.INLAY, 0.05))
        whites.append(panel_on(face, white, mb.INLAY))
    body, text = with_bottom_text(body)
    white = trimesh.util.concatenate(whites + [text])
    write_3mf(os.path.join(OUT, "bailey_final_onepiece.3mf"),
              [("bailey_final_onepiece", (90, 90), [("body_black", body, 1), ("white", white, 2)])])
    return body, white


def swap_layers(mesh, top):
    """mesh の色が入る層の数（立てて印刷, 0.2mm）。"""
    t = mesh.triangles[:, :, 2]
    lo, hi = t.min(1), t.max(1)
    zs = np.arange(LAYER / 2, top, LAYER)
    return int(sum(((lo <= z) & (hi >= z)).any() for z in zs))


def main():
    os.makedirs(OUT, exist_ok=True)
    box, text, plates = build_plates()
    body, white = build_onepiece()
    n_box = swap_layers(text, 5)
    n_plate = int(round(mp.INLAY / LAYER))
    n_one = swap_layers(white, mp.H + 1)
    print(f"box: {box.extents.round(1)} watertight={box.is_watertight}; text layers={n_box}")
    for f, (b, a) in plates.items():
        print(f"plate {f:5s} {b.extents.round(1)} watertight={b.is_watertight}/{a.is_watertight}")
    print(f"onepiece: {body.extents.round(1)} watertight={body.is_watertight}/{white.is_watertight}; "
          f"white layers={n_one}")
    print(f"swaps (about one per layer that has both colors): plates job ~{n_plate}, box job ~{n_box}, "
          f"onepiece ~{n_one}")
    return box, text, plates, body, white


if __name__ == "__main__":
    main()
