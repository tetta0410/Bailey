"""ベイリーのリトファン・ランタン — Bambu Lab A1 mini 用（白1色）。

耳まで写った写真 (photos/bailey_ears.jpg) を、正面の壁の厚みに変える（暗いところほど厚い）。
市販の LED ティーライトの上にかぶせると、写真のベイリーが光の中に浮かぶ。
背景（カーテン・テレビ・床）は外して白（いちばん薄い = いちばん明るい）にする。

形: 上下が開いた四角い筒（正面がリトファン、ほか3面は白い壁）。立てたまま印刷、サポート不要。
外側の表面は平らで、凹凸は内側。窓の下には BAILEY の文字を外から彫る（薄くなるので点灯すると光る）。

出力: 3d/out/bailey_lantern.stl
"""
import os
import sys

import numpy as np
import shapely
import trimesh
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from scipy import ndimage as ndi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ittetsu"))
from make_bailey import extrude, text_poly  # noqa: E402

OUT = os.path.join(HERE, "out")
PHOTO = os.path.join(HERE, "photos", "bailey_ears.jpg")

# ---- 形 (mm)
W, D, H = 86.0, 48.0, 116.0     # 外形（幅・奥行き・高さ）。内側の奥行き 43mm に直径38mm前後のティーライトが入る
WALL = 2.0                      # 側面・背面の壁
FRONT = 3.0                     # 正面の壁（リトファン以外の部分）
WIN_W, WIN_H = 76.0, 93.0       # リトファンの窓
WIN_BOTTOM = 14.0               # 窓の下端（下にティーライトが隠れる）
T_MIN, T_MAX = 0.6, 3.0         # リトファンの厚み（明るい〜暗い）
PX = 0.25                       # リトファンの格子の細かさ
CORNER = 4.0
NAME = "BAILEY"
NAME_H = 6.0                    # 文字の高さ
NAME_DEPTH = 1.8                # 彫る深さ（残りの壁 1.2mm が光を通す）

# ---- 写真
CROP = (80, 0, 1180, 1340)      # 使う範囲（写真の画素）
# ベイリーの輪郭（耳の先〜首）。これより外は背景として白にする
OUTLINE = [(110, 0), (330, 0), (380, 120), (430, 230), (560, 230), (690, 230), (740, 120), (770, 0),
           (1130, 0), (1150, 150), (1110, 300), (1080, 450), (1050, 560), (1030, 680), (990, 800),
           (990, 950), (1010, 1100), (990, 1250), (960, 1340), (320, 1340), (300, 1200), (290, 1050),
           (290, 900), (260, 760), (240, 640), (210, 520), (180, 400), (140, 260), (110, 120)]
FADE = 120                      # 首の下端を何画素かけて白へぼかすか


def photo_brightness():
    """写真 → 0..1 の明るさ（窓の大きさ, 上が行0）。背景は 1（白）。"""
    im = ImageOps.exif_transpose(Image.open(PHOTO)).convert("L")
    mask = Image.new("L", im.size, 0)
    ImageDraw.Draw(mask).polygon(OUTLINE, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(6))
    im, mask = im.crop(CROP), mask.crop(CROP)
    nx, ny = int(WIN_W / PX), int(WIN_H / PX)
    im = ImageOps.autocontrast(im.resize((nx, ny), Image.LANCZOS), cutoff=1)
    m = np.asarray(mask.resize((nx, ny), Image.LANCZOS), float) / 255
    v = np.asarray(im, float) / 255
    fade = np.clip((ny - np.arange(ny)) / (FADE * ny / (CROP[3] - CROP[1])), 0, 1)[:, None]
    m = m * fade
    return v * m + (1 - m)


def heightmap_mesh(t, px):
    """厚みマップ t[行, 列]（行0が上）→ 裏が z=0、表が z=t の閉じたメッシュ（x 右, y 上）。"""
    t = t[::-1]
    ny, nx = t.shape
    X, Y = np.meshgrid(np.arange(nx) * px, np.arange(ny) * px)
    top = np.column_stack([X.ravel(), Y.ravel(), t.ravel()])
    bot = np.column_stack([X.ravel(), Y.ravel(), np.zeros(t.size)])
    idx = np.arange(nx * ny).reshape(ny, nx)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    n = nx * ny
    faces = [np.column_stack([a, b, d]), np.column_stack([a, d, c]),
             np.column_stack([a + n, d + n, b + n]), np.column_stack([a + n, c + n, d + n])]
    ring = np.concatenate([idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]])
    r2 = np.roll(ring, -1)
    faces += [np.column_stack([ring, ring + n, r2]), np.column_stack([r2, ring + n, r2 + n])]
    m = trimesh.Trimesh(np.vstack([top, bot]), np.vstack(faces), process=True)
    if m.volume < 0:
        m.invert()
    return m


def rbox(x0, y0, x1, y1, r):
    return shapely.box(x0 + r, y0 + r, x1 - r, y1 - r).buffer(r, quad_segs=16)


def make_lantern():
    v = photo_brightness()
    t = T_MIN + (1 - v) * (T_MAX - T_MIN)
    win = heightmap_mesh(t, PX)                      # x: 0..WIN_W, y: 0..WIN_H, z: 0..t（外側の面が z=0）
    # 格子の端を窓ぴったりより 0.5mm ずつ外へ広げ、まわりの枠と確実に重ねる（すき間を作らない）
    v_ = win.vertices.copy()
    sx = (WIN_W + 1.0) / v_[:, 0].max()
    sy = (WIN_H + 1.0) / v_[:, 1].max()
    # 立てる: (x, y, z) → (x + 窓の左端, 奥行き = z, 高さ = y + 窓の下端)。外側の面が y=0
    win.vertices = np.column_stack([v_[:, 0] * sx + (W - WIN_W) / 2 - 0.5, v_[:, 2],
                                    v_[:, 1] * sy + WIN_BOTTOM - 0.5])
    win.fix_normals()
    # 筒: 外形 − 内側。正面の壁は窓の部分をくり抜く
    shell = extrude(rbox(0, 0, W, D, CORNER), H).difference(
        extrude(rbox(WALL, FRONT, W - WALL, D - WALL, 1.5), H + 2, -1))
    hole = shapely.box((W - WIN_W) / 2, WIN_BOTTOM, (W + WIN_W) / 2, WIN_BOTTOM + WIN_H)
    cut = extrude(hole, FRONT + 0.2, -0.1)
    v_ = cut.vertices.copy()
    cut.vertices = np.column_stack([v_[:, 0], v_[:, 2], v_[:, 1]])
    cut.fix_normals()
    shell = shell.difference(cut)
    # 窓の下の帯に名前を彫る（外側から。文字の部分だけ壁が薄くなり、点灯すると光る）
    name = text_poly(NAME, NAME_H)
    nb = name.bounds
    name = shapely.affinity.scale(name, NAME_H / (nb[3] - nb[1]) * 1.0, NAME_H / (nb[3] - nb[1]), origin=(0, 0))
    name = shapely.affinity.translate(name, W / 2, WIN_BOTTOM / 2)
    eng = extrude(name, NAME_DEPTH + 0.1, -0.1)
    v_ = eng.vertices.copy()
    eng.vertices = np.column_stack([v_[:, 0], v_[:, 2], v_[:, 1]])
    eng.fix_normals()
    return shell.difference(eng).union(win)


def main():
    m = make_lantern()
    os.makedirs(OUT, exist_ok=True)
    m.export(os.path.join(OUT, "bailey_lantern.stl"))
    print(f"lantern {m.extents.round(1)} mm watertight={m.is_watertight} vol={m.volume / 1000:.1f}cm3 "
          f"faces={len(m.faces)}")
    return m


if __name__ == "__main__":
    main()
