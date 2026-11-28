"""ベイリーの看板「Welcome to Bailey's House」— Bambu Lab A1 mini で刷れる最大（175mm角）。

ペン立てのイラスト版と同じく、黒の上に白を重ねる厚みで濃淡を出す（色替えは黒→白の1回）。
文字は白をのせない所（＝黒）で表す。

レイアウト:
  side  … 正方形。左に全身のイラスト、右に文字を縦に積む
  house … 三角屋根の家の形。屋根に "Welcome to"、まん中に顔まわり、下に "Bailey's House"

このファイルはまず見え方の見本（明るさマップ）を作る。
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bailey_illust as bi  # noqa: E402

OUT = os.path.join(HERE, "out")
SIZE = 175.0            # 看板の外形 (mm)
FRAME = 3.0             # まわりの黒い縁 (mm)
PX = 0.1                # 明るさマップの細かさ (mm)
MPL_TTF = os.path.join(os.path.dirname(__import__("matplotlib").__file__), "mpl-data", "fonts", "ttf")
SERIF_BOLD = os.path.join(MPL_TTF, "STIXGeneralBol.ttf")      # "Bailey's" 用の太いセリフ体
SANS_BOLD = os.path.join(MPL_TTF, "DejaVuSans-Bold.ttf")


def mm(v):
    return int(round(v / PX))


def paste_illust(canvas, key, box, crop_rows=None):
    """イラストを box (x0, y0, x1, y1 mm, 上が y0) に縦横比を保って入れる。crop_rows=(上, 下) の割合で切り取り。"""
    g = bi.load(key)
    if crop_rows:
        g = g[int(g.shape[0] * crop_rows[0]):int(g.shape[0] * crop_rows[1])]
    x0, y0, x1, y1 = (mm(v) for v in box)
    k = min((x1 - x0) / g.shape[1], (y1 - y0) / g.shape[0])
    w, h = round(g.shape[1] * k), round(g.shape[0] * k)
    g = cv2.resize(g.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    ox, oy = x0 + (x1 - x0 - w) // 2, y0 + (y1 - y0 - h) // 2
    canvas[oy:oy + h, ox:ox + w] = np.minimum(canvas[oy:oy + h, ox:ox + w], g)


def draw_text(canvas, text, font_path, height_mm, center, max_w_mm=None):
    """文字を黒（0）で書く。center=(x, y) mm。文字の高さ height_mm（幅が max_w を超えるなら縮める）。"""
    size = mm(height_mm) * 1.35
    font = ImageFont.truetype(font_path, int(size))
    l, t, r, b = font.getbbox(text)
    if max_w_mm and (r - l) > mm(max_w_mm):
        font = ImageFont.truetype(font_path, int(size * mm(max_w_mm) / (r - l)))
        l, t, r, b = font.getbbox(text)
    im = Image.fromarray((canvas * 255).astype(np.uint8))
    d = ImageDraw.Draw(im)
    cx, cy = mm(center[0]), mm(center[1])
    d.text((cx - (l + r) / 2, cy - (t + b) / 2), text, font=font, fill=0)
    canvas[:] = np.asarray(im).astype(np.float32) / 255


def layout_side():
    n = mm(SIZE)
    c = np.ones((n, n), np.float32)
    shape = np.ones((n, n), bool)
    paste_illust(c, "tongue", (FRAME + 3, FRAME + 3, 92, SIZE - FRAME - 3))
    xc, w = 133, 72
    draw_text(c, "WELCOME", SANS_BOLD, 9, (xc, 52), w)
    draw_text(c, "to", SERIF_BOLD, 10, (xc, 68), w)
    draw_text(c, "Bailey's", SERIF_BOLD, 22, (xc, 92), w)
    draw_text(c, "HOUSE", SANS_BOLD, 12, (xc, 120), w)
    return c, shape


def layout_house():
    n = mm(SIZE)
    c = np.ones((n, n), np.float32)
    roof_h = 48.0
    yy, xx = np.mgrid[0:n, 0:n] * PX
    roof = yy < roof_h
    inside_roof = yy >= roof_h * (1 - np.minimum(xx, SIZE - xx) / (SIZE / 2))
    shape = ~roof | inside_roof
    draw_text(c, "WELCOME TO", SANS_BOLD, 8, (SIZE / 2, roof_h - 11), 70)
    paste_illust(c, "tongue", (FRAME + 6, roof_h + 2, SIZE - FRAME - 6, SIZE - 34), crop_rows=(0.0, 0.62))
    draw_text(c, "Bailey's House", SERIF_BOLD, 15, (SIZE / 2, SIZE - 19), SIZE - 2 * FRAME - 12)
    return c, shape


def frame_mask(shape):
    """形のふちから FRAME mm を黒い縁にする。"""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * mm(FRAME) + 1,) * 2)
    inner = cv2.erode(np.pad(shape.astype(np.uint8), 1), k)[1:-1, 1:-1]          # 画像の外は形の外として扱う
    return shape & ~inner.astype(bool)


def appearance(bright, shape, w_max=1.6, layer=0.08):
    """仕上がりの目安（白の厚みを層ごとにそろえ、ノズル幅でにじませる）。形の外は背景色。"""
    t = np.round(bright * w_max / layer) * layer
    a = cv2.GaussianBlur(np.clip(t / w_max, 0, 1), (0, 0), 0.2 / PX)
    a[frame_mask(shape)] = 0
    black, white = np.array([26, 26, 28.0]), np.array([240, 238, 232.0])
    img = black + (white - black) * a[..., None]
    img[~shape] = (205, 200, 190)
    return img.astype(np.uint8)


def main():
    os.makedirs(OUT, exist_ok=True)
    tiles = []
    for name, fn in (("side", layout_side), ("house", layout_house)):
        b, s = fn()
        img = Image.fromarray(appearance(b, s))
        img.save(os.path.join(OUT, f"preview_bailey_sign_{name}.png"))
        tiles.append(img.resize((875, 875), Image.LANCZOS))
    sheet = Image.new("RGB", (875 * 2 + 30, 875 + 20), (205, 200, 190))
    for i, t in enumerate(tiles):
        sheet.paste(t, (10 + i * 885, 10))
    sheet.save(os.path.join(OUT, "preview_bailey_sign_layouts.png"))


if __name__ == "__main__":
    main()
