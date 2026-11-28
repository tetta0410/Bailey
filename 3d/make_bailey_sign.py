"""ベイリーの看板「Welcome to Bailey's House」— Bambu Lab A1 mini で刷れる最大サイズ。

ペン立てのイラスト版と同じく、黒の上に白を重ねる厚みで濃淡を出す。舌だけは白の代わりに赤を重ねる
（赤が厚い所は赤、薄い所は黒が透けて暗い赤）。文字と縁は黒。

  A: 1枚もの  179×179mm（A1 mini のプレートいっぱい）… 左にベイリーの全身、右に文字（舌も白黒）。四隅にワイヤー穴
  B: 2枚つなぎ 340×174mm … 左の板にベイリーの顔（大きく）、右の板に文字。
     裏側のアリ継ぎ（あり・ほぞ）で横につなぐ（表からは見えない）
B は四隅に結束バンド用の穴（φ3mm）。B を作るときは --with-b。

板は絵の面を上にして印刷する（裏が平ら）。色: 1=白, 2=黒, 3=赤（いまの AMS lite の並び）。

出力 (3d/out/):
  bailey_sign_A.3mf / bailey_sign_B_left.3mf / bailey_sign_B_right.3mf
  preview_bailey_sign_A.png / preview_bailey_sign_B.png / preview_bailey_sign_sizes.png
"""
import os
import sys

import cv2
import fast_simplification
import numpy as np
import shapely
import trimesh
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ittetsu"))
import bailey_illust as bi  # noqa: E402
from make_bailey import extrude  # noqa: E402
from make_bailey_lantern import heightmap_mesh  # noqa: E402
from make_pencase import write_3mf  # noqa: E402

OUT = os.path.join(HERE, "out")
MPL_TTF = os.path.join(os.path.dirname(__import__("matplotlib").__file__), "mpl-data", "fonts", "ttf")
SERIF = os.path.join(MPL_TTF, "STIXGeneralBol.ttf")
SANS = os.path.join(MPL_TTF, "DejaVuSans-Bold.ttf")

WHITE_FIL, BLACK_FIL, RED_FIL = 1, 2, 3
T = 4.0                 # 板の厚み（屋外でたわみにくいように）
W_MAX = 1.6             # 白（赤）のいちばん厚い所
ZB = T - W_MAX          # 黒→白（赤）に替わる高さ
LAYER = 0.08
FRAME = 3.0             # 黒い縁
GRID = 0.2              # 厚みマップの細かさ（0.4mm ノズルには十分）
HOLE_R, RING_R, HOLE_IN = 1.5, 2.8, 6.0       # ワイヤー穴の半径（φ3mm）・黒い輪の半径・縁からの位置
KEEP = 0.3

# 舌の輪郭（bailey_illust.load("tongue") の画素座標）
TONGUE = [(345, 722), (380, 716), (410, 714), (445, 716), (482, 722), (490, 745), (497, 765), (504, 790),
          (506, 815), (503, 840), (496, 862), (487, 882), (470, 900), (448, 913), (425, 918), (410, 914),
          (395, 918), (372, 915), (352, 903), (337, 885), (328, 862), (323, 835), (325, 805), (330, 775),
          (337, 748)]

# アリ継ぎ（B の左右の板をつなぐ）: 裏側の下半分だけ。表の絵には出ない
TAB_Y = (50.0, 124.0)   # 板の下からの位置（中心）
TAB_NECK, TAB_END, TAB_LEN = 9.0, 13.0, 6.0
TAB_Z, SOCKET_Z, CLEAR = 1.8, 2.0, 0.2


# ---------------------------------------------------------------- 絵（明るさマップ）
class Canvas:
    """板の明るさマップ（行0が上, 1画素 px mm）。red=舌の所, keep=黒にする所。"""

    def __init__(self, w, h, px):
        self.w, self.h, self.px = w, h, px
        self.nx, self.ny = int(round(w / px)) + 1, int(round(h / px)) + 1
        self.b = np.ones((self.ny, self.nx), np.float32)
        self.red = np.zeros((self.ny, self.nx), bool)

    def p(self, v):
        return int(round(v / self.px))

    def illust(self, key, box, crop=None, tongue=False):
        g = bi.load(key)
        r0 = int(g.shape[0] * crop[0]) if crop else 0
        r1 = int(g.shape[0] * crop[1]) if crop else g.shape[0]
        g = g[r0:r1]
        x0, y0, x1, y1 = (self.p(v) for v in box)
        k = min((x1 - x0) / g.shape[1], (y1 - y0) / g.shape[0])
        w, h = round(g.shape[1] * k), round(g.shape[0] * k)
        ox, oy = x0 + (x1 - x0 - w) // 2, y0 + (y1 - y0 - h) // 2
        gg = cv2.resize(g.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        self.b[oy:oy + h, ox:ox + w] = np.minimum(self.b[oy:oy + h, ox:ox + w], gg)
        if tongue:
            m = np.zeros((self.ny, self.nx), np.uint8)
            pts = np.array([[ox + x * k, oy + (y - r0) * k] for x, y in TONGUE], np.int32)
            cv2.fillPoly(m, [pts], 1)
            self.red |= m.astype(bool)

    def text(self, s, font, height_mm, cx, cy, max_w=None):
        size = self.p(height_mm) * 1.35
        f = ImageFont.truetype(font, max(1, int(size)))
        l, t, r, b = f.getbbox(s)
        if max_w and (r - l) > self.p(max_w):
            f = ImageFont.truetype(font, max(1, int(size * self.p(max_w) / (r - l))))
            l, t, r, b = f.getbbox(s)
        im = Image.fromarray((self.b * 255).astype(np.uint8))
        ImageDraw.Draw(im).text((self.p(cx) - (l + r) / 2, self.p(cy) - (t + b) / 2), s, font=f, fill=0)
        self.b = np.asarray(im).astype(np.float32) / 255

    def finish(self, holes):
        """縁と取り付け穴のまわりを黒に。"""
        f = self.p(FRAME)
        self.b[:f + 1], self.b[-f - 1:], self.b[:, :f + 1], self.b[:, -f - 1:] = 0, 0, 0, 0
        yy, xx = np.mgrid[0:self.ny, 0:self.nx] * self.px
        for hx, hy in holes:
            ring = (xx - hx) ** 2 + (yy - hy) ** 2 <= RING_R ** 2
            self.b[ring] = 0
            self.red[ring] = False


def corner_holes(w, h):
    return [(HOLE_IN, HOLE_IN), (w - HOLE_IN, HOLE_IN), (HOLE_IN, h - HOLE_IN), (w - HOLE_IN, h - HOLE_IN)]


def edge_holes(w, h):
    """上下左右の辺のまん中（ワイヤー用）。"""
    return [(w / 2, HOLE_IN), (w / 2, h - HOLE_IN), (HOLE_IN, h / 2), (w - HOLE_IN, h / 2)]


def text_block(c, cx, cy, width):
    """WELCOME / to / Bailey's / HOUSE を (cx, cy) を中心に並べる（幅 width）。"""
    s = width / 72.0
    top = cy - 40 * s
    c.text("WELCOME", SANS, 9 * s, cx, top + 6 * s, width)
    c.text("to", SERIF, 10 * s, cx, top + 21 * s, width)
    c.text("Bailey’s", SERIF, 22 * s, cx, top + 45 * s, width)
    c.text("HOUSE", SANS, 12 * s, cx, top + 72 * s, width)


SIGN = 179.0            # A1 mini のプレート 180mm 角いっぱい（縁から 0.5mm ずつ残す）


def layout_A(px):
    W = H = SIGN
    k = SIGN / 174.0                                # 174mm で決めた配置をそのまま拡大
    c = Canvas(W, H, px)
    c.illust("tongue", (14 * k, 13 * k, 90 * k, 161 * k))
    text_block(c, 130 * k, H / 2, 62 * k)
    holes = corner_holes(W, H)
    c.finish(holes)
    return c, holes


def layout_B_left(px):
    W, H = 170.0, 174.0
    c = Canvas(W, H, px)
    c.illust("tongue", (12, 10, W - 12, H - 10), crop=(0.0, 0.64), tongue=True)
    holes = corner_holes(W, H)
    c.finish(holes)
    return c, holes


def layout_B_right(px):
    W, H = 170.0, 174.0
    c = Canvas(W, H, px)
    text_block(c, W / 2, H / 2, 118)
    holes = corner_holes(W, H)
    c.finish(holes)
    return c, holes


# ---------------------------------------------------------------- 立体
def thickness(b):
    return np.clip(np.round(b * W_MAX / LAYER) * LAYER, 0.02, W_MAX)


def decimate(m):
    # 面数を減らす。減らし方によってまれに閉じた形が崩れるので、崩れたら別の割合で試す
    for keep in (KEEP, 0.25, 0.2, 0.15, 0.45, 0.6, 0.8):
        pts, tri = fast_simplification.simplify(m.vertices, m.faces, target_reduction=1 - keep)
        d = trimesh.Trimesh(pts, tri, process=True)
        d.fix_normals()
        if d.is_watertight:
            return d
    return m


def relief(t, x0, y_top, px):
    """厚みマップ t（行0が上）を、左上 (x0, y_top) mm の位置に ZB の高さから載せる。"""
    m = heightmap_mesh(t, px)
    m.apply_translation([x0, y_top - (t.shape[0] - 1) * px, ZB])
    return decimate(m)


def tabs(W):
    """左の板の右側に出るアリ継ぎのほぞ（裏側 TAB_Z の高さまで）。"""
    shapes = [Polygon([(W, y - TAB_NECK / 2), (W + TAB_LEN, y - TAB_END / 2),
                       (W + TAB_LEN, y + TAB_END / 2), (W, y + TAB_NECK / 2)]) for y in TAB_Y]
    return extrude(shapely.union_all(shapes).buffer(0), TAB_Z, 0)


def sockets():
    """右の板の左側に彫るあり（ほぞより CLEAR 大きい）。"""
    shapes = [Polygon([(-0.1, y - TAB_NECK / 2 - CLEAR), (TAB_LEN + CLEAR, y - TAB_END / 2 - CLEAR),
                       (TAB_LEN + CLEAR, y + TAB_END / 2 + CLEAR), (-0.1, y + TAB_NECK / 2 + CLEAR)]) for y in TAB_Y]
    return extrude(shapely.union_all(shapes).buffer(0), SOCKET_Z + 0.1, -0.1)


def build(layout, joint=None):
    c, holes = layout(GRID)
    W, H = c.w, c.h
    f = c.p(FRAME)
    art = (slice(f, c.ny - f), slice(f, c.nx - f))
    tb = thickness(c.b)
    t_white = np.where(c.red, 0.02, tb)[art]
    body = extrude(shapely.box(0, 0, W, H), T, 0).difference(
        extrude(shapely.box(FRAME, FRAME, W - FRAME, H - FRAME), T, ZB))
    if joint == "tabs":
        body = body.union(tabs(W))
    if joint == "sockets":
        body = body.difference(sockets())
    cyl = trimesh.util.concatenate([extrude(shapely.Point(x, H - y).buffer(HOLE_R, quad_segs=24), T + 2, -1)
                                    for x, y in holes])
    body = body.difference(cyl)
    white = relief(t_white, FRAME, H - FRAME, GRID).difference(cyl)
    parts = [("sign_black", body, BLACK_FIL), ("sign_white", white, WHITE_FIL)]
    if c.red.any():
        rows, cols = np.nonzero(c.red)
        r0, r1, c0, c1 = rows.min() - 2, rows.max() + 3, cols.min() - 2, cols.max() + 3
        t_red = np.where(c.red, tb, 0.02)[r0:r1, c0:c1]
        red = relief(t_red, c0 * GRID, H - r0 * GRID, GRID)
        parts.append(("tongue_red", red, RED_FIL))
    return parts


# ---------------------------------------------------------------- 見え方
def appearance(layout, px=0.1):
    c, holes = layout(px)
    a = cv2.GaussianBlur(np.clip(np.round(c.b * W_MAX / LAYER) * LAYER / W_MAX, 0, 1), (0, 0), 0.2 / px)
    black = np.array([26, 26, 28.0])
    top = np.where(c.red[..., None], np.array([196, 32, 38.0]), np.array([240, 238, 232.0]))
    img = black + (top - black) * a[..., None]
    yy, xx = np.mgrid[0:c.ny, 0:c.nx] * px
    for hx, hy in holes:
        img[(xx - hx) ** 2 + (yy - hy) ** 2 <= HOLE_R ** 2] = (150, 165, 175)   # 穴（フェンスが見える）
    return Image.fromarray(img.astype(np.uint8))


def main(build_meshes=True, with_b=False):
    os.makedirs(OUT, exist_ok=True)
    A = appearance(layout_A)
    A.save(os.path.join(OUT, "preview_bailey_sign_A.png"))
    if not with_b:                                  # いまは1枚もの（A）だけ作る
        if build_meshes:
            parts = build(layout_A)
            write_3mf(os.path.join(OUT, "bailey_sign_A.3mf"), [("bailey_sign_A", (90, 90), parts)])
            for n, m, _ in parts:
                print(f"A        {n:11s} {m.extents.round(1)} faces={len(m.faces)} watertight={m.is_watertight}")
        return
    L, R = appearance(layout_B_left), appearance(layout_B_right)
    B = Image.new("RGB", (L.width + R.width, L.height))
    B.paste(L, (0, 0)); B.paste(R, (L.width, 0))
    B.save(os.path.join(OUT, "preview_bailey_sign_B.png"))
    # 大きさくらべ（同じ縮尺で並べる）
    k = 0.45
    a2, b2 = A.resize((int(A.width * k), int(A.height * k))), B.resize((int(B.width * k), int(B.height * k)))
    sheet = Image.new("RGB", (a2.width + b2.width + 90, max(a2.height, b2.height) + 60), (170, 180, 186))
    sheet.paste(a2, (30, 30)); sheet.paste(b2, (a2.width + 60, 30))
    sheet.save(os.path.join(OUT, "preview_bailey_sign_sizes.png"))
    if not build_meshes:
        return
    for name, layout, joint in (("A", layout_A, None), ("B_left", layout_B_left, "tabs"),
                                ("B_right", layout_B_right, "sockets")):
        parts = build(layout, joint)
        write_3mf(os.path.join(OUT, f"bailey_sign_{name}.3mf"), [(f"bailey_sign_{name}", (90, 90), parts)])
        for n, m, _ in parts:
            print(f"{name:8s} {n:11s} {m.extents.round(1)} faces={len(m.faces)} watertight={m.is_watertight}")


if __name__ == "__main__":
    main(build_meshes="--preview" not in sys.argv, with_b="--with-b" in sys.argv)
