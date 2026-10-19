"""ベイリー（ボーダーコリー）の3Dプリント用モデルを生成する。

出力 (3d/out/):
  bailey_keychain.3mf  … 顔のキーホルダー（黒/白/ピンクのマルチカラー、AMS lite向け）
  bailey_tag.3mf       … 骨型の名札（表: BAILEY 浮き文字 / 裏: 電話番号を彫刻）
  bailey_lithophane.stl… 写真のリトファン（白PLAで印刷し、光に透かす）
各パーツの STL も出力する（単色で印刷したい場合用）。
"""
import os
from functools import reduce

import numpy as np
import shapely
import trimesh
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from PIL import Image, ImageOps
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
BOLD = FontProperties(fname="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")


def ellipse(cx, cy, rx, ry, rot=0):
    e = affinity.scale(Point(0, 0).buffer(1, 64), rx, ry)
    return affinity.translate(affinity.rotate(e, rot), cx, cy)


def mirror(g):
    return affinity.scale(g, -1, 1, origin=(0, 0))


def sym(g):
    return unary_union([g, mirror(g)])


def text_poly(s, size, font=BOLD):
    tp = TextPath((0, 0), s, size=size, prop=font)
    rings = [Polygon(r) for r in tp.to_polygons() if len(r) >= 3]
    g = reduce(lambda a, b: a.symmetric_difference(b), rings)  # even-odd fill
    minx, miny, maxx, maxy = g.bounds
    return affinity.translate(g, -(minx + maxx) / 2, -(miny + maxy) / 2)


def extrude(g, h, z0=0.0):
    g = g.buffer(0)
    parts = list(g.geoms) if hasattr(g, "geoms") else [g]
    meshes = [trimesh.creation.extrude_polygon(p, h) for p in parts if p.area > 1e-3]
    m = trimesh.util.concatenate(meshes)
    m.apply_translation([0, 0, z0])
    return m


def save(name, parts):
    """parts: {部品名: mesh}。3MF (複数パーツ) と個別 STL を書き出す。"""
    os.makedirs(OUT, exist_ok=True)
    scene = trimesh.Scene()
    for pname, m in parts.items():
        scene.add_geometry(m, node_name=pname, geom_name=pname)
        m.export(os.path.join(OUT, f"{name}_{pname}.stl"))
    scene.export(os.path.join(OUT, f"{name}.3mf"))


# ---------------------------------------------------------------- キーホルダー
def bailey_face():
    """顔の2D形状 (mm)。正面を向いた写真がモデル。"""
    head = unary_union([
        ellipse(0, 0, 14, 13),
        ellipse(0, -8.5, 9, 8),                         # マズル
        sym(ellipse(-10.5, -4, 5, 6.5, 15)),            # 頬の毛
    ])
    ear = Polygon([(-11.5, 6), (-14.5, 22), (-4.5, 11)]).buffer(1.4, join_style=1)
    head = unary_union([head, sym(ear)])
    # 白い部分: 額のブレーズ〜マズル〜あご
    blaze = Polygon([(-1.4, 13.5), (1.4, 13.5), (2.6, 6), (5, -1), (-5, -1), (-2.6, 6)])
    white = unary_union([blaze.buffer(0.8), ellipse(0, -9, 8, 7),
                         ellipse(0, -15, 5.8, 3)]).intersection(head.buffer(-0.01))
    # 目: 黒い毛の中なので、白い「にっこり目」(∩) で表現
    arc = [(-6.5 + 2.4 * np.cos(t), 2.2 + 1.6 * np.sin(t)) for t in np.linspace(0.15, np.pi - 0.15, 20)]
    eyes = sym(LineString(arc).buffer(0.55, cap_style=1))
    nose = unary_union([ellipse(0, -5, 3.6, 2.5),
                        Polygon([(-2.4, -5.8), (2.4, -5.8), (0, -8)]).buffer(0.5)])
    smile = [(0, -7.5), (0, -9.4)] + [(-1.8 + 1.8 * np.cos(t), -9.4 + 1.2 * np.sin(t))
                                       for t in np.linspace(0, -np.pi, 16)]
    mouth = sym(LineString(smile + [(-4.2, -8.9)]).buffer(0.35, cap_style=1))
    tongue = unary_union([ellipse(0, -13.2, 2.6, 3.0),
                          Polygon([(-2.6, -12), (2.6, -12), (2.2, -10.4), (-2.2, -10.4)])])
    white = white.difference(nose.buffer(0.01)).difference(mouth).difference(tongue.buffer(0.01))
    white = unary_union([white, eyes])
    # 首のバンダナ（赤）をあごの下に
    bandana = Polygon([(-10.5, -14.5), (10.5, -14.5), (0, -22)]).buffer(1.2, join_style=1)
    bandana_full = bandana
    bandana = bandana.difference(ellipse(0, -15, 6.2, 3.4).buffer(0.3))
    # キーリング用の穴（耳と耳の間、額の上）
    loop = Point(0, 17.5).buffer(4.0, 48)
    hole = Point(0, 17.5).buffer(1.9, 48)
    return dict(head=head, white=white, nose=nose, tongue=tongue,
                bandana=bandana, bandana_full=bandana_full, loop=loop, hole=hole)


def keychain():
    f = bailey_face()
    base_t, top_t = 3.0, 0.8  # 土台の厚み / 浮き上がり
    silhouette = unary_union([f["head"], f["bandana_full"], f["loop"]]).difference(f["hole"])
    black = extrude(silhouette, base_t)
    top = lambda g: extrude(g.difference(f["hole"]), top_t, base_t)
    save("bailey_keychain", {
        "black_base": black,
        "white": top(f["white"]),
        "black_nose": top(f["nose"]),
        "pink_tongue": top(f["tongue"]),
        "red_bandana": top(f["bandana"].difference(f["tongue"].buffer(0.01))),
    })
    return f


# ---------------------------------------------------------------- 骨型の名札
def bone(length=58, width=14, knob=7.5):
    shaft = shapely.box(-length / 2 + knob, -width / 2, length / 2 - knob, width / 2)
    knobs = [Point(sx * (length / 2 - knob), sy * 5.5).buffer(knob, 64)
             for sx in (-1, 1) for sy in (-1, 1)]
    return unary_union([shaft, *knobs])


def tag():
    shape = bone()
    hole = Point(-24.5, 0).buffer(2.2, 48)
    base_t, top_t, engrave = 3.0, 0.8, 0.6
    name = text_poly("BAILEY", 8.5)
    name = affinity.translate(name, 3.5, 0)
    phone = mirror(affinity.translate(text_poly("090-7828-9179", 3.6), 2.5, 0))  # 裏面用に左右反転
    body = extrude(shape.difference(hole), base_t)
    cut = extrude(phone, engrave + 0.01, -0.01)
    body = body.difference(cut)
    save("bailey_tag", {"black_base": body, "white_name": extrude(name, top_t, base_t)})
    return shape, hole, name, phone


# ---------------------------------------------------------------- リトファン
def lithophane(img_path, width=100.0, t_min=0.8, t_max=3.2, frame=3.0, px=0.25):
    im = ImageOps.exif_transpose(Image.open(img_path)).convert("L")
    w, h = im.size
    ncols = int(width / px)
    nrows = int(ncols * h / w)
    im = ImageOps.autocontrast(im.resize((ncols, nrows), Image.LANCZOS), cutoff=1)
    a = np.asarray(im, dtype=float) / 255.0
    z = t_max - a * (t_max - t_min)             # 暗い所ほど厚く
    fpx = int(frame / px)                        # 外枠
    z = np.pad(z, fpx, constant_values=t_max + 1.0)
    z = z[::-1]                                  # 画像の上 → +Y
    ny, nx = z.shape
    xs, ys = np.arange(nx) * px, np.arange(ny) * px
    X, Y = np.meshgrid(xs, ys)
    top = np.column_stack([X.ravel(), Y.ravel(), z.ravel()])
    bot = np.column_stack([X.ravel(), Y.ravel(), np.zeros(z.size)])
    verts = np.vstack([top, bot])
    idx = np.arange(nx * ny).reshape(ny, nx)
    a0, b0, c0, d0 = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel()
    n = nx * ny
    faces = [np.column_stack([a0, b0, d0]), np.column_stack([a0, d0, c0]),          # 上面
             np.column_stack([a0 + n, d0 + n, b0 + n]), np.column_stack([a0 + n, c0 + n, d0 + n])]  # 底面
    ring = np.concatenate([idx[0, :], idx[1:, -1], idx[-1, -2::-1], idx[-2:0:-1, 0]])
    r2 = np.roll(ring, -1)
    faces += [np.column_stack([ring, ring + n, r2]), np.column_stack([r2, ring + n, r2 + n])]
    m = trimesh.Trimesh(verts, np.vstack(faces), process=True)
    if m.volume < 0:
        m.invert()
    os.makedirs(OUT, exist_ok=True)
    m.export(os.path.join(OUT, "bailey_lithophane.stl"))
    return m


if __name__ == "__main__":
    keychain()
    tag()
    photo = os.path.join(HERE, "photos", "bailey_front.jpeg")
    if os.path.exists(photo):
        lithophane(photo)
    for fn in sorted(os.listdir(OUT)):
        if fn.endswith((".stl", ".3mf")):
            m = trimesh.load(os.path.join(OUT, fn), force="mesh")
            print(f"{fn:36s} {m.extents.round(1)} mm  watertight={m.is_watertight}")
