"""一鉄（いってつ）の手描きイラスト入りペン立て — Bambu Lab A1 mini (180mm角) 用。

手描きの絵と「いってつ」の文字を写真から線データとして取り出し、
円筒のペン立ての表面に浮き彫り（レリーフ）として巻き付ける。

出力 (3d/ittetsu/out/):
  ittetsu_penholder.stl       … 1色印刷用（本体＋浮き彫りを一体化）★おすすめ
  ittetsu_penholder_2color.3mf… 2色用（body / relief の2パーツ, AMS lite）
  ittetsu_penholder_view.stl  … ブラウザ表示用の軽量版
"""
import os

import numpy as np
import shapely
import trimesh
from PIL import Image
from scipy import ndimage as ndi
from shapely import affinity
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from skimage import measure, morphology

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# ---- ペン立ての寸法 (mm)
R = 40.0           # 外半径（直径80mm）
H = 105.0          # 高さ
WALL = 2.4         # 壁の厚み
FLOOR = 3.0        # 底の厚み
ART_BOTTOM = 8.0   # 絵の下端の高さ
ART_HEIGHT = 90.0  # 絵（文字含む）の高さ
RELIEF = 1.0       # 浮き彫りの高さ
EMBED = 0.4        # 浮き彫りを壁に埋め込む深さ（くっつけるため）
MIN_STROKE = 0.9   # 線の最小太さ（ノズル0.4mmで確実に出る太さ）


def ink_mask(path, crop=(20, 200, 460, 900), up=3):
    """写真からペンの線だけを取り出す（影や罫線、透かし文字は除外）。"""
    im = Image.open(path).convert("L").crop(crop)
    im = im.resize((im.width * up, im.height * up), Image.BICUBIC)
    g = np.asarray(im, float)
    # 紙の明るさのムラ（影）を推定して割り算で打ち消す
    bg = ndi.median_filter(ndi.zoom(g, 0.1, order=1), size=9)
    bg = ndi.zoom(bg, (g.shape[0] / bg.shape[0], g.shape[1] / bg.shape[1]), order=1)
    ink = (g / np.maximum(bg, 1)) < 0.55
    ink = morphology.remove_small_objects(ink, max_size=60)
    return ink


def mask_to_polygons(ink, mm_per_px):
    """2値画像 → shapely ポリゴン (mm)。線は MIN_STROKE まで太らせる。"""
    r = max(1, int(round((MIN_STROKE / mm_per_px - 12) / 2)))   # 元の線幅は約12px
    ink = morphology.dilation(ink, morphology.disk(r))
    ink = morphology.remove_small_holes(ink, max_size=int((0.8 / mm_per_px) ** 2))
    pad = np.pad(ink, 2).astype(float)
    rings = []
    for c in measure.find_contours(pad, 0.5):
        if len(c) < 4:
            continue
        p = Polygon(np.column_stack([c[:, 1], -c[:, 0]]))  # 行→下向き, 列→右
        p = p.buffer(0)
        if p.area > 4:
            rings.append(p)
    # 入れ子の深さで「塗り / 穴」を判定
    tree = STRtree(rings)
    depth = []
    for i, p in enumerate(rings):
        pt = p.representative_point()
        depth.append(sum(1 for j in tree.query(pt, predicate="within") if j != i and rings[j].area > p.area))
    res = shapely.Polygon()
    for d in range(max(depth) + 1):
        layer = shapely.union_all([rings[i] for i in range(len(rings)) if depth[i] == d])
        res = res.union(layer) if d % 2 == 0 else res.difference(layer)
    res = affinity.scale(res, mm_per_px, mm_per_px, origin=(0, 0))
    return res


def place_art(g):
    """絵を中央・下端 ART_BOTTOM に配置（u=周方向, v=高さ）。"""
    minx, miny, maxx, maxy = g.bounds
    return affinity.translate(g, -(minx + maxx) / 2, ART_BOTTOM - miny)


def wrap(mesh, radius):
    """平面上のメッシュ (x=周方向, y=高さ, z=外向き) を円筒に巻き付ける。正面は -Y。"""
    mesh = mesh.copy()
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=0.8, max_iter=20)
    th = v[:, 0] / radius
    r = radius + v[:, 2]
    out = np.column_stack([r * np.sin(th), -r * np.cos(th), v[:, 1]])
    return trimesh.Trimesh(out, f, process=True)


def relief_mesh(art, tol):
    art = art.simplify(tol, preserve_topology=True).buffer(0)
    parts = [trimesh.creation.extrude_polygon(p, RELIEF + EMBED) for p in getattr(art, "geoms", [art]) if p.area > 0.05]
    flat = trimesh.util.concatenate(parts)
    flat.apply_translation([0, 0, -EMBED])
    return wrap(flat, R)


def body_mesh(sections=360):
    outer = trimesh.creation.cylinder(radius=R, height=H, sections=sections)
    outer.apply_translation([0, 0, H / 2])
    inner = trimesh.creation.cylinder(radius=R - WALL, height=H, sections=sections)
    inner.apply_translation([0, 0, H / 2 + FLOOR])
    return outer.difference(inner)


def main():
    os.makedirs(OUT, exist_ok=True)
    ink = ink_mask(os.path.join(HERE, "ittetsu_drawing.jpg"))
    mm_per_px = ART_HEIGHT / ink.shape[0]
    art = place_art(mask_to_polygons(ink, mm_per_px))
    print(f"art: {art.bounds[2]-art.bounds[0]:.1f} x {art.bounds[3]-art.bounds[1]:.1f} mm, "
          f"{len(getattr(art, 'geoms', [art]))} pieces")

    body = body_mesh()
    relief = relief_mesh(art, tol=0.04)
    scene = trimesh.Scene()
    scene.add_geometry(body.difference(relief), node_name="body", geom_name="body")  # 重なりをなくす
    scene.add_geometry(relief, node_name="relief", geom_name="relief")
    scene.export(os.path.join(OUT, "ittetsu_penholder_2color.3mf"))

    single = body.union(relief)
    single.export(os.path.join(OUT, "ittetsu_penholder.stl"))

    view = body_mesh(sections=180).union(relief_mesh(art, tol=0.12))
    view.export(os.path.join(OUT, "ittetsu_penholder_view.stl"))

    for name, m in [("single", single), ("body", body), ("relief", relief), ("view", view)]:
        print(f"{name:7s} {m.extents.round(1)} mm  faces={len(m.faces)}  watertight={m.is_watertight}  vol={m.volume/1000:.1f}cm3")


if __name__ == "__main__":
    main()
