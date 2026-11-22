"""ベイリーの4面アルバム（プレート差し込み式）— Bambu Lab A1 mini + AMS lite。

4枚の写真を白黒のプレートにして、箱の4面のアリ溝へ上から差し込む。
  - プレート: 絵の面を下にして平置きで印刷。黒は最初の 0.6mm だけなので色替えは数回（4枚まとめて1回で印刷）。
  - 舌はピンクを使わず、白い舌を黒い線で縁取る（白・黒の2色だけ）。
  - 箱: 単色で立てて印刷（色替えなし、サポートなし）。色は自由（緑・白・赤・黒など）。
    中は真ん中の仕切りで左右2部屋。

出力 (3d/out/):
  bailey_album_box.stl      … 箱（単色）
  bailey_album_plates.3mf   … プレート4枚（各 plate_white / art_black の2パーツ）を1枚のビルドプレートに配置
"""
import math
import os
import sys

import numpy as np
import shapely
import trimesh
from shapely import affinity
from shapely.geometry import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ittetsu"))
import make_bailey_album as ma  # noqa: E402
import make_bailey_penholder as mb  # noqa: E402
from make_bailey import extrude  # noqa: E402
from make_pencase import write_3mf  # noqa: E402

OUT = mb.OUT

# ---- 箱 (mm)
W, D, H = 90.0, 75.0, 100.0
CORNER = 3.0
WALL = 5.2              # 溝（深さ 3.2）＋ 裏に 2mm 残す
FLOOR = 2.4
DIV, DIV_TOP = 2.0, H - 10.0

# ---- プレートとアリ溝
T = 3.0                 # プレートの厚み
SLOPE = math.radians(30)
INSET = T * math.tan(SLOPE)          # 表（絵の面）が裏より片側これだけ狭い
CLEAR = 0.2             # 溝の奥行き方向のすき間
CLEAR_X = 0.35          # 溝の幅方向の広げ量（斜面で片側 約0.2mm）
GROOVE_Z = 16.0         # 溝の底の高さ（この下は箱の台座になる）
PH = H - GROOVE_Z + 1.0              # プレートの長さ（縁から 1mm 出してつまめるように）
PW = {"front": 74.0, "back": 74.0, "left": 59.0, "right": 59.0}   # プレートの裏側の幅
INLAY = 0.6             # 黒の深さ（0.2mm で3層）
TONGUE_LINE = 0.9       # 舌の縁取りの太さ

PHOTOS = {"front": ma.FRONT, "left": ma.LEFT, "right": ma.RIGHT, "back": ma.BACK}


def art_for(face):
    """面の写真 → 黒の図形（舌は白のまま黒で縁取る）。プレートの見える範囲の中央 (0, 中央) に置く。"""
    visible_w = PW[face] - 2 * INSET
    fit = (visible_w - 4.0, PH - 1.0 - 4.0)
    b, p = ma.fit_face(PHOTOS[face], fit)
    b = ma.clean(b)
    if not p.is_empty:
        p = ma.clean(p)
        b = shapely.union_all([b.difference(p), p.difference(p.buffer(-TONGUE_LINE))]).buffer(0)
    # fit_face は中央が (0, H/2)。プレートの長さ方向の中央へ
    return affinity.translate(b, 0, (PH - 1.0) / 2 - H / 2)


def plate(face):
    """平置きのプレート（絵の面が z=0、x 中央 0、y 0..PH）と黒のパーツ。
    下から見て正しく読めるよう、絵は左右反転して入れる。"""
    wb = PW[face] / 2
    wf = wb - INSET
    sec = Polygon([(-wf, 0), (wf, 0), (wb, T), (-wb, T)])          # (x, z) の断面: 絵の面が狭い
    m = trimesh.creation.extrude_polygon(sec, PH)                   # (x, z断面, y長さ)
    v = m.vertices.copy()
    m.vertices = np.column_stack([v[:, 0], v[:, 2], v[:, 1]])
    m.fix_normals()
    art = affinity.scale(art_for(face), -1, 1, origin=(0, 0))
    art_m = extrude(art, INLAY, 0)
    body = m.difference(extrude(art, INLAY + 0.05, -0.05))
    return body, art_m


def face_map(face, u, d, z):
    """面の座標（u=外から見て右, d=表面からの深さ, z=高さ）→ 箱の座標。"""
    if face == "front":
        return np.column_stack([W / 2 + u, d, z])
    if face == "back":
        return np.column_stack([W / 2 - u, D - d, z])
    if face == "left":
        return np.column_stack([d, D / 2 - u, z])
    return np.column_stack([W - d, D / 2 + u, z])


def groove(face):
    wb = PW[face] / 2 + CLEAR_X
    wf = wb - INSET - CLEAR * math.tan(SLOPE)
    sec = Polygon([(-wf, -0.1), (wf, -0.1), (wb, T + CLEAR), (-wb, T + CLEAR)])   # (u, d)
    m = trimesh.creation.extrude_polygon(sec, H + 1 - GROOVE_Z)
    v = m.vertices.copy()
    m.vertices = face_map(face, v[:, 0], v[:, 1], v[:, 2] + GROOVE_Z)
    m.fix_normals()
    return m


def rbox(x0, y0, x1, y1, r):
    return shapely.box(x0 + r, y0 + r, x1 - r, y1 - r).buffer(r, quad_segs=16)


def make_box():
    box = extrude(rbox(0, 0, W, D, CORNER), H).difference(
        extrude(rbox(WALL, WALL, W - WALL, D - WALL, 1.5), H, FLOOR))
    wall = shapely.box(W / 2 - DIV / 2, WALL - 0.5, W / 2 + DIV / 2, D - WALL + 0.5)
    box = box.union(extrude(wall, DIV_TOP - FLOOR + 0.5, FLOOR - 0.5))
    for face in PW:
        box = box.difference(groove(face))
    return box


def installed(face, m):
    """平置きのプレートを、箱の面に差し込んだ位置へ（プレビュー用）。"""
    m = m.copy()
    v = m.vertices.copy()
    m.vertices = face_map(face, -v[:, 0], v[:, 2], v[:, 1] + GROOVE_Z)
    m.fix_normals()
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    box = make_box()
    box.export(os.path.join(OUT, "bailey_album_box.stl"))
    print(f"box {box.extents.round(1)} mm watertight={box.is_watertight} vol={box.volume / 1000:.1f}cm3")
    plates = {f: plate(f) for f in PW}
    # 1枚のビルドプレート (180x180) に 2x2 で並べる
    centers = {"front": (49, 46), "back": (127, 46), "left": (41.5, 135), "right": (104.5, 135)}
    objs = []
    for f, (body, art) in plates.items():
        objs.append((f"plate_{f}", centers[f], [("plate_white", body, 1), ("art_black", art, 2)]))
        print(f"plate {f:5s} {body.extents.round(1)} mm watertight={body.is_watertight}/{art.is_watertight}")
    write_3mf(os.path.join(OUT, "bailey_album_plates.3mf"), objs)
    return box, plates


if __name__ == "__main__":
    main()
