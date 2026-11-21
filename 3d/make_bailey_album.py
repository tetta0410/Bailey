"""ベイリーの成長アルバム・ペン立て — 4面すべてにベイリー。Bambu Lab A1 mini + AMS lite。

正面: 黒（と舌のピンク）をはめ込んだ写し絵。
左・右・背面: 写真から取った黒い毛の形を、本体と同じ色のまま浅く彫る（色替えなし）。

顔（正面）を下にして倒して印刷すると、色が入るのは最初の 0.8mm だけで、色替えは十数回。
左右の彫りは立った壁、背面の彫りは一番上の面になるので、どちらもそのまま印刷できる。
箱の中は真ん中の仕切りで左右2部屋。背面が天井になるので、中にツリーサポートが必要。

出力 (3d/out/):
  bailey_album_facedown.3mf … 【おすすめ】顔を下にして倒した向き
  bailey_album.3mf          … 立てた向き（サポート不要だが正面の絵の高さで毎層色替え）
"""
import os
import sys

import numpy as np
import shapely
from shapely import affinity

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ittetsu"))
import make_bailey_penholder as mb  # noqa: E402
from make_bailey import extrude  # noqa: E402
from make_pencase import write_3mf  # noqa: E402

W, D, H, CORNER = mb.W, mb.D, mb.H, mb.CORNER
ENGRAVE = 0.6                 # 左右・背面の彫りの深さ

# 面ごとの写真（mb.placed_art の名前）
FRONT = "okaeri"
LEFT = "puppy"
RIGHT = "ears_in"
BACK = "tongue"

SIDE_FIT = (D - 2 * CORNER - 6, 90.0)    # 左右の面に収める幅・高さ
BACK_FIT = (W - 2 * CORNER - 6, 90.0)    # 背面に収める幅・高さ


def engrave_shape(photo, fit):
    """写真の黒い毛（と舌）の形を、面に収まる大きさにして中央 (0, H/2) に置く。"""
    art = mb.placed_art(photo)
    g = shapely.union_all([art[0], art[1]]).buffer(0)
    x0, y0, x1, y1 = g.bounds
    k = min(fit[0] / (x1 - x0), fit[1] / (y1 - y0))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return affinity.translate(affinity.scale(g, k, k, origin=(cx, cy)), -cx, H / 2 - cy)


def on_face(g, face, depth):
    """2D 図形 g（u=右, v=上, 面の中央が u=0）を、外から見た向きのまま各面に depth の深さで置く。"""
    m = extrude(g, depth + 0.1, -0.1)          # z: 外側(-0.1) → 内側(depth)
    u, v, z = m.vertices[:, 0], m.vertices[:, 1], m.vertices[:, 2]
    if face == "left":      # x=0 の面。外から見ると右が -y
        xyz = np.column_stack([z, D / 2 - u, v])
    elif face == "right":   # x=W の面。外から見ると右が +y
        xyz = np.column_stack([W - z, D / 2 + u, v])
    elif face == "back":    # y=D の面。外から見ると右が -x
        xyz = np.column_stack([W / 2 - u, D - z, v])
    else:
        raise ValueError(face)
    m.vertices = xyz
    m.fix_normals()
    return m


def main():
    black, pink = mb.placed_art(FRONT)[:2]
    body = mb.make_body()
    body = body.difference(mb.on_front(shapely.union_all([black, pink]), mb.INLAY, 0.05))
    for face, photo, fit in [("left", LEFT, SIDE_FIT), ("right", RIGHT, SIDE_FIT), ("back", BACK, BACK_FIT)]:
        body = body.difference(on_face(engrave_shape(photo, fit), face, ENGRAVE))
    parts = [("body_white", body, 1), ("face_black", mb.on_front(black, mb.INLAY), 2)]
    if not pink.is_empty:
        parts.append(("tongue_pink", mb.on_front(pink, mb.INLAY), 3))
    os.makedirs(mb.OUT, exist_ok=True)
    write_3mf(os.path.join(mb.OUT, "bailey_album.3mf"), [("bailey_album", (90, 90), parts)])
    write_3mf(os.path.join(mb.OUT, "bailey_album_facedown.3mf"), [(
        "bailey_album_facedown", (90, 90), [(n, mb.face_down(m), e) for n, m, e in parts])])
    for n, m, _ in parts:
        print(f"{n:12s} {m.extents.round(1)} mm watertight={m.is_watertight} vol={m.volume / 1000:.2f}cm3")
    return parts


if __name__ == "__main__":
    main()
