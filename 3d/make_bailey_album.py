"""ベイリーの成長アルバム・ペン立て — 4面すべてにベイリー。Bambu Lab A1 mini + AMS lite。

正面: 黒（と舌のピンク）をはめ込んだ写し絵。
左・右・背面: 写真から取った黒い毛の形を、本体と同じ色のまま浅く彫る（色替えなし）。

顔（正面）を下にして倒して印刷すると、色が入るのは最初の 0.8mm だけで、色替えは十数回。
左右の彫りは立った壁、背面の彫りは一番上の面になるので、どちらもそのまま印刷できる。
箱の中は真ん中の仕切りで左右2部屋。背面が天井になるので、中にツリーサポートが必要。

出力 (3d/out/):
  bailey_album_facedown.3mf … 【おすすめ】顔を下にして倒した向き
  bailey_album.3mf          … 立てた向き（サポート不要だが正面の絵の高さで毎層色替え）

白黒版（python3 3d/make_bailey_album.py bw）: 4面とも黒（と舌のピンク）をはめ込み、輪郭の細かい
ギザギザや点を整理してくっきりさせる。側面にも色が入るので倒しても色替えは減らず、立てて印刷する。
  bailey_album_bw.3mf … 立てた向き（サポート不要）
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

# 面ごとの写真（mb.placed_art の名前）。2回目にもらった4枚:
#   tongue2 = 舌ぺろ（耳まで）, puppy2 = 子犬（切り抜き）, dandelion = たんぽぽの横顔, nose = 鼻アップ
FRONT = "tongue2"
LEFT = "puppy2"
RIGHT = "dandelion"
BACK = "nose"

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


def clean(g, smooth=0.35, min_area=1.2, min_hole=0.25):
    """くっきりさせる: 毛先の細かいギザギザをならし、小さな点を消す（目の光などの小さな穴は残す）。"""
    g = g.buffer(smooth).buffer(-2 * smooth).buffer(smooth)
    keep = []
    for p in getattr(g, "geoms", [g]):
        if p.area < min_area:
            continue
        holes = [h for h in p.interiors if shapely.Polygon(h).area >= min_hole]
        keep.append(shapely.Polygon(p.exterior, holes))
    return shapely.union_all(keep).buffer(0).simplify(0.02)   # 丸めで増えた点を間引く（0.02mm は印刷に出ない）


def fit_face(photo, fit):
    """黒とピンクを、同じ倍率のまま面に収めて中央 (0, H/2) に置く。"""
    art = mb.placed_art(photo)
    b, p = art[0], art[1]
    x0, y0, x1, y1 = shapely.union_all([b, p]).bounds
    k = min(fit[0] / (x1 - x0), fit[1] / (y1 - y0))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    f = lambda g: affinity.translate(affinity.scale(g, k, k, origin=(cx, cy)), -cx, H / 2 - cy)
    return f(b), f(p)


def layers_with(shapes, lh=0.2):
    """高さ方向に、どの層に色が入るかを数える（立てて印刷したときの色替えの目安）。"""
    zs = np.arange(lh / 2, H, lh)
    hit = np.zeros(len(zs), bool)
    for g in shapes:
        if g.is_empty:
            continue
        for i, z in enumerate(zs):
            if not hit[i] and g.intersects(shapely.LineString([(-W, z), (W, z)])):
                hit[i] = True
    return hit


def main_bw(tol=0.0, write=True):
    """4面とも黒をはめ込んだ白黒版（立てて印刷）。tol>0 はビューア用の軽い形。"""
    faces = {"front": (FRONT, (W - 2 * CORNER - 4, 92.0)), "left": (LEFT, SIDE_FIT),
             "right": (RIGHT, SIDE_FIT), "back": (BACK, BACK_FIT)}
    body = mb.make_body()
    blacks, pinks, blk_shapes, pnk_shapes = [], [], [], []
    for face, (photo, fit) in faces.items():
        b, p = fit_face(photo, fit)
        b, p = clean(b), (clean(p) if not p.is_empty else p)
        if tol:
            b, p = b.simplify(tol).buffer(0), (p.simplify(tol).buffer(0) if not p.is_empty else p)
        place = (lambda g, d, e=0.0: mb.on_front(g, d, e)) if face == "front" else \
                (lambda g, d, e=0.0, f=face: on_face(g, f, d, e))
        body = body.difference(place(shapely.union_all([b, p]), mb.INLAY, 0.05))
        blacks.append(place(b, mb.INLAY))
        blk_shapes.append(b)
        if not p.is_empty:
            pinks.append(place(p, mb.INLAY))
            pnk_shapes.append(p)
    import trimesh
    parts = [("body_white", body, 1), ("face_black", trimesh.util.concatenate(blacks), 2)]
    if pinks:
        parts.append(("tongue_pink", trimesh.util.concatenate(pinks), 3))
    if write:
        write_3mf(os.path.join(mb.OUT, "bailey_album_bw.3mf"), [("bailey_album_bw", (90, 90), parts)])
    nb, npk = layers_with(blk_shapes), layers_with(pnk_shapes)
    print(f"standing, 0.2mm: {nb.sum()} layers with black, {npk.sum()} with pink -> about "
          f"{int((nb | npk).sum() + (nb & npk).sum())} swaps")
    for n, m, _ in parts:
        print(f"{n:12s} {m.extents.round(1)} mm watertight={m.is_watertight}")
    return parts


def on_face(g, face, depth, extra=0.1):
    """2D 図形 g（u=右, v=上, 面の中央が u=0）を、外から見た向きのまま各面に depth の深さで置く。
    extra は外側へはみ出させる量（くり抜き用。はめ込むパーツは 0 で面一）。"""
    m = extrude(g, depth + extra, -extra)      # z: 外側(-extra) → 内側(depth)
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
    main_bw() if sys.argv[1:] == ["bw"] else main()
