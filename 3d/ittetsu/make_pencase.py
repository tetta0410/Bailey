"""一鉄（いってつ）の四角いペン立て（仕切り付き）— Bambu Lab A1 mini + AMS lite 用。

絵は本体に直接ではなく「差し替えプレート」に入れる。プレートは平置きで印刷するので、
黒い絵の色替えは最後の数層だけで済む（縦の側面に絵を入れると毎層色替えになる）。
プレートは本体正面のアリ溝に上から差し込む。

  本体 (90 x 75 x 100 mm)
  ┌──────────────┬────────┐  奥
  │              │ 深い   │  はさみ・定規・マーカー
  │  ペン・鉛筆  ├────────┤
  │  (深さ97mm)  │ 浅い   │  短くなった鉛筆・消しゴム（底上げで深さ50mm）
  └──────────────┴────────┘  正面（プレート）

出力 (3d/ittetsu/out/):
  ittetsu_pencase.3mf       … 本体＋プレートを1枚のプレートに配置（AMS lite でそのまま印刷）
  ittetsu_pencase_body.stl  … 本体だけ
  ittetsu_plate.3mf         … 絵のプレートだけ（base / art の2パーツ）
  ittetsu_pencase_onepiece.3mf … 一体型（本体の正面に絵を直接はめ込み。色替えは絵の高さ分の毎層）
"""
import math
import os
import sys

import numpy as np
import shapely
import trimesh
from shapely import affinity
from shapely.geometry import Polygon

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_penholder import ink_mask, mask_to_polygons  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")

# ---- 本体 (mm)。X=幅, Y=奥行き(正面が y=0), Z=高さ
W, D, H = 90.0, 75.0, 100.0
WALL = 2.4          # 側面・背面の壁
FRONT = 5.2         # 正面の壁（プレート用の溝が入るので厚め）
FLOOR = 2.4
CORNER = 6.0        # 外側の角の丸み
DIV = 1.6           # 仕切りの厚み
DIV_TOP = H - 10    # 仕切りは縁より 10mm 低く（物を取りやすく）
SPLIT_X = 48.4      # 左の大部屋の右端
RAISE = 50.0        # 右手前の部屋の底上げ高さ（深さ ≒ 50mm）

# ---- プレート
T = 3.0             # 厚み
SLOPE = math.radians(30)   # アリ溝の傾き
PW = 74.0           # 裏側の幅
GROOVE_Z = 8.0      # 溝の底の高さ
PH = H - GROOVE_Z + 1.0    # 縁から 1mm 出して指でつまめるように
CLEAR = 0.2         # 溝の奥行き方向のすき間
CLEAR_X = 0.35      # 溝の幅方向の広げ量（斜面で片側 約0.2mm のすき間になる）
INLAY = 0.6         # 黒い絵の深さ（プレート表面に面一ではめ込み）
ART_H = 80.0        # 絵の高さ
INLAY_1P = 0.8      # 一体型での絵の深さ
ART_Z = 48.0        # 一体型での絵の中心の高さ


def rbox(x0, y0, x1, y1, r):
    return shapely.box(x0 + r, y0 + r, x1 - r, y1 - r).buffer(r, quad_segs=16)


def extrude(g, h, z0=0.0):
    parts = [trimesh.creation.extrude_polygon(p, h) for p in getattr(g, "geoms", [g]) if p.area > 0.05]
    m = trimesh.util.concatenate(parts)
    m.apply_translation([0, 0, z0])
    return m


def dovetail(width_back, length, thick, grow=0.0):
    """プレートの断面（台形）を長さ方向に押し出す。平置き: X=幅, Y=長さ, Z=厚み(裏が z=0)。"""
    inset = thick * math.tan(SLOPE)
    wb, wf = width_back / 2 + grow, width_back / 2 - inset + grow
    sec = Polygon([(-wb, 0), (wb, 0), (wf, thick), (-wf, thick)])
    m = trimesh.creation.extrude_polygon(sec, length)      # XY 断面を Z へ押し出し
    m.apply_transform(trimesh.transformations.rotation_matrix(-math.pi / 2, [1, 0, 0]))  # Z→Y, Y→-Z ...
    m.apply_transform(trimesh.transformations.rotation_matrix(math.pi, [0, 1, 0]))
    return m


def plate_profile(length):
    """平置きのプレート（裏 z=0, 表 z=T, 長さ方向 Y 0..length, 中心 X=0）。"""
    m = dovetail(PW, length, T)
    b = m.bounds
    m.apply_translation([0, -b[0][1], -b[0][2]])
    return m


def art_shapes():
    ink = ink_mask(os.path.join(HERE, "ittetsu_drawing.jpg"))
    art = mask_to_polygons(ink, ART_H / ink.shape[0])
    minx, miny, maxx, maxy = art.bounds
    return affinity.translate(art, -(minx + maxx) / 2, -(miny + maxy) / 2)


def make_plate():
    base = plate_profile(PH)
    art = art_shapes()
    # 絵の中心: 本体の縁より上に出る 1mm と溝の底を除いた、見える範囲の中央
    art = affinity.translate(art, 0, (PH - 1.0) / 2)
    art_m = extrude(art, INLAY + 0.01, T - INLAY)
    base = base.difference(art_m)
    art_m = extrude(art, INLAY, T - INLAY)
    return base, art_m, art


def make_body(front=FRONT, groove=True):
    outer = rbox(0, 0, W, D, CORNER)
    cavity = rbox(WALL, front, W - WALL, D - WALL, 2.0)
    body = extrude(outer, H).difference(extrude(cavity, H, FLOOR))
    mid_y = (front + D - WALL) / 2
    walls = shapely.union_all([
        shapely.box(SPLIT_X, front - 0.5, SPLIT_X + DIV, D - WALL + 0.5),          # 縦の仕切り
        shapely.box(SPLIT_X, mid_y - DIV / 2, W - WALL + 0.5, mid_y + DIV / 2),     # 右の前後仕切り
    ])
    body = body.union(extrude(walls, DIV_TOP - 0.5, 0.5))
    # 右手前の部屋は底上げ（中身は印刷時にインフィルになる）
    raised = shapely.box(SPLIT_X, front - 0.5, W - WALL + 0.5, mid_y)
    body = body.union(extrude(raised, RAISE - 0.5, 0.5))
    if not groove:
        return body
    # 正面のアリ溝（上から差し込む, 表面側が狭い）
    groove = dovetail(PW, H, T + CLEAR, grow=CLEAR_X)
    # 平置き座標 → 立てた座標: (x, y=長さ, z=厚み) → (x, 厚み方向=+Y, 高さ=Z)
    g = trimesh.Trimesh(groove.vertices.copy(), groove.faces.copy())
    b = g.bounds
    g.apply_translation([W / 2, -b[0][1], -b[0][2]])
    v = g.vertices.copy()
    g.vertices = np.column_stack([v[:, 0], (T + CLEAR) - v[:, 2], v[:, 1] + GROOVE_Z])
    g.fix_normals()
    g.apply_translation([0, -0.01, 0])
    body = body.difference(g)
    return body


def make_onepiece():
    """一体型: 本体の正面に絵を直接はめ込む（正面の壁は WALL の厚み, 絵の深さ INLAY_1P）。"""
    body = make_body(front=WALL + 0.4, groove=False)
    art = affinity.translate(art_shapes(), 0, ART_Z)

    def on_front(depth, extra=0.0):
        m = extrude(art, depth + extra, -extra)         # z: 外側へ extra はみ出させて確実に切る
        v = m.vertices.copy()
        m.vertices = np.column_stack([v[:, 0] + W / 2, v[:, 2], v[:, 1]])
        m.fix_normals()
        return m

    body = body.difference(on_front(INLAY_1P, 0.05))
    return body, on_front(INLAY_1P)


def stand_plate(m):
    """平置きプレートを本体に差し込んだ位置へ（プレビュー用）。"""
    m = m.copy()
    v = m.vertices.copy()
    m.vertices = np.column_stack([v[:, 0] + W / 2, T - v[:, 2], v[:, 1] + GROOVE_Z])
    m.fix_normals()
    return m


def main():
    os.makedirs(OUT, exist_ok=True)
    body = make_body()
    base, art_m, art = make_plate()

    plate = trimesh.Scene()
    plate.add_geometry(base, node_name="plate_base", geom_name="plate_base")
    plate.add_geometry(art_m, node_name="plate_art_black", geom_name="plate_art_black")
    plate.export(os.path.join(OUT, "ittetsu_plate.3mf"))
    body.export(os.path.join(OUT, "ittetsu_pencase_body.stl"))

    # 1枚のビルドプレート (180x180) に並べる: 本体を左、プレートを右に
    sc = trimesh.Scene()
    b2 = body.copy(); b2.apply_translation([-W / 2 - 42, -D / 2, 0])
    pb, pa = base.copy(), art_m.copy()
    for m in (pb, pa):
        m.apply_translation([W / 2 - 42 + 8 + PW / 2, -PH / 2, 0])
    sc.add_geometry(b2, node_name="pencase_body", geom_name="pencase_body")
    sc.add_geometry(pb, node_name="plate_base", geom_name="plate_base")
    sc.add_geometry(pa, node_name="plate_art_black", geom_name="plate_art_black")
    sc.export(os.path.join(OUT, "ittetsu_pencase.3mf"))
    allb = sc.bounds
    print("build plate footprint:", (allb[1] - allb[0]).round(1))

    # 一体型（本体に直接プリント）
    one, one_art = make_onepiece()
    sc1 = trimesh.Scene()
    sc1.add_geometry(one, node_name="pencase_body", geom_name="pencase_body")
    sc1.add_geometry(one_art, node_name="art_black", geom_name="art_black")
    sc1.export(os.path.join(OUT, "ittetsu_pencase_onepiece.3mf"))
    one.export(os.path.join(OUT, "view_1p_body.stl"))
    one_art.export(os.path.join(OUT, "view_1p_art.stl"))
    for name, m in [("1p_body", one), ("1p_art", one_art)]:
        print(f"{name:7s} {m.extents.round(1)} mm watertight={m.is_watertight} vol={m.volume/1000:.1f}cm3")
    ab = one_art.bounds
    print("1p art z range:", ab[:, 2].round(1), " y range:", ab[:, 1].round(2))

    # プレビュー用（組み立て状態）
    stand_plate(base).export(os.path.join(OUT, "view_plate.stl"))
    stand_plate(art_m).export(os.path.join(OUT, "view_art.stl"))
    body.export(os.path.join(OUT, "view_body.stl"))
    for name, m in [("body", body), ("plate", base), ("art", art_m)]:
        print(f"{name:6s} {m.extents.round(1)} mm watertight={m.is_watertight} vol={m.volume/1000:.1f}cm3")
    print("art size:", (np.array(art.bounds[2:]) - art.bounds[:2]).round(1),
          " visible plate width:", round(PW - 2 * T * math.tan(SLOPE), 1))


if __name__ == "__main__":
    main()
