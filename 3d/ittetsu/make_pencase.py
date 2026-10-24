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


def write_3mf(path, objects):
    """Bambu Studio 向けの 3MF を書く。

    objects: [(オブジェクト名, (x, y) ベッド上の中心, [(パーツ名, mesh, フィラメント番号), ...]), ...]
    各オブジェクトは「パーツを持つ1つのオブジェクト」(components) として書くので、
    読み込み時にパーツが別オブジェクトに分かれない。
    """
    import zipfile
    from xml.sax.saxutils import quoteattr

    res, build, cfg = [], [], []
    oid = 1
    for oname, (cx, cy), parts in objects:
        lo = np.min([m.bounds[0] for _, m, _ in parts], axis=0)
        hi = np.max([m.bounds[1] for _, m, _ in parts], axis=0)
        center = np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        comp_ids, part_cfg = [], []
        for pname, m, ext in parts:
            v = m.vertices - center
            vx = "".join(f'<vertex x="{a:.4f}" y="{b:.4f}" z="{c:.4f}"/>' for a, b, c in v)
            tx = "".join(f'<triangle v1="{a}" v2="{b}" v3="{c}"/>' for a, b, c in m.faces)
            res.append(f'<object id="{oid}" type="model" name={quoteattr(pname)}><mesh><vertices>{vx}</vertices>'
                       f'<triangles>{tx}</triangles></mesh></object>')
            comp_ids.append(oid)
            part_cfg.append(f'<part id="{oid}" subtype="normal_part"><metadata key="name" value={quoteattr(pname)}/>'
                            f'<metadata key="extruder" value="{ext}"/></part>')
            oid += 1
        comps = "".join(f'<component objectid="{i}"/>' for i in comp_ids)
        res.append(f'<object id="{oid}" type="model" name={quoteattr(oname)}><components>{comps}</components></object>')
        build.append(f'<item objectid="{oid}" transform="1 0 0 0 1 0 0 0 1 {cx} {cy} 0" printable="1"/>')
        cfg.append(f'<object id="{oid}"><metadata key="name" value={quoteattr(oname)}/>'
                   f'<metadata key="extruder" value="{parts[0][2]}"/>{"".join(part_cfg)}</object>')
        oid += 1
    model = ('<?xml version="1.0" encoding="UTF-8"?>\n<model unit="millimeter" xml:lang="en-US" '
             'xmlns="http://schemas.microsoft.com/3dmanufacturing/core/2015/02">'
             f'<resources>{"".join(res)}</resources><build>{"".join(build)}</build></model>')
    types = ('<?xml version="1.0" encoding="UTF-8"?>\n<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
             '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
             '<Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>'
             '<Default Extension="config" ContentType="text/xml"/></Types>')
    rels = ('<?xml version="1.0" encoding="UTF-8"?>\n<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Target="/3D/3dmodel.model" Id="rel0" '
            'Type="http://schemas.microsoft.com/3dmanufacturing/2013/01/3dmodel"/></Relationships>')
    settings = f'<?xml version="1.0" encoding="UTF-8"?>\n<config>{"".join(cfg)}</config>'
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", types)
        z.writestr("_rels/.rels", rels)
        z.writestr("3D/3dmodel.model", model)
        z.writestr("Metadata/model_settings.config", settings)


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

    write_3mf(os.path.join(OUT, "ittetsu_plate.3mf"),
              [("ittetsu_plate", (90, 90), [("plate_base", base, 1), ("plate_art_black", art_m, 2)])])
    body.export(os.path.join(OUT, "ittetsu_pencase_body.stl"))

    # 1枚のビルドプレート (180x180) に並べる: 本体を左、プレートを右に
    write_3mf(os.path.join(OUT, "ittetsu_pencase.3mf"), [
        ("pencase_body", (50, 90), [("pencase_body", body, 1)]),
        ("ittetsu_plate", (50 + W / 2 + 8 + PW / 2, 90), [("plate_base", base, 1), ("plate_art_black", art_m, 2)]),
    ])

    # 一体型（本体に直接プリント）
    one, one_art = make_onepiece()
    write_3mf(os.path.join(OUT, "ittetsu_pencase_onepiece.3mf"),
              [("ittetsu_pencase", (90, 90), [("pencase_body", one, 1), ("art_black", one_art, 2)])])
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
