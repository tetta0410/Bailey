"""ベイリーの4面アルバム・ペン立て（完成版）— 黒い箱 × 白い板 × 黒いベイリー。

写真: 正面=舌ぺろ, 左=寝顔, 右=肉球, 背面=鼻アップ。
まわりに細い額縁の線を入れる。
絵は描き足しをせず、bailey_sketch.py で写真そのものを白黒の線画に写し取る
（目・毛並み・ヒゲ・舌・肉球・ホクロまで写真のまま）。
箱の底には白で「Bailey / 2026.1.15」。

2つの作り方:
  plates   … 白黒の板4枚を箱の4面のアリ溝に差し込む（色替えはごく少ない）
             bailey_final_box.3mf    箱（黒）＋底の文字（白）
             bailey_final_plates.3mf 板4枚（白地＋黒の絵）を1枚のビルドプレートに
  onepiece … 板と同じ見た目を、箱と一体で立てたまま印刷（組み立てなし・色替え多い）
             bailey_final_onepiece.3mf

使い方: python3 3d/make_bailey_album_final.py
"""
import functools
import os
import sys

import numpy as np
import shapely
import trimesh
from shapely import affinity

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ittetsu"))
import bailey_sketch as bs  # noqa: E402
import make_bailey_album as ma  # noqa: E402
import make_bailey_album_plates as mp  # noqa: E402
import make_bailey_penholder as mb  # noqa: E402
from make_bailey import extrude, text_poly  # noqa: E402
from make_pencase import write_3mf  # noqa: E402

OUT = mb.OUT
PHOTOS = {"front": "tongue2", "left": "sleep2", "right": "paw2", "back": "nose"}

FRAME_INSET = 2.5       # 額縁の線の、板の縁からの距離
FRAME_W = 0.9           # 額縁の線の太さ
ART_MARGIN = 4.0        # 額縁の内側の余白

BOTTOM_TEXT = ["Bailey", "2026.1.15"]
BOTTOM_SIZES = [11.0, 5.0]
BOTTOM_DEPTH = 0.6

LAYER = 0.2


def panel_size(face):
    """板の見える部分（幅, 高さ）。"""
    return mp.PW[face] - 2 * mp.INSET, mp.PH - 1.0


@functools.lru_cache()
def art(face):
    """面の絵（黒い部分）: 写真から写し取った線画＋額縁の線。板の見える部分の中央 (0, 高さ/2) 基準。"""
    w, h = panel_size(face)
    fw, fh = w - 2 * (FRAME_INSET + FRAME_W + ART_MARGIN), h - 2 * (FRAME_INSET + FRAME_W + ART_MARGIN)
    key = PHOTOS[face]
    ar = bs.aspect(key)
    width = min(fw, fh * ar)                      # 縦横比を保って枠に収める
    b = mb.mask_to_shape(bs.sketch(key, width), bs.PX_MM)          # 左上が原点, 下向きが -y
    x0, y0, x1, y1 = b.bounds
    b = affinity.translate(b, -(x0 + x1) / 2, h / 2 - (y0 + y1) / 2)
    r = shapely.box(-w / 2 + FRAME_INSET, FRAME_INSET, w / 2 - FRAME_INSET, h - FRAME_INSET)
    frame = r.difference(r.buffer(-FRAME_W, join_style="mitre"))
    g = shapely.union_all([b, frame]).buffer(0)
    # ほとんど接している点や細すぎるすき間をなくして、立体にできる形に整える
    g = g.buffer(0.02, join_style="round").buffer(-0.02, join_style="round").simplify(0.01).buffer(0)
    return shapely.union_all([q for q in getattr(g, "geoms", [g]) if q.area > 0.2])


def bottom_text():
    """底の文字（外から＝下から見て読める向き）。箱の座標 (x, y) で返す。"""
    lines, y = [], 0.0
    for t, s in zip(BOTTOM_TEXT, BOTTOM_SIZES):
        g = text_poly(t, s)
        lines.append(affinity.translate(g, 0, -y))
        y += s * 0.9 + 3.0
    g = shapely.union_all(lines)
    x0, y0, x1, y1 = g.bounds
    g = affinity.translate(g, -(x0 + x1) / 2, -(y0 + y1) / 2)
    # 箱を手前に倒して底をのぞくと、正面側の縁が上・右が +x になる。その向きで読めるよう上下を反転
    return affinity.translate(affinity.scale(g, 1, -1, origin=(0, 0)), mp.W / 2, mp.D / 2)


def with_bottom_text(box):
    t = bottom_text()
    box = box.difference(extrude(t, BOTTOM_DEPTH + 0.05, -0.05))
    return box, extrude(t, BOTTOM_DEPTH, 0)


# ---------------------------------------------------------------- 板差し込み版
def build_plates():
    mp.art_for = lambda face: art(face)     # 板の絵をこの完成版の絵に差し替える
    box, text = with_bottom_text(mp.make_box())
    write_3mf(os.path.join(OUT, "bailey_final_box.3mf"),
              [("bailey_final_box", (90, 90), [("box_black", box, 1), ("bottom_text_white", text, 2)])])
    plates = {f: mp.plate(f) for f in mp.PW}
    centers = {"front": (49, 46), "back": (127, 46), "left": (41.5, 135), "right": (104.5, 135)}
    write_3mf(os.path.join(OUT, "bailey_final_plates.3mf"),
              [(f"plate_{f}", centers[f], [("plate_white", b, 1), ("art_black", a, 2)])
               for f, (b, a) in plates.items()])
    return box, text, plates


# ---------------------------------------------------------------- 一体版
def panel_on(face, g, depth, extra=0.0):
    """板と同じ位置（溝の底 GROOVE_Z から上）に、面の外から見た向きで図形を置く。"""
    if face == "front":
        g2 = affinity.translate(g, 0, mp.GROOVE_Z)
        return mb.on_front(g2, depth, extra)
    g2 = affinity.translate(g, 0, mp.GROOVE_Z)
    return ma.on_face(g2, face, depth, extra)


def build_onepiece():
    body = mb.make_body()                     # 仕切り付き 90x75x100（溝なし）
    whites = []
    for face in mp.PW:
        w, h = panel_size(face)
        panel = shapely.box(-w / 2, 0, w / 2, h - 1.0)          # 縁から少し下までの白い板の部分
        white = panel.difference(art(face)).buffer(0)          # 黒い絵の部分は本体の黒がそのまま見える
        body = body.difference(panel_on(face, white, mb.INLAY, 0.05))
        whites.append(panel_on(face, white, mb.INLAY))
    body, text = with_bottom_text(body)
    white = trimesh.util.concatenate(whites + [text])
    write_3mf(os.path.join(OUT, "bailey_final_onepiece.3mf"),
              [("bailey_final_onepiece", (90, 90), [("body_black", body, 1), ("white", white, 2)])])
    return body, white


def swap_layers(mesh, top):
    """mesh の色が入る層の数（立てて印刷, 0.2mm）。"""
    t = mesh.triangles[:, :, 2]
    lo, hi = t.min(1), t.max(1)
    zs = np.arange(LAYER / 2, top, LAYER)
    return int(sum(((lo <= z) & (hi >= z)).any() for z in zs))


def main():
    os.makedirs(OUT, exist_ok=True)
    box, text, plates = build_plates()
    body, white = build_onepiece()
    n_box = swap_layers(text, 5)
    n_plate = int(round(mp.INLAY / LAYER))
    n_one = swap_layers(white, mp.H + 1)
    print(f"box: {box.extents.round(1)} watertight={box.is_watertight}; text layers={n_box}")
    for f, (b, a) in plates.items():
        print(f"plate {f:5s} {b.extents.round(1)} watertight={b.is_watertight}/{a.is_watertight}")
    print(f"onepiece: {body.extents.round(1)} watertight={body.is_watertight}/{white.is_watertight}; "
          f"white layers={n_one}")
    print(f"swaps (about one per layer that has both colors): plates job ~{n_plate}, box job ~{n_box}, "
          f"onepiece ~{n_one}")
    return box, text, plates, body, white


if __name__ == "__main__":
    main()
