"""ベイリーの顔入りペン立て（仕切り付き）— Bambu Lab A1 mini + AMS lite で1回印刷。

本体は白。顔の白い部分は本体の白をそのまま使い、黒とピンク（舌）だけを
正面に面一ではめ込む。立てて印刷するので、顔のある高さでは毎層色替えになるが、
  - 色は 白(本体)・黒・ピンク の3色、ピンクは舌の高さだけ
  - 顔は低めに置き、右手前の底上げ部分（インフィル）と高さを重ねて
    「インフィルにフラッシュ」で捨てる樹脂を中に流せるようにする
ことでプープを抑える。

出力 (3d/out/):
  bailey_penholder.3mf  … body_white / face_black / tongue_pink の3パーツ（1つのオブジェクト）
"""
import os
import sys

import numpy as np
import shapely
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ittetsu"))
from make_bailey import extrude, mirror, sym  # noqa: E402
from make_pencase import RAISE, W, WALL, make_body, write_3mf  # noqa: E402

OUT = os.path.join(HERE, "out")
INLAY = 0.8          # 顔のはめ込み深さ
FACE_SCALE = 0.88    # 顔全体の大きさ（1.0 で高さ約66mm）
FACE_BOTTOM = 14.0   # 顔（舌の先）の下端の高さ
OUTLINE = 1.0        # 頭のまわりの黒いフチ


def ellipse(cx, cy, rx, ry, rot=0.0):
    e = affinity.scale(Point(0, 0).buffer(1, quad_segs=32), rx, ry)
    return affinity.translate(affinity.rotate(e, rot), cx, cy)


def superellipse(a, b, p, cx=0.0, cy=0.0, n=240):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s = np.cos(t), np.sin(t)
    return Polygon(np.column_stack([cx + a * np.sign(c) * np.abs(c) ** (2 / p),
                                    cy + b * np.sign(s) * np.abs(s) ** (2 / p)]))


def bailey_face():
    """正面写真のベイリー（顔の中心が原点, mm, scale 前）。黒とピンクの2D形状を返す。"""
    # 頭: 幅広の丸い四角 + 頬のもふもふ
    head = shapely.union_all([
        superellipse(23.5, 19.5, 2.6, 0, 1.0),
        sym(ellipse(-18.5, -7.5, 7.0, 7.5, -20)),     # 頬の毛
        ellipse(0, -11.0, 13.5, 9.5),                 # マズル〜あご
    ])
    # 大きな立ち耳（少し外向き, 先は丸い）
    ear = Polygon([(-7.5, 15.0), (-20.0, 9.0), (-19.5, 33.0)]).buffer(3.4, quad_segs=24)
    ears = sym(ear)
    inner_ears = sym(Polygon([(-10.5, 17.0), (-17.0, 13.5), (-17.0, 28.0)]).buffer(1.0, quad_segs=12))
    silhouette = shapely.union_all([head, ears])

    # 黒い毛: 頭の上半分〜目のまわり。真ん中に白いブレーズ（写真では少し左寄り）
    cap = Polygon([(-30, 40), (30, 40), (30, -9), (17, -6.5), (11, -3.5), (5, -1.5),
                   (-5, -1.5), (-11, -3.5), (-17, -6.5), (-30, -9)])
    blaze = Polygon([(-3.4, 21), (1.0, 21), (2.2, 10), (7.5, -0.5), (-8.5, -0.5), (-4.4, 10)]
                    ).buffer(1.2, join_style="round")
    muzzle = ellipse(0, -9.0, 13.5, 9.5)
    black = silhouette.intersection(cap).difference(blaze).difference(muzzle)
    black = shapely.union_all([black, ears.difference(head.buffer(-0.01)).buffer(0)])
    black = black.difference(inner_ears)                # 耳の内側は白

    # 目: 黒い毛の中に白い縁どり → 黒目 → 白いハイライト
    eye_ring = sym(Point(-9.4, 3.2).buffer(3.3, quad_segs=24))
    pupils = sym(Point(-9.4, 3.2).buffer(2.6, quad_segs=24))
    shine = sym(Point(-8.5, 4.2).buffer(0.85, quad_segs=12))
    black = black.difference(eye_ring)
    black = shapely.union_all([black, pupils.difference(shine)])

    # 鼻（大きめの丸い逆三角）
    nose = Polygon([(-4.4, -3.6), (4.4, -3.6), (0, -8.4)]).buffer(1.6, join_style="round")
    # 口: 鼻の下から左右へ。口の中（暗い部分）から舌が垂れる
    smile = [(0, -9.5), (0, -10.8)] + [(-3.2 + 3.2 * np.cos(t), -10.8 + 2.2 * np.sin(t))
                                       for t in np.linspace(0, -np.pi, 18)]
    mouth_line = sym(LineString(smile + [(-7.6, -10.0)]).buffer(0.55, cap_style="round"))
    mouth_open = ellipse(0, -14.2, 5.8, 2.3)
    tongue = shapely.union_all([
        Polygon([(-4.6, -14.2), (4.6, -14.2), (4.4, -24.0), (-4.4, -24.0)]),
        ellipse(0, -24.0, 4.4, 4.2),
    ])

    # 頭のまわりの黒いフチ（白い本体の上で顔の形が分かるように）
    outline = silhouette.difference(silhouette.buffer(-OUTLINE))
    black = shapely.union_all([black, nose, mouth_line, mouth_open, outline])
    black = black.difference(tongue.buffer(0.01))
    # 舌の下側の黒フチ（舌は顔の外に垂れるので輪郭を足す）
    tongue_edge = tongue.buffer(OUTLINE).difference(tongue).difference(silhouette.buffer(-0.5))
    black = shapely.union_all([black, tongue_edge]).buffer(0)
    return black, tongue, silhouette


def placed_face():
    """スケールして正面の中央・下端 FACE_BOTTOM に置く（u=横, v=高さ）。"""
    black, pink, sil = bailey_face()
    allg = shapely.union_all([black, pink])
    g = lambda x: affinity.scale(x, FACE_SCALE, FACE_SCALE, origin=(0, 0))
    black, pink, sil, allg = g(black), g(pink), g(sil), g(allg)
    dy = FACE_BOTTOM - allg.bounds[1]
    t = lambda x: affinity.translate(x, 0, dy)
    # 細かすぎる頂点やほぼ接する点を整理（押し出し時の三角形分割を安定させる）
    clean = lambda x: x.buffer(0.02, join_style="round").buffer(-0.02, join_style="round").simplify(0.01)
    return clean(t(black)), clean(t(pink)), t(sil)


def on_front(g, depth, extra=0.0):
    """正面 (y=0) に、平面図形 g を奥行き depth で配置したメッシュ。"""
    m = extrude(g, depth + extra, -extra)
    v = m.vertices.copy()
    m.vertices = np.column_stack([v[:, 0] + W / 2, v[:, 2], v[:, 1]])
    m.fix_normals()
    return m


def main():
    black, pink, sil = placed_face()
    body = make_body(front=WALL + 0.4, groove=False)
    body = body.difference(on_front(shapely.union_all([black, pink]), INLAY, 0.05))
    black_m, pink_m = on_front(black, INLAY), on_front(pink, INLAY)
    os.makedirs(OUT, exist_ok=True)
    write_3mf(os.path.join(OUT, "bailey_penholder.3mf"), [("bailey_penholder", (90, 90), [
        ("body_white", body, 1), ("face_black", black_m, 2), ("tongue_pink", pink_m, 3)])])
    for name, m in [("body", body), ("black", black_m), ("pink", pink_m)]:
        print(f"{name:6s} {m.extents.round(1)} mm watertight={m.is_watertight} vol={m.volume/1000:.2f}cm3")
    zb, zp = black_m.bounds[:, 2], pink_m.bounds[:, 2]
    print(f"face z {zb[0]:.1f}-{zb[1]:.1f} mm (raised floor up to {RAISE}), tongue z {zp[0]:.1f}-{zp[1]:.1f} mm")
    for lh in (0.2, 0.24):
        n2 = (zb[1] - zb[0]) / lh
        n3 = (zp[1] - zp[0]) / lh
        print(f"layer {lh}: ~{n2:.0f} layers with black, ~{n3:.0f} with pink -> about {n2 + n3:.0f} swaps")
    return body, black_m, pink_m


if __name__ == "__main__":
    main()
