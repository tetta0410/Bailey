"""ベイリーの迷子札QRチャーム — Bambu Lab A1 mini + AMS lite 用。

QRコードはベイリーのページ (GitHub Pages) を指す。
平置きで印刷し、QR・名前・耳の黒は上面 0.6mm だけのはめ込みなので、色替えは最後の数層だけ。
裏面には Dad / Mom の電話番号を黒ではめ込み（QRが読めないとき用）。裏の黒は最初の数層だけ。

出力: 3d/out/bailey_qr_charm.3mf  (charm_white / charm_black の2パーツ, 1つのオブジェクト)
"""
import os
import sys

import numpy as np
import segno
import shapely
from shapely import affinity
from matplotlib.font_manager import FontProperties
from shapely.geometry import Point, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "ittetsu"))
from make_bailey import extrude, mirror, text_poly  # noqa: E402
from make_pencase import write_3mf  # noqa: E402

URL = "https://tetta0410.github.io/Bailey/"
CONTACTS = ["Dad 090-7828-9179", "Mom 080-4472-0384"]
MONO_BOLD = FontProperties(fname="/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")
OUT = os.path.join(HERE, "out")

MODULE = 0.9      # QR 1マスの大きさ (mm)。0.4mm ノズルで2本強の線幅
QUIET = 2         # QR まわりの白い余白（マス数）
THICK = 5.0       # 厚み
INLAY = 0.6       # 黒のはめ込み深さ（0.2mm で3層）
BACK_INLAY = 0.6  # 裏の電話番号（黒）の深さ


def qr_shapes():
    qr = segno.make(URL, error="m", boost_error=False)
    rows = list(qr.matrix_iter(scale=1, border=0))
    n = len(rows)
    cells = [shapely.box(c * MODULE, (n - 1 - r) * MODULE, (c + 1) * MODULE, (n - r) * MODULE)
             for r, row in enumerate(rows) for c, v in enumerate(row) if v]
    return shapely.union_all(cells), n, qr.version, qr.error


# ポップな頭: 角の丸い四角（スーパー楕円）。HEAD_A=横半径, HEAD_B=縦半径, HEAD_P=角の丸さ
HEAD_A, HEAD_B, HEAD_P = 20.5, 24.5, 3.2
# 右の耳（大きめの丸い三角, 外向き）。三角を太らせて角を丸くする
EAR_TRI = [(10.5, 41.0), (17.0, 35.5), (20.5, 50.0)]
EAR_ROUND = 4.6
BORDER = 1.0      # 黒いフチの幅（シール風）


def superellipse(a, b, p, cy, n=360):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    c, s_ = np.cos(t), np.sin(t)
    x = a * np.sign(c) * np.abs(c) ** (2 / p)
    y = b * np.sign(s_) * np.abs(s_) ** (2 / p) + cy
    return Polygon(np.column_stack([x, y]))


def paw(cx, cy, size=4.2):
    """肉球（大きいパッド＋指4つ）。size はだいたいの高さ。"""
    k = size / 4.2
    pad = affinity.scale(Point(0, 0).buffer(1.0, quad_segs=24), 1.45 * k, 1.15 * k)
    toes = [Point(x * k, y * k).buffer(0.62 * k, quad_segs=16)
            for x, y in [(-1.55, 1.25), (-0.55, 2.05), (0.55, 2.05), (1.55, 1.25)]]
    g = shapely.union_all([affinity.translate(pad, 0, -0.35 * k), *toes])
    return affinity.translate(g, cx, cy - 0.5 * k)


def charm():
    qr, n, ver, err = qr_shapes()
    side = (n + 2 * QUIET) * MODULE                 # 白い余白込みの QR の大きさ
    qr_cy = 25.0
    qr_y0 = qr_cy - side / 2
    qr = affinity.translate(qr, -n * MODULE / 2, qr_y0 + QUIET * MODULE)
    head = superellipse(HEAD_A, HEAD_B, HEAD_P, HEAD_B)
    ear = Polygon(EAR_TRI).buffer(EAR_ROUND, quad_segs=24)
    ears = shapely.union_all([ear, mirror(ear)])
    hole = Point(0, 44.2).buffer(2.0, quad_segs=24)
    outline = shapely.union_all([head, ears]).difference(hole)
    ears_black = ears.difference(head.buffer(0.4))
    border = outline.difference(outline.buffer(-BORDER))
    name = affinity.translate(text_poly("BAILEY", 4.6), 0, 6.3)
    nb = name.bounds
    paws = shapely.union_all([paw(nb[0] - 3.8, 6.3, 3.9), paw(nb[2] + 3.8, 6.3, 3.9)])
    qr = qr.buffer(0.02, join_style="mitre")  # 斜めに角だけ接するマスをつなげる（非多様体を防ぐ）
    front = shapely.union_all([qr, name, paws, ears_black, border]).intersection(outline)
    # 裏の電話番号（等幅フォントで桁をそろえ、裏から読めるよう左右反転）＋ 耳とフチも黒
    lines = [text_poly(t, 3.2, MONO_BOLD) for t in CONTACTS]
    gap = 6.6
    phone = shapely.union_all([affinity.translate(mirror(l), 0, 24.0 + gap / 2 - i * gap)
                               for i, l in enumerate(lines)])
    back = shapely.union_all([phone, ears_black, border]).intersection(outline)
    # フチの内側に白が十分あるか確認
    inner = outline.buffer(-BORDER).exterior
    info = dict(n=n, version=ver, error=err, qr_side=side,
                size=(outline.bounds[2] - outline.bounds[0], outline.bounds[3] - outline.bounds[1]),
                margin_qr=inner.distance(shapely.box(-side / 2, qr_y0, side / 2, qr_y0 + side)),
                margin_name=inner.distance(shapely.union_all([name, paws])), margin_phone=inner.distance(phone),
                hole_wall=min(hole.exterior.distance(outline.exterior), hole.distance(qr)))
    return outline, front, back, info


def main():
    outline, front, back, info = charm()
    white = extrude(outline, THICK)
    white = white.difference(extrude(front, INLAY + 0.05, THICK - INLAY))
    white = white.difference(extrude(back, BACK_INLAY + 0.05, -0.05))
    black_m = extrude(front, INLAY, THICK - INLAY).union(extrude(back, BACK_INLAY, 0))
    os.makedirs(OUT, exist_ok=True)
    write_3mf(os.path.join(OUT, "bailey_qr_charm.3mf"),
              [("bailey_qr_charm", (90, 90), [("charm_white", white, 1), ("charm_black", black_m, 2)])])
    print(f"QR v{info['version']}-{info['error']} {info['n']}x{info['n']} modules, "
          f"QR(余白込み) {info['qr_side']:.1f}mm, charm {info['size'][0]:.1f} x {info['size'][1]:.1f} x {THICK} mm")
    print("white margin (mm): qr %.1f / name %.1f / phone %.1f / hole wall %.1f" % (
        info["margin_qr"], info["margin_name"], info["margin_phone"], info["hole_wall"]))
    print("watertight:", white.is_watertight, black_m.is_watertight)


if __name__ == "__main__":
    main()
