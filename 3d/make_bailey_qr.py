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
from scipy.interpolate import splev, splprep
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


def smooth(points, n=240, mirror_x=False):
    """制御点を通るなめらかな閉曲線 (周期スプライン)。mirror_x なら右半分の点から左右対称に作る。"""
    pts = np.array(points, float)
    if mirror_x:
        pts = np.vstack([pts, pts[-2:0:-1] * [-1, 1]])
    tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=True)
    x, y = splev(np.linspace(0, 1, n, endpoint=False), tck)
    return Polygon(np.column_stack([x, y])).buffer(0)


# ボーダーコリーの頭（右半分, 下のあご → 頬のもふもふ → 頭のてっぺん）
HEAD = [(0, 0.0), (6, 0.6), (11.5, 2.6), (15.6, 6.0), (18.9, 10.4), (18.6, 15.6), (20.4, 20.6),
        (19.7, 27.0), (19.5, 33.0), (18.6, 39.0), (16.2, 44.6), (12.2, 48.8), (6.2, 50.7), (0, 51.2)]
# 右の耳（立ち耳, 先は丸く少し外向き）
EAR = [(8.6, 47.6), (10.8, 52.8), (13.8, 58.0), (17.0, 61.2), (19.4, 61.0), (20.5, 57.6),
       (20.3, 51.4), (18.8, 44.8), (14.0, 44.6)]


def charm():
    qr, n, ver, err = qr_shapes()
    side = (n + 2 * QUIET) * MODULE                 # 白い余白込みの QR の大きさ
    qr_y0 = 9.3                                     # 余白込み QR の下端
    qr = affinity.translate(qr, -n * MODULE / 2, qr_y0 + QUIET * MODULE)
    head = smooth(HEAD, mirror_x=True)
    ear = smooth(EAR)
    ears = shapely.union_all([ear, mirror(ear)])
    hole = Point(0, 46.3).buffer(2.1, quad_segs=24)
    outline = shapely.union_all([head, ears]).difference(hole)
    ears_black = ears.difference(head.buffer(0.3)).intersection(outline.buffer(-0.5))
    name = affinity.translate(text_poly("BAILEY", 5.0), 0, 5.6)
    qr = qr.buffer(0.02, join_style="mitre")  # 斜めに角だけ接するマスをつなげる（非多様体を防ぐ）
    front = shapely.union_all([qr, name, ears_black]).intersection(outline.buffer(-0.6))
    # 裏の電話番号（等幅フォントで桁をそろえ、裏から読めるよう左右反転）＋ 耳も黒
    lines = [text_poly(t, 3.2, MONO_BOLD) for t in CONTACTS]
    gap = 6.6
    phone = shapely.union_all([affinity.translate(mirror(l), 0, 25.0 + gap / 2 - i * gap)
                               for i, l in enumerate(lines)])
    back = shapely.union_all([phone, ears_black])
    # 白い縁が十分あるか確認
    margin = lambda g: outline.exterior.distance(g)
    info = dict(n=n, version=ver, error=err, qr_side=side,
                size=(outline.bounds[2] - outline.bounds[0], outline.bounds[3] - outline.bounds[1]),
                margin_qr=margin(shapely.box(-side / 2, qr_y0, side / 2, qr_y0 + side)),
                margin_name=margin(name), margin_phone=margin(phone),
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
