"""ベイリーの迷子札QRチャーム — Bambu Lab A1 mini + AMS lite 用。

QRコードはベイリーのページ (GitHub Pages) を指す。
平置きで印刷し、QR・名前・耳の黒は上面 0.6mm だけのはめ込みなので、色替えは最後の数層だけ。
裏面には Dad / Mam の電話番号を黒ではめ込み（QRが読めないとき用）。裏の黒は最初の数層だけ。

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
CONTACTS = ["Dad 090-7828-9179", "Mam 080-4472-0384"]
MONO_BOLD = FontProperties(fname="/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf")
OUT = os.path.join(HERE, "out")

MODULE = 1.0      # QR 1マスの大きさ (mm)。0.4mm ノズルで2本分の線幅
QUIET = 2         # QR まわりの白い余白（マス数）
W = 39.0          # チャームの幅
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


def charm():
    qr, n, ver, err = qr_shapes()
    side = (n + 2 * QUIET) * MODULE                 # 白い余白込みの QR の大きさ
    qr_y0 = 10.5                                    # 余白込み QR の下端
    qr = affinity.translate(qr, (W - n * MODULE) / 2, qr_y0 + QUIET * MODULE)
    top = qr_y0 + side + 2.0                        # 本体（四角部分）の上端
    cx = W / 2
    body = shapely.box(0, 0, W, top).buffer(-4).buffer(4, quad_segs=16)
    # ボーダーコリーの耳（黒）
    ear = Polygon([(3, top - 7), (6.5, top + 9), (15, top - 1)]).buffer(1.5, join_style="round")
    ears = shapely.union_all([ear, affinity.scale(ear, -1, 1, origin=(cx, 0))])
    # キーリング用の穴（耳と耳の間）
    boss = Point(cx, top + 1.5).buffer(4.6, quad_segs=24)
    hole = Point(cx, top + 1.5).buffer(2.1, quad_segs=24)
    outline = shapely.union_all([body, ears, boss]).difference(hole)
    ears_black = ears.difference(shapely.box(0, 0, W, top - 2.5)).intersection(outline)
    name = text_poly("BAILEY", 6.2)
    name = affinity.translate(name, cx, 5.2)
    qr = qr.buffer(0.02, join_style="mitre")  # 斜めに角だけ接するマスをつなげる（非多様体を防ぐ）
    black = shapely.union_all([qr, name, ears_black]).intersection(outline.buffer(-0.6))
    # 裏の電話番号（等幅フォントで桁をそろえ、裏から読めるよう左右反転）
    lines = [text_poly(t, 3.4, MONO_BOLD) for t in CONTACTS]
    gap = 7.0
    phone = shapely.union_all([affinity.translate(mirror(l), cx, top / 2 + gap / 2 - i * gap)
                               for i, l in enumerate(lines)])
    return outline, black, phone, dict(n=n, version=ver, error=err, size=(W, outline.bounds[3]), qr_side=side)


def main():
    outline, black, phone, info = charm()
    white = extrude(outline, THICK).difference(extrude(black, INLAY + 0.05, THICK - INLAY))
    white = white.difference(extrude(phone, BACK_INLAY + 0.05, -0.05))
    black_m = extrude(black, INLAY, THICK - INLAY)
    black_m = black_m.union(extrude(phone, BACK_INLAY, 0))
    os.makedirs(OUT, exist_ok=True)
    write_3mf(os.path.join(OUT, "bailey_qr_charm.3mf"),
              [("bailey_qr_charm", (90, 90), [("charm_white", white, 1), ("charm_black", black_m, 2)])])
    print(f"QR v{info['version']}-{info['error']} {info['n']}x{info['n']} modules, "
          f"QR(余白込み) {info['qr_side']:.1f}mm, charm {info['size'][0]:.1f} x {info['size'][1]:.1f} x {THICK} mm")
    print("watertight:", white.is_watertight, black_m.is_watertight)
    pb = phone.bounds
    print(f"back text: {pb[2]-pb[0]:.1f} x {pb[3]-pb[1]:.1f} mm (charm width {W})")
    return outline, black, phone


if __name__ == "__main__":
    main()
