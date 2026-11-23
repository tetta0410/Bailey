"""ベイリーの鼻のキーホルダー（立体）— Bambu Lab A1 mini、黒1色。

写真 (photos/bailey_nose_close.jpg) から、鼻の輪郭・左右の鼻の穴・真ん中の縦の溝の位置を写し、
ふっくらした立体にする。表面は写真で測ったベイリーの鼻のシボ（約1mmの粒）を入れる。
（右側は写真では影で粒が見えないので、同じ大きさの粒を全体に入れている）

裏は平らで、そのままビルドプレートに置いて印刷する（サポート不要）。上にキーリング用の輪。

出力: 3d/out/bailey_nose_keychain.stl / .3mf
"""
import os
import sys

import cv2
import fast_simplification
import numpy as np
import shapely
import shapely.affinity
import trimesh
from scipy import ndimage as ndi
from scipy.interpolate import splev, splprep
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ittetsu"))
from make_bailey import extrude  # noqa: E402
from make_bailey_lantern import heightmap_mesh  # noqa: E402
from make_pencase import write_3mf  # noqa: E402

OUT = os.path.join(HERE, "out")

# ---- 写真から写した形（写真の画素座標）
OUTLINE = [(420, 880), (470, 855), (540, 845), (620, 842), (700, 845), (780, 860), (830, 890), (850, 940),
           (852, 1000), (840, 1060), (810, 1110), (760, 1140), (700, 1155), (650, 1166), (620, 1160),
           (560, 1150), (500, 1135), (455, 1100), (420, 1050), (400, 990), (402, 930)]
NOSTRIL_L = [(445, 1020), (470, 1010), (510, 1012), (540, 1025), (552, 1045), (548, 1062), (530, 1072),
             (505, 1075), (480, 1070), (460, 1058), (447, 1040)]
NOSTRIL_R = [(705, 1000), (740, 990), (780, 995), (805, 1010), (812, 1035), (800, 1060), (770, 1075),
             (735, 1072), (712, 1055), (700, 1030)]
GROOVE = [(640, 1080), (640, 1170)]          # 鼻の下の真ん中の溝（上 → 下）

# ---- 大きさ (mm)
WIDTH = 32.0          # 鼻の幅
GRID = 0.08           # 厚みマップの細かさ
EDGE = 1.2            # ふちの厚み
DOME = 10.5           # ふくらみの高さ
ROUND = 10.0          # ふちの丸み（この距離でふくらみきる）
NOSTRIL_DEPTH = 6.5   # 鼻の穴の深さ
GROOVE_DEPTH = 1.2    # 縦の溝の深さ
GROOVE_W = 0.55       # 縦の溝の幅（ガウスのσ）
CELL = 0.95           # シボの粒の大きさ（写真では約13〜14画素 ≒ 1mm）
CELL_DEPTH = 0.3      # 粒のあいだの溝の深さ
RING_OUT, RING_IN, RING_T = 4.6, 2.3, 3.2    # キーリングの輪（外半径・内半径・厚み）


def to_mm(pts, s, x0, y1):
    """写真の画素 → mm（x 右, y 上）。"""
    return [((x - x0) * s, (y1 - y) * s) for x, y in pts]


def smooth(pts, n=240):
    """写した点をなめらかな閉じた曲線にする（点の間を角ばらせない）。"""
    p = np.array(pts, float)
    tck, _ = splprep([p[:, 0], p[:, 1]], s=len(p) * 0.5, per=True)
    x, y = splev(np.linspace(0, 1, n, endpoint=False), tck)
    return Polygon(np.column_stack([x, y])).buffer(0)


def raster(poly, nx, ny):
    m = np.zeros((ny, nx), np.uint8)
    pts = np.array([[x / GRID, (ny - 1) - y / GRID] for x, y in poly.exterior.coords], np.int32)
    cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def voronoi_grooves(nx, ny, seed=7):
    """粒（ボロノイ）の境目を溝にした高さ（0 〜 -CELL_DEPTH）。"""
    rng = np.random.default_rng(seed)
    w, h = nx * GRID, ny * GRID
    gx, gy = np.meshgrid(np.arange(-1, w / CELL + 2), np.arange(-1, h / CELL + 2))
    pts = np.column_stack([gx.ravel(), gy.ravel()]) * CELL + rng.uniform(-0.38, 0.38, (gx.size, 2)) * CELL
    X, Y = np.meshgrid(np.arange(nx) * GRID, np.arange(ny) * GRID)
    d, _ = cKDTree(pts).query(np.column_stack([X.ravel(), Y.ravel()]), k=2)
    edge = (d[:, 1] - d[:, 0]).reshape(ny, nx)
    f1 = d[:, 0].reshape(ny, nx)
    groove = -CELL_DEPTH * np.exp(-(edge / (0.2 * CELL)) ** 2)        # 粒のあいだの溝
    dome = 0.12 * CELL_DEPTH * (1 - np.clip(f1 / (0.6 * CELL), 0, 1) ** 2)   # 粒をほんの少し丸く
    return groove + dome


def build():
    xs = [p[0] for p in OUTLINE]
    ys = [p[1] for p in OUTLINE]
    s = WIDTH / (max(xs) - min(xs))
    x0, y1 = min(xs), max(ys)
    m = lambda pts: to_mm(pts, s, x0, y1)
    outline = smooth(m(OUTLINE))
    # 写真の鼻の穴は暗い部分のふちで写しているので、なめらかにした分だけ少し広げて元の大きさに戻す
    nl, nr = smooth(m(NOSTRIL_L), 120).buffer(0.6), smooth(m(NOSTRIL_R), 120).buffer(0.6)
    # まわりに 1mm の余白をとって格子に収める（格子の端に形がかかると、ふちが高い壁になってしまう）
    ox, oy = 1.0 - outline.bounds[0], 1.0 - outline.bounds[1]
    sh = lambda g: shapely.affinity.translate(g, ox, oy)
    outline, nl, nr = sh(outline), sh(nl), sh(nr)
    m0 = m
    m = lambda pts: [(x + ox, y + oy) for x, y in m0(pts)]
    W, H = outline.bounds[2] + 1, outline.bounds[3] + 1
    nx, ny = int(W / GRID) + 1, int(H / GRID) + 1

    inside = raster(outline, nx, ny)
    d = ndi.distance_transform_edt(inside) * GRID
    r = np.clip(d / ROUND, 0, 1)
    t = EDGE + DOME * np.sqrt(1 - (1 - r) ** 2)                     # 丸いふくらみ

    # 鼻の穴: 縁はなめらかに落ち込み、奥は丸いおわん形に深くなる
    nos = (raster(nl, nx, ny) | raster(nr, nx, ny))
    dn = ndi.distance_transform_edt(nos) * GRID                       # 穴のふちからの距離
    bowl = np.clip(dn / 1.6, 0, 1) ** 0.6
    rim = np.clip(ndi.gaussian_filter(nos.astype(float), 0.45 / GRID) * 2, 0, 1)
    t -= NOSTRIL_DEPTH * np.maximum(bowl, 0.35 * rim)
    # 縦の溝
    (gx0, gy0), (gx1, gy1) = m(GROOVE)
    X, Y = np.meshgrid(np.arange(nx) * GRID, np.arange(ny) * GRID)
    Yimg = (ny - 1) * GRID - Y                                       # raster と同じ向き（行0が上）に合わせる
    along = np.clip((gy0 - Yimg) / (gy0 - gy1), 0, 1)
    fade = np.clip(along * 4, 0, 1)
    thin = np.clip((t - 1.0) / 4.0, 0, 1)                              # ふちの薄い所では浅く（下に切れ込みすぎない）
    t -= GROOVE_DEPTH * fade * thin * np.exp(-((X - gx0) / GROOVE_W) ** 2) * ((Yimg <= gy0 + 0.5) & (Yimg >= gy1 - 2))
    # シボ（鼻の穴の中と溝には入れない）
    calm = ndi.binary_dilation(nos > 0.5, iterations=int(0.6 / GRID))
    tex = voronoi_grooves(nx, ny)[::-1]
    t += tex * ~calm
    t = np.where(inside, np.maximum(t, 0.8), 0.02)

    mesh = heightmap_mesh(t, GRID)                                   # x 右, y 上, 裏が z=0
    pts, tri = fast_simplification.simplify(mesh.vertices, mesh.faces, target_reduction=0.55)
    mesh = trimesh.Trimesh(pts, tri, process=True)
    mesh.fix_normals()
    nose = mesh.intersection(extrude(outline, 30, -1))

    # キーリングの輪（鼻の上のまん中、鼻のふちに少し重ねる）
    top = outline.bounds[3]
    cx = (outline.bounds[0] + outline.bounds[2]) / 2
    ring = Point(cx, top + RING_OUT - 1.6).buffer(RING_OUT, quad_segs=32).difference(
        Point(cx, top + RING_OUT - 1.6).buffer(RING_IN, quad_segs=32))
    ring = ring.difference(outline.buffer(-0.8))                      # 鼻の中には入れない
    keychain = nose.union(extrude(ring, RING_T, 0))
    return keychain, t, outline


def main():
    os.makedirs(OUT, exist_ok=True)
    kc, t, outline = build()
    kc.export(os.path.join(OUT, "bailey_nose_keychain.stl"))
    write_3mf(os.path.join(OUT, "bailey_nose_keychain.3mf"), [("bailey_nose", (90, 90), [("nose_black", kc, 2)])])
    print(f"keychain {kc.extents.round(1)} mm watertight={kc.is_watertight} vol={kc.volume / 1000:.2f}cm3 "
          f"faces={len(kc.faces)}")
    return kc


if __name__ == "__main__":
    main()
