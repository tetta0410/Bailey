"""ベイリーの4面アルバム（イラスト版）— 用意してもらった白黒イラストを、そのまま板にする。

イラストのペン線は 0.1〜0.2mm で、0.4mm ノズルでは白と黒の2色にくっきり分けると細部が消える。
そこで「黒の上に白を重ねる厚み」で灰色を出す方法（フィラメント・ペインティング）を使う:
  - 板は絵の面を上にして印刷する。下は黒、上に白を重ねる
  - 白が厚い所ほど白く、薄い所ほど下の黒が透けて灰色、白がない所は黒
  - イラストの明るさ → 白の厚み（0〜W_MAX）。絵は描き足さない
  - 色替えは黒→白の1回だけ（4枚とも同じ高さで替わる）
板のまわり FRAME mm は黒の縁（箱と同じ黒、表面と同じ高さ）。

箱は make_bailey_album_final.py と同じ（黒、底に白で Bailey 2026.1.15）。

出力 (3d/out/):
  bailey_illust_plates.3mf … 板4枚（plate_black / plate_white）を1枚のビルドプレートに
  bailey_final_box.3mf     … 箱（make_bailey_album_final.py で作る）
  bailey_illust_graytest.3mf … 灰色の出方を確かめるテスト片（白の厚み 8段階）
  preview_bailey_illust_*.png
"""
import math
import os
import sys

import cv2
import fast_simplification
import numpy as np
import shapely
import trimesh
from shapely.geometry import Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "ittetsu"))
import bailey_illust as bi  # noqa: E402
import make_bailey_album_plates as mp  # noqa: E402
from make_bailey import extrude  # noqa: E402
from make_bailey_lantern import heightmap_mesh  # noqa: E402
from make_pencase import write_3mf  # noqa: E402

OUT = os.path.join(HERE, "out")
ILLUSTS = {"front": "tongue", "left": "sleep", "right": "paw", "back": "nose"}

T = mp.T                 # 板の厚み（溝に合わせる）
W_MAX = 1.6              # 白のいちばん厚い所 (mm)。白PLAはだいたいこの厚みで下の黒が見えなくなる
LAYER = 0.08             # 積層ピッチ（細かいほど灰色の段階が増える）
ZB = T - W_MAX           # 黒→白に替わる高さ
FRAME = 2.5              # 板のまわりの黒い縁 (mm)
GRID = 0.15              # 厚みマップの細かさ (mm)
KEEP = 0.35              # メッシュの面数をどこまで減らすか（残す割合）


def face_size(face):
    """板の表（見える面）の幅と、箱の縁より下の長さ。"""
    return mp.PW[face] - 2 * mp.INSET, mp.PH - 1.0


def brightness(face):
    """板の絵の部分（縁の内側）の明るさマップ 0..1（行0が上）。イラストを縦横比のまま中央に置き、余白は白。"""
    w, h = face_size(face)
    aw, ah = w - 2 * FRAME, h - 2 * FRAME
    nx, ny = round(aw / GRID), round(ah / GRID)
    g = bi.load(ILLUSTS[face]).astype(np.float32)
    k = min(nx / g.shape[1], ny / g.shape[0])
    gw, gh = round(g.shape[1] * k), round(g.shape[0] * k)
    g = cv2.resize(g, (gw, gh), interpolation=cv2.INTER_AREA)
    canvas = np.ones((ny, nx), np.float32)
    y0, x0 = (ny - gh) // 2, (nx - gw) // 2
    canvas[y0:y0 + gh, x0:x0 + gw] = g
    return canvas


def plate(face):
    """絵の面を上にした板（x 中央 0, y 0..PH, z 0..T）。黒の部分と白の部分を返す。"""
    w, h = face_size(face)
    wb = mp.PW[face] / 2
    wt = wb - mp.INSET                                     # 表（上）が狭いアリ溝の断面
    sec = Polygon([(-wb, 0), (wb, 0), (wt, T), (-wt, T)])
    body = trimesh.creation.extrude_polygon(sec, mp.PH)    # (x, z断面) を y 方向へ
    v = body.vertices.copy()
    body.vertices = np.column_stack([v[:, 0], v[:, 2], v[:, 1]])
    body.fix_normals()
    # 絵の部分は黒を ZB まで下げ、その上に白の厚みマップを載せる
    aw, ah = w - 2 * FRAME, h - 2 * FRAME
    pocket = extrude(shapely.box(-aw / 2, FRAME, aw / 2, FRAME + ah), 2 * T, 0)
    v = pocket.vertices.copy()
    pocket.vertices = np.column_stack([v[:, 0], v[:, 1], v[:, 2] + ZB])
    black = body.difference(pocket)

    b = brightness(face)
    t = np.round(b * W_MAX / LAYER) * LAYER                # 層の厚みにそろえる
    t = np.clip(t, 0.02, W_MAX)                            # 0 だと閉じた形にならないので、ごく薄く残す（印刷されない）
    white = heightmap_mesh(t, GRID)
    white.apply_scale([aw / (t.shape[1] - 1) / GRID, ah / (t.shape[0] - 1) / GRID, 1])
    white.apply_translation([-aw / 2, FRAME, ZB])
    # 面数を減らす（平らな所が多いので見た目は変わらない）。閉じた形が崩れたら残す割合を増やしてやり直す
    full = white
    for keep in (KEEP, 0.45, 0.6, 0.8):
        pts, tri = fast_simplification.simplify(full.vertices, full.faces, target_reduction=1 - keep)
        white = trimesh.Trimesh(pts, tri, process=True)
        white.fix_normals()
        if white.is_watertight:
            break
    else:
        white = full
    return black, white, b


def installed(face, m):
    """面を上にした板を、箱の面に差し込んだ位置へ（プレビュー用）。外から見て x が右、y が上。"""
    m = m.copy()
    v = m.vertices.copy()
    m.vertices = mp.face_map(face, v[:, 0], T - v[:, 2], v[:, 1] + mp.GROOVE_Z)
    m.fix_normals()
    return m


def appearance(face, b, px=0.1):
    """仕上がりの見え方の目安: 白の厚み（層ごと）で黒が透ける割合を決め、ノズル幅でぼかす。"""
    w, h = face_size(face)
    t = np.round(b * W_MAX / LAYER) * LAYER
    a = np.clip(t / W_MAX, 0, 1)
    a = cv2.resize(a, (round((w - 2 * FRAME) / px), round((h - 2 * FRAME) / px)), interpolation=cv2.INTER_LINEAR)
    a = cv2.GaussianBlur(a, (0, 0), 0.2 / px)              # 0.4mm ノズルでにじむ分
    fr = round(FRAME / px)
    full = np.zeros((a.shape[0] + 2 * fr, a.shape[1] + 2 * fr), np.float32)
    full[fr:fr + a.shape[0], fr:fr + a.shape[1]] = a
    black, white = np.array([26, 26, 28.0]), np.array([240, 238, 232.0])
    return (black + (white - black) * full[..., None]).astype(np.uint8)


def gray_test():
    """白の厚みごとの灰色を確かめるテスト片（40×16mm、8段階）。本番と同じ黒の高さ・白の厚みで作る。"""
    steps = 8
    sw, sh = 5.0, 12.0
    black = extrude(shapely.box(0, 0, steps * sw + 4, sh + 4), ZB, 0)
    whites = []
    for i in range(steps):
        t = (i + 1) * W_MAX / steps
        whites.append(extrude(shapely.box(2 + i * sw, 2, 2 + (i + 1) * sw, 2 + sh), t, ZB))
    white = trimesh.util.concatenate(whites)
    black = black.union(extrude(shapely.difference(shapely.box(0, 0, steps * sw + 4, sh + 4),
                                                   shapely.box(2, 2, 2 + steps * sw, 2 + sh)), W_MAX, ZB))
    write_3mf(os.path.join(OUT, "bailey_illust_graytest.3mf"),
              [("graytest", (90, 90), [("test_black", black, 1), ("test_white", white, 2)])])
    return black, white


def main():
    os.makedirs(OUT, exist_ok=True)
    centers = {"front": (49, 46), "back": (127, 46), "left": (41.5, 135), "right": (104.5, 135)}
    objs, plates = [], {}
    for f in ("front", "left", "right", "back"):
        black, white, b = plate(f)
        plates[f] = (black, white, b)
        objs.append((f"plate_{f}", centers[f], [("plate_black", black, 1), ("plate_white", white, 2)]))
        print(f"{f:5s} black {black.extents.round(1)} watertight={black.is_watertight}  "
              f"white faces={len(white.faces)} watertight={white.is_watertight}")
    write_3mf(os.path.join(OUT, "bailey_illust_plates.3mf"), objs)
    print(f"color change: black up to z={ZB:.2f} mm, white above ({int(round(W_MAX / LAYER))} layers of {LAYER} mm)")
    gb, gw = gray_test()
    print(f"gray test {gb.extents.round(1)} watertight={gb.is_watertight}/{gw.is_watertight}")
    return plates


if __name__ == "__main__":
    main()
