"""用意してもらったベイリーのイラスト（白黒の版画風）を、印刷できる白黒の図にする。

絵は描き足さない。イラストをそのまま白黒にしたうえで、0.4mm ノズルで出せない細かさだけを整える:
  - 細すぎる黒い点や白い点は消す（ノズルより細いものは印刷で出ないため）
  - 長い線（ヒゲ・毛の流れ）は、形はそのままで MIN_LINE の太さにそろえる
  - 目の光など、イラストではっきり明るい小さな点は、消さずに印刷できる大きさまで広げる
"""
import functools
import os

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import measure, morphology

HERE = os.path.dirname(os.path.abspath(__file__))
ILLUST = os.path.join(HERE, "illust")
PX_MM = 0.05            # 1画素 (mm)
MIN_LINE = 0.42         # 印刷できる線の太さ (mm)。0.4mm ノズル1本分
MIN_DOT = 0.45          # 印刷できる点の大きさ (mm)
LONG_LINE = 2.5         # これより長い細い線は、消さずに MIN_LINE の太さで残す (mm)
THRESHOLD = 0.5         # これより暗い所を黒にする（0..1）

FILES = {
    "nose": "bailey_nose_illust.png",
    "tongue": "bailey_tongue_illust.png",
    "sleep": "bailey_sleep_illust.png",
    "paw": "bailey_paw_illust.jpg",
}


@functools.lru_cache()
def load(key):
    """イラスト → 明るさ 0..1（透明な背景は白）。絵のある範囲で切り抜く。"""
    im = Image.open(os.path.join(ILLUST, FILES[key]))
    a = np.asarray(im.convert("RGBA")).astype(float) / 255
    g = a[..., :3].mean(2) * a[..., 3] + (1 - a[..., 3])
    ys, xs = np.nonzero(g < 0.9)
    return g[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def aspect(key):
    g = load(key)
    return g.shape[1] / g.shape[0]


def _thin_long(mask, r):
    """mask のうち、太さが 2r より細い部分で、長い線になっているものを MIN_LINE の太さにして返す。"""
    thin = mask & ~morphology.opening(mask, morphology.disk(r))
    lab = measure.label(thin)
    keep = np.zeros(mask.shape, bool)
    for reg in measure.regionprops(lab):
        if reg.axis_major_length * PX_MM >= LONG_LINE and reg.axis_major_length > 4 * max(reg.axis_minor_length, 1):
            sl = reg.slice
            keep[sl] |= lab[sl] == reg.label
    if not keep.any():
        return keep
    return ndi.binary_dilation(morphology.skeletonize(keep), structure=morphology.disk(r))


def ink(key, width_mm):
    """イラスト → 白黒（True=黒）。幅 width_mm、1画素 PX_MM。"""
    g = load(key)
    k = width_mm / PX_MM / g.shape[1]
    g = cv2.resize(g.astype(np.float32), (round(g.shape[1] * k), round(g.shape[0] * k)), interpolation=cv2.INTER_AREA)
    g = ndi.gaussian_filter(g, 0.04 / PX_MM)
    black = g < THRESHOLD
    r = max(1, round(MIN_LINE / 2 / PX_MM))

    # 長い細い線（ヒゲ・毛の流れ）は形を保って太さをそろえる。黒い線と白い線の両方
    black_lines = _thin_long(black, r)
    white_lines = _thin_long(~black, r)
    # 細すぎる部分を消す（ノズルで出ない太さ）
    solid = morphology.opening(black, morphology.disk(r))          # 細い黒を消す
    solid = morphology.closing(solid, morphology.disk(r))          # 細い白を埋める
    out = (solid | black_lines) & ~white_lines

    # 目の光など: イラストではっきり明るい小さな点は、印刷できる大きさに広げて残す
    peaks = (g > 0.75) & ndi.binary_erosion(ndi.binary_dilation(black, iterations=r * 2), iterations=r)
    lab = measure.label(peaks)
    dots = np.zeros(out.shape, bool)
    for reg in measure.regionprops(lab):
        if reg.area * PX_MM ** 2 < (MIN_DOT ** 2):
            cy, cx = reg.centroid
            yy, xx = np.ogrid[:out.shape[0], :out.shape[1]]
            dots |= (yy - cy) ** 2 + (xx - cx) ** 2 <= (MIN_DOT / 2 / PX_MM) ** 2
    out &= ~dots

    # 仕上げ: 印刷で出ない小さな黒い点・白い穴を整理
    out = morphology.remove_small_objects(out, max_size=int((MIN_DOT / PX_MM) ** 2))
    out = ~morphology.remove_small_objects(~out, max_size=int((MIN_DOT / PX_MM) ** 2 * 0.6))
    return out
