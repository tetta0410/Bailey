"""ベイリーの写真を、白黒の線画（版画風）に写し取る。

描き足しや置き換えはせず、写真そのものから:
  - 背景を外して犬だけにする（GrabCut。写真ごとに犬のまわりの目安を指定）
  - 暗い毛は黒く塗る
  - 明暗の境目（目の縁・毛の流れ・ヒゲ・口元）を黒い線で写す
  - 目のまわりだけ、黒い毛の中の明るい線（目のふち）や目の光を白で抜く
    （黒い毛全体でやると白く飛んで見えるので、毛は黒のまま）
  - 舌・肉球など赤みのある所は塗らず、ふちを線で描く
最後に 0.4mm ノズルで印刷できる太さ・大きさにそろえる。
"""
import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage as ndi
from skimage import morphology

HERE = os.path.dirname(os.path.abspath(__file__))
PHOTOS = os.path.join(HERE, "photos")
PX_MM = 0.05            # 線画の 1画素 (mm)

NOSE_OUTLINE = [(330, 340), (390, 310), (470, 350), (560, 420), (630, 420), (710, 400), (800, 300),
                (870, 240), (1030, 150), (1108, 150), (1108, 1477), (390, 1477), (330, 1300),
                (250, 1150), (170, 1000), (220, 920), (270, 840), (275, 700), (270, 560), (290, 470)]
TONGUE_OUTLINE = [(100, 0), (540, 0), (600, 230), (700, 230), (770, 0), (1180, 0), (1170, 250),
                  (1120, 500), (1090, 650), (1060, 800), (1080, 1000), (1110, 1290), (390, 1290),
                  (400, 1100), (330, 850), (250, 650), (170, 450), (110, 250)]
SIT_OUTLINE = [(x * 2, y * 2) for x, y in [
    (100, 45), (160, 55), (250, 150), (330, 170), (410, 130), (470, 40), (520, 0), (540, 80), (560, 200),
    (580, 300), (560, 360), (540, 430), (520, 520), (520, 620), (500, 660), (430, 700), (410, 720), (300, 730),
    (250, 720), (220, 640), (200, 580), (180, 520), (170, 440), (150, 350), (120, 250), (100, 150)]]
SLEEP2_OUTLINE = [(0, 230), (470, 215), (580, 60), (610, 0), (840, 0), (920, 200), (1000, 300), (1150, 350),
                  (1280, 450), (1320, 620), (1430, 780), (1570, 840), (1580, 980), (1450, 1060), (1250, 1080),
                  (1050, 1020), (870, 950), (780, 1100), (600, 1180), (300, 1200), (0, 1250)]
PAW2_OUTLINE = [(x * 2, y * 2) for x, y in [
    (110, 300), (140, 230), (220, 180), (320, 160), (440, 150), (540, 160), (610, 190), (600, 260), (590, 330),
    (600, 420), (570, 480), (520, 520), (440, 520), (360, 500), (270, 470), (200, 440), (150, 400), (120, 350)]]
DARK = {"sit": 0.40}    # 黒い毛が日に当たって灰色に写っている写真は、黒く塗る範囲を少し広げる
IRIS = {"tongue2": 0.12, "sit": 0.11, "nose": 0.14}   # 目玉の中で白にする明るさ
# 目玉（元の写真の画素: 中心x, 中心y, 半径）
EYEBALLS = {
    "tongue2": [(478, 514, 50), (857, 475, 47)],
    "sit": [(596, 624, 46), (926, 580, 40)],
    "nose": [(842, 560, 48)],
}
# 閉じた目: 黒い毛の中のまぶたの線・目の切れ込みを、写真の明暗の線のまま白い線で入れる
CLOSED_EYES = {"sleep2"}
NO_PINK = {"sleep2"}      # 口元のうっすらしたピンクは肌の色なので、舌のように縁取らない
EYE_BRIGHT = {"sleep2": 0.99, "tongue2": 0.40, "sit": 0.42, "nose": 0.40}   # 目のまわりで白にする明るさ（犬の中で正規化した 0..1）
# 目のまわり（元の写真の画素: 中心x, 中心y, 半径）。ここだけ黒い毛の中の細部を白で抜く
EYE_ZONES = {
    "tongue2": [(478, 514, 75), (857, 475, 72)],
    "nose": [(840, 560, 75)],
    "sit": [(596, 624, 58), (926, 580, 52)],
    "sleep2": [(1110, 495, 95)],
}
PAW_OUTLINE = [(330, 640), (460, 480), (660, 430), (920, 420), (1120, 460), (1280, 540), (1400, 660),
               (1560, 780), (1680, 860), (1720, 960), (1680, 1080), (1580, 1160), (1440, 1240), (1280, 1300),
               (1120, 1280), (900, 1240), (720, 1140), (500, 1040), (360, 940), (320, 800)]


def _load(name):
    return np.asarray(ImageOps.exif_transpose(Image.open(os.path.join(PHOTOS, name))).convert("RGB"))


def _poly(shape, pts):
    m = Image.new("L", (shape[1], shape[0]), 0)
    ImageDraw.Draw(m).polygon(pts, fill=255)
    return np.asarray(m) > 0


def _largest(m):
    lab, n = ndi.label(m)
    return lab == (np.argmax(ndi.sum(m, lab, range(1, n + 1))) + 1) if n else m


def _grabcut(a, poly, scale=800):
    """犬のまわりの目安 poly から、GrabCut で犬の形を切り抜く。"""
    h, w = a.shape[:2]
    k = scale / max(h, w)
    sm = cv2.resize(a, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
    pm = _poly(sm.shape, [(x * k, y * k) for x, y in poly])
    mask = np.full(sm.shape[:2], cv2.GC_BGD, np.uint8)
    mask[ndi.binary_dilation(pm, iterations=25)] = cv2.GC_PR_BGD
    mask[pm] = cv2.GC_PR_FGD
    mask[ndi.binary_erosion(pm, iterations=40)] = cv2.GC_FGD
    cv2.grabCut(cv2.cvtColor(sm, cv2.COLOR_RGB2BGR), mask, None, np.zeros((1, 65)), np.zeros((1, 65)), 6,
                cv2.GC_INIT_WITH_MASK)
    m = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8) * 255
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR) > 127
    return ndi.binary_fill_holes(_largest(m))


def prepare(key):
    """写真 → (画像, 犬の形, 輪郭線を引かない所, 目のまわり)。犬の外枠で切り抜いて返す。"""
    nol = None
    rot = 0
    if key == "nose":
        a = _load("bailey_nose.jpg")
        m = _grabcut(a, NOSE_OUTLINE)
    elif key == "tongue2":
        a = _load("bailey_tongue2.jpg")[:1290]                 # 名札より下は使わない
        m = _grabcut(a, TONGUE_OUTLINE) & _poly(a.shape, TONGUE_OUTLINE)
        nol = np.zeros(m.shape, bool)
        nol[:330, 500:820] = True                              # 耳と耳の間: 白い毛と白いカーテンの境目は描かない
    elif key == "puppy2":
        a = cv2.resize(_load("bailey_puppy_cutout.png"), (1512, 1890), interpolation=cv2.INTER_AREA)
        f = a.astype(int)
        green = (f[..., 1] > f[..., 0] + 10) & (f[..., 1] > f[..., 2] + 5)   # 切り抜きの縁に残った芝生
        m = ndi.binary_fill_holes(ndi.binary_closing((a.min(2) < 235) & ~green, iterations=3))
        m = ndi.binary_opening(_largest(m), iterations=4)
    elif key == "sit":
        a = _load("bailey_sit.jpg")[:1480]                     # 足先まで
        m = _grabcut(a, SIT_OUTLINE) & _poly(a.shape, SIT_OUTLINE)
    elif key == "sleep2":
        a = _load("bailey_sleep2.jpg")
        m = _grabcut(a, SLEEP2_OUTLINE) & _poly(a.shape, SLEEP2_OUTLINE)
        nol = np.zeros(m.shape, bool)
        nol[1080:, :800] = True                                # 体の下側は写真の途中で切るので輪郭線を引かない
    elif key == "paw":
        a = _load("bailey_paw.jpg")
        m = _grabcut(a, PAW_OUTLINE)
        rot = 1                                                 # 指を上に
    elif key == "paw2":
        a = _load("bailey_paw2c.jpg")
        m = _grabcut(a, PAW2_OUTLINE) & _poly(a.shape, PAW2_OUTLINE)
        rot = -1                                                # 指を上に
    else:
        raise ValueError(key)
    if nol is None:
        nol = np.zeros(m.shape, bool)
    eye = np.zeros(m.shape, bool)
    ball = np.zeros(m.shape, bool)
    yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
    for cx, cy, r in EYE_ZONES.get(key, []):
        eye |= (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
    for cx, cy, r in EYEBALLS.get(key, []):
        ball |= (xx - cx) ** 2 + (yy - cy) ** 2 < r * r
    eye = np.where(ball, 2, eye.astype(np.uint8))                          # 0=外, 1=目のまわり, 2=目玉
    if rot:
        a, m, nol, eye = (np.ascontiguousarray(np.rot90(v, rot)) for v in (a, m, nol, eye))
    ys, xs = np.nonzero(m)
    sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
    return a[sl], m[sl], nol[sl], eye[sl]


def sketch(key, width_mm, dark=0.30, edge=0.035, s_mm=0.22):
    """写真 → 白黒の線画（True=黒）。1画素 = PX_MM。"""
    paw = key in ("paw", "paw2")
    a, m, nol, eye = prepare(key)
    k = width_mm / PX_MM / a.shape[1]
    size = (int(a.shape[1] * k), int(a.shape[0] * k))
    a = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    m = cv2.resize(m.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    nol = cv2.resize(nol.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    eye = cv2.resize(eye.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
    eyeball = eye == 2
    eye = eye > 0
    f = a.astype(float)
    g = f.mean(2) / 255
    lo, hi = np.percentile(g[m], [2, 98])                      # 写真ごとの露出差をそろえる
    g = np.clip((g - lo) / (hi - lo), 0, 1)
    s = s_mm / PX_MM

    lines = (ndi.gaussian_filter(g, s) - ndi.gaussian_filter(g, 1.6 * s)) < -edge   # 明暗の境目の暗い側
    fill = ndi.gaussian_filter(g, 1.0) < DARK.get(key, dark)                       # 暗い毛
    red = ndi.gaussian_filter(f[..., 0] - (f[..., 1] + f[..., 2]) / 2, 2)
    pinkish = (red > 55) & (key not in NO_PINK)                                    # 舌など
    fill &= ~pinkish
    if paw:
        pinkish = ndi.gaussian_filter(f[..., 0], 4) / (ndi.gaussian_filter(f[..., 1], 4) + 1) > (1.48 if key == "paw" else 1.3)
        fill = ndi.gaussian_filter(f.mean(2), 1.5) < 45                           # 暗い写真なので本当に黒いホクロだけ
        if key == "paw2":
            fill[:] = False                                                       # ホクロのない足。影の肉球は塗らない
            # 白い毛の流れ: 細かい明暗の境目のうち、細長いもの（毛の筋）だけを線にする
            fs = 0.12 / PX_MM
            dd = ndi.gaussian_filter(g, fs) - ndi.gaussian_filter(g, 2.5 * fs)
            strands = (dd < -0.035) & ~ndi.binary_dilation(pinkish, iterations=int(0.6 / PX_MM))
            from skimage import measure
            lab = measure.label(strands)
            keep = [r.label for r in measure.regionprops(lab)
                    if r.area > (0.5 / PX_MM) ** 2 and r.axis_major_length > 4 * max(r.axis_minor_length, 1)]
            lines |= np.isin(lab, keep)
        r2 = ndi.gaussian_filter(red / 255, s) - ndi.gaussian_filter(red / 255, 1.6 * s)
        lines |= np.abs(r2) > 0.03

    sil = m ^ ndi.binary_erosion(m, iterations=int(0.45 / PX_MM))                  # 犬の輪郭
    sil &= ~nol
    e = int(0.8 / PX_MM)
    sil[:e] = sil[-e:] = False
    sil[:, :e] = sil[:, -e:] = False                                               # 写真の端で切れた所は描かない
    pk = morphology.remove_small_objects(ndi.binary_opening(pinkish & m, iterations=3),
                                         max_size=int((2.0 / PX_MM) ** 2))
    pk = ndi.binary_fill_holes(ndi.binary_closing(pk, iterations=4))
    pk_line = pk ^ ndi.binary_erosion(pk, iterations=int(0.5 / PX_MM))             # 舌・肉球のふち
    black = ((fill | lines | pk_line) & m) | sil

    # 目のまわりだけ: 写真で実際に明るい所（目の光・瞳の下の明るい縁）だけを白で抜く。
    # コントラストを強めたりせず、写真の明るさそのものを使う。
    bright = (ndi.gaussian_filter(g, 0.08 / PX_MM) > EYE_BRIGHT.get(key, 0.4)) & fill & eye
    bright = morphology.remove_small_objects(bright, max_size=int((0.25 / PX_MM) ** 2))
    # 印刷で潰れない大きさ（0.4mm 程度）に少しだけ広げる
    black &= ~ndi.binary_dilation(bright, structure=morphology.disk(int(0.1 / PX_MM)))
    if key in CLOSED_EYES:
        loc = np.clip(g * 3, 0, 1)                                      # 暗い所を見やすく（写真の明暗の順番は変えない）
        loc = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)).apply((loc * 255).astype(np.uint8)) / 255.0
        cs = 0.15 / PX_MM
        crease = ((ndi.gaussian_filter(loc, cs) - ndi.gaussian_filter(loc, 1.6 * cs)) < -0.03) & eye & fill
        black &= ~morphology.remove_small_objects(crease, max_size=200)
    # 目玉の中: まわりの黒い毛より明るく写っている瞳の茶色い部分も白にする（真ん中の黒い瞳はそのまま黒）
    if key in IRIS:
        gi = ndi.gaussian_filter(g, 0.1 / PX_MM)
        iris = (gi > IRIS[key]) & fill & eyeball
        iris = morphology.remove_small_objects(iris, max_size=int((0.3 / PX_MM) ** 2))
        black &= ~iris

    # 印刷できない小さな黒い点を消し、小さすぎる白い穴は埋める
    black = morphology.remove_small_objects(black, max_size=int((0.6 / PX_MM) ** 2))
    black = ~morphology.remove_small_objects(~black, max_size=int((0.38 / PX_MM) ** 2))
    return black


def aspect(key):
    a = prepare(key)[0]
    return a.shape[1] / a.shape[0]
