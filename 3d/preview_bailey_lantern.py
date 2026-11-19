"""ランタンの見え方: 明かりを付けたとき（透過のシミュレーション）と、本体の形。"""
import importlib.util
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_bailey_lantern as ml  # noqa: E402

_spec = importlib.util.spec_from_file_location("ittetsu_preview", os.path.join(HERE, "ittetsu", "preview.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# 1) 明かりを付けたとき: 正面全体の厚みマップを作り、白PLAの透過（厚みに対しおおよそ指数的に減る）で明るさにする
from PIL import ImageDraw  # noqa: E402
import shapely  # noqa: E402
P = 0.25
nx, ny = int(ml.W / P), int(ml.H / P)
front = np.full((ny, nx), ml.FRONT)
t = ml.T_MIN + (1 - ml.photo_brightness()) * (ml.T_MAX - ml.T_MIN)
x0 = int((ml.W - ml.WIN_W) / 2 / P); y0 = ny - int((ml.WIN_BOTTOM + ml.WIN_H) / P)
front[y0:y0 + t.shape[0], x0:x0 + t.shape[1]] = t
name = ml.text_poly(ml.NAME, ml.NAME_H)
k = ml.NAME_H / (name.bounds[3] - name.bounds[1])
name = shapely.affinity.translate(shapely.affinity.scale(name, k, k, origin=(0, 0)), ml.W / 2, ml.WIN_BOTTOM / 2)
mask = Image.new("L", (nx, ny), 0)
d = ImageDraw.Draw(mask)
for g in getattr(name, "geoms", [name]):
    d.polygon([(x / P, ny - y / P) for x, y in g.exterior.coords], fill=255)
    for hole in g.interiors:
        d.polygon([(x / P, ny - y / P) for x, y in hole.coords], fill=0)
front[np.asarray(mask) > 0] = ml.FRONT - ml.NAME_DEPTH
glow = np.exp(-1.25 * (front - ml.T_MIN))
warm = np.array([1.0, 0.86, 0.62])                      # ティーライトの暖かい色
lit = Image.fromarray((glow[..., None] * warm * 255).clip(0, 255).astype(np.uint8))
lit = lit.resize((380, int(380 * ny / nx)))
frame = Image.new("RGB", (lit.width + 40, lit.height + 40), (40, 34, 28))
frame.paste(lit, (20, 20))

# 2) 本体の形（粗い格子で描く）
ml.PX = 0.6
m = ml.make_lantern()
cols = np.tile(np.array([0.96, 0.95, 0.92]), (len(m.faces), 1))
shape = _mod.render(m, -28, 15, cols, size=frame.height, scale=frame.height / 150)

sheet = Image.new("RGB", (frame.width + shape.width, frame.height), (245, 243, 238))
sheet.paste(frame, (0, 0))
sheet.paste(shape, (frame.width, 0))
sheet.save(os.path.join(ml.OUT, "preview_bailey_lantern.png"))
