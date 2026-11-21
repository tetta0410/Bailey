"""成長アルバム・ペン立てを4方向（正面・左・右・背面）から描く。
彫った部分は実物では白い影で見えるが、画像では分かりにくいので少し灰色にして描く。"""
import importlib.util
import os
import sys

import numpy as np
import trimesh
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_bailey_album as ma  # noqa: E402

_spec = importlib.util.spec_from_file_location("ittetsu_preview", os.path.join(HERE, "ittetsu", "preview.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

WHITE, BLACK, PINK, CARVED = (np.array(c) for c in ([0.97, 0.96, 0.93], [0.11, 0.11, 0.13],
                                                     [0.94, 0.53, 0.60], [0.74, 0.73, 0.70]))
parts = ma.main()
meshes, cols = [], []
for name, m, _ in parts:
    c = np.tile({"body_white": WHITE, "face_black": BLACK, "tongue_pink": PINK}[name], (len(m.faces), 1))
    if name == "body_white":
        ctr, n = m.triangles_center, m.face_normals
        lo, hi = ma.ENGRAVE - 0.2, ma.ENGRAVE + 0.2
        carved = (((ctr[:, 0] > lo) & (ctr[:, 0] < hi) & (n[:, 0] < -0.9)) |
                  ((ctr[:, 0] > ma.W - hi) & (ctr[:, 0] < ma.W - lo) & (n[:, 0] > 0.9)) |
                  ((ctr[:, 1] > ma.D - hi) & (ctr[:, 1] < ma.D - lo) & (n[:, 1] > 0.9)))
        c[carved] = CARVED
    meshes.append(m)
    cols.append(c)
m = trimesh.util.concatenate(meshes)
cols = np.vstack(cols)
views = [_mod.render(m, a, 12, cols, size=520, scale=4.0) for a in (0, -90, 90, 180)]
sheet = Image.new("RGB", (sum(v.width for v in views), views[0].height))
x = 0
for v in views:
    sheet.paste(v, (x, 0)); x += v.width
sheet.save(os.path.join(ma.mb.OUT, "preview_bailey_album.png"))
