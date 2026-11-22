"""プレート差し込み式アルバムの完成イメージ（箱は緑、プレートは白黒）。4方向＋斜め上。"""
import importlib.util
import os
import sys

import numpy as np
import trimesh
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_bailey_album_plates as mp  # noqa: E402

_spec = importlib.util.spec_from_file_location("ittetsu_preview", os.path.join(HERE, "ittetsu", "preview.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

BOX = np.array([0.13, 0.33, 0.22]) if "--white" not in sys.argv else np.array([0.97, 0.96, 0.93])
WHITE, BLACK = np.array([0.97, 0.96, 0.93]), np.array([0.11, 0.11, 0.13])

box = mp.make_box()
meshes, cols = [box], [np.tile(BOX, (len(box.faces), 1))]
for f in mp.PW:
    body, art = mp.plate(f)
    for m, c in ((mp.installed(f, body), WHITE), (mp.installed(f, art), BLACK)):
        meshes.append(m)
        cols.append(np.tile(c, (len(m.faces), 1)))
m = trimesh.util.concatenate(meshes)
cols = np.vstack(cols)
views = [_mod.render(m, a, e, cols, size=460, scale=3.6) for a, e in ((0, 12), (-90, 12), (90, 12), (180, 12), (30, 35))]
sheet = Image.new("RGB", (sum(v.width for v in views), views[0].height))
x = 0
for v in views:
    sheet.paste(v, (x, 0)); x += v.width
sheet.save(os.path.join(mp.OUT, "preview_bailey_album_plates.png"))
