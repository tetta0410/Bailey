"""完成版（黒い箱×白い板×黒いベイリー）の2パターンを描く: 上段=板差し込み、下段=一体型。底の文字も1枚。"""
import importlib.util
import os
import sys

import numpy as np
import trimesh
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_bailey_album_final as mf  # noqa: E402

_spec = importlib.util.spec_from_file_location("ittetsu_preview", os.path.join(HERE, "ittetsu", "preview.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

BLACK, WHITE = np.array([0.13, 0.13, 0.14]), np.array([0.96, 0.95, 0.92])
box, text, plates, body, white = mf.main()


def scene(parts):
    meshes = [m for m, _ in parts]
    cols = np.vstack([np.tile(c, (len(m.faces), 1)) for m, c in parts])
    return trimesh.util.concatenate(meshes), cols


a_parts = [(box, BLACK), (text, WHITE)]
for f, (b, a) in plates.items():
    a_parts += [(mf.mp.installed(f, b), WHITE), (mf.mp.installed(f, a), BLACK)]
A = scene(a_parts)
B = scene([(body, BLACK), (white, WHITE)])
rows = []
for m, cols in (A, B):
    views = [_mod.render(m, az, el, cols, size=420, scale=3.3) for az, el in ((0, 12), (-90, 12), (90, 12), (180, 12), (30, 30))]
    rows.append(views)
bottom = _mod.render(A[0], 0, -88, A[1], size=420, scale=3.3)     # 真下から見た底
W = 420 * 5
sheet = Image.new("RGB", (W + 420, 840), (245, 243, 238))
for r, views in enumerate(rows):
    for i, v in enumerate(views):
        sheet.paste(v, (i * 420, r * 420))
sheet.paste(bottom, (W, 210))
sheet.save(os.path.join(mf.OUT, "preview_bailey_final.png"))
