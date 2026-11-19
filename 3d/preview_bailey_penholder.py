"""ベイリーのペン立ての完成イメージ（正面／斜め／上から）。"""
import os
import sys

import numpy as np
import trimesh
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_bailey_penholder as mb  # noqa: E402

# 3d/preview.py と名前がぶつかるので、ittetsu/preview.py をファイル指定で読み込む
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("ittetsu_preview", os.path.join(HERE, "ittetsu", "preview.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
render = _mod.render

WHITE, BLACK, PINK = np.array([0.97, 0.96, 0.93]), np.array([0.11, 0.11, 0.13]), np.array([0.94, 0.53, 0.60])

photo = sys.argv[1] if len(sys.argv) > 1 else "tongue"
mb.main(photo)
COLORS = {"body_white": WHITE, "body_green": np.array([0.13, 0.33, 0.22]), "fur_white": WHITE,
          "face_black": BLACK, "tongue_pink": PINK}
meshes = [m for _, m, _ in mb.LAST_PARTS]
cols = np.vstack([np.tile(COLORS[n], (len(m.faces), 1)) for n, m, _ in mb.LAST_PARTS])
m = trimesh.util.concatenate(meshes)
views = [render(m, 0, 10, cols, scale=5.2), render(m, 32, 28, cols, scale=5.0), render(m, 12, 68, cols, scale=5.2)]
sheet = Image.new("RGB", (sum(v.width for v in views), views[0].height))
x = 0
for v in views:
    sheet.paste(v, (x, 0)); x += v.width
sheet.save(os.path.join(mb.OUT, "preview_bailey_penholder.png" if photo == "tongue"
                         else f"preview_bailey_penholder_{photo}.png"))
