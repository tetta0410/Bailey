"""四角いペン立ての完成イメージ（組み立て状態・上から・ビルドプレート上）。"""
import os
import sys

import numpy as np
import trimesh
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preview import render  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
BLUE, BLACK = np.array([0.50, 0.72, 0.86]), np.array([0.12, 0.12, 0.14])


def colored(parts):
    meshes = [trimesh.load(os.path.join(OUT, f)) for f, _ in parts]
    cols = np.vstack([np.tile(c, (len(m.faces), 1)) for m, (_, c) in zip(meshes, parts)])
    return trimesh.util.concatenate(meshes), cols


m, cols = colored([("view_body.stl", BLUE), ("view_plate.stl", BLUE), ("view_art.stl", BLACK)])
views = [render(m, 0, 12, cols, scale=5.2), render(m, 35, 30, cols, scale=5.0), render(m, 15, 70, cols, scale=5.2)]
sheet = Image.new("RGB", (sum(v.width for v in views), views[0].height))
x = 0
for v in views:
    sheet.paste(v, (x, 0)); x += v.width
sheet.save(os.path.join(OUT, "preview_pencase.png"))
