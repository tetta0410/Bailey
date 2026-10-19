"""生成したモデルの上面プレビュー画像を描く。"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import trimesh

sys.path.insert(0, os.path.dirname(__file__))
import make_bailey as mb

def fill(ax, g, color, ec="none", under="#222"):
    for p in getattr(g, "geoms", [g]):
        if p.is_empty:
            continue
        ax.fill(*p.exterior.xy, color=color, ec=ec, lw=0.6)
        for i in p.interiors:
            ax.fill(*i.xy, color=under)

fig, axs = plt.subplots(1, 4, figsize=(16, 5.5), facecolor="#e8e2d6")
f = mb.bailey_face()
ax = axs[0]
from shapely.ops import unary_union
fill(ax, unary_union([f["head"], f["bandana_full"], f["loop"]]).difference(f["hole"]), "#222", under="#e8e2d6")
fill(ax, f["white"], "#f4f4f4"); fill(ax, f["nose"], "#000", "#555")
fill(ax, f["tongue"], "#f08aa0"); fill(ax, f["bandana"].difference(f["tongue"]), "#d62839")
shape, hole, name, phone = mb.tag()
fill(axs[1], shape.difference(hole), "#222", under="#e8e2d6"); fill(axs[1], name, "#f4f4f4")
fill(axs[2], mb.mirror(shape.difference(hole)), "#222", under="#e8e2d6"); fill(axs[2], mb.mirror(phone), "#777")
# リトファン: 厚みから透過光をシミュレーション
m = trimesh.load(os.path.join(mb.OUT, "bailey_lithophane.stl"))
v = m.vertices[m.vertices[:, 2] > 0]
xi = np.round(v[:, 0] / 0.25).astype(int); yi = np.round(v[:, 1] / 0.25).astype(int)
img = np.zeros((yi.max() + 1, xi.max() + 1)); img[yi, xi] = v[:, 2]
light = np.exp(-1.1 * img)
axs[3].imshow(light, cmap="gray", origin="lower")
for ax in axs:
    ax.set_aspect("equal"); ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(mb.OUT, "preview.png"), dpi=110)
