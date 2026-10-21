"""ペン立ての完成イメージを描く（Zバッファによる簡易レンダラー）。"""
import os

import numpy as np
import trimesh
from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def render(mesh, azim, elev, colors, size=700, scale=5.5):
    a, e = np.radians(azim), np.radians(elev)
    # カメラ座標系（正射影）
    fwd = np.array([-np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), -np.sin(e)])  # 視線の向き
    right = np.array([np.cos(a), np.sin(a), 0.0])
    up = np.array([0, 0, 1.0]) - fwd * fwd[2]
    up /= np.linalg.norm(up)
    c = mesh.bounds.mean(axis=0)
    v = mesh.vertices - c
    sx = v @ right * scale + size / 2
    sy = size / 2 - v @ up * scale
    depth = v @ fwd
    n = mesh.face_normals
    light = 0.6 * fwd * -1 + 0.5 * right * -1 + 0.6 * up
    light /= np.linalg.norm(light)
    shade = 0.3 + 0.7 * np.clip(n @ light, 0, 1)
    vis = (n @ fwd) < 0
    zbuf = np.full((size, size), np.inf)
    img = np.ones((size, size, 3)) * np.array([0.95, 0.94, 0.91])
    for fi in np.nonzero(vis)[0]:
        i0, i1, i2 = mesh.faces[fi]
        xs, ys, zs = sx[[i0, i1, i2]], sy[[i0, i1, i2]], depth[[i0, i1, i2]]
        x0, x1 = int(max(xs.min(), 0)), int(min(xs.max() + 1, size))
        y0, y1 = int(max(ys.min(), 0)), int(min(ys.max() + 1, size))
        if x0 >= x1 or y0 >= y1:
            continue
        X, Y = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        d = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(d) < 1e-12:
            continue
        w0 = ((ys[1] - ys[2]) * (X - xs[2]) + (xs[2] - xs[1]) * (Y - ys[2])) / d
        w1 = ((ys[2] - ys[0]) * (X - xs[2]) + (xs[0] - xs[2]) * (Y - ys[2])) / d
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        z = w0 * zs[0] + w1 * zs[1] + w2 * zs[2]
        sub = zbuf[y0:y1, x0:x1]
        upd = inside & (z < sub)
        sub[upd] = z[upd]
        img[y0:y1, x0:x1][upd] = colors[fi] * shade[fi]
    return Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))


if __name__ == "__main__":
    m = trimesh.load(os.path.join(OUT, "ittetsu_penholder_view.stl"))
    r = np.linalg.norm(m.triangles_center[:, :2], axis=1)
    is_relief = (r > 40.15)[:, None]
    white, black, blue = np.array([0.96, 0.94, 0.88]), np.array([0.13, 0.13, 0.15]), np.array([0.55, 0.75, 0.86])
    views = [render(m, 0, 10, np.where(is_relief, white, white)),
             render(m, 0, 10, np.where(is_relief, black, white)),
             render(m, 35, 28, np.where(is_relief, blue, blue))]
    W = sum(v.width for v in views)
    sheet = Image.new("RGB", (W, views[0].height))
    x = 0
    for v in views:
        sheet.paste(v, (x, 0)); x += v.width
    sheet.save(os.path.join(OUT, "preview.png"))
