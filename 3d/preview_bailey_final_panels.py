"""完成版の4枚の板を、実寸の比率で並べて描く（白い板に黒い絵）。"""
import os
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_bailey_album_final as mf  # noqa: E402

PX = 12
tiles = []
for face in ("front", "left", "right", "back"):
    g = mf.art(face)
    w, h = mf.panel_size(face)
    im = Image.new("RGB", (int(w * PX), int(h * PX)), (246, 245, 240))
    tot = Image.new("L", im.size, 0)
    tr = lambda c: [((x + w / 2) * PX, (h - y) * PX) for x, y in c]
    for q in getattr(g, "geoms", [g]):
        m = Image.new("L", im.size, 0)
        d = ImageDraw.Draw(m)
        d.polygon(tr(q.exterior.coords), fill=255)
        for hole in q.interiors:
            d.polygon(tr(hole.coords), fill=0)
        tot.paste(255, mask=m)
    im.paste((22, 22, 24), mask=tot)
    tiles.append(im)
sheet = Image.new("RGB", (sum(t.width for t in tiles) + 20 * 5, max(t.height for t in tiles) + 40), (20, 20, 22))
x = 20
for t in tiles:
    sheet.paste(t, (x, 20)); x += t.width + 20
sheet.save(os.path.join(mf.OUT, "preview_bailey_final_panels.png"))
