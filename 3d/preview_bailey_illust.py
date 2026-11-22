"""イラスト版の板の仕上がり目安を、元のイラストと並べて描く（上: イラスト、下: 印刷の見え方）。"""
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_bailey_album_relief as mr  # noqa: E402

PX = 0.1
H_PX = 640
tops, bottoms = [], []
for f in ("front", "left", "right", "back"):
    b = mr.brightness(f)
    look = mr.appearance(f, b, PX)
    w = round(look.shape[1] * H_PX / look.shape[0])
    illust = np.zeros(look.shape, np.uint8)
    fr = round(mr.FRAME / PX)
    src = (np.asarray(Image.fromarray((b * 255).astype(np.uint8)).resize((look.shape[1] - 2 * fr, look.shape[0] - 2 * fr))))
    illust[fr:-fr, fr:-fr] = src[..., None]
    illust[:fr] = illust[-fr:] = illust[:, :fr] = illust[:, -fr:] = 26
    tops.append(Image.fromarray(illust).resize((w, H_PX), Image.LANCZOS))
    bottoms.append(Image.fromarray(look).resize((w, H_PX), Image.LANCZOS))
W = sum(t.width for t in tops) + 10 * 5
sheet = Image.new("RGB", (W, H_PX * 2 + 30), (120, 120, 120))
x = 10
for t, bt in zip(tops, bottoms):
    sheet.paste(t, (x, 10)); sheet.paste(bt, (x, H_PX + 20)); x += t.width + 10
sheet.save(os.path.join(mr.OUT, "preview_bailey_illust_compare.png"))
# 板の見え方だけ（実寸比で並べる）
sheet2 = Image.new("RGB", (W, H_PX + 20), (40, 40, 42))
x = 10
for bt in bottoms:
    sheet2.paste(bt, (x, 10)); x += bt.width + 10
sheet2.save(os.path.join(mr.OUT, "preview_bailey_illust_plates.png"))
