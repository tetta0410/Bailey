"""QRチャームの表・裏のプレビューを描き、表のQRが読み取れるか OpenCV で確認する。"""
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_bailey_qr as mq  # noqa: E402

PX = 12  # 1mm あたりのピクセル
BG, WHITE, BLACK, ENG = (232, 226, 214), (246, 245, 240), (30, 30, 32), (200, 198, 190)


def draw(img, g, color, ox, h, flip=False):
    """ポリゴンごとに穴を抜いたマスクを作って塗る（穴の中の別ポリゴンを消さないように）。"""
    tr = lambda xy: [((mq.W - x if flip else x) * PX + ox, (h - y) * PX + 20) for x, y in xy]
    total = Image.new("L", img.size, 0)
    for p in getattr(g, "geoms", [g]):
        if p.is_empty:
            continue
        m = Image.new("L", img.size, 0)
        d = ImageDraw.Draw(m)
        d.polygon(tr(p.exterior.coords), fill=255)
        for i in p.interiors:
            d.polygon(tr(i.coords), fill=0)
        total.paste(255, mask=m)
    img.paste(color, mask=total)


outline, black, phone, _ = mq.charm()
h = outline.bounds[3]
img = Image.new("RGB", (int(mq.W * PX * 2 + 100), int(h * PX + 40)), BG)
draw(img, outline, WHITE, 20, h)
draw(img, black, BLACK, 20, h)
ox = int(mq.W * PX + 80)
draw(img, outline, WHITE, ox, h, flip=True)        # 裏（左右反転して見る）
draw(img, phone, ENG, ox, h, flip=True)
out = os.path.join(mq.OUT, "preview_qr_charm.png")
img.save(out)

data, pts, _ = cv2.QRCodeDetector().detectAndDecode(cv2.imread(out))
print("decoded:", repr(data), "OK" if data == mq.URL else "NG")
