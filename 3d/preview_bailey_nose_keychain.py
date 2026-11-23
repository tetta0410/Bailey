"""鼻のキーホルダーの見え方（黒マットのイメージ）と、元の写真を並べる。"""
import importlib.util
import os
import sys

import numpy as np
import trimesh
from PIL import Image, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("ittetsu_preview", os.path.join(HERE, "ittetsu", "preview.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

m = trimesh.load(os.path.join(HERE, "out", "bailey_nose_keychain.stl"))
cols = np.tile(np.array([0.30, 0.30, 0.32]), (len(m.faces), 1))
S = 520
views = [_mod.render(m, 0, 88, cols, size=S, scale=14), _mod.render(m, 0, 35, cols, size=S, scale=13),
         _mod.render(m, 35, 25, cols, size=S, scale=13)]
photo = ImageOps.exif_transpose(Image.open(os.path.join(HERE, "photos", "bailey_nose_close.jpg"))).convert("RGB")
photo = photo.crop((360, 780, 890, 1220)).resize((S, round(S * 440 / 530)))
sheet = Image.new("RGB", (S * 4, S), (242, 240, 236))
sheet.paste(photo, (0, (S - photo.height) // 2))
for i, v in enumerate(views):
    sheet.paste(v, (S * (i + 1), 0))
sheet.save(os.path.join(HERE, "out", "preview_bailey_nose_keychain.png"))
