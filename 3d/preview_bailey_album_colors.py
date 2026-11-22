"""プレート差し込み式アルバムの配色案を並べて描く（手持ちの 赤・白・黒・緑 のみ）。"""
import importlib.util
import os
import sys

import numpy as np
import trimesh
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import make_bailey_album_plates as mp  # noqa: E402

_spec = importlib.util.spec_from_file_location("ittetsu_preview", os.path.join(HERE, "ittetsu", "preview.py"))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

C = {"white": [0.96, 0.95, 0.92], "black": [0.12, 0.12, 0.13], "red": [0.78, 0.13, 0.15], "green": [0.13, 0.36, 0.22]}
# (名前, 箱, {面: (プレートの地, 絵)})
SCHEMES = [
    ("A  Classic", "green", {f: ("white", "black") for f in mp.PW}),
    ("B  Monotone", "black", {f: ("white", "black") for f in mp.PW}),
    ("C  Christmas", "red", {f: ("white", "black") for f in mp.PW}),
    ("D  Two-tone", "green", {"front": ("white", "black"), "back": ("white", "black"),
                           "left": ("red", "black"), "right": ("red", "black")}),
    ("E  Pop art", "black", {"front": ("white", "black"), "left": ("red", "black"),
                             "right": ("green", "black"), "back": ("red", "black")}),
]

box = mp.make_box()
plates = {f: mp.plate(f) for f in mp.PW}
installed = {f: (mp.installed(f, b), mp.installed(f, a)) for f, (b, a) in plates.items()}
tiles = []
for name, boxc, faces in SCHEMES:
    meshes, cols = [box], [np.tile(C[boxc], (len(box.faces), 1))]
    for f, (pb, pa) in installed.items():
        bg, fg = faces[f]
        meshes += [pb, pa]
        cols += [np.tile(C[bg], (len(pb.faces), 1)), np.tile(C[fg], (len(pa.faces), 1))]
    m = trimesh.util.concatenate(meshes)
    cols = np.vstack(cols)
    # 正面と左が見える角度、右と背面が見える角度の2つ
    a = _mod.render(m, -35, 22, cols, size=360, scale=2.7)
    b = _mod.render(m, 145, 22, cols, size=360, scale=2.7)
    tile = Image.new("RGB", (360, 720 + 40), (245, 243, 238))
    tile.paste(a, (0, 40)); tile.paste(b, (0, 400))
    # 見出し（日本語フォントがない環境でも出るよう英字にする）
    from PIL import ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except OSError:
        font = None
    ImageDraw.Draw(tile).text((14, 8), name, fill=(40, 40, 40), font=font)
    tiles.append(tile)
sheet = Image.new("RGB", (360 * len(tiles), tiles[0].height), (245, 243, 238))
for i, t in enumerate(tiles):
    sheet.paste(t, (360 * i, 0))
sheet.save(os.path.join(mp.OUT, "preview_bailey_album_colors.png"))
