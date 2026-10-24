# ベイリーの3Dプリント（Bambu Lab A1 用）

`out/` のファイルを Bambu Studio に読み込めばそのまま印刷できます。

| ファイル | 内容 | サイズ | おすすめ設定 |
|---|---|---|---|
| `bailey_keychain.3mf` | 顔のキーホルダー（黒/白/ピンク/赤） | 32×47×3.8 mm | 0.12〜0.16 mm, AMS lite |
| `bailey_tag.3mf` | 骨型の名札（表 BAILEY / 裏 電話番号を彫刻） | 58×26×3.8 mm | 0.12 mm, AMS lite |
| `bailey_qr_charm.3mf` | **QR付き迷子札チャーム**。QR → https://tetta0410.github.io/Bailey/ 、ボーダーコリーの頭の形, 耳は表裏とも黒, 裏に Dad / Mom の電話番号（黒） | 41×61×5 mm | 0.2 mm, AMS lite（白＋黒）, QR面を上にしたまま |
| `bailey_lithophane.stl` | 写真のリトファン（光に透かすと写真が浮かぶ） | 106×139×4.2 mm | 白PLA, 0.12 mm, インフィル100%, 縦置き |

## マルチカラーの読み込み方
1. `.3mf` を Bambu Studio にドラッグ。「1つのオブジェクトとして読み込みますか？」→ **はい**。
2. 右の「オブジェクト」一覧の各パーツ（`black_base`, `white`, `pink_tongue` …）にフィラメントを割り当てる。
3. AMS lite がない場合は、色替えしたい高さ（3.0 mm）に「一時停止」を入れて手動でフィラメント交換。

## QRチャームのコツ
- `charm_white` に白、`charm_black` に黒を割り当てる。黒は上面と裏面の 0.6mm だけなので色替えは最初と最後の数層だけ。
- QR面を上にしたまま印刷する（裏の電話番号はベッド側の最初の3層）。アイロン（ironing）は不要。
- 同じ時間で何個か並べて印刷しても、色替え回数はほぼ同じ。
- 生成スクリプト `make_bailey_qr.py` の `URL` を変えれば別のページにも使える。`preview_bailey_qr.py` で読み取りテストができる。

## リトファンのコツ
- 必ず**白PLA**・**インフィル100%**。ブリムを付け、縦（写真が立つ向き）に置くと綺麗に出ます。
- 窓辺やLEDライトの前に置くと写真が見えます。

## 作り直し
```
pip install trimesh shapely manifold3d mapbox_earcut matplotlib pillow
python3 3d/make_bailey.py   # out/ に再生成
python3 3d/preview.py       # out/preview.png を描く
```
サイズや厚みは `make_bailey.py` 内の数値で調整できます。
