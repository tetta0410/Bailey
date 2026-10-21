# いってつのペン立て（Bambu Lab A1 mini 用）

手描きの絵と「いってつ」の文字を写真から線データとして取り出し、
直径80mm・高さ105mm のペン立ての表面に 1.0mm の浮き彫りにしたもの。

| ファイル | 用途 |
|---|---|
| `out/ittetsu_penholder.stl` | **1色印刷用（おすすめ）**。本体と浮き彫りが一体 |
| `out/ittetsu_penholder_2color.3mf` | 2色用（`body` と `relief` の2パーツ、AMS lite） |
| `out/ittetsu_penholder_view.stl` | 3Dビューア用の軽量版 |
| `out/preview.png` | 完成イメージ |

## Bambu Studio での印刷
1. プリンター: Bambu Lab A1 mini / ノズル 0.4mm、フィラメント: PLA
2. `ittetsu_penholder.stl` を読み込む（そのままの向きで底が下）
3. プロファイル: 0.12mm Fine か 0.16mm Optimal。サポート不要
4. スライスして送信

2色版は絵が縦に続くのでほぼ毎層で色替えになり、時間と捨てフィラメントが大きく増えます。

## 寸法を変えたいとき
`make_penholder.py` 冒頭の `R`（半径）, `H`（高さ）, `RELIEF`（浮き彫りの高さ）などを変えて
```
python3 3d/ittetsu/make_penholder.py
python3 3d/ittetsu/preview.py
```
