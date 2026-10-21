# いってつのペン立て（Bambu Lab A1 mini + AMS lite 用）

## 四角いペン立て（仕切り付き）★最新版

本体 90×75×100mm。手描きの絵と「いってつ」は**差し替えプレート**に黒ではめ込み、
本体正面のアリ溝に上から差し込む。プレートは平置きで印刷するので、黒への色替えは表面の数層だけで済む。

```
  ┌──────────────┬────────┐ 奥
  │              │ 深い   │ はさみ・定規・マーカー（深さ97mm）
  │  ペン・鉛筆  ├────────┤
  │  (深さ97mm)  │ 浅い   │ 短い鉛筆・消しゴム（底上げで深さ50mm）
  └──────────────┴────────┘ 正面（絵のプレート）
```
- 新品の鉛筆（約175mm）は約75mm 上に出るのでつまみやすい
- 仕切りは縁より 10mm 低い

| ファイル | 内容 |
|---|---|
| `out/ittetsu_pencase.3mf` | 本体＋プレートを 1 枚のビルドプレートに配置（これを開けばOK） |
| `out/ittetsu_pencase_body.stl` | 本体だけ |
| `out/ittetsu_plate.3mf` | プレートだけ（`plate_base` / `plate_art_black`） |
| `out/preview_pencase.png` | 完成イメージ |

### 印刷手順
1. A1 mini / 0.4mm ノズル。AMS lite に好きな色と黒の PLA
2. `ittetsu_pencase.3mf` を開く →「1つのオブジェクトとして読み込みますか？」→ はい
3. `pencase_body` と `plate_base` に好きな色、`plate_art_black` に黒
4. 0.16mm Optimal、サポートなし → スライス → 送信
5. プレートを絵を外向きにして溝へ上から差し込む（固ければ溝を軽くヤスリがけ）

寸法は `make_pencase.py` 冒頭の数値で変えられる。
```
python3 3d/ittetsu/make_pencase.py
python3 3d/ittetsu/preview_pencase.py
```

---

## 旧版: 丸いペン立て

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
