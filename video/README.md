# 五楊高架救護派遣動畫

約 4 分 46 秒的 3D 資訊圖動畫，給一般民眾看：為什麼五楊高架上的事故，依「事故在哪個縣市」派車，往往不是最快到場的分隊。旁白稿在 [`narration.md`](narration.md)。

**觀看**：[YouTube「汐五、五楊高架救護反應困境」](https://youtu.be/lhAl9ud9Hpc)，也嵌在五楊頁 `output/wuyang.html` 的開頭（首頁連結列有「▶ 動畫」連到這裡）。

成品影片（1080p，分享版約 82 MB、高畫質版約 300 MB）太大，沒有放進 repo（`.gitignore` 已排除 `video/wuyang_ems*.mp4`），依下面的步驟可以完整重現。

## 內容

| 段落 | 畫面 | 數字來源 |
|---|---|---|
| 1 開場 | 從北台灣上空降到五股前，看高架與平面上下兩層 | — |
| 2 認識高架 | 汐五約 18 公里、五楊約 40 公里，依縣市上色 | `output/points_100m.csv` |
| 3 北上入口 | 校前路、中豐、機場系統；機場系統（50.8K）到五股約 20 公里沒有入口 | `data/processed/chains.json`（03） |
| 4 案例 | 北上 30.5K（新北市）：興國 19.7、青埔 21.9、隆恩 31.6 分同時出發；比隆恩先到的有 29 個分隊，27 個在桃園 | `model_paths.py`、07_wuyang |
| 5 縣市界對不上 | 高架依「最快到場分隊」上色，北上、南下各一次 | `data/processed/wuyang_points.csv`（07） |
| 6 實務安排 | 青埔第一優先；大園、中壢、復旦比隆恩快 | 07_wuyang |
| 7 結語 | 資料來源與限制 | — |

救護車路線和計時器都來自本專案的路網模型（與 `07_wuyang.py` 同一個 Dijkstra），所以畫面上的抵達時間與網站上的數字一致。

## 重現步驟

前置：已跑過 `01_fetch_osm.py`、`03_route.py`（產生 `data/raw/osm_roads.json`、`data/processed/`），以及 `07_wuyang.py` 的計算（`data/processed/wuyang_points.csv`）。需要 Node（Playwright + Chromium）與 ffmpeg。

```bash
python3 video/model_paths.py     # 各分隊到北上 30.5K 的模型路線（含每個節點的經過秒數）
python3 video/prep_imagery.py    # 下載並處理衛星／航照底圖（圖磚快取在 data/raw/tiles/）
python3 video/tts.py             # 旁白語音，每段一個請求；已有的 scene{N}.wav 會略過
python3 video/prep_full.py       # 場景資料與時間軸（依語音長度排，字幕對齊語音中的停頓）
cd video && python3 -m http.server 8766 &    # 給 Chromium 讀頁面
node video/render_film.js <影格資料夾> 3        # 逐格算圖，中斷後重跑會從缺的影格接續
python3 video/assemble.py <影格資料夾> video/wuyang_ems.mp4
```

改某一段旁白：改 `narration.md` 那段 → 刪掉對應的 `audio/scene{N}.wav` → 重跑 `tts.py`、`prep_full.py`，再從那一段的開始時間重算影格。

`scene.html`、`prep_sample.py`、`render.js` 與 `samples/` 是選路線畫法時做的 8–14 秒範例（示意路線與真實道路、修正鏡頭後的版本）。

## 語音

Google Gemini TTS（`gemini-3.8-flash-tts`，聲音 `Charon`），經 Vertex AI 呼叫；風格指令放在 `speechMetadata.style`（放進文字會被唸出來）。成品約 6,400 個語音 token；加上試錄和重錄的第 4 段，共約 8,000 個，費用約 US$0.16。`audio/usage.json` 記錄每段目前版本的用量。

## 圖資與授權

- 路網：© OpenStreetMap contributors（ODbL）
- 遠景衛星影像：Sentinel-2 cloudless 2023 by EOX IT Services，**CC BY-NC-SA 4.0（限非商業用途）**，含修改的 Copernicus Sentinel 資料
- 近景航照：內政部國土測繪中心 114 年正射影像（WMTS `PHOTO2025`）；授權條款尚待確認，影片中已標示出處
- 消防分隊：內政部消防署「救援與應變單位點位」
- 縣市界：taiwan-atlas

影片中的時間皆為模型估算：依速限、不塞車、不含出勤準備時間（約 1–1.5 分鐘）。「青埔第一優先、新北以隆恩優先」的派遣做法為口頭得知，沒有正式文件。
