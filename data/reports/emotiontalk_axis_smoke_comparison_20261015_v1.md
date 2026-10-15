# EmotionTalk 卷積方向小規模對照（2026-10-15）

## 目的與範圍

既有模型直接把 `[batch, 13 MFCC, 93 time]` 交給 channels-last Conv1D，因此卷積沿 13 個 MFCC 係數移動，93 個時間位置被視為通道。這不代表模型完全無法利用時間資訊，但不是沿時間軸的局部卷積。

候選版在模型內加入 `Permute((2, 1))`，讓 Conv1D 收到 `[batch, 93 time, 13 MFCC]`。外部輸入仍為 `[batch, 13, 93]`，四類順序仍為 neutral / happy / sad / anger。只新增可選架構，原版保留且仍為預設；既有模型不能靠改輸入轉置就變成候選架構，需重新訓練權重。

本次只驗證建模、訓練及匯出流程，不判定哪個架構正式較好。沒有修改正式推論、config 或原始資料，也沒有使用 test 音訊。

## 相同條件

- 使用既有 speaker-disjoint 清單，train／val 每類各 8 筆，各 32 筆錄音。
- 2 epochs、seed=42、batch size=16、Adam learning rate=0.001。
- 同一 `emotiontalk-mfcc-v1.1` 前處理；隨機裁切為唯一 augmentation，未新增音量或雜訊增強。
- 每輪 train 有效窗口 31 筆、VAD 拒絕 1 筆；val 有效窗口 320 個，按原始 32 筆錄音平均機率評分，coverage 均為 100%。
- 兩次 `run_config.json` 只有 `architecture` 與 `architecture_variant` 不同；所選錄音、前處理、val 音訊紀錄及每輪裁切 CSV 逐位元相同。
- 同一 seed 不代表不同形狀的網路會有相同權重或相同最佳化歷程；本次不是參數量配平的消融實驗。

## 小樣本結果

| 項目 | 原版 legacy | 時間卷積 temporal |
| --- | ---: | ---: |
| 總參數量（含 BN 非訓練參數） | 25,828 | 13,028 |
| 第一層 Conv1D kernel | `(5, 93, 32)` | `(5, 13, 32)` |
| 最佳 epoch（依 val macro-F1） | 2 | 2 |
| 32 筆 val accuracy | 34.375% | 28.125% |
| 32 筆 val macro-F1 | 0.245798 | 0.158120 |
| FP32 ONNX 大小 | 106,900 bytes | 55,511 bytes |
| 8 個真實特徵窗口 TF／ONNX 最大絕對差 | 3.5763e-7 | 2.3842e-7 |

**候選版這次沒有優於原版。** 每類只有 8 筆、訓練 2 輪、單一種子，不能據此認定候選版一定無效，也不能宣稱已改善正式準確率。轉置同時讓第一層通道數由 93 變為 13，減少 12,800 個參數，因此不能將分數差異全歸因於卷積方向。這兩個模型都是 `smoke_only.onnx`，不應直接切換為正式模型。

## 輸出與重現

- 原版：[result.json](../../models/emotiontalk_axis_legacy_smoke_20261015_v1/result.json)、[model_structure.json](../../models/emotiontalk_axis_legacy_smoke_20261015_v1/model_structure.json)。
- 候選版：[result.json](../../models/emotiontalk_axis_temporal_smoke_20261015_v1/result.json)、[model_structure.json](../../models/emotiontalk_axis_temporal_smoke_20261015_v1/model_structure.json)。
- 原版 ONNX SHA-256：`26c340fb2791c5620cc30d23d9998a5e644628f33e8c769ebdb373ceeea6b1c5`。
- 候選版 ONNX SHA-256：`7b9edc4211c463954c2b0aaed0f14535764432a260d0f41f69ba970a12cdbce9`。

以下省略 `--output-dir`，讓每次執行建立新時間戳目錄；不要刪除或重用既有輸出目錄：

```powershell
python -B scripts/train_audio_emotiontalk_manifest.py --architecture legacy --epochs 2 --limit-per-class 8 --seed 42 --batch-size 16
python -B scripts/train_audio_emotiontalk_manifest.py --architecture temporal --epochs 2 --limit-per-class 8 --seed 42 --batch-size 16
python -B -m unittest discover -s tests -q
```

## 檢查與限制

- 全專案 66 項單元測試通過，包括新增 5 項架構測試：原版預設層序／seeded weights、轉置數值、卷積形狀／參數量、四類機率及錯誤架構拒絕。
- 兩版匯出皆通過 8 個真實 val 特徵窗口的 TF／ONNX 比對（rtol=1e-4、atol=1e-5）；另測 ORT batch=1／3，以及現有 `AudioEmotionModel` 對 50／93／120 格特徵補齊／截短，輸出皆為有限且正規化的四類機率。合成特徵只驗證介面，不代表準確率。
- 檢查 ONNX 第一層 Conv 權重為原版 `[32,93,1,5]`、候選版 `[32,13,1,5]`，確認轉置後的通道方向確實反映在匯出圖。
- 兩個輸出目錄各 12 個檔案，重跑被拒絕後雜湊均不變。修改前既有 models／data/reports／config／人工聽音 WAV 共 130 個檔案 SHA-256 均不變。
- 訓練入口會讀取 train／val／test 清單做切分檢查，但本次只解碼選中的 train／val 錄音，沒有 test 音訊推論或評分。
- 未驗證真人耳機、樹莓派、長時間執行或完整前端；外部 ONNX 介面相容不等於已完成部署驗證。未安裝套件、刪除資料或執行 git add／commit／push。
- 下一步需另行確認：先加入可保存與恢復訓練進度的 checkpoint，再考慮固定資料與預算的較充分架構對照。這些尚未實作或啟動；不能因小規模流程成功就直接部署。
