# EmotionTalk 驗證集誤判診斷

只分析既有驗證紀錄，沒有重訓、重新推論、修改標籤或讀取 test 音訊。

- 最佳輪次：30；錄音 2194 筆。
- Accuracy：47.58%；Macro-F1：0.4089；拒判 16 筆，仍計入分母。
- 全猜中性 accuracy 基準：52.32%，不代表能辨識其他類別。

## 優先看什麼

- 已有有效窗口但判錯 1134 筆；完全拒判 16 筆。錯誤不能全部歸因於 VAD 沒收到聲音，也不能因通過 VAD 就認定片段有清楚的人聲情緒。
- 正確與誤判的 RMS 中位數見下表；它是描述統計，不足以證明調音量就能修好模型。
- 最多的混淆：neutral 判成 anger（217 筆）、neutral 判成 sad（202 筆）、anger 判成 happy（173 筆）；這些是人工聽音的優先方向。
- 聽音清單包含高分誤判；分數是原錄音的有效窗口平均機率，沒有做機率校準，不能視為真實正確率。

## 四類結果

| 原始標籤 | Precision | Recall | F1 | 筆數 |
|---|---:|---:|---:|---:|
| neutral | 78.88% | 50.44% | 0.6153 | 1148 |
| happy | 23.76% | 38.33% | 0.2933 | 287 |
| sad | 18.93% | 42.11% | 0.2612 | 152 |
| anger | 45.26% | 47.94% | 0.4656 | 607 |

## 混淆矩陣

列是原始標籤，欄是模型預測；每筆原錄音只計一次。

| 原始／預測 | neutral | happy | sad | anger | 拒判 |
|---|---:|---:|---:|---:|---:|
| neutral | 579 | 136 | 202 | 217 | 14 |
| happy | 53 | 110 | 24 | 100 | 0 |
| sad | 8 | 44 | 64 | 35 | 1 |
| anger | 94 | 173 | 48 | 291 | 1 |

## 最常見混淆

- neutral 被判為 anger：217 筆，占原始 neutral 的 18.90%。
- neutral 被判為 sad：202 筆，占原始 neutral 的 17.60%。
- anger 被判為 happy：173 筆，占原始 anger 的 28.50%。
- neutral 被判為 happy：136 筆，占原始 neutral 的 11.85%。
- happy 被判為 anger：100 筆，占原始 happy 的 34.84%。

## 長度、音量與有效窗口

以下為中位數；incorrect 不含拒判，abstain 另外列出。音量是原始整段左右平均後的 RMS，不是重取樣、補零與切片後的窗口 RMS。低 RMS 不能直接證明沒有人聲。

### 整體正確／誤判／拒判

| 分組 | 筆數 | 秒數中位 | RMS 中位 | 有效窗口中位 | 無效窗口比例中位 |
|---|---:|---:|---:|---:|---:|
| correct | 1044 | 3.378 | 0.06177 | 5.0 | 0.00% |
| incorrect | 1134 | 4.227 | 0.06336 | 7.0 | 0.00% |
| abstain | 16 | 0.878 | 0.00776 | 0.0 | 100.00% |
### 各真實類別

| 分組 | 筆數 | 秒數中位 | RMS 中位 | 有效窗口中位 | 無效窗口比例中位 |
|---|---:|---:|---:|---:|---:|
| neutral/correct | 579 | 1.979 | 0.05603 | 2.0 | 0.00% |
| happy/correct | 110 | 3.830 | 0.06958 | 6.0 | 0.00% |
| sad/correct | 64 | 6.956 | 0.04771 | 10.5 | 11.44% |
| anger/correct | 291 | 5.034 | 0.06891 | 9.0 | 0.00% |
| neutral/incorrect | 555 | 3.945 | 0.06047 | 6.0 | 0.00% |
| happy/incorrect | 177 | 3.943 | 0.06617 | 6.0 | 0.00% |
| sad/incorrect | 87 | 6.407 | 0.06284 | 11.0 | 0.00% |
| anger/incorrect | 315 | 4.564 | 0.06697 | 8.0 | 0.00% |
| neutral/abstain | 14 | 0.853 | 0.00901 | 0.0 | 100.00% |
| sad/abstain | 1 | 2.960 | 0.00745 | 0.0 | 100.00% |
| anger/abstain | 1 | 1.595 | 0.00700 | 0.0 | 100.00% |

### 分桶與說話者

分桶僅供描述，不是挑選新門檻；各組情緒比例不同，不能直接推論因果。

| 分組 | 筆數 | Accuracy（含拒判） | 拒判 | neutral/happy/sad/anger 筆數 |
|---|---:|---:|---:|---|
| duration: <1.5s | 401 | 65.59% | 12 | 302/35/5/59 |
| duration: 1.5-3s | 495 | 44.85% | 4 | 282/72/13/128 |
| duration: 3-6s | 622 | 41.32% | 0 | 284/101/44/193 |
| duration: >=6s | 676 | 44.67% | 0 | 280/79/90/227 |
| rms: <0.01 | 10 | 0.00% | 10 | 8/0/1/1 |
| rms: >=0.01 | 2184 | 47.80% | 6 | 1140/287/151/606 |
| speaker: 01 | 406 | 44.83% | 1 | 196/72/33/105 |
| speaker: 02 | 612 | 30.72% | 1 | 260/91/71/190 |
| speaker: 13 | 773 | 67.01% | 10 | 470/73/32/198 |
| speaker: 15 | 403 | 38.71% | 4 | 222/51/16/114 |

## 人工聽音清單

案例為刻意挑選的診斷樣本，不是隨機抽樣，不能用其正確率代表整體。每類選一筆中位信心度正確案例與最高信心度誤判，另補 happy/sad 誤報、短錄音誤判及拒判，重複來源合併。

音檔是原始完整 WAV 的副本，沒有增益、轉檔或裁切；預測則是有效窗口平均結果，不是每個瞬間的情緒。副本留在已忽略的 data/datasets，不隨報告提交；仍受原資料集授權限制。

請先聽音再看答案，填寫 listening_cases.csv 的 heard_label、label_agreement、audio_issue、notes。可以填不確定，不要因模型判錯就直接改標籤。助理尚未聽音，以下不代表人工標註結論。

| 音檔 | 原標籤 | 預測 | 分數 | 秒數 | 挑選理由 |
|---|---|---|---:|---:|---|
| [01_neutral_to_neutral.wav](../../datasets/emotiontalk_review_20261015_v1/01_neutral_to_neutral.wav) | neutral | neutral | 51.00% | 1.61 | neutral: median-confidence correct case |
| [02_neutral_to_sad.wav](../../datasets/emotiontalk_review_20261015_v1/02_neutral_to_sad.wav) | neutral | sad | 84.21% | 0.87 | neutral: highest-confidence error; false sad: highest-confidence case |
| [03_happy_to_happy.wav](../../datasets/emotiontalk_review_20261015_v1/03_happy_to_happy.wav) | happy | happy | 45.53% | 3.28 | happy: median-confidence correct case |
| [04_happy_to_neutral.wav](../../datasets/emotiontalk_review_20261015_v1/04_happy_to_neutral.wav) | happy | neutral | 84.46% | 0.67 | happy: highest-confidence error |
| [05_sad_to_sad.wav](../../datasets/emotiontalk_review_20261015_v1/05_sad_to_sad.wav) | sad | sad | 56.13% | 9.42 | sad: median-confidence correct case |
| [06_sad_to_happy.wav](../../datasets/emotiontalk_review_20261015_v1/06_sad_to_happy.wav) | sad | happy | 84.25% | 1.99 | sad: highest-confidence error |
| [07_anger_to_anger.wav](../../datasets/emotiontalk_review_20261015_v1/07_anger_to_anger.wav) | anger | anger | 49.42% | 2.11 | anger: median-confidence correct case |
| [08_anger_to_neutral.wav](../../datasets/emotiontalk_review_20261015_v1/08_anger_to_neutral.wav) | anger | neutral | 91.49% | 1.03 | anger: highest-confidence error; short recording: highest-confidence error |
| [09_anger_to_happy.wav](../../datasets/emotiontalk_review_20261015_v1/09_anger_to_happy.wav) | anger | happy | 91.41% | 1.51 | false happy: highest-confidence case |
| [10_neutral_to_abstain.wav](../../datasets/emotiontalk_review_20261015_v1/10_neutral_to_abstain.wav) | neutral | abstain | -- | 0.45 | abstention example |

## 結論邊界

- 這是同一模型的驗證集診斷，不是獨立 test 成績，也不是新舊模型公平比較。
- 可從混淆與分組找線索，但不能據此斷定是麥克風、標籤、MFCC、模型容量或切片造成。
- 檔案正常可解碼、分數高或 VAD 通過，都不代表情緒正確；目前不建議自動替換正式模型。
- 下一步先由組員聽音核對，保留不確定案例；再決定是否需要進一步窗口診斷或新舊模型耳機比較。
