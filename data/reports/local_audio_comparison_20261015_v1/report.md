# 本機錄音新舊模型比較

這是使用者預期情緒的診斷性對照，不是獨立標註的正式測試集。助理未人工聽音或判斷標籤正確性。

old：既有 EmotionTalk MFCC 模型；new：完整 30 輪 manifest 模型，不是兩輪試跑模型。
raw：未加權未平滑；calibrated：目前類別加權與門檻；smoothed：再加目前時間平滑。

| 模型／模式 | 與預期標籤一致 | Macro-F1 | 覆蓋率 | 整段拒判 |
| --- | ---: | ---: | ---: | ---: |
| old/raw | 25.00% | 0.1000 | 100.00% | 0 |
| old/calibrated | 25.00% | 0.1000 | 100.00% | 0 |
| old/smoothed | 25.00% | 0.1000 | 100.00% | 0 |
| new/raw | 50.00% | 0.3375 | 100.00% | 0 |
| new/calibrated | 50.00% | 0.3506 | 100.00% | 0 |
| new/smoothed | 50.00% | 0.3506 | 100.00% | 0 |

## 逐段結果

以下是有效窗口機率的整段平均分類，不是串流最後一次分類。

| 錄音 | 預期 | old/raw | old/calibrated | old/smoothed | new/raw | new/calibrated | new/smoothed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| neutral_01.wav | neutral | anger | anger | anger | neutral | neutral | neutral |
| neutral_02.wav | neutral | anger | anger | anger | neutral | neutral | neutral |
| neutral_03.wav | neutral | anger | anger | anger | neutral | neutral | neutral |
| happy_01.wav | happy | anger | anger | anger | anger | neutral | neutral |
| happy_02.wav | happy | anger | anger | anger | neutral | neutral | neutral |
| happy_03.wav | happy | anger | anger | anger | anger | anger | anger |
| sad_01.wav | sad | anger | anger | anger | neutral | neutral | neutral |
| sad_02.wav | sad | anger | anger | anger | neutral | neutral | neutral |
| sad_03.wav | sad | anger | anger | anger | neutral | neutral | neutral |
| anger_01.wav | anger | anger | anger | anger | anger | anger | anger |
| anger_02.wav | anger | anger | anger | anger | anger | anger | anger |
| anger_03.wav | anger | anger | anger | anger | anger | anger | anger |

## 方法與限制

- 同一錄音以固定 1.5 秒窗口、0.5 秒步長及尾窗處理；兩模型共享完全相同的特徵。
- 使用目前正式 VAD、peak normalization、高通與 MFCC。訓練模組的快速高通有細小數值差異，本次沒有切換它。
- 每段錄音獨立重設平滑，遇到 VAD 拒判也重設；這是確定性的離線重播，不重現即時執行緒排程或跨錄音歷史。
- 整段無有效預測保留為拒判，仍計入分母；詳見 windows.csv 的逐窗口分數與 recordings.csv 的最後窗口結果。
- confidence-floor fallback 另外標記；smoothed 忠實沿用目前程式，可能把 confidence=0 的 neutral fallback 再平滑成有效分數，不能將它當真實中性證據。
- 個人錄音未複製到報告、不開啟麥克風、不重訓、不改設定或模型；沒有用 EmotionTalk test 音訊。
- 同一位使用者少量錄音不足以估計泛化準確率，模型分數也不等於正確率；不得只依此調參後仍把同一批當獨立測試。
