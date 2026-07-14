# Training Models

這個專案可以自己訓練模型，但需要先準備資料集。建議順序是：

1. 先做影像情緒模型。
2. 確認 `python app.py --vision-only` 能使用 ONNX 模型。
3. 再做音訊情緒模型。

## Video Model

目前提供 `scripts/train_video_mini_xception.py`，用 Keras 訓練一個小型 Mini-Xception-like CNN，最後輸出：

```text
models/mini_xception_int8.onnx
```

資料集資料夾格式：

```text
data/datasets/fer/
  train/
    neutral/
    happy/
    sad/
    anger/
  val/
    neutral/
    happy/
    sad/
    anger/
```

每個情緒資料夾放臉部圖片。圖片可以是彩色或灰階，訓練程式會轉成 `48x48` grayscale。

安裝訓練依賴：

```powershell
pip install -r requirements_train.txt
```

開始訓練：

```powershell
python scripts/train_video_mini_xception.py --data-dir data/datasets/fer --epochs 30
```

訓練完成後測試：

```powershell
python app.py --vision-only
```

## Dataset Notes

可使用的公開資料集方向：

- FER-2013：常見臉部表情資料集，4 類情緒。
- CK+：表情資料較乾淨，但資料量較小。
- RAF-DB：品質較好，但授權和申請流程要另外確認。

如果是專題展示，建議先用 FER-2013 或自己拍攝少量資料做 proof of concept。正式報告要寫清楚資料來源和授權。

## Audio Model

音訊模型可以自己做，但建議放到第二階段，因為音訊情緒需要資料集、特徵處理和環境音處理。可參考資料集：

- RAVDESS
- CREMA-D
- TESS

目前 `src/audio/` 已保留 Tiny 1D-CNN ONNX 入口，等資料集和模型確定後再補訓練腳本。
