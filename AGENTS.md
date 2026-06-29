# Multimodal-Recognition

## 專案概述
多模態情緒辨識系統，即時分析 Webcam 臉部表情 + 麥克風語調，輸出 7 類情緒（neutral, happy, sad, anger, fear, surprise, disgust）。支援 PC (Windows) 與 Raspberry Pi 5 部署。

## 已完成
- 完整推論管線：src/video/、src/audio/、src/fusion/、src/output/
- config.yaml 驅動設定 + src/config.py 預設值深層合併
- 人臉偵測雙後端：MediaPipe → Haar cascade 自動降級
- 視訊 heuristic fallback（smile detection + contrast）
- 音訊 heuristic fallback（energy-based）
- ONNX Runtime 統一 helper：create_ort_session() 在 shared_types.py
  - ARM (aarch64/armv7l) → XNNPACKExecutionProvider
  - 其他 → CPUExecutionProvider
- 相依清理：移除 faster-whisper、RPi 需求移除 mediapipe
- WSL 腳本：setup_wsl.sh、run_vision.sh、run_smoke_test.sh

## 專案結構
```
Multimodal-Recognition/
├── app.py                         # 入口點
├── src/
│   ├── main.py                    # 主循環、CLI 參數
│   ├── config.py                  # 設定載入 + 預設值
│   ├── shared_types.py            # 共用型別、create_ort_session()
│   ├── video/                     # 視覺管線
│   │   ├── capture.py             # OpenCV 相機初始化
│   │   ├── face_detection.py      # MediaPipe / Haar 人臉偵測
│   │   ├── model.py               # Mini-Xception ONNX 載入 + 推論
│   │   └── inference.py           # Frame skip 封裝
│   ├── audio/                     # 音訊管線
│   │   ├── features.py            # 輕量 MFCC（無 librosa）
│   │   ├── model.py               # Tiny 1D-CNN ONNX 載入 + 推論
│   │   └── inference.py           # sounddevice 擷取 + 推論執行緒
│   ├── fusion/
│   │   └── late_fusion.py         # 信心度加權融合
│   └── output/
│       ├── display.py             # OpenCV 疊加顯示
│       └── logger.py              # CSV 日誌
├── config/config.yaml             # 可調參數
├── models/                        # ONNX 模型放這裡（.gitkeep）
├── scripts/                       # 訓練 + 環境初始化腳本
│   ├── train_video_mini_xception.py  # TensorFlow 訓練視覺模型
│   ├── setup_windows.ps1
│   ├── setup_wsl.sh
│   ├── run_vision.sh / .ps1
│   └── run_smoke_test.sh / .ps1
├── AGENTS.md                      # 本檔案
├── ARCHITECTURE.md                # 完整設計文件
├── TRAINING.md                    # 訓練說明
├── requirements.txt               # PC 依賴
├── requirements_rpi.txt           # RPi 5 依賴（無 mediapipe）
└── requirements_train.txt         # 訓練用依賴（tensorflow + tf2onnx）
```

## 目前限制
- 尚無真實 ONNX 模型（mini_xception_int8.onnx、tiny_cnn_audio_int8.onnx）
- 無訓練資料集（FER2013、RAVDESS 等）
- 無音訊模型訓練腳本（需寫 train_audio_tiny_cnn.py）
- 所有 emotion 推論目前走 heuristic fallback（僅供開發測試，不具備真實情緒辨識能力）

## 待辦事項
- [ ] 下載 FER2013 資料集到 data/datasets/fer/（需寫 download_fer2013.py）
- [ ] 訓練 mini_xception_int8.onnx：python scripts/train_video_mini_xception.py --data-dir data/datasets/fer
- [ ] 撰寫 scripts/train_audio_tiny_cnn.py（Tiny 1D-CNN + int8 量化）
- [ ] 下載音訊資料集（RAVDESS、CREMA-D、TESS）
- [ ] 訓練 tiny_cnn_audio_int8.onnx
- [ ] 驗證：python app.py --vision-only --no-display --max-frames 10

## WSL 開發環境
```bash
# 安裝 Node.js 20（opencode 需要）
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
node --version   # v20.x

# 安裝 opencode
npm install -g opencode-ai

# 啟動
source .venv/bin/activate
python app.py --vision-only --no-display --max-frames 10
```
