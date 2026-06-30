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
- 訓練腳本全數完成：download_fer2013.py、download_audio_datasets.py、train_video_mini_xception.py、train_audio_tiny_cnn.py
- 環境建置：.venv（Python 3.11），訓練/執行依賴已安裝
- FER2013 資料集下載 + 訓練：mini_xception_fp32.onnx（28,709 張圖片，30 epochs）
- RAVDESS 音訊資料集下載 + 訓練：tiny_cnn_audio_fp32.onnx（1,440 個音檔，50 epochs）
- 第二輪訓練：新增 TESS（2,800 筆）+ CREMA-D（7,442 筆），總計 11,682 筆，60 epochs
  - FP32 模型：106KB，int8 模型：37KB
- 驗證通過：模型載入 + dummy inference 正常

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
│   ├── train_audio_tiny_cnn.py       # TensorFlow 訓練音訊模型
│   ├── download_fer2013.py           # 下載 FER2013 資料集
│   ├── download_audio_datasets.py    # 下載音訊資料集
│   ├── run_train_vision.sh           # WSL 背景執行視覺訓練
│   ├── run_train_audio.sh            # WSL 背景執行音訊訓練
│   ├── run_download_audio.sh         # WSL 背景執行音訊下載
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
- 音訊含三個資料集：RAVDESS（1,440）+ TESS（2,800）+ CREMA-D（7,442），總計 11,682 筆
- MFCC 特徵改用真實 mel-scale filterbank + DCT（numpy-only），無額外相依
- WSL 無 webcam，無法在本機執行 app.py（需 Windows 實體機）
- protobuf/mpl-dtypes 版本衝突（mediapipe vs tf2onnx/tensorflow），訓練仍可正常運作

## 待辦事項
- [x] 下載 FER2013 資料集到 data/datasets/fer
- [x] 訓練 mini_xception_fp32.onnx（int8 版本因 ConvInteger 限制改用 FP32）
- [x] 下載 RAVDESS 音訊資料集
- [x] 訓練 tiny_cnn_audio_fp32.onnx（int8 版本因 ConvInteger 限制改用 FP32）
- [x] 驗證：模型載入 + dummy 推論通過
- [x] 找 CREMA-D/TESS 替代下載來源
  - TESS 替代：PaddleAudio 鏡像（2,800 筆）
  - CREMA-D 替代：Zenodo WebDataset（7,442 筆）或 Hugging Face snapshot
- [ ] 在 Windows 本機 Python 執行 app.py 測試 webcam

## 音訊精度改善 TODO（依優先級排列）
- [x] 音量正規化：load_audio_files 中 peak normalization（`audio /= max(|audio|)`）
- [x] 資料擴充：訓練時加入高斯雜訊、音量擾動、SpecAugment
- [x] Learning rate schedule：ReduceLROnPlateau
- [x] Temporal smoothing：推論時對 softmax scores 做 moving average（alpha=0.7）
- [ ] MFCC 13 → 26 + delta/delta-delta 特徵
- [ ] Dropout 0.3 → 0.5 + SpatialDropout1D + L2 regularization
- [ ] 模型輕微擴大：Conv1D 32→64→64 + Dense 128

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

## Windows 開發環境
```powershell
# 第一次：安裝依賴
.\scripts\setup_windows.ps1

# 每次執行
.\.venv\Scripts\Activate.ps1
python app.py --vision-only
```

## 環境狀態（2025-06-30）
- Python 3.11.15 + .venv（位於專案根目錄）
- 訓練依賴已安裝（tensorflow 2.16.2、tf2onnx 1.16.1、onnxruntime 1.17.3）
- 執行依賴已安裝（opencv 4.11.0、mediapipe 0.10.14、sounddevice 0.4.6）
- ⚠️ sounddevice 需要 `libportaudio2` 系統套件：`sudo apt-get install -y libportaudio2`
- ⚠️ TensorFlow 以 CPU 模式執行（WSL 內無 CUDA 驅動）
- ⚠️ protobuf/ml-dtypes 版本衝突（mediapipe vs tf2onnx/tensorflow），訓練仍可正常運作
