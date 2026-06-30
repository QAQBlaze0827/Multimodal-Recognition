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
- RAVDESS + TESS 音訊訓練：tiny_cnn_audio_fp32.onnx（4,240 樣本，80 epochs），**val_acc 76.1%**
- 音訊精度改善（Phase 1+2）：peak normalization、data augmentation（noise/volume/SpecAugment）、ReduceLROnPlateau、temporal smoothing（alpha=0.7）
- **CREMA-D 測試判定為雜訊**：含入後 val_acc 從 76% 暴跌至 53%，已排除
- **Phase 3 測試判定無效**：N_MFCC=26+delta、Conv1D 64→64、Dense 128、Dropout 0.5+L2 等變動均導致準確度下降
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
- 音訊最佳模型使用 RAVDESS（1,440）+ TESS（2,800）= 4,240 樣本（val_acc 76.1%）
- CREMA-D（7,442 筆）判定為雜訊，含入後 val_acc 降至 53%
- MFCC 特徵改用真實 mel-scale filterbank + DCT（numpy-only），無額外相依
- WSL 無 webcam，無法在本機執行 app.py（需 Windows 實體機）
- protobuf/mpl-dtypes 版本衝突（mediapipe vs tf2onnx/tensorflow），訓練仍可正常運作

## 待辦事項
- [x] 下載 FER2013 資料集到 data/datasets/fer
- [x] 訓練 mini_xception_fp32.onnx（int8 版本因 ConvInteger 限制改用 FP32）
- [x] 下載 RAVDESS + TESS 音訊資料集
- [x] 訓練 tiny_cnn_audio_fp32.onnx（Phase 2：peak norm + augmentation + LR schedule，val_acc 76.1%）
- [x] 驗證：模型載入 + dummy 推論通過
- [x] 下載 CREMA-D 並測試 → 判定為雜訊，含入後從 76%→53%
- [x] Phase 3 測試（MFCC 26、delta、bigger model、dropout 0.5+L2）→ 無效，全數 < 54%
- [ ] 在 Windows 本機 Python 執行 app.py 測試 webcam

## 音訊精度改善結果
| Phase | 變更 | 訓練資料 | val_acc |
|-------|------|----------|---------|
| 初始 | 原始 tiny_cnn | ravdess 1,440 | 62% |
| Phase 1 | temporal smoothing (推論) | ravdess 1,440 | inference 改善 |
| Phase 2 | peak norm + augment + ReduceLROnPlateau | ravdess+tess 4,240 | **76.1%** |
| Phase 2 | 同上 | ravdess+tess+crema_d 11,682 | 53% |
| Phase 3 | N_MFCC=26 + delta + Conv1D 64→64 + Dense 128 + Dropout 0.5+L2 | ravdess+tess 4,240 | 54% |
| Phase 3 | N_MFCC=26 + Conv1D 64→64 + Dense 128 + Dropout 0.3 | ravdess+tess 4,240 | 52% |
| Phase 3 | N_MFCC=13 + Conv1D 32→64 + Dense 64 + Dropout 0.3 | ravdess+tess 4,240 | 54% |

**結論**：Phase 2 為最佳配置（N_MFCC=13, Conv1D 32→64, Dense 64, Dropout 0.3, augment=True），另外 CREMA-D 資料集與 RAVDESS/TESS 不相容導致 val_acc 雪崩，已排除。Phase 3 所有嘗試均無正向效果。

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

## 環境狀態（2026-06-30）
- Python 3.11.15 + .venv（位於專案根目錄）
- 訓練依賴已安裝（tensorflow 2.16.2、tf2onnx 1.16.1、onnxruntime 1.17.3）
- 執行依賴已安裝（opencv 4.11.0、mediapipe 0.10.14、sounddevice 0.4.6）
- ⚠️ sounddevice 需要 `libportaudio2` 系統套件：`sudo apt-get install -y libportaudio2`
- ⚠️ TensorFlow 以 CPU 模式執行（WSL 內無 CUDA 驅動）
- ⚠️ protobuf/ml-dtypes 版本衝突（mediapipe vs tf2onnx/tensorflow），訓練仍可正常運作
