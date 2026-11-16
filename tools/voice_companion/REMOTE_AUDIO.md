# Pi 麥克風傳到筆電：終端機驗收

這一步只完成以下流程，**還不會讓 Pi 網頁顯示語音對話**：

```text
Pi 的 INMP441 麥克風（既有單一 InputStream）
  └─ 選左聲道
      ├─ 原本的語音情緒辨識（保持原流程）
      └─ 選用 PCM 傳送 → Tailscale → 筆電接收
                                    └─ 重採樣到 16 kHz
                                        → Whisper base → 既有固定回覆 → 終端機
```

Whisper 不在 Pi 上跑；筆電接收程式不開筆電麥克風。沒有 LLM、語音朗讀、新回覆規則或模型重訓。原來的筆電收音網頁模式仍可使用，但本次測試請停止 `transcribe_mic.py`，避免混淆來源。

## 1. 先確認兩邊的程式與環境

兩台都需要取得包含本功能的版本；目前開發分支是 `codex/rpi-voice-companion`。先由使用者檢查、提交、推送，再在 Pi 工作目錄乾淨的前提下更新對應分支。**不要把本文件當成已經替你推送或更新 Pi 的證明，也不用現在合併 main。**

筆電 PowerShell 在專題根目錄執行：

```powershell
git status -sb
Test-Path .\tools\voice_companion\receive_audio.py
$sttPython = "$env:LOCALAPPDATA\emotion-stt-test\.venv\Scripts\python.exe"
& $sttPython -c "import av, faster_whisper; print('PyAV:', av.__version__); print('Import OK')"
```

這台筆電已確認 PyAV 18.1.0 存在，不須重裝。新電腦需先依根目錄 README 準備獨立語音環境與模型；缺 PyAV 時只在此獨立環境安裝 `tools/voice_companion/requirements.txt`。**不要在 Pi 安裝這份 Whisper 依賴。**

Pi 的 SSH 終端機執行：

```bash
cd ~/Multimodal-Recognition
source .venv/bin/activate
git status -sb
python app.py --help
```

應看得到 `--stt-url`。沒有就表示 Pi 尚未取得本次版本，先不要測。先在原本辨識程式視窗按 Ctrl+C，等待停止；同時間只開一份 `app.py`。

沿用 Pi 已驗證的音訊設定：`input_sample_rate: 48000`、`sample_rate: 16000`、`channels: 2`、`channel_select: left`。`device_id` 必須是當次的 INMP441 裝置，之前的 `0` 不是所有機器永久固定。傳輸沿用這些設定，不會自行強制換裝置或聲道。本功能沒有修改 `config/config.yaml`。

## 2. 確認 Tailscale 位址

在筆電與 Pi 各自執行：

```text
tailscale ip -4
tailscale status
```

記下兩台各自的 `100.x.x.x` 位址，確認兩台都在線且屬於你管理的 tailnet。不要把筆電與 Pi 位址填反。

接收器只綁定明確的 Tailscale IPv4（`100.64.0.0/10`）或測試用 `127.0.0.1`，不接受 `0.0.0.0`、LAN IP 或主機名稱。但**符合位址範圍不代表程式已驗證 Tailscale 實際啟用**；仍需確認路由與 tailnet 權限。

此工具使用 HTTP，不是 HTTPS；跨裝置保密依賴實際走 Tailscale 的加密連線。不要用公網轉發、反向代理、Funnel 或路由器 port forwarding 暴露此服務。Tailscale 的加密說明見[官方文件](https://tailscale.com/docs/concepts/tailscale-encryption)，存取規則仍需由你設定，不能以安裝 Tailscale 代替授權檢查。

## 3. 筆電產生本次金鑰

在將要啟動接收程式的同一個 PowerShell，執行：

```powershell
$sttPython = "$env:LOCALAPPDATA\emotion-stt-test\.venv\Scripts\python.exe"
$env:COMPANION_AUDIO_TOKEN = & $sttPython -c "import secrets; print(secrets.token_hex(32))"
$env:COMPANION_AUDIO_TOKEN
```

最後一行只為了讓你把這串 **64 字元金鑰** 貼到自己的 Pi SSH 視窗。不要貼到聊天、文件、Git、截圖或群組。Pi 必須用**同一串**，不要兩邊各產生一串。金鑰只放本次程序環境；關掉視窗後下次重新設定，不用 `setx` 或寫進 config。

在 Pi 將要啟動 `app.py` 的同一個 SSH 視窗輸入：

```bash
read -r -s -p 'Paste the laptop audio token: ' COMPANION_AUDIO_TOKEN
printf '\n'
export COMPANION_AUDIO_TOKEN
```

第一行會等你貼上金鑰並按 Enter，不顯示輸入內容。完成後再執行後兩行。不要將金鑰直接寫進命令列，以免留下 shell 歷史。環境變數不是對本機管理員或同權限惡意程式的安全隔離，請只在可信的兩台裝置使用。

## 4. 筆電先開接收器

回到已設定金鑰的筆電 PowerShell，從專題根目錄執行：

```powershell
$laptopIp = Read-Host 'Laptop Tailscale IPv4'
$piIp = Read-Host 'Pi Tailscale IPv4'
& $sttPython -B .\tools\voice_companion\receive_audio.py --host $laptopIp --allow-peer $piIp --port 8765
```

看到以下訊息才到 Pi 開始傳音：

```text
Model cache: ...emotion-stt-test\models
[remote] Listening on ...:8765; allowed source: ...
[remote] Terminal replies only. No laptop microphone, recording files or web delivery.
```

模型使用已有的多語言 base 快取、CPU INT8、離線模式。缺檔就報錯，不會自動下載。可加 `--model-dir` 指向當初 `download_root` 的快取根目錄；不是 ONNX 模型路徑。

如果 port 已占用，改成另一個未占用的 port，Pi URL 也要一致。不要關閉整台電腦的防火牆；若被擋，只允許 Pi 的 Tailscale IP 存取筆電這個 TCP port，並檢查 tailnet 存取規則。本次程式不會自動修改防火牆、Tailscale 或管理員設定。

## 5. Pi 先只測語音

在已 activate `.venv`、設定金鑰且位於專題根目錄的 SSH 視窗執行：

```bash
read -r -p 'Laptop Tailscale IPv4: ' laptop_ip
python app.py --audio-only --stt-url "http://${laptop_ip}:8765"
```

這個模式不開相機或正式情緒資料庫，方便先單獨驗證收音。應看到 Pi 的 `[stt-send] Laptop audio receiver connected.`，筆電的 `[remote] Source connected: 48000 Hz mono...`。

對 **Pi 麥克風** 說話，說完停頓約一秒。筆電應顯示例如：

```text
[remote] Phrase 001 queued.
[001] 我想玩
  audio=...s | STT=...s
  Dog [play]: 好呀！我們一起玩！
```

實際文字可能有 ASR 錯字，回覆仍是既有有限規則；不是自由聊天。終端機保留原始轉錄，所以仍可能有簡體。先前「網頁轉繁體」只影響網頁顯示，此步不改原文。

## 6. 驗收與停止

依序測試並記錄，不必追求新句型準確度：

1. 說「我想玩」「我想休息一下」「我好難過可不可以陪我一下」「我不想要你陪我」：筆電有轉錄與對應 play／rest／company／quiet。
2. 只有對 Pi 麥克風說話才有收音；此接收程式不開筆電麥克風。
3. 安靜 30 秒觀察是否亂出字。一次通過不代表能保證所有噪音下都無幻覺。
4. Pi 按 Ctrl+C：筆電顯示 Source stopped；若是網路突然中斷，通常最後一包後約 3 秒判斷逾時。未完成、排隊與已失效的轉錄會捨棄，不補播舊句子。
5. 用相同 token 重開 Pi 指令，應能恢復。筆電接收器停止時，Pi 原本語音情緒辨識應繼續，傳輸端顯示 Unavailable。

最後先停 Pi `app.py`，再停筆電接收器。停止接收器本身**不會停止 Pi 麥克風**；只要 Pi 帶 `--stt-url` 的程序仍在跑，就會嘗試重連。若要恢復原功能，去掉 `--stt-url` 後重開即可。

這一步不用開 `backend/app.py` 或瀏覽器；網頁沒有新文字是預期。只要終端機驗收通過，再進行下一階段「筆電回覆 → Pi 後端 → 網頁」。不要把既有 `--companion-local` 改綁 `0.0.0.0`。

## 7. 通過後才與影像一起測

先停上一個 `--audio-only` 程序，沿用同一個已設定環境的 SSH 視窗：

```bash
python app.py --no-display --video-model ferplus --stt-url "http://${laptop_ip}:8765"
```

FERPlus 須已有模型；用原本模型則拿掉 `--video-model ferplus`。不要同時另外開一份 `app.py`。既有情緒網頁仍依原方式啟動 `backend/app.py`，但本步語音文字仍只到筆電終端機。完整模式沿用既有 CSV／SQLite 與資料保留規則，並非新的「不刪歷史資料」模式。

在另一個 Pi SSH 視窗記錄傳送前後相同條件下的溫度、FPS、辨識更新情況及體感延遲：

```bash
vcgencmd measure_temp
vcgencmd get_throttled
```

Pi 不需裝 Whisper 或 PyAV，但新增複製、PCM 轉換與網路工作仍有負擔。48 kHz 單聲道 16-bit PCM 的音訊本體約 96,000 bytes/s（約 0.77 Mbps），尚未包含 HTTP／Tailscale 開銷。使用者曾回報無風扇／散熱片時約 70°C；本次未量測 Pi，不能保證效能或溫度不變，若持續升溫、降頻或卡頓，先停止加載測試並處理散熱。

## 傳輸與隱私限制

- 必須明確加 `--stt-url` 才傳音；沒有此參數就不建立傳送器。**啟用後連靜音與背景聲都會傳送，不受有沒有臉、情緒 VAD 或中性結果控制**，請先取得在場人同意。
- 共用原 InputStream，選好的單聲道在情緒正規化之前分流。新傳輸支援 16/48 kHz，每 0.5 秒一包 signed 16-bit little-endian PCM；網路在另一執行緒，callback 不等待 HTTP。
- Pi 收音佇列最多 10 個 100 ms 塊；逾期超過 1 秒的塊捨棄，丟塊／停頓／網路失敗會讓接收器清掉半句，避免把不連續內容當一句。網路失敗不重送舊音訊；401／403 需修正設定並重啟 Pi 程序。
- 筆電以 PyAV AudioResampler 做有濾波的連續重採樣，再沿用 100 ms 切句、約 0.8 秒停頓、12 秒上限與 Whisper VAD。新 STT 路徑不使用每三點取一點；原情緒辨識的重採樣、VAD、模型與融合沒有順便改動。
- 筆電轉錄佇列最多 3 段，超過 5 秒的待處理／完成結果捨棄。單一有效來源、session 與序號檢查、3 秒來源逾時，不依賴兩台的牆上時鐘同步；慢速接收請求可能延後逾時檢查，不是硬即時保證。
- 拒絕錯誤金鑰、來源 IP、Host、Origin、格式、長度與重複序號。不信任代理來源標頭，也不跟隨重新導向。它是可信 tailnet 的實驗接收器，不是經完整資安稽核、可承受公開網路攻擊的服務。
- 音訊只在兩台記憶體處理；此功能不建立錄音檔、逐字稿檔或 SQLite 對話。終端機會印文字，作業系統記憶體／交換檔、終端機紀錄與使用者截圖不在安全抹除保證內。金鑰外洩請停兩邊並換新金鑰。

PyAV 用法參考[官方 AudioResampler 文件](https://pyav.basswood.io/docs/stable/api/audio.html)；目前本機測試版本為 18.1.0，不代表官網 stable 文件永遠是該版本。

## 故障排查

| 訊息或現象 | 先檢查 |
|---|---|
| 找不到 `--stt-url` 或 `receive_audio.py` | 兩台分支／提交是否包含本次更動 |
| Set COMPANION_AUDIO_TOKEN | 同一個終端機有沒有設定並 export，是否 64 個十六進位字元 |
| Unavailable / ConnectionRefusedError / TimeoutError | 筆電接收器是否已 Listening、兩台 Tailscale 在線、URL／port、防火牆與存取規則 |
| HTTP 401 | 兩邊金鑰不同；改正後重新啟動 Pi 傳送程序 |
| HTTP 403 | `--allow-peer` 是否填 Pi 位址、URL 是否填筆電位址；修正後重啟 |
| HTTP 409 | 是否還有另一個傳送器在跑；停止舊來源後等數秒 |
| 接上但沒有文字 | 對 Pi 麥克風說話並停頓；檢查 device_id、左聲道、音量；不要先改模型 |
| Capture gap / Transcription busy / Stale transcription | 丟塊、網路或運算落後，舊資料已捨棄；先停影像只測 audio-only，不要加大無限佇列 |
| 終端機有文字但網頁沒有 | 此階段尚未接回網頁，屬預期 |
