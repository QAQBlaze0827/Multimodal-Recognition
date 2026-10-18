# 同照片影像模型比較

這是單人 12 張照片與預期表情的一致率，不是正式測試準確率。
兩模型使用相同原始裁切，各自套用規定的前處理；沒有重拍、重新偵測、調參或上傳照片。
四類只選出中性／開心／難過／生氣後正規化；其他類別不硬合併，完整分類結果另列。

| 預期表情 | 舊模型完整／四類符合 | FERPlus 完整／四類符合 |
|---|---|---|
| neutral | 2/3 / 2/3 | 3/3 / 3/3 |
| happy | 2/3 / 2/3 | 2/3 / 2/3 |
| sad | 2/3 / 2/3 | 1/3 / 1/3 |
| anger | 0/3 / 0/3 | 2/3 / 2/3 |

| 編號 | 預期 | 舊完整 | 舊四類 | 舊排除機率 | FER+完整 | FER+四類 | FER+排除機率 |
|---|---|---|---|---|---|---|---|
| 01 | neutral | sad | sad | 17.6% | neutral | neutral | 0.8% |
| 02 | neutral | neutral | neutral | 18.9% | neutral | neutral | 1.4% |
| 03 | neutral | neutral | neutral | 7.4% | neutral | neutral | 0.8% |
| 04 | happy | happy | happy | 3.1% | happy | happy | 0.9% |
| 05 | happy | neutral | neutral | 17.7% | neutral | neutral | 1.2% |
| 06 | happy | happy | happy | 0.5% | happy | happy | 0.3% |
| 07 | sad | neutral | neutral | 10.6% | sad | sad | 10.1% |
| 08 | sad | sad | sad | 10.1% | neutral | neutral | 6.5% |
| 09 | sad | sad | sad | 15.9% | neutral | neutral | 5.3% |
| 10 | anger | neutral | neutral | 8.7% | anger | anger | 12.3% |
| 11 | anger | neutral | neutral | 17.6% | anger | anger | 13.9% |
| 12 | anger | sad | sad | 19.7% | neutral | neutral | 7.3% |

完整機率、裁切雜湊、模型來源與計時見 summary.json；本報告不複製個人照片。
計時只含 CPU session.run（預熱後），不含讀圖、前處理、相機或人臉偵測，不代表樹莓派效能。
本批曾用於診斷，不能作為最終獨立測試；需新樣本、其他使用者與樹莓派實測再決定部署。
FERPlus 上游模型 README 標 MIT，倉庫 LICENSE 標 Apache-2.0；已保留原始聲明，重新散布前應釐清。
