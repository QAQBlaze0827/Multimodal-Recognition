# Audio Model Diagnostics

- data_dir: `data\datasets\emotiontalk_clean`
- total_samples: 8

## Summary

| Model | Kind | Accuracy | Macro F1 | Total |
|---|---:|---:|---:|---:|
| smoke_mfcc | mfcc | 0.0000 | 0.0000 | 8 |
| smoke_mel | mel | 0.1250 | 0.0833 | 8 |

## smoke_mfcc

- path: `models\tiny_cnn_audio_fp32.onnx`
- accuracy: 0.0000
- macro_f1: 0.0000

### Confusion Matrix

| True \ Pred | neutral | happy | sad | anger |
|---|---:|---:|---:|---:|
| neutral | 0 | 1 | 1 | 0 |
| happy | 0 | 0 | 0 | 2 |
| sad | 0 | 1 | 0 | 1 |
| anger | 0 | 2 | 0 | 0 |

### Per Class

| Emotion | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| neutral | 0.0000 | 0.0000 | 0.0000 | 2 |
| happy | 0.0000 | 0.0000 | 0.0000 | 2 |
| sad | 0.0000 | 0.0000 | 0.0000 | 2 |
| anger | 0.0000 | 0.0000 | 0.0000 | 2 |

### Prediction Counts

```json
{
  "sad": 1,
  "happy": 4,
  "anger": 3
}
```

### First Mistakes

```json
[
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0098.wav",
    "true": "neutral",
    "pred": "sad",
    "confidence": 0.6963297724723816
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0859.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.9982901215553284
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\happy\\emotiontalk_clean_happy_0480.wav",
    "true": "happy",
    "pred": "anger",
    "confidence": 0.9502944946289062
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\happy\\emotiontalk_clean_happy_0486.wav",
    "true": "happy",
    "pred": "anger",
    "confidence": 0.113870769739151
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\sad\\emotiontalk_clean_sad_0095.wav",
    "true": "sad",
    "pred": "happy",
    "confidence": 3.3145804536616197e-06
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\sad\\emotiontalk_clean_sad_0774.wav",
    "true": "sad",
    "pred": "anger",
    "confidence": 0.45426061749458313
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\anger\\emotiontalk_clean_anger_0104.wav",
    "true": "anger",
    "pred": "happy",
    "confidence": 0.9926005005836487
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\anger\\emotiontalk_clean_anger_0584.wav",
    "true": "anger",
    "pred": "happy",
    "confidence": 0.8124580383300781
  }
]
```

## smoke_mel

- path: `models\audio_emotiontalk_mel_cnn_fp32.onnx`
- accuracy: 0.1250
- macro_f1: 0.0833

### Confusion Matrix

| True \ Pred | neutral | happy | sad | anger |
|---|---:|---:|---:|---:|
| neutral | 1 | 1 | 0 | 0 |
| happy | 2 | 0 | 0 | 0 |
| sad | 1 | 1 | 0 | 0 |
| anger | 0 | 2 | 0 | 0 |

### Per Class

| Emotion | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| neutral | 0.2500 | 0.5000 | 0.3333 | 2 |
| happy | 0.0000 | 0.0000 | 0.0000 | 2 |
| sad | 0.0000 | 0.0000 | 0.0000 | 2 |
| anger | 0.0000 | 0.0000 | 0.0000 | 2 |

### Prediction Counts

```json
{
  "happy": 4,
  "neutral": 4
}
```

### First Mistakes

```json
[
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0098.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.4696851074695587
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\happy\\emotiontalk_clean_happy_0480.wav",
    "true": "happy",
    "pred": "neutral",
    "confidence": 0.40585002303123474
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\happy\\emotiontalk_clean_happy_0486.wav",
    "true": "happy",
    "pred": "neutral",
    "confidence": 0.5104573369026184
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\sad\\emotiontalk_clean_sad_0095.wav",
    "true": "sad",
    "pred": "neutral",
    "confidence": 0.3668247163295746
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\sad\\emotiontalk_clean_sad_0774.wav",
    "true": "sad",
    "pred": "happy",
    "confidence": 0.4214034974575043
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\anger\\emotiontalk_clean_anger_0104.wav",
    "true": "anger",
    "pred": "happy",
    "confidence": 0.3724346458911896
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\anger\\emotiontalk_clean_anger_0584.wav",
    "true": "anger",
    "pred": "happy",
    "confidence": 0.42446690797805786
  }
]
```
