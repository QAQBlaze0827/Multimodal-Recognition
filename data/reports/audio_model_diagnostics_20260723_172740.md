# Audio Model Diagnostics

- data_dir: `data\datasets\emotiontalk_clean`
- total_samples: 4440

## Summary

| Model | Kind | Accuracy | Macro F1 | Total |
|---|---:|---:|---:|---:|
| current_mfcc | mfcc | 0.2473 | 0.1924 | 4440 |
| emotiontalk_mfcc | mfcc | 0.6435 | 0.6447 | 4440 |
| emotiontalk_mel_cnn | mel | 0.5050 | 0.4752 | 4440 |

## current_mfcc

- path: `models\tiny_cnn_audio_fp32.onnx`
- accuracy: 0.2473
- macro_f1: 0.1924

### Confusion Matrix

| True \ Pred | neutral | happy | sad | anger |
|---|---:|---:|---:|---:|
| neutral | 28 | 347 | 69 | 666 |
| happy | 4 | 502 | 98 | 506 |
| sad | 15 | 418 | 53 | 624 |
| anger | 15 | 552 | 28 | 515 |

### Per Class

| Emotion | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| neutral | 0.4516 | 0.0252 | 0.0478 | 1110 |
| happy | 0.2760 | 0.4523 | 0.3428 | 1110 |
| sad | 0.2137 | 0.0477 | 0.0781 | 1110 |
| anger | 0.2228 | 0.4640 | 0.3011 | 1110 |

### Prediction Counts

```json
{
  "happy": 1819,
  "anger": 2311,
  "sad": 248,
  "neutral": 62
}
```

### First Mistakes

```json
[
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0000.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.5720745921134949
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0001.wav",
    "true": "neutral",
    "pred": "anger",
    "confidence": 0.9931566119194031
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0002.wav",
    "true": "neutral",
    "pred": "sad",
    "confidence": 0.9208062887191772
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0003.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.7861824035644531
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0004.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.7655011415481567
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0005.wav",
    "true": "neutral",
    "pred": "anger",
    "confidence": 0.8726575374603271
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0006.wav",
    "true": "neutral",
    "pred": "anger",
    "confidence": 0.9939026832580566
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0007.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.325774222612381
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0008.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.9774361252784729
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0009.wav",
    "true": "neutral",
    "pred": "anger",
    "confidence": 0.14679723978042603
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0010.wav",
    "true": "neutral",
    "pred": "anger",
    "confidence": 0.028001708909869194
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0011.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 8.817851266940124e-06
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0012.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.6476307511329651
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0013.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.8888645172119141
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0014.wav",
    "true": "neutral",
    "pred": "anger",
    "confidence": 0.4426630437374115
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0015.wav",
    "true": "neutral",
    "pred": "anger",
    "confidence": 0.018044287338852882
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0016.wav",
    "true": "neutral",
    "pred": "anger",
    "confidence": 0.9949630498886108
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0017.wav",
    "true": "neutral",
    "pred": "anger",
    "confidence": 0.9868964552879333
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0018.wav",
    "true": "neutral",
    "pred": "anger",
    "confidence": 0.015751369297504425
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0019.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.727355420589447
  }
]
```

## emotiontalk_mfcc

- path: `models\tiny_cnn_audio_emotiontalk_fp32.onnx`
- accuracy: 0.6435
- macro_f1: 0.6447

### Confusion Matrix

| True \ Pred | neutral | happy | sad | anger |
|---|---:|---:|---:|---:|
| neutral | 697 | 164 | 77 | 172 |
| happy | 234 | 556 | 32 | 288 |
| sad | 131 | 47 | 869 | 63 |
| anger | 175 | 173 | 27 | 735 |

### Per Class

| Emotion | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| neutral | 0.5635 | 0.6279 | 0.5939 | 1110 |
| happy | 0.5915 | 0.5009 | 0.5424 | 1110 |
| sad | 0.8647 | 0.7829 | 0.8217 | 1110 |
| anger | 0.5843 | 0.6622 | 0.6208 | 1110 |

### Prediction Counts

```json
{
  "anger": 1258,
  "happy": 940,
  "neutral": 1237,
  "sad": 1005
}
```

### First Mistakes

```json
[
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0000.wav",
    "true": "neutral",
    "pred": "anger",
    "confidence": 0.4368375837802887
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0001.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.5517613887786865
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0002.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.71591717004776
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0003.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.3759746551513672
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0006.wav",
    "true": "neutral",
    "pred": "anger",
    "confidence": 0.5283643007278442
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0010.wav",
    "true": "neutral",
    "pred": "anger",
    "confidence": 0.4212987720966339
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0011.wav",
    "true": "neutral",
    "pred": "anger",
    "confidence": 0.39775165915489197
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0017.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.3897143006324768
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0018.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.5141794681549072
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0020.wav",
    "true": "neutral",
    "pred": "anger",
    "confidence": 0.361187219619751
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0021.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.39312565326690674
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0024.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.7844412326812744
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0025.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.5065213441848755
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0026.wav",
    "true": "neutral",
    "pred": "anger",
    "confidence": 0.4511638283729553
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0033.wav",
    "true": "neutral",
    "pred": "anger",
    "confidence": 0.39798665046691895
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0036.wav",
    "true": "neutral",
    "pred": "anger",
    "confidence": 0.35457760095596313
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0038.wav",
    "true": "neutral",
    "pred": "sad",
    "confidence": 0.7545977234840393
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0040.wav",
    "true": "neutral",
    "pred": "anger",
    "confidence": 0.4537905156612396
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0045.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.4772959053516388
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0046.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.5396780371665955
  }
]
```

## emotiontalk_mel_cnn

- path: `models\audio_emotiontalk_mel_cnn_fp32.onnx`
- accuracy: 0.5050
- macro_f1: 0.4752

### Confusion Matrix

| True \ Pred | neutral | happy | sad | anger |
|---|---:|---:|---:|---:|
| neutral | 724 | 239 | 141 | 6 |
| happy | 326 | 694 | 83 | 7 |
| sad | 254 | 156 | 694 | 6 |
| anger | 247 | 687 | 46 | 130 |

### Per Class

| Emotion | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| neutral | 0.4668 | 0.6523 | 0.5442 | 1110 |
| happy | 0.3908 | 0.6252 | 0.4809 | 1110 |
| sad | 0.7199 | 0.6252 | 0.6692 | 1110 |
| anger | 0.8725 | 0.1171 | 0.2065 | 1110 |

### Prediction Counts

```json
{
  "sad": 964,
  "happy": 1776,
  "neutral": 1551,
  "anger": 149
}
```

### First Mistakes

```json
[
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0000.wav",
    "true": "neutral",
    "pred": "sad",
    "confidence": 0.3348326086997986
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0001.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.34373950958251953
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0002.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.48570892214775085
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0003.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.6174219250679016
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0004.wav",
    "true": "neutral",
    "pred": "sad",
    "confidence": 0.46580740809440613
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0006.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.3906958997249603
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0016.wav",
    "true": "neutral",
    "pred": "sad",
    "confidence": 0.5509030818939209
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0017.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.5391789078712463
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0020.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.36652541160583496
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0021.wav",
    "true": "neutral",
    "pred": "sad",
    "confidence": 0.3469863533973694
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0022.wav",
    "true": "neutral",
    "pred": "sad",
    "confidence": 0.41956835985183716
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0023.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.443036824464798
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0026.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.40227219462394714
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0030.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.3690144419670105
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0031.wav",
    "true": "neutral",
    "pred": "sad",
    "confidence": 0.4589948356151581
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0036.wav",
    "true": "neutral",
    "pred": "sad",
    "confidence": 0.785243570804596
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0038.wav",
    "true": "neutral",
    "pred": "sad",
    "confidence": 0.851748526096344
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0042.wav",
    "true": "neutral",
    "pred": "sad",
    "confidence": 0.5804482102394104
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0046.wav",
    "true": "neutral",
    "pred": "happy",
    "confidence": 0.4502464234828949
  },
  {
    "file": "data\\datasets\\emotiontalk_clean\\neutral\\emotiontalk_clean_neutral_0047.wav",
    "true": "neutral",
    "pred": "sad",
    "confidence": 0.4380187690258026
  }
]
```
