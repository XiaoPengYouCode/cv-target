# Target Detector

本项目是一个箭靶检测，旨在实现高效、准确的目标识别与定位。

## 识别效果

<p align="center">
    <img src="output/1.png", height="300pt"> </img>
    <img src="output/2.png", height="300pt"> </img>
</p>

## 目录结构

```
target_detector/
├── data/           # 数据集与样本
├── models/         # 检测模型
├── utils/          # 工具函数
├── utils/          # 工具函数
├── requirements.txt
├── src
|   ├── circle_detector.py # 色环识别
|   ├── corner_detect.py   # 角落识别(效果不好)
|   ├── corner_detector.py # 角落识别(效果不好)
|   ├── moblie_lsd.py      # 神经网络直线识别(效果一般)
|   └── sold2.py           # 神经网络直线识别(效果一般)
└── README.md
```


